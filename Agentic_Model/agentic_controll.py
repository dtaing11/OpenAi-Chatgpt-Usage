"""
agentic_controller.py

This is just a  reference implementation of a minimal agentic controller loop with planning.

What is in this file:
- Tool catalog with strict JSON Schemas (prevents tool/field hallucination)
- Argument validation + one-shot LLM repair when validation fails
- Budgets: max steps, tokens, and cost with simple accounting
- Loop detection to stop repeated ineffective actions
- Rolling history summarization to keep context small
- Planner that chooses the next action (tool or 'answer')
- Executor stub that simulates two tools (replace with your backends)
- Final synthesis step that composes the final answer

Notes on LLM Provider:
- This version uses OpenAI's SDK for planning, repair, summarization, and synthesis.
- To swap providers, replace client calls in the following functions:
  - repair_args_with_llm()
  - update_summary()
  - plan_next_action()
  - synthesize_answer()

Run:
  python agentic_controller.py
"""

# Imports & Setup ------------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from jsonschema import Draft202012Validator
from dotenv import load_dotenv
from openai import OpenAI
import re
import hashlib
import json
import os
import time
import requests
import argparse

# Chroma (env-configurable, manual embeddings style like your snippet)
import chromadb  
from chromadb.config import Settings  

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENFOODFACTS_PRODUCT = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"  
OPENFOODFACTS_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"

WEATHER_CODE_TEXT = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle (light)",
    57: "freezing drizzle (dense)",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain (light)",
    67: "freezing rain (heavy)",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers (light)",
    81: "rain showers (moderate)",
    82: "rain showers (violent)",
    85: "snow showers (light)",
    86: "snow showers (heavy)",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}
# Tool Catalog ---------------------------------------------------------------------------------------------------------
# the tool catalog provides precise JSON Schemas for arguments. This helps to prevent the model from inventing fields or
# tools, and helps validation & auto-repair.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Tool 1: a weather tool
    # STUDENT_COMPLETE --> You need to replace this with the correct one for the real weather tool call
    "weather.get_current": {
        "type": "object",
        "description": "Get the current weather",
        "properties": {
            "city":  {"type": "string", "minLength": 1},
            "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}
        },
        "required": ["city"],
        "additionalProperties": False
    },
    # Tool 2: knowledge-base search tool
    "kb.search": {
        "type": "object",
        "description": "search a knowledge base for information",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "k":     {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
        },
        "required": ["query"],
        "additionalProperties": False
    }
    # STUDENT_COMPLETE --> You need to add a new tool schema for your custom tool
}

#  Add a third tool for OpenFoodFacts ----------------------------------------
TOOL_SCHEMAS["food.lookup"] = { 
    "type": "object",
    "description": "Lookup OpenFoodFacts by barcode (preferred) or search by keywords.",
    "properties": {
        "barcode": {"type": "string", "minLength": 5},
        "query":   {"type": "string", "minLength": 2},
        "k":       {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
    },
    "anyOf": [
        {"required": ["barcode"]},
        {"required": ["query"]}
    ],
    "additionalProperties": False
}
# -------------------------------------------------------------------------------

# Optional hints: rough latency/cost so planner can reason about budgets. I recommend replacing the default values
# with estimates that are accurate on measurements.
TOOL_HINTS: Dict[str, Dict[str, Any]] = {
    "weather.get_current": {"avg_ms": 400, "avg_tokens": 50},
    "kb.search":           {"avg_ms": 120, "avg_tokens": 30},
    "food.lookup":         {"avg_ms": 350, "avg_tokens": 30},  
}

# Controller State -----------------------------------------------------------------------------------------------------
@dataclass
class StepRecord:
    """Telemetry for each executed step (action)."""
    action: str                   
    args: Dict[str, Any]         
    ok: bool                    
    latency_ms: int              
    info: Dict[str, Any] = field(default_factory=dict)  

@dataclass
class ControllerState:
    """Mutable task state carried through the controller loop."""
    goal: str                     
    history_summary: str = ""    
    tool_trace: List[StepRecord] = field(default_factory=list)
    tokens_used: int = 0          
    cost_cents: float = 0.0      
    steps_taken: int = 0          
    last_observation: str = ""    
    done: bool = False            


# Budgets & Accounting -------------------------------------------------------------------------------------------------
# Hard ceilings to avoid runaway cost
MAX_STEPS = 8
MAX_TOKENS = 20_000
MAX_COST_CENTS = 75.0


def within_budget(s: ControllerState) -> bool:
    """
    Check hard ceilings for steps, tokens, and cost.

    :param s: instance of ControllerState
    :return: True if still within budget, false if over-budget
    """
    return (
        s.steps_taken < MAX_STEPS and
        s.tokens_used < MAX_TOKENS and
        s.cost_cents < MAX_COST_CENTS
    )


def record_usage(s: ControllerState, usage) -> None:
    """
    Update token/cost counters using the response.usage object if available.
    This is a simplified accounting model for demonstration purposes.

    :param s: instance of ControllerState object
    :param usage: a response.usage object from OpenAI model response
    :return: None
    """
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    total = pt + ct
    s.tokens_used += total
    # gpt-5-mini is $0.25/million token
    s.cost_cents += total * 0.25/1E4


# Loop Detection -------------------------------------------------------------------------------------------------------
# Detect repeated (action, args) to avoid "stuck" ReAct oscillations.
LAST_ACTIONS = deque(maxlen=3)

# Deduplicate repeated identical tool calls (prevents planner loops from doing work)
ACTION_CACHE: Dict[str, Dict[str, Any]] = {}  
def _act_key(action: str, args: Dict[str, Any]) -> str: 
    return hashlib.sha256(json.dumps({"a": action, "x": args}, sort_keys=True).encode()).hexdigest()

def fingerprint_action(action: str, args: Dict[str, Any]) -> str:
    """
    Hash the tool call pair (action,args) to compare recent moves.

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: A sha256 hash
    """
    blob = json.dumps({"a": action, "x": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def looks_stuck(action: str, args: Dict[str, Any]) -> bool:
    """
    Return True if the last N actions are identical (loop).

    :param action: the action the model selected
    :param args: the arguments the model selected for the action
    :return: True if this action is the same as the N actions, False otherwise
    """
    fp = fingerprint_action(action, args)
    LAST_ACTIONS.append(fp)
    return (
        len(LAST_ACTIONS) == LAST_ACTIONS.maxlen and
        len(set(LAST_ACTIONS)) == 1
    )


# Arg Validation & Repair ----------------------------------------------------------------------------------------------

def validate_args(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate args against the JSON Schema for the given tool. Return (ok, error_message). Error is concise for LLM
    repair prompt.

    :param tool_name: the name of the tool that the model selected
    :param args: the arguments the model selected for that tool
    :return: (True, None) if validates, (False, error message) if not
    """
    schema = TOOL_SCHEMAS[tool_name]
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
    if errors:
        e = errors[0]
        path = "/".join([str(p) for p in e.path]) or "(root)"
        return False, f"Invalid arguments at {path}: {e.message}"
    return True, None


def repair_args_with_llm(tool_name: str, bad_args: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """
    Ask the LLM to fix only the invalid parts to satisfy the JSON Schema.
    We enforce JSON-only output and re-validate after repair.

    :param tool_name: name of the selected tool
    :param bad_args: dictionary of bad arguments provided by the model
    :param error_msg: the error message provided by the validator
    :return: corrected (hopefully) arguments
    """
    schema = TOOL_SCHEMAS[tool_name]
    dev = (
        "You fix JSON arguments to match a JSON Schema. "
        "Return VALID JSON only—no prose, no code fences, no comments."
    )
    user = json.dumps({
        "tool_name": tool_name,
        "schema": schema,
        "invalid_args": bad_args,
        "validator_error": error_msg
    })
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},  # force JSON
        messages=[
            {"role": "developer", "content": dev},
            {"role": "user", "content": user}
        ]
    )
    return json.loads(resp.choices[0].message.content)


# History Summarization ------------------------------------------------------------------------------------------------

def update_summary(state: ControllerState, new_evidence: str) -> None:
    """
    Compress the prior summary + new evidence into a short rolling memory.
    Keeps context small but preserves key facts and decisions.

    :param state: instance of ControllerState
    :param new_evidence: this would be the response from the tool call
    :return:
    """
    sys = "Compress facts and decisions into ≤120 tokens. Keep IDs and key numbers. Do not include anything that is " \
          "unnecessary, only things that are strictly useful for the goal."
    user = json.dumps({
        "prior_summary": state.history_summary,
        "new_evidence": new_evidence
    })
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    content = resp.choices[0].message.content.strip()
    state.history_summary = content
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)


# Planner --------------------------------------------------------------------------------------------------------------

def plan_next_action(state: ControllerState) -> Tuple[str, Dict[str, Any], str]:
    """
    Ask the LLM to pick ONE next action:
      - a known tool from TOOL_SCHEMAS with arguments, OR
      - the literal string 'answer' when it can synthesize the final answer.

    :param state: instance of ControllerState
    :return: (action, args, rationale)
    """
    # Pass the schema to the model. We also pass tool latency/token count for budget control and example to help the
    # model choose.
    tool_specs = []
    for name, schema in TOOL_SCHEMAS.items():
        spec = {
            "name": name,
            "schema": schema,  # including the full JSON Schema
            "budget_hint": {
                "avg_ms": TOOL_HINTS[name]["avg_ms"],
                "avg_tokens": TOOL_HINTS[name]["avg_tokens"],
            },
            # (Optional Few-shot prompting approach) keep examples tiny and schema-compliant
            # STUDENT_COMPLETE --> You may need to change this to be in line with your custom weather tool implementation
            "examples": {
                "weather.get_current": [
                    {"city": "Paris", "units": "metric"},
                    {"city": "New York"}  # units defaults via schema
                ],
                "kb.search": [
                    {"query": "VPN policy for contractors", "k": 3}
                ],
                # show usage examples for the OpenFoodFacts tool
                "food.lookup": [
                    {"barcode": "3017620429484"},
                    {"query": "Chobani Greek Yogurt strawberry", "k": 5}
                ]
            }.get(name, [])
        }
        tool_specs.append(spec)

    dev = (
        "You are a planner. Choose ONE next action toward the goal. Do not call actions towards "
        "information already contained in the history summary provided below.\n"
        "Use ONLY tools from `tool_catalog` OR choose 'answer' if you can respond now.\n"
        "You can only answer with information provided by the tools."
        "When using a tool, produce arguments that VALIDATE against its JSON Schema.\n"
        "Allowed output format (JSON only):\n"
        '{"action":"<tool_name|answer>","args":{...}, "rationale":"<brief reason>"}'
    )

    user = json.dumps({
        "goal": state.goal,
        "budget": {
            "steps_remaining": MAX_STEPS - state.steps_taken,
            "tokens_remaining": MAX_TOKENS - state.tokens_used,
            "cost_cents_remaining": round(MAX_COST_CENTS - state.cost_cents, 2)
        },
        "history_summary": state.history_summary,
        "tool_catalog": tool_specs,
        "last_observation": state.last_observation
    })

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},  # we could use strict mode if we wanted to
        messages=[{"role": "developer", "content": dev},
                  {"role": "user", "content": user}]
    )
    obj = json.loads(resp.choices[0].message.content)
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return obj["action"], obj.get("args", {}), obj.get("rationale", "")


# Executor -------------------------------------------------------------------------------------------------------------

def geocode_city(city: str) -> Optional[Dict[str, Any]]:
    """
    Robust geocoder that:
      - accepts 'lat,lon' directly with no external calls,
      - queries Open-Meteo Geocoding with the string *as-is* (no appends),
      - if not found, falls back to OSM Nominatim with the string *as-is*,
      - returns {'name','lat','lon','country'} or None.
    No hardcoded locations, states, or countries are introduced.
    """
    s = city.strip()

    # 1) If user gave coordinates like "32.78,-79.93", use them directly.
    m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$', s)
    if m:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return {"name": f"{lat},{lon}", "lat": lat, "lon": lon, "country": None}

    # 2) Try Open-Meteo Geocoder exactly as provided (no string modifications).
    try:
        r = requests.get(
            OPEN_METEO_GEOCODE,
            params={"name": s, "count": 5, "language": "en", "format": "json"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if results:
            # choose the most likely by population if present, otherwise the first
            results.sort(key=lambda x: x.get("population") or 0, reverse=True)
            top = results[0]
            return {
                "name": top.get("name") or s,
                "lat": float(top["latitude"]),
                "lon": float(top["longitude"]),
                "country": top.get("country"),
            }
    except Exception:
        pass 

    # 3) Fallback: OSM Nominatim, exact string (no edits). Nominatim requires a User-Agent.
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": s, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "agentic-controller/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        arr = r.json() or []
        if arr:
            top = arr[0]
            name = top.get("display_name") or s
            lat = float(top["lat"])
            lon = float(top["lon"])
            # country is optional in this API; keep None if not parsable
            country = None
            # if address blob exists, try to extract country (still not hardcoded)
            if isinstance(top.get("address"), dict):
                country = top["address"].get("country")
            return {"name": name, "lat": lat, "lon": lon, "country": country}
    except Exception:
        pass

    # Nothing found
    return None



def fetch_current_weather(lat: float, lon: float, units: str) -> Dict[str, Any]:
    """Call Open-Meteo current weather and normalize to a stable schema."""
    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"

    r = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": temp_unit,
            "windspeed_unit": wind_unit,
            "timezone": "auto",
        },
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()

    # v3 'current' block; fallback to v2 'current_weather'
    cur = data.get("current", {})
    if not cur and "current_weather" in data:
        cw = data["current_weather"]
        cur = {
            "temperature_2m": cw.get("temperature"),
            "wind_speed_10m": cw.get("windspeed"),
            "weather_code": cw.get("weathercode"),
        }

    code = cur.get("weather_code", cur.get("weathercode"))
    conditions = WEATHER_CODE_TEXT.get(int(code) if code is not None else -1, "unknown")

    return {
        "temperature": cur.get("temperature_2m"),
        "apparent_temperature": cur.get("apparent_temperature"),
        "humidity": cur.get("relative_humidity_2m"),
        "wind_speed": cur.get("wind_speed_10m"),
        "weather_code": code,
        "conditions": conditions,
        "observed_at": data.get("current_units", {}).get("time") or data.get("timezone", ""),
        "units": {
            "temperature": "°C" if units == "metric" else "°F",
            "wind_speed": "km/h" if units == "metric" else "mph",
        },
        "raw": cur,
    }

# Chroma RAG helpers (same approach as your snippet)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")  
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "squad_chunks")  
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  

# Keep your variable names for SQuAD compatibility
SquaAd = os.getenv("SQUAD_DEV_FILE", "dev-v2.0.json")  
FIVEHUNDREDQUESTION = os.getenv("FIVEHUNDREDQUESTION", "500-Question.jsonl") 

def _get_chroma_client():  
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

def _get_collection(client):  
    return client.get_or_create_collection(CHROMA_COLLECTION)

def _oa_embed(texts: List[str]) -> List[List[float]]:  
    oa = OpenAI()
    out: List[List[float]] = []
    B = 512
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        res = oa.embeddings.create(model=EMBED_MODEL, input=batch).data
        out.extend([e.embedding for e in res])
    return out

def seed_chroma_if_needed(col):  
    if col.count() > 0:
        return
    if not os.path.exists(SquaAd):
        print("[chroma] dev-v2.0.json not found; skipping auto-seed")
        return
    with open(SquaAd, "r", encoding="utf-8") as f:
        dev = json.load(f)
    docs, ids = [], []
    idx = 0
    for art in dev.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = " ".join(para.get("context","").split())
            size, overlap = 420, 60
            for start in range(0, len(ctx), size - overlap):
                chunk = ctx[start:start+size].strip()
                if len(chunk) < 100:
                    continue
                docs.append(chunk); ids.append(f"p{idx}"); idx += 1
    if not docs:
        return
    embeds = _oa_embed(docs)
    col.add(documents=docs, embeddings=embeds, ids=ids)
    print(f"[chroma] collection ready: {CHROMA_COLLECTION} (count={col.count()})")

def _embed_query(q: str) -> List[float]:  
    oa = OpenAI()
    e = oa.embeddings.create(model=EMBED_MODEL, input=q)
    return e.data[0].embedding

def get_sources(col, question: str, k: int = 5) -> List[str]: 
    qvec = _embed_query(question)
    res = col.query(query_embeddings=[qvec], n_results=k, include=["documents","ids","distances","metadatas"])
    docs = (res.get("documents") or [[]])[0]
    return docs

def extractQuestion(dev):
    count = 0
    with open(FIVEHUNDREDQUESTION, "w", encoding="utf-8") as w:
        for art in dev["data"]:
            for para in art["paragraphs"]:
                for qa in para["qas"]:
                    if qa.get("is_impossible", False):
                        continue
                    answers = qa.get("answers", [])
                    if not answers:
                        continue
                    gold = answers[0]["text"]
                    item = {"id": qa["id"], "question": qa["question"].strip(), "gold": gold.strip()}
                    w.write(json.dumps(item) + "\n")
                    count += 1
                    if count >= 500:
                        return

# OpenFoodFacts helpers ------------------------------------------------------                  

def _normalize_off_product(p: Dict[str, Any]) -> Dict[str, Any]: 
    if not p:
        return {}
    nutr = p.get("nutriments", {}) or {}
    return {
        "product_name": p.get("product_name") or p.get("product_name_en"),
        "brands": p.get("brands"),
        "quantity": p.get("quantity"),
        "categories": p.get("categories"),
        "labels": p.get("labels"),
        "image_url": p.get("image_front_url") or p.get("image_url"),
        "nutriments": {
            "energy_kcal_100g": nutr.get("energy-kcal_100g"),
            "fat_100g": nutr.get("fat_100g"),
            "saturated_fat_100g": nutr.get("saturated-fat_100g"),
            "carbohydrates_100g": nutr.get("carbohydrates_100g"),
            "sugars_100g": nutr.get("sugars_100g"),
            "fiber_100g": nutr.get("fiber_100g"),
            "proteins_100g": nutr.get("proteins_100g"),
            "salt_100g": nutr.get("salt_100g"),
            "sodium_100g": nutr.get("sodium_100g"),
        },
        "barcode": p.get("code"),
        "countries": p.get("countries"),
        "quantity_unit": p.get("serving_size"),
    }

def _off_lookup_by_barcode(barcode: str) -> Tuple[bool, str, Dict[str, Any]]: 
    r = requests.get(OPENFOODFACTS_PRODUCT.format(barcode=barcode), timeout=10)
    r.raise_for_status()
    data = r.json()
    status = data.get("status")
    if status != 1:
        return False, f"Barcode {barcode} not found.", {}
    prod = _normalize_off_product(data.get("product") or {})
    obs = f"Found product '{prod.get('product_name')}' (barcode {barcode})."
    return True, obs, {"product": prod}

def _off_search(query: str, k: int) -> Tuple[bool, str, Dict[str, Any]]: 
    params = {
        "search_terms": query,
        "search_simple": 1,
        "json": 1,
        "action": "process",
        "page_size": k
    }
    r = requests.get(OPENFOODFACTS_SEARCH, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    prods = data.get("products") or []
    results = []
    for p in prods:
        results.append(_normalize_off_product(p))
    if not results:
        return True, "No products found.", {"results": []}
    top = results[0]
    obs = f"Retrieved {len(results)} products (top: {top.get('product_name')})."
    return True, obs, {"results": results}
# -------------------------------------------------------------------------------

def execute_action(action: str, args: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Execute the selected action with validation, repair, retries, and error handling.

    Replace the stubbed tool bodies with your real backends (APIs, DBs, etc.). Right now, they are just dummie entries.

    :param action: tool selected by the model
    :param args: arguments selected by the model
    :return: (ok, observation_text, normalized_payload, latency_ms)
    """

    t0 = time.time()

    # 'answer' is a "virtual tool" signaling that we should synthesize the final answer.
    if action == "answer":
        obs = "Ready to synthesize final answer from working memory and evidence."
        return True, obs, {}, int((time.time() - t0) * 1000)

    # Guard: only call known tools from the catalog.
    if action not in TOOL_SCHEMAS:
        return False, f"Unknown tool: {action}", {}, int((time.time() - t0) * 1000)
    # 1) Validate arguments against schema.
    ok, msg = validate_args(action, args)
    if not ok:
        # 2) One-shot repair via LLM; re-validate.
        fixed = repair_args_with_llm(action, args, msg)
        ok2, msg2 = validate_args(action, fixed)
        if not ok2:
            return False, f"Arg repair failed: {msg2}", {}, int((time.time() - t0) * 1000)
        args = fixed

    # 3) Execute the tool with basic retry on transient failures (e.g., timeouts).
    try:
        # Replace these with your real integrations
        if action == "weather.get_current":
            city = args["city"].strip()
            units = args.get("units", "metric")

            geo = geocode_city(city)
            if not geo:
                return False, f"Could not find coordinates for '{city}'.", {}, int((time.time() - t0) * 1000)

            wx = fetch_current_weather(geo["lat"], geo["lon"], units)
            payload = {
                "city": f"{geo['name']}, {geo.get('country') or ''}".strip().strip(","),
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "units": units,
                "temperature": wx["temperature"],
                "apparent_temperature": wx["apparent_temperature"],
                "humidity": wx["humidity"],
                "wind_speed": wx["wind_speed"],
                "conditions": wx["conditions"],
                "weather_code": wx["weather_code"],
                "observed_at": wx["observed_at"],
            }

            temp_unit = "°C" if units == "metric" else "°F"
            wind_unit = "km/h" if units == "metric" else "mph"
            obs = (
                f"{payload['city']}: {payload['temperature']}{temp_unit}, {payload['conditions']}; "
                f"wind {payload['wind_speed']} {wind_unit}"
            )
            return True, obs, payload, int((time.time() - t0) * 1000)

        elif action == "kb.search":
            # Real vector search with local Chroma using manual OpenAI embeddings (style from your snippet)
            q = args["query"].strip()
            k = int(args.get("k", 5))

            client = _get_chroma_client()
            col = _get_collection(client)
            # Seed if empty (quietly skips if SQuAD file not present)
            seed_chroma_if_needed(col)

            # Use richer include to compute similarity score and keep metadata if available
            qvec = _embed_query(q)
            res = col.query(query_embeddings=[qvec], n_results=k, include=["ids","documents","distances","metadatas"])
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]

            results = []
            for i, doc_id in enumerate(ids):
                dist = dists[i] if i < len(dists) else None
                score = (1.0 - float(dist)) if dist is not None else None  # cosine similarity
                snippet = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                results.append({
                    "doc_id": doc_id,
                    "snippet": snippet,
                    "score": None if score is None else round(score, 4),
                    "source": meta.get("source"),
                    "metadata": meta
                })

            if not results:
                obs = "No KB results."
                return True, obs, {"results": []}, int((time.time() - t0) * 1000)

            top = results[0]
            top_score = top.get("score")
            score_str = f"{top_score:.3f}" if isinstance(top_score, (int, float)) else "n/a"
            obs = f"Retrieved {len(results)} snippets (top similarity {score_str})."
            return True, obs, {"results": results}, int((time.time() - t0) * 1000)

        elif action == "food.lookup":  #OpenFoodFacts tool executor
            barcode = args.get("barcode")
            query = args.get("query")
            k = int(args.get("k", 5))

            # Duplicate suppression (cache)
            key = _act_key(action, args)
            if key in ACTION_CACHE:
                cached = ACTION_CACHE[key]
                obs_cached = cached.get("_obs_hint") or "Using cached OpenFoodFacts result. Ready to answer."
                return True, obs_cached, cached["payload"], int((time.time() - t0) * 1000)

            if barcode:
                ok2, obs0, payload = _off_lookup_by_barcode(barcode)
                if not ok2:
                    return ok2, obs0, payload, int((time.time() - t0) * 1000)

                # Build a compact nutrition summary for the planner + summary
                p = payload.get("product") or {}
                n = p.get("nutriments") or {}
                kcal = n.get("energy_kcal_100g")
                fat = n.get("fat_100g")
                carbs = n.get("carbohydrates_100g")
                sugar = n.get("sugars_100g")
                protein = n.get("proteins_100g")
                salt = n.get("salt_100g")

                nutline_parts = []
                if kcal is not None:   nutline_parts.append(f"{kcal} kcal/100g")
                if fat is not None:    nutline_parts.append(f"fat {fat}g")
                if carbs is not None:  nutline_parts.append(f"carb {carbs}g")
                if sugar is not None:  nutline_parts.append(f"sugar {sugar}g")
                if protein is not None:nutline_parts.append(f"protein {protein}g")
                if salt is not None:   nutline_parts.append(f"salt {salt}g")
                nutline = ", ".join(nutline_parts) if nutline_parts else "nutrition data present"

                # Strong observation the planner can use to stop looping
                obs = f"{obs0} Nutrition (per 100g): {nutline}. Ready to answer."

                # Cache and return
                ACTION_CACHE[key] = {"payload": payload, "_obs_hint": obs}
                return True, obs, payload, int((time.time() - t0) * 1000)

            else:
                ok2, obs, payload = _off_search(query.strip(), k)
                ACTION_CACHE[key] = {"payload": payload, "_obs_hint": obs + " Ready to answer."}
                return ok2, obs + " Ready to answer.", payload, int((time.time() - t0) * 1000)

        else:
            # Safety: no executor wired for this tool
            return False, f"No executor bound for tool: {action}", {}, int((time.time() - t0) * 1000)

    except Exception as e:
        # Non-transient or unexpected error
        return False, f"Tool error: {type(e).__name__}: {e}", {}, int((time.time() - t0) * 1000)


# Final Synthesis ------------------------------------------------------------------------------------------------------
def synthesize_answer(state: ControllerState) -> str:
    """
    Compose the final answer using the compact working summary accumulated in state.history_summary. The full raw trace
    can be logged elsewhere.

    :param state: instance of ControllerState
    :return: model's response
    """
    sys = "Your goal is to produce a final answer to a goal (likely a question) using only evidence provided in the " \
          "working summary."
    user = (
        f"Goal: {state.goal}\n\n"
        f"Working summary:\n{state.history_summary}\n\n"
        f"Produce the final answer in ≤ 200 tokens."
    )
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "developer", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    if hasattr(resp, "usage"):
        record_usage(state, resp.usage)
    return resp.choices[0].message.content.strip()


# Controller Loop ------------------------------------------------------------------------------------------------------

def run_agent(goal: str) -> str:
    """
    Main controller loop:
      while budgets remain and not done:
        1) Build context (we keep a rolling summary in state)
        2) Plan next action (tool or 'answer')
        3) Loop detection guard
        4) Execute with validation/repair/retry
        5) Update summary and telemetry
        6) If 'answer', synthesize final output and stop

    :param goal:
    :return:
    """
    state = ControllerState(goal=goal)

    while within_budget(state) and not state.done:
        # Ask the planner to choose the next action
        action, args, rationale = plan_next_action(state)
        print(f"Action selected: {action}\n\targuments: {args}\n\trationale: {rationale}")

        # Prevent infinite ReAct loops by hashing last few actions
        if looks_stuck(action, args):
            print("\tdetected being stuck in loop...")
            # Very explicit hint that the planner should switch to 'answer'
            state.last_observation = "Loop detected: information already retrieved. Proceed to 'answer' using collected evidence."
            # Do not increment steps or execute; let planner try again
            continue

        # Execute the chosen action (or 'answer' pseudo-tool)
        ok, obs, payload, ms = execute_action(action, args)
        print(f"\t\ttool payload: {payload}")

        # Record step telemetry
        state.steps_taken += 1
        state.tool_trace.append(StepRecord(
            action=action,
            args=args,
            ok=ok,
            latency_ms=ms,
            info=payload
        ))

        # Provide short observation back to planner for next turn
        state.last_observation = obs

        # Summarize new evidence into compact working memory
        update_summary(state, f"{action}({args}) -> {obs}")

        # If planner signaled 'answer', produce final answer and exit
        if action == "answer" and ok:
            final = synthesize_answer(state)
            state.done = True
            print("hello")
            return final

        # If a tool failed, we do not crash; the planner sees the observation
        # and can pivot on the next iteration. The loop will also stop on budgets.
    print(within_budget(state), state.done)
    # If we exit naturally, budgets are exhausted or we never reached 'answer'
    return "Stopped: budget exhausted or no progress."


# Demo -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the agent with a natural language goal/query. Query for food must include barcode number, Example: What's the current weather in Paris (metric) and nutrition for Jben (barcode 6111242106949)?"
    )
    parser.add_argument(
        "goal",
        type=str,
        help="The natural language query for the agent. Example: \"Example: What's the current weather in Paris (metric) and nutrition for Jben (barcode 6111242106949)?\""
    )
    args = parser.parse_args()
    goal = args.goal 

    print("\n--- Running Agent ---\n")
    answer = run_agent(goal)
    print("\n--- Final Answer ---\n")
    print(answer)


    # You could also print telemetry for inspection:
    # - steps taken, tokens used, cost, brief trace, etc.
