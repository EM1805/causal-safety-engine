"""
Gemini-governed proposal agent (Gemini 3).

STRICT ROLE SEPARATION
---------------------
Gemini 3:
- generates ONE conservative action proposal
- outputs JSON ONLY
- has NO execution authority
- has NO policy authority

Causal Safety Engine (PCB CLI):
- validates proposal
- decides (ALLOW / BLOCK / SILENCE)
- executes deterministically

This file implements the proposer side only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from google import genai

from gemini_adapter import evaluate_with_causal_engine, normalize_proposal


# ---------------------------------------------------------------------
# Configuration (fail fast)
# ---------------------------------------------------------------------

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"

DEFAULT_DATA_PATH = "IMPLEMENTATION/pcb_one_click/data.csv"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------
# Gemini 3 client
# ---------------------------------------------------------------------

def get_client() -> genai.Client:
    api_key = _require_env(GEMINI_API_KEY_ENV)
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    """
    Expect an explicit Gemini 3 model.
    Example:
      GEMINI_MODEL=gemini-3-fast
    """
    model = _require_env(GEMINI_MODEL_ENV)

    if not model.startswith("gemini-3"):
        raise RuntimeError(
            f"{GEMINI_MODEL_ENV} must be a Gemini 3 model, got: {model}"
        )

    return model


# ---------------------------------------------------------------------
# Defensive JSON extraction (LLMs are never trusted)
# ---------------------------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output.

    Handles:
    - markdown fences
    - leading / trailing text
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Gemini output")

    return json.loads(cleaned[start : end + 1])


# ---------------------------------------------------------------------
# Prompt (STRICT JSON CONTRACT)
# ---------------------------------------------------------------------

def build_prompt(context: str) -> str:
    return f"""
You must output ONLY a valid JSON object.
No markdown. No explanations. No extra text.

Context:
{context}

Rules:
- Propose exactly ONE conservative action
- If stress is high, DO NOT increase activity
- Use small numeric deltas only

JSON schema:
{{
  "action": "adjust_features",
  "params": {{
    "deltas": {{
      "activity": number,
      "stress": number,
      "sleep": number
    }}
  }},
  "rationale": "string"
}}
"""


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    client = get_client()
    model_name = get_model_name()

    # Observational context (example – can come from CSV, sensors, API, etc.)
    context = """
User metrics summary:
- mood: low
- stress: high
- activity: medium
- sleep: good
"""

    # -----------------------------------------------------------------
    # Gemini 3 proposal (proposal-only role)
    # -----------------------------------------------------------------

    response = client.models.generate_content(
        model=model_name,
        contents=build_prompt(context),
    )

    try:
        raw_proposal = extract_json(response.text)
        proposal = normalize_proposal(raw_proposal)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid JSON proposal from Gemini:\n{response.text}"
        ) from exc

    # -----------------------------------------------------------------
    # Deterministic safety evaluation (PCB decides)
    # -----------------------------------------------------------------

    verdict = evaluate_with_causal_engine(
        proposal=proposal,
        data_path=DEFAULT_DATA_PATH,
    )

    # -----------------------------------------------------------------
    # Output (audit-friendly)
    # -----------------------------------------------------------------

    print("\n=== Gemini proposal ===")
    print(json.dumps(proposal, indent=2))

    print("\n=== Causal verdict ===")
    print(json.dumps(
        {k: v for k, v in verdict.items() if k != "proposal"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
