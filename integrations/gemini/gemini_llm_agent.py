"""
Gemini-governed proposal agent for the Causal Safety Engine.

STRICT role separation:

- Gemini:
  - generates ONE conservative action proposal
  - outputs JSON only
  - has no execution or policy authority

- Causal Safety Engine (PCB CLI):
  - validates proposals
  - enforces causal guardrails
  - decides ALLOW / BLOCK / SILENCE
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import google.generativeai as genai

from gemini_adapter import evaluate_with_causal_engine, normalize_proposal


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"

REQUIRED_MODEL_PREFIX = "gemini-3"
DEFAULT_DATA_PATH = "IMPLEMENTATION/pcb_one_click/data.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _require_env(var_name: str) -> str:
    """Read a required environment variable or fail fast."""
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def _normalize_model_name(model_name: str) -> str:
    """Remove optional 'models/' prefix from Gemini model names."""
    return model_name.removeprefix("models/")


# ---------------------------------------------------------------------
# Gemini model selection (explicit + constrained)
# ---------------------------------------------------------------------

def get_model() -> genai.GenerativeModel:
    """
    Return a configured Gemini model.

    Constraints:
    - Model must be explicitly provided via GEMINI_MODEL
    - Only Gemini 3 family models are allowed
    """
    model_name = _require_env(GEMINI_MODEL_ENV)
    normalized = _normalize_model_name(model_name)

    if not normalized.startswith(REQUIRED_MODEL_PREFIX):
        raise RuntimeError(
            f"{GEMINI_MODEL_ENV} must target Gemini 3 family, got: {model_name}"
        )

    return genai.GenerativeModel(model_name)


# ---------------------------------------------------------------------
# Defensive JSON extraction
# ---------------------------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from model output.

    Defends against:
    - markdown fences
    - extra text before/after JSON
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
        raise ValueError("No valid JSON object found in Gemini output")

    return json.loads(cleaned[start : end + 1])


# ---------------------------------------------------------------------
# Prompt construction (proposal-only)
# ---------------------------------------------------------------------

def build_prompt(context: str) -> str:
    """Build a strict JSON-only prompt for proposal generation."""
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
# Main agent flow
# ---------------------------------------------------------------------

def main() -> None:
    """Run Gemini proposal flow and submit it to the causal safety layer."""
    api_key = _require_env(GEMINI_API_KEY_ENV)
    genai.configure(api_key=api_key)

    model = get_model()

    # Observational context (can be loaded from CSV, sensors, API, etc.)
    context = """
User metrics summary:
- mood: low
- stress: high
- activity: medium
- sleep: good
"""

    response = model.generate_content(build_prompt(context))

    try:
        raw_proposal = extract_json(response.text)
        proposal = normalize_proposal(raw_proposal)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid proposal returned by Gemini:\n{response.text}"
        ) from exc

    verdict = evaluate_with_causal_engine(
        proposal=proposal,
        data_path=DEFAULT_DATA_PATH,
    )

    print("\n=== Gemini proposal ===")
    print(json.dumps(proposal, indent=2))

    print("\n=== Causal verdict ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "proposal"}, indent=2))


if __name__ == "__main__":
    main()
