"""
Gemini-governed proposal agent.

Role separation (STRICT):
- Gemini: proposes ONE conservative action (JSON only)
- Causal Safety Engine (PCB CLI): validates, decides, executes

Gemini has:
- no execution authority
- no policy authority
- no override capability
"""

from __future__ import annotations

import json
import os
from typing import Dict

import google.generativeai as genai

from gemini_adapter import evaluate_with_causal_engine, normalize_proposal


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def get_model() -> genai.GenerativeModel:
    """
    Select a valid Gemini model.

    Priority:
    1. GEMINI_MODEL env var (if set)
    2. First model supporting generateContent
    """
    preferred = os.getenv("GEMINI_MODEL")
    if preferred:
        return genai.GenerativeModel(preferred)

    for model_info in genai.list_models():
        if "generateContent" in model_info.supported_generation_methods:
            return genai.GenerativeModel(model_info.name)

    raise RuntimeError("No Gemini model supports generateContent")


# ---------------------------------------------------------------------
# Defensive JSON extraction
# ---------------------------------------------------------------------

def extract_json(text: str) -> Dict:
    """
    Extract the first JSON object from Gemini output.

    Handles:
    - markdown fences
    - leading/trailing text
    """
    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Gemini output")

    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_prompt(context: str) -> str:
    """
    Build a STRICT JSON-only prompt.
    """
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
# Main agent loop
# ---------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = get_model()

    # Observational context (could come from CSV, sensors, API, etc.)
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
            f"Invalid JSON proposal from Gemini:\n{response.text}"
        ) from exc

    verdict = evaluate_with_causal_engine(
        proposal=proposal,
        data_path="IMPLEMENTATION/pcb_one_click/data.csv",
    )

    print("\n=== Gemini proposal ===")
    print(json.dumps(proposal, indent=2))

    print("\n=== Causal verdict ===")
    print(json.dumps(
        {k: v for k, v in verdict.items() if k != "proposal"},
        indent=2
    ))


if __name__ == "__main__":
    main()
