"""Gemini-governed proposal agent.

Gemini only produces a proposal JSON.
The Causal Safety Engine (PCB CLI) remains the decision/execution authority.
"""

import json
import os

import google.generativeai as genai

from gemini_adapter import evaluate_with_causal_engine, normalize_proposal


def get_model() -> genai.GenerativeModel:
    """Return configured Gemini model.

    - Use GEMINI_MODEL if provided.
    - Otherwise pick first model that supports generateContent.
    """
    preferred = os.getenv("GEMINI_MODEL")
    if preferred:
        return genai.GenerativeModel(preferred)

    for model_info in genai.list_models():
        if "generateContent" in model_info.supported_generation_methods:
            return genai.GenerativeModel(model_info.name)

    raise RuntimeError("No Gemini model supports generateContent")


def extract_json(text: str) -> dict:
    """Extract first JSON object defensively from model output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in Gemini output")

    return json.loads(text[start : end + 1])


def build_prompt(context: str) -> str:
    return f"""
You must output ONLY a valid JSON object.
No markdown, no extra text.

Context:
{context}

Rules:
- Propose one conservative action.
- If stress is high, do not increase activity.
- Follow exactly this schema:

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


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = get_model()

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
        raise RuntimeError(f"Invalid JSON proposal from Gemini:\n{response.text}") from exc

    verdict = evaluate_with_causal_engine(
        proposal=proposal,
        data_path="IMPLEMENTATION/pcb_one_click/data.csv",
    )

    print("\n=== Gemini proposal ===")
    print(json.dumps(proposal, indent=2))
    print("\n=== Causal verdict ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "proposal"}, indent=2))


if __name__ == "__main__":
    main()
