"""
Gemini-governed agent for Causal Safety Engine

Gemini:
- generates ACTION PROPOSALS only
- never executes
- never decides

Causal Safety Engine:
- evaluates
- decides ALLOW / BLOCK / SILENCE
"""

import os
import json
import google.generativeai as genai

from gemini_adapter import evaluate_with_causal_engine


# -------------------------------------------------------------------
# 1. Configure Gemini safely (API key from env)
# -------------------------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)


# -------------------------------------------------------------------
# 2. Select a VALID model dynamically (no hardcoded names)
# -------------------------------------------------------------------

def get_text_generation_model() -> genai.GenerativeModel:
    """
    Selects the first Gemini model that supports generateContent.
    This avoids all 404 / version / availability issues.
    """
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)

    raise RuntimeError("No Gemini model supports generateContent")


model = get_text_generation_model()


# -------------------------------------------------------------------
# 3. Defensive JSON extraction (LLMs are never trusted)
# -------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Extracts the first valid JSON object from Gemini output.
    Handles markdown fences and extra text safely.
    """
    text = text.strip()

    # Remove ``` blocks if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Gemini output")

    return json.loads(text[start:end + 1])


# -------------------------------------------------------------------
# 4. Observational context (from CSV, sensors, etc.)
# -------------------------------------------------------------------

context = """
User metrics summary:
- mood: low
- stress: high
- activity: medium
- sleep: good
"""


# -------------------------------------------------------------------
# 5. STRICT JSON-only prompt (still parsed defensively)
# -------------------------------------------------------------------

prompt = f"""
You are a system that outputs ONLY valid JSON.
No markdown. No explanations. No text outside JSON.

Context:
{context}

Rules:
- Propose exactly ONE action
- Be conservative
- If stress is high, DO NOT increase activity

JSON schema:
{{
  "action": "string",
  "params": {{
    "delta": number
  }},
  "rationale": "string"
}}
"""


# -------------------------------------------------------------------
# 6. Gemini proposal (proposal-only role)
# -------------------------------------------------------------------

response = model.generate_content(prompt)

try:
    proposal = extract_json(response.text)
except Exception as e:
    raise RuntimeError(
        f"Invalid JSON from Gemini:\n{response.text}"
    ) from e


# -------------------------------------------------------------------
# 7. SAFETY-FIRST evaluation (engine decides, not Gemini)
# -------------------------------------------------------------------

verdict = evaluate_with_causal_engine(
    proposal=proposal,
    data_path="IMPLEMENTATION/pcb_one_click/data.csv",
)


# -------------------------------------------------------------------
# 8. Output (audit-friendly)
# -------------------------------------------------------------------

print("\n=== Gemini proposal ===")
print(json.dumps(proposal, indent=2))

print("\n=== Causal verdict ===")
print(json.dumps(verdict, indent=2))
