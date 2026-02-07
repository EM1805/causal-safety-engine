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

if "GEMINI_API_KEY" not in os.environ:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# -------------------------------------------------------------------
# 2. Select a VALID model dynamically
#    (avoid 404s forever)
# -------------------------------------------------------------------

def get_text_generation_model() -> genai.GenerativeModel:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)

    raise RuntimeError("No Gemini model supports generateContent")


model = get_text_generation_model()


# -------------------------------------------------------------------
# 3. Observational context (can come from CSV, sensors, etc.)
# -------------------------------------------------------------------

context = """
User metrics summary:
- mood: low
- stress: high
- activity: medium
- sleep: good
"""


# -------------------------------------------------------------------
# 4. STRICT JSON-only prompt
# -------------------------------------------------------------------

prompt = f"""
You are a system that ONLY outputs valid JSON.
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
# 5. Gemini proposal
# -------------------------------------------------------------------

response = model.generate_content(prompt)

try:
    proposal = json.loads(response.text)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Invalid JSON from Gemini:\n{response.text}") from e


# -------------------------------------------------------------------
# 6. SAFETY-FIRST evaluation (your engine decides)
# -------------------------------------------------------------------

verdict = evaluate_with_causal_engine(
    proposal=proposal,
    data_path="IMPLEMENTATION/pcb_one_click/data.csv",
)


# -------------------------------------------------------------------
# 7. Output
# -------------------------------------------------------------------

print("\n=== Gemini proposal ===")
print(json.dumps(proposal, indent=2))

print("\n=== Causal verdict ===")
print(json.dumps(verdict, indent=2))
