import os
import google.generativeai as genai
import json
from gemini_adapter import evaluate_with_causal_engine

# --- Setup Gemini ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# --- Input osservazionale (può venire dal CSV, sensori, ecc.) ---
context = """
User metrics summary:
- mood: low
- stress: high
- activity: medium
- sleep: good
"""

# --- Prompt vincolato ---
prompt = f"""
You are an assistant that ONLY outputs valid JSON.

Given this context:
{context}

Propose ONE action to improve mood.

Rules:
- action must be conservative
- do NOT increase activity if stress is high
- output JSON only

Schema:
{{
  "action": "string",
  "params": {{
    "delta": number
  }},
  "rationale": "string"
}}
"""

response = model.generate_content(prompt)

# --- Parse JSON ---
proposal = json.loads(response.text)

# --- SAFETY EVALUATION ---
verdict = evaluate_with_causal_engine(
    proposal=proposal,
    data_path="IMPLEMENTATION/pcb_one_click/data.csv"
)

print("Gemini proposal:", proposal)
print("Causal verdict:", verdict)
