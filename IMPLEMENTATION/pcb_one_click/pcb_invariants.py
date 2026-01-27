# pcb_invariants.py
# Python 3.7 compatible
# Hard invariants for PCB (alignment & safety)
from typing import Dict, Tuple

def check_invariants(context: Dict) -> Tuple[bool, str]:
    """Return (ok, reason). Hard, auditable rules that must never be violated."""
    # Global invariants
    if int(context.get("guardrail_flag", 0)) == 1:
        return False, "invariant:guardrail_block"
    if int(context.get("lag", 1)) <= 0:
        return False, "invariant:no_zero_lag"
    eff = abs(float(context.get("effect_size", 0.0)))
    if eff < 0.05:
        return False, "invariant:effect_too_small"

    # Level-specific invariants
    if str(context.get("level")) == "2.8":
        action = str(context.get("action", "")).strip()
        if action and action not in ("increase", "reduce", "observe", "increase_source", "reduce_source"):
            return False, "invariant:unknown_action"

    if str(context.get("level")) == "3.2":
        if int(context.get("n_treated", 0)) < 3:
            return False, "invariant:too_few_treated_events"
        if int(context.get("n_matched", 0)) < int(context.get("n_treated", 0)):
            return False, "invariant:insufficient_matches"
        overlap = context.get("overlap_score", None)
        if overlap is not None:
            try:
                if float(overlap) < 0.2:
                    return False, "invariant:no_overlap"
            except Exception:
                pass
        if int(context.get("placebo_fail", 0)) == 1:
            return False, "invariant:placebo_fail"

    return True, ""
