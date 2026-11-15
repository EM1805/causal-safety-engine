"""Shared trial metadata and explicit scope for Level 3.2 intervention evidence."""

from datetime import date, timedelta
from hashlib import sha256
import json


TRIAL_FIELDS = (
    "trial_id", "insight_id", "t_index", "adherence_flag", "adherence_verified",
    "dose", "lag_days", "regime_id",
)
SCOPE_FIELDS = ("insight_id", "action_name", "dose", "target_col", "lag_days", "regime_id", "regime_start")


def text(value):
    if value is None or value != value:
        return ""
    return str(value).strip()


def integer(value):
    clean = text(value)
    try:
        parsed = int(clean)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def trial_id(row):
    return "trial-" + digest({k: text(row.get(k)) for k in ("insight_id", "action_name", "date", "t_index", "dose", "regime_id")})


def load_scope(path):
    with open(path, encoding="utf-8") as handle:
        scope = json.load(handle)
    if not isinstance(scope, dict) or set(scope) != set(SCOPE_FIELDS):
        raise ValueError("scope must specify exactly: " + ", ".join(SCOPE_FIELDS))
    if any(not isinstance(scope[key], str) or not scope[key].strip() for key in SCOPE_FIELDS if key != "lag_days"):
        raise ValueError("invalid or missing scope field")
    if type(scope["lag_days"]) is not int or not 0 <= scope["lag_days"] <= 365:
        raise ValueError("lag_days must be an integer from 0 to 365")
    for key in SCOPE_FIELDS:
        if isinstance(scope[key], str):
            scope[key] = scope[key].strip()
    scope["regime_start"] = iso_day(scope["regime_start"]).isoformat()
    return scope


def iso_day(value):
    day = date.fromisoformat(value)
    if day.isoformat() != value:
        raise ValueError("expected a calendar date in YYYY-MM-DD format")
    return day


def scoped_trials(records, days, scope, active):
    """Return unique, dated, adherent trial/outcome pairs and eligible controls."""
    indices = {day: idx for idx, day in enumerate(days)}
    start = iso_day(scope["regime_start"])
    lag = timedelta(days=scope["lag_days"])
    eligible, counts, ids = [], {}, {}
    for record in records:
        idx = integer(record.get("t_index"))
        identifier = text(record.get("trial_id"))
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
        if identifier:
            ids[identifier] = ids.get(identifier, 0) + 1
    for record in records:
        idx = integer(record.get("t_index"))
        identifier = text(record.get("trial_id"))
        if idx is None or idx >= len(days) or counts.get(idx) != 1 or ids.get(identifier) != 1:
            continue
        when = days[idx]
        if when < start or text(record.get("date")) != when.isoformat():
            continue
        if any(text(record.get(key)) != scope[key] for key in ("insight_id", "action_name", "dose", "target_col", "regime_id")):
            continue
        if integer(record.get("lag_days")) != scope["lag_days"]:
            continue
        if integer(record.get("adherence_flag")) != 1 or integer(record.get("adherence_verified")) != 1:
            continue
        outcome = indices.get(when + lag)
        if outcome is not None and not any(i in active for i in range(idx + 1, outcome + 1)):
            eligible.append((record, outcome))
    controls = {}
    for idx, day in enumerate(days):
        endpoint = indices.get(day + lag)
        if day >= start and endpoint is not None and not any(i in active for i in range(idx, endpoint + 1)):
            controls[idx] = endpoint
    return eligible, controls
