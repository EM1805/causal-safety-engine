# PostHog integration (MVP)

This integration follows the architecture:

`PostHog UI -> PostHog App Plugin -> CSE Backend API -> Causal Safety Engine -> PostHog App Plugin`

## 1) Backend API

### Run locally

```bash
pip install fastapi uvicorn pandas requests numpy
uvicorn integrations.posthog.backend.app:app --reload --port 8000
```

### Endpoint

`POST /analyze`

Request:

```json
{
  "rows": [
    {"timestamp": "2026-04-01T10:00:00Z", "conversion": 0.12, "price": 99, "traffic": 1200}
  ]
}
```

Response contract (UI-safe):

```json
{
  "causes": [
    {
      "feature": "price_lag1",
      "effect": -0.34,
      "stability": 0.77,
      "bootstrap_p10": -0.42,
      "bootstrap_p90": -0.19,
      "explanation": "Higher Price Lag1 likely contributed to the drop.",
      "falsification_pass": true
    }
  ],
  "confidence": 0.72,
  "suggestion": "Prioritize an A/B test on price_lag1 and validate impact over the next 7 days."
}
```

## 2) Pulling PostHog events

Use `integrations/posthog/backend/posthog_client.py`.

```python
from integrations.posthog.backend.posthog_client import PostHogClient

client = PostHogClient(api_key="<key>", project_id="<project_id>")
events = client.get_events(limit=500)
```

## 3) Event -> time-series transform

Use `transform_events_to_timeseries` in `integrations/posthog/backend/transform.py`.

## 4) PostHog app UI

A minimal React button component is provided in:

- `integrations/posthog/posthog_app/src/CausalButton.tsx`

Place it on Trends and Funnel pages near metric titles.

## 5) MVP checklist

- [x] `/analyze` endpoint
- [x] `Explain why` button
- [x] lag-aware and filter-aware ranking
- [x] confidence with bootstrap + temporal stability + falsification signal

## 6) Important note

This MVP produces **causal hypotheses** for product decision support.
It is not a definitive causal proof and should be validated with controlled experiments.
