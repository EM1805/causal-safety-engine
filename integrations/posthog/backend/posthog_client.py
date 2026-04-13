from __future__ import annotations

from typing import Any

import requests


class PostHogClient:
    def __init__(self, api_key: str, project_id: str, base_url: str = "https://app.posthog.com") -> None:
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = base_url.rstrip("/")

    def get_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        url = f"{self.base_url}/api/projects/{self.project_id}/events/"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, params={"limit": limit}, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict) and "results" in payload:
            return payload["results"]

        if isinstance(payload, list):
            return payload

        return []
