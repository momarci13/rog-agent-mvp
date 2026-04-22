"""Ollama client with JSON-mode + retry.

One model, many roles: we switch behaviour via system prompts rather than
loading different models — a 7B Q4 fits in VRAM; two don't.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMConfig:
    model: str
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    temperature: float = 0.2
    timeout_s: int = 180


class OllamaLLM:
    """Minimal Ollama client. Uses the /api/chat endpoint directly so we
    don't need the `ollama` package at runtime (though it's in requirements
    for convenience)."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if json_mode:
            payload["format"] = "json"

        r = self._client.post(f"{self.cfg.host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    def chat_json(
        self,
        messages: list[dict],
        schema_hint: str = "",
        *,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> dict:
        """Ask the model for JSON and parse it with retries on malformed output."""
        if schema_hint:
            messages = messages + [{
                "role": "system",
                "content": f"Respond with JSON only. Schema hint:\n{schema_hint}",
            }]
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                txt = self.chat(messages, temperature=temperature, json_mode=True)
                return json.loads(txt)
            except (json.JSONDecodeError, httpx.HTTPError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Failed to get valid JSON after {max_retries+1} tries: {last_err}")

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.cfg.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
