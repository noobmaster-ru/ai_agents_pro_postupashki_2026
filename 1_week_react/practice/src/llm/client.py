import json
import random
import threading
import time

import requests

from ..config import Settings
from .ledger import Ledger

RETRY_CODES = (408, 429, 500, 502, 503, 504)
MAX_BACKOFF = 65.0


def backoff_seconds(response: requests.Response | None, attempt: int) -> float:
    if response is None:
        return min(10.0, 2.0 * (attempt + 1)) + random.uniform(0, 1.0)
    delay = min(MAX_BACKOFF, 4.0 * 2 ** attempt)
    if response.status_code == 429:
        try:
            reset_ms = int(response.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"])
            delay = max(3.0, min(MAX_BACKOFF, reset_ms / 1000 - time.time() + 1.0))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            delay = min(MAX_BACKOFF, 15.0 * (attempt + 1))
    return delay + random.uniform(0, 2.0)


class HungRequest(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings, ledger: Ledger, attempts: int = 7, timeout: int = 90, hard_deadline: int = 120):
        self.settings = settings
        self.ledger = ledger
        self.attempts = attempts
        self.timeout = timeout
        self.hard_deadline = hard_deadline

    def chat(self, messages: list, model: str, tools: list | None = None, tag: str = "chat",
             temperature: float | None = None, run_id: str | None = None) -> dict:
        body = {"model": model, "messages": messages, "usage": {"include": True}}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if temperature is not None:
            body["temperature"] = temperature
        started = time.perf_counter()
        data = self._post(body)
        self.ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started, run_id)
        return data["choices"][0]["message"]

    def _request(self, body: dict) -> requests.Response:
        box: dict = {}

        def work():
            try:
                box["response"] = requests.post(self.settings.chat_url, json=body, headers=self.settings.headers,
                                                timeout=self.timeout)
            except Exception as e:
                box["error"] = e

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(self.hard_deadline)
        if worker.is_alive():
            raise HungRequest(f"запрос завис дольше {self.hard_deadline} с")
        if "error" in box:
            raise box["error"]
        return box["response"]

    def _post(self, body: dict) -> dict:
        problem = "нет ответа от OpenRouter"
        for attempt in range(self.attempts):
            response = None
            try:
                response = self._request(body)
            except (requests.RequestException, HungRequest) as e:
                problem = type(e).__name__
            else:
                if response.status_code == 200:
                    return response.json()
                problem = f"HTTP {response.status_code}: {response.text[:5000]}"
                if response.status_code not in RETRY_CODES:
                    break
            if attempt < self.attempts - 1:
                time.sleep(backoff_seconds(response, attempt))
        raise RuntimeError(problem)
