"""Completion backends for LLM-based auditing methods.

A backend is any callable ``complete(prompt: str) -> str``. Two are provided:

- :class:`OpenAICompatibleBackend` posts one chat completion to any OpenAI-compatible
  ``/v1`` endpoint (the OpenAI API, a gateway, or a local vLLM or Ollama server) using the
  standard library only, at temperature 0 by default;
- :class:`FixedResponseBackend` returns scripted replies and records every prompt, for
  tests and for replaying recorded runs without network access.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Union

Complete = Callable[[str], str]


class FixedResponseBackend:
    """Scripted replies. ``responses`` is a list consumed in call order, a dict from prompt
    to reply, or a function of the prompt. ``prompts`` records every prompt received."""

    def __init__(self, responses: Union[Sequence[str], Dict[str, str], Complete], default: Optional[str] = None):
        self.responses = responses
        self.default = default
        self.prompts: List[str] = []
        self._i = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        r = self.responses
        if callable(r):
            return r(prompt)
        if isinstance(r, dict):
            if prompt in r:
                return r[prompt]
        elif self._i < len(r):
            out = r[self._i]
            self._i += 1
            return out
        if self.default is not None:
            return self.default
        raise LookupError("FixedResponseBackend has no reply for this prompt")


class OpenAICompatibleBackend:
    """One chat completion per prompt against ``{base_url}/chat/completions``.

    ``api_key`` defaults to the ``OPENAI_API_KEY`` environment variable and ``base_url`` to
    ``OPENAI_BASE_URL`` or the OpenAI API. Network and parse errors return an empty string,
    which the parsers treat as no answer, so a long generation pass survives one bad call;
    set ``raise_errors=True`` to raise instead. Some reasoning models reject
    ``temperature=0``; pass ``temperature=None`` to omit the field.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: Optional[float] = 0.0, max_tokens: int = 512, timeout: float = 240.0,
                 raise_errors: bool = False) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.raise_errors = raise_errors

    def __call__(self, prompt: str) -> str:
        import urllib.error
        import urllib.request

        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key or 'EMPTY'}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return (payload["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            if self.raise_errors:
                raise
            return ""
