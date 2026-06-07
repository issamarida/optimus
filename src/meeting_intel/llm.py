from __future__ import annotations

import os
from typing import Optional, Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None):
        from anthropic import Anthropic
        self._client = Anthropic()
        self.model = model or os.environ.get("MEETING_INTEL_MODEL", "claude-3-5-haiku-latest")

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        message = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def get_llm(backend: str = "auto") -> Optional[LLM]:
    backend = backend or "auto"
    if backend in ("none", "offline"):
        return None
    if backend in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:
            return None
    return None
