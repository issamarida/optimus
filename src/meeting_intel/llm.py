"""A thin, provider-agnostic LLM client used by summarization and coaching.

Why an abstraction: summarization (abstractive) and coaching feedback are
genuinely generative tasks where an LLM is the right tool in production, but the
repo must also run with no network and no API key (CI, demos, offline review).
So callers ask for an LLM, get one if a key + SDK are present, and otherwise get
``None`` and fall back to deterministic methods. No silent failure, no hard dep.

Configure via env: ``MEETING_INTEL_LLM=anthropic`` and ``ANTHROPIC_API_KEY=...``.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None):
        from anthropic import Anthropic  # optional dep
        self._client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model or os.environ.get("MEETING_INTEL_MODEL", "claude-3-5-haiku-latest")

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def get_llm(backend: str = "auto") -> Optional[LLM]:
    """Return a usable LLM or None. ``auto`` enables it only when a key is present."""
    backend = backend or "auto"
    if backend in ("none", "offline"):
        return None
    if backend in ("auto", "anthropic"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return AnthropicLLM()
            except Exception:
                return None
    return None
