"""Small OpenAI-compatible translator used for OCR table cells."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .config import Settings


class TranslationError(RuntimeError):
    pass


class Translator(Protocol):
    async def translate(
        self,
        text: str,
        *,
        context: str = "",
        required_literals: Sequence[str] = (),
    ) -> str: ...

    async def aclose(self) -> None: ...


def normalize_openai_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base if base.endswith("/v1") else f"{base}/v1"


class OpenAITranslator:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=normalize_openai_base_url(settings.base_url),
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def translate(
        self,
        text: str,
        *,
        context: str = "",
        required_literals: Sequence[str] = (),
    ) -> str:
        if not text.strip():
            return text
        prompt = self.settings.system_prompt
        if self.settings.domain_prompt.strip():
            prompt = f"{prompt}\n\n{self.settings.domain_prompt.strip()}"
        user = (
            f"Target language: {self.settings.target_language}\n"
            f"Context: {context or 'document text'}\n"
            "Return only the translation. Preserve every [[...|...]] protected wrapper exactly.\n\n"
            f"Text:\n{text}"
        )
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(1, self.settings.llm_max_attempts + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.llm_key}"},
                    json=payload,
                )
                if response.status_code >= 400:
                    detail = " ".join(response.text.split())[:500]
                    error = TranslationError(f"LLM request failed (HTTP {response.status_code}): {detail}")
                    if response.status_code < 500 and response.status_code != 429:
                        raise error
                    last_error = error
                else:
                    body = response.json()
                    result = str(body["choices"][0]["message"]["content"] or "").strip()
                    if not result:
                        raise TranslationError("LLM returned an empty translation")
                    missing = [literal for literal in required_literals if literal not in result]
                    if missing:
                        raise TranslationError(f"LLM changed or omitted {len(missing)} protected literal(s)")
                    return result
            except (httpx.TimeoutException, httpx.TransportError, KeyError, ValueError) as exc:
                last_error = exc
            if attempt < self.settings.llm_max_attempts:
                await asyncio.sleep(min(4.0, 2 ** (attempt - 1)))
        raise TranslationError(f"LLM translation failed after bounded retries: {last_error}") from last_error


class StaticTranslator:
    """Deterministic translator for offline route and rendering tests."""

    def __init__(self, translations: dict[str, str] | None = None, prefix: str = "TR: ") -> None:
        self.translations = translations or {}
        self.prefix = prefix
        self.calls: list[str] = []
        self.contexts: list[str] = []

    async def translate(
        self,
        text: str,
        *,
        context: str = "",
        required_literals: Sequence[str] = (),
    ) -> str:
        del required_literals
        self.calls.append(text)
        self.contexts.append(context)
        return self.translations.get(text, f"{self.prefix}{text}")

    async def aclose(self) -> None:
        return None
