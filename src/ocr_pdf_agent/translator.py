"""Compatibility bridge for the extracted production OCR translation stages.

The parent pipeline expects a small ``translate_chunk`` function.  The
standalone service keeps one OpenAI-compatible client implementation and does
not add an alternate model or silent fallback here.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .llm import OpenAITranslator


def _build_client(settings: Settings) -> OpenAITranslator:
    return OpenAITranslator(settings)


async def translate_chunk(
    client: OpenAITranslator,
    text: str,
    context_prev: str,
    context_next: str,
    settings: Settings | None = None,
    *,
    seg_type: str = "para",
    source_kind: str = "pdf",
    has_layout: bool = False,
    has_visual_context: bool = False,
    route: Any = None,
    layout_retry_reason: str | None = None,
    required_literals: tuple[str, ...] = (),
    rejected_translation: str | None = None,
) -> str:
    del settings, source_kind, has_layout, has_visual_context, route
    parts = [
        f"Segment type: {seg_type}.",
        "Translate only the current segment; neighbouring text is context only.",
    ]
    if context_prev:
        parts.append(f"Previous context: {context_prev}")
    if context_next:
        parts.append(f"Following context: {context_next}")
    if layout_retry_reason:
        parts.append(
            "This is a strict retry: preserve every protected wrapper and "
            "factual literal exactly, without adding commentary."
        )
    if rejected_translation:
        parts.append(
            "The previous candidate failed literal or target-language validation; "
            "translate the complete source again and repair it."
        )
    return await client.translate(
        text,
        context="\n".join(parts),
        required_literals=required_literals,
    )


__all__ = ["_build_client", "translate_chunk"]
