"""Small helpers for supplying bounded document context to fragment translators."""

from __future__ import annotations


def context_excerpt(value: object, needle: str = "", *, limit: int = 1200) -> str:
    """Return a whitespace-normalized window around ``needle`` within ``limit`` chars."""
    content = " ".join(str(value or "").split())
    if not content or limit <= 0:
        return ""
    if len(content) <= limit:
        return content
    normalized_needle = " ".join(needle.split()).casefold()
    index = content.casefold().find(normalized_needle) if normalized_needle else -1
    if index < 0:
        return content[: limit - 1].rstrip() + "…"
    start = max(0, min(index - limit // 3, len(content) - limit))
    excerpt = content[start : start + limit]
    return ("…" if start else "") + excerpt.rstrip() + ("…" if start + limit < len(content) else "")


__all__ = ["context_excerpt"]
