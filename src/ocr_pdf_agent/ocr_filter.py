"""Shared conservative filters for non-semantic OCR detections."""

from __future__ import annotations

from typing import Any

VISUAL_KINDS = {
    "image",
    "figure",
    "chart",
    "seal",
    "stamp",
    "logo",
    "barcode",
    "qr_code",
    "header_image",
    "footer_image",
    "page_header_image",
    "page_footer_image",
    "equation",
    "formula",
    "inline_equation",
    "interline_equation",
}


def is_low_confidence_speck(
    text: str,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    confidence: Any,
) -> bool:
    """Identify only tiny, low-confidence, isolated OCR glyph noise.

    A real single-character label normally occupies materially more than one
    percent of a page dimension at OCR resolution. Requiring all conditions
    keeps this filter deliberately narrower than a generic confidence cutoff.
    """
    compact = "".join(text.split())
    if len(compact) != 1:
        return False
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return False
    if score >= 0.5:
        return False
    width_ratio = (bbox[2] - bbox[0]) / page_size[0]
    height_ratio = (bbox[3] - bbox[1]) / page_size[1]
    return width_ratio <= 0.01 and height_ratio <= 0.01


__all__ = ["VISUAL_KINDS", "is_low_confidence_speck"]
