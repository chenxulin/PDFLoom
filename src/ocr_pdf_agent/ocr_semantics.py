"""Shared PP-StructureV3 semantic policy for scanned-PDF translation.

The OCR service deliberately returns more than plain text.  This module keeps
the meaning of those labels consistent across searchable-layer construction,
translation routing, and quality checks:

* document text and headings become source text;
* tables stay out of the intermediate source layer because a dedicated final
  pass translates description cells and rebuilds the complete vector grid;
  images and figure captions remain untouched;
* page headers and footers are omitted from the text layer and masked on the
  scanned background; page numbers are masked here and redrawn numerically
  after translation;
* formulas are left as pixels for PDFMathTranslate v2 to detect and protect;
* figures, charts, images, and seals stay untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_TYPE_ALIASES = {
    "header": "page_header",
    "footer": "page_footer",
    "number": "page_number",
    "page_no": "page_number",
    "pagenumber": "page_number",
    "display_formula": "equation",
    "formula": "equation",
    "inline_formula": "inline_equation",
    "interline_formula": "interline_equation",
    "image_body": "image",
    "figure": "image",
    "chart_body": "chart",
    "stamp": "seal",
    "figure_title": "figure_caption",
    "image_title": "figure_caption",
    "table_title": "table_caption",
}

# Header/footer artwork is page furniture rather than document content, so it
# is removed together with textual headers and footers. Figures in the body
# remain covered by ``PRESERVED_VISUAL_TYPES``.
FURNITURE_TEXT_TYPES = frozenset({"page_header", "page_footer", "page_number"})
FURNITURE_REGION_TYPES = frozenset(set(FURNITURE_TEXT_TYPES) | {"header_image", "footer_image"})
FORMULA_TYPES = frozenset(
    {
        "equation",
        "inline_equation",
        "interline_equation",
        "display_equation",
        "formula_number",
    }
)
PRESERVED_VISUAL_TYPES = frozenset(
    {
        "image",
        "chart",
        "seal",
    }
)

# Tables are intentionally excluded from the v1/v2 source layer: their cells
# are handled by the strict post-translation vector-table rebuilder. Figures
# remain source pixels. Captions and normalized table-cell OCR lines stay out
# of this intermediate layer so the layout engine cannot duplicate them.
NON_SOURCE_TEXT_TYPES = frozenset(
    set(FURNITURE_REGION_TYPES)
    | set(FORMULA_TYPES)
    | set(PRESERVED_VISUAL_TYPES)
    | {"figure_caption", "table", "table_caption", "table_text"}
)


def canonical_ocr_type(value: Any) -> str:
    """Return a stable lower-case semantic label."""
    label = str(value or "text").strip().lower().replace("-", "_").replace(" ", "_")
    return _TYPE_ALIASES.get(label, label or "text")


def is_furniture_text_type(value: Any) -> bool:
    return canonical_ocr_type(value) in FURNITURE_TEXT_TYPES


def is_furniture_region_type(value: Any) -> bool:
    return canonical_ocr_type(value) in FURNITURE_REGION_TYPES


def is_formula_type(value: Any) -> bool:
    return canonical_ocr_type(value) in FORMULA_TYPES


def is_preserved_visual_type(value: Any) -> bool:
    return canonical_ocr_type(value) in PRESERVED_VISUAL_TYPES


def should_inject_source_text(value: Any) -> bool:
    """Whether OCR text with this semantic type belongs in the source layer.

    The exclusion list is intentionally explicit. Unknown future textual labels
    are retained so a new Paddle layout class cannot silently cause漏译.
    """
    return canonical_ocr_type(value) not in NON_SOURCE_TEXT_TYPES


def iter_source_text_blocks(ocr_result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for block in ocr_result.get("blocks") or []:
        if not isinstance(block, dict) or not str(block.get("text") or "").strip():
            continue
        if should_inject_source_text(block.get("type") or block.get("sub_type")):
            yield block


def source_text_by_page(ocr_result: dict[str, Any]) -> dict[int, list[str]]:
    """Return injectable OCR text in reading order, grouped by zero-based page."""
    grouped: dict[int, list[tuple[float, float, str]]] = {}
    for block in iter_source_text_blocks(ocr_result):
        try:
            page_idx = int(block.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = block.get("bbox")
        try:
            x0 = float(bbox[0])
            y0 = float(bbox[1])
        except (TypeError, ValueError, IndexError):
            x0 = y0 = 0.0
        grouped.setdefault(page_idx, []).append((y0, x0, str(block["text"]).strip()))
    return {page_idx: [text for _, _, text in sorted(items)] for page_idx, items in grouped.items()}


__all__ = [
    "FORMULA_TYPES",
    "FURNITURE_REGION_TYPES",
    "FURNITURE_TEXT_TYPES",
    "NON_SOURCE_TEXT_TYPES",
    "PRESERVED_VISUAL_TYPES",
    "canonical_ocr_type",
    "is_formula_type",
    "is_furniture_region_type",
    "is_furniture_text_type",
    "is_preserved_visual_type",
    "iter_source_text_blocks",
    "should_inject_source_text",
    "source_text_by_page",
]
