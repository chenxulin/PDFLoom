"""Restore heading hierarchy after coordinate-OCR PDF translation.

PP-StructureV3 returns tight glyph boxes.  They are ideal for masking scanned
source text, but target-language headings can be much longer than the source.
PDFMathTranslate then has no choice but to shrink the translation into the
source glyph width.  This module uses the retained ``title`` and
``section_heading`` semantics to redraw only those translated blocks in the
available content column, preserving the surrounding scan and body layout.
"""
from __future__ import annotations

import logging
import math
import os
import re
import statistics
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .ocr_bilingual_layout import detect_ocr_bilingual_layout
from .ocr_pdf_coordinates import (
    add_visual_redaction,
    insert_visual_textbox,
    is_visually_horizontal,
    map_ocr_rect_to_visual,
    pdf_rect_to_visual,
)
from .ocr_semantics import (
    canonical_ocr_type,
    is_numbered_heading_text,
    should_inject_source_text,
    visually_preserved_page_indices,
)

logger = logging.getLogger(__name__)

_HEADING_TYPES = frozenset({"title", "section_heading"})
_HEADER_TYPES = frozenset({"page_header", "header_image"})
_FOOTER_TYPES = frozenset({"page_footer", "footer_image"})
_FURNITURE_BAND_NEIGHBOUR_RATIO = 0.008
_FURNITURE_BAND_MAX_EXTENSION_RATIO = 0.03
_COLUMN_EXCLUDED_TYPES = frozenset(
    {
        "figure_caption",
        "page_aside_text",
        "page_footnote",
        "table_caption",
        "table_text",
    }
)
_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_CJK_GAP_RE = re.compile(
    r"(?<=[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af])\s+"
    r"(?=[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af])"
)
_TRAILING_SECTION_NUMBER_RE = re.compile(r"(?:^|\s)(\d+(?:\.\d+)+)[.)、]?\s*$")
_JOINED_SECTION_LABEL_RE = re.compile(
    r"^(\d+(?:\.\d+)+[.)]?)(?=[A-Za-z\u3400-\u9fff])"
)


@dataclass(frozen=True)
class OcrHeadingTypographyResult:
    translated_pdf_path: Path
    bilingual_pdf_path: Path | None
    heading_font_path: Path | None
    heading_count: int
    repaired_headings: int
    repaired_pages: int
    bilingual_repaired_headings: int
    bilingual_repaired_pages: int
    deferred_headings: tuple[OcrDeferredHeading, ...] = ()
    bilingual_deferred_headings: tuple[OcrDeferredHeading, ...] = ()


@dataclass(frozen=True)
class OcrDeferredHeading:
    page_idx: int
    source_rect: tuple[float, float, float, float]
    target_rect: tuple[float, float, float, float] | None
    text: str
    font_path: Path
    font_size: float
    align: int
    target_rects: tuple[tuple[float, float, float, float], ...] = ()


class OcrHeadingTranslationError(ValueError):
    """Raised when a complete OCR heading translation cannot be applied safely."""


@dataclass(frozen=True)
class OcrHeadingRegionTranslation:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    source_text: str
    target_text: str
    protected_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class OcrHeadingTranslationPlan:
    regions: tuple[OcrHeadingRegionTranslation, ...]
    region_count: int
    translated_regions: int
    protected_values: int


@dataclass(frozen=True)
class _HeadingBlock:
    index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    block_type: str


@dataclass(frozen=True)
class _HeadingRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    block_type: str
    member_bboxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class _SourceItem:
    rect: fitz.Rect
    block_type: str


@dataclass(frozen=True)
class _PdfTextBlock:
    rect: fitz.Rect
    text: str
    fonts: tuple[str, ...]
    horizontal: bool = True


@dataclass(frozen=True)
class _RepairPlan:
    source_rect: fitz.Rect
    target_rect: fitz.Rect | None
    draw_rect: fitz.Rect
    text: str
    font_path: Path
    font_size: float
    align: int
    deferred: bool = False
    target_rects: tuple[fitz.Rect, ...] = ()


@dataclass(frozen=True)
class _DocumentRepair:
    repaired_headings: int
    repaired_pages: int
    deferred_headings: tuple[OcrDeferredHeading, ...] = ()


def _positive_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and item > 0 for item in (width, height)):
        return None
    return width, height


def _positive_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        bbox = tuple(float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(item) for item in bbox)
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        return None
    return bbox


def _page_sizes(ocr_result: dict[str, Any]) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    for raw in ocr_result.get("pages") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        page_size = _positive_pair(raw.get("page_size"))
        if page_idx >= 0 and page_size is not None:
            sizes[page_idx] = page_size
    return sizes


def _bbox_union(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )


def _bbox_overlap_score(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1.0, min(first_area, second_area))


def _merge_adjacent_heading_regions(
    headings: list[_HeadingRegion],
) -> list[_HeadingRegion]:
    """Merge split number/label regions that form one visual heading line."""
    pending = sorted(headings, key=lambda item: (item.page_idx, item.bbox[1], item.bbox[0]))
    merged: list[_HeadingRegion] = []
    for heading in pending:
        match_index: int | None = None
        for index in range(len(merged) - 1, -1, -1):
            existing = merged[index]
            if existing.page_idx != heading.page_idx:
                break
            vertical_overlap = max(
                0.0,
                min(existing.bbox[3], heading.bbox[3])
                - max(existing.bbox[1], heading.bbox[1]),
            )
            vertical_ratio = vertical_overlap / max(
                1.0,
                min(
                    existing.bbox[3] - existing.bbox[1],
                    heading.bbox[3] - heading.bbox[1],
                ),
            )
            horizontal_gap = max(
                0.0,
                max(existing.bbox[0], heading.bbox[0])
                - min(existing.bbox[2], heading.bbox[2]),
            )
            if vertical_ratio >= 0.65 and horizontal_gap <= heading.page_size[0] * 0.03:
                match_index = index
                break
        if match_index is None:
            merged.append(heading)
            continue
        existing = merged[match_index]
        merged[match_index] = _HeadingRegion(
            page_idx=heading.page_idx,
            bbox=_bbox_union([existing.bbox, heading.bbox]),
            page_size=heading.page_size,
            block_type=(
                "title"
                if "title" in {existing.block_type, heading.block_type}
                else "section_heading"
            ),
            member_bboxes=existing.member_bboxes + heading.member_bboxes,
        )
    return merged


def _semantic_block_type(raw: dict[str, Any]) -> str:
    block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
    if block_type == "text" and is_numbered_heading_text(raw.get("text")):
        return "section_heading"
    return block_type


def _normalise_heading_regions(ocr_result: dict[str, Any]) -> list[_HeadingRegion]:
    preserved_pages = visually_preserved_page_indices(ocr_result)
    sizes = _page_sizes(ocr_result)
    header_bottoms: dict[int, float] = {}
    footer_tops: dict[int, float] = {}
    markers_by_page: dict[
        int, list[tuple[tuple[float, float, float, float], str, tuple[float, float]]]
    ] = {}
    blocks_by_page: dict[int, list[tuple[tuple[float, float, float, float], str]]] = {}
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if bbox is None or page_size is None:
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if region_type in _HEADER_TYPES | _FOOTER_TYPES:
            markers_by_page.setdefault(page_idx, []).append((bbox, region_type, page_size))

    represented_marker_types = {
        (page_idx, region_type)
        for page_idx, markers in markers_by_page.items()
        for _, region_type, _ in markers
    }
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if bbox is None or page_size is None:
            continue
        block_type = _semantic_block_type(raw)
        blocks_by_page.setdefault(page_idx, []).append((bbox, block_type))
        markers = markers_by_page.setdefault(page_idx, [])
        if block_type in _HEADER_TYPES | _FOOTER_TYPES and (
            page_idx,
            block_type,
        ) not in represented_marker_types:
            markers.append((bbox, block_type, page_size))

    excluded_markers = _HEADER_TYPES | _FOOTER_TYPES | {"page_number"}
    for page_idx, markers in markers_by_page.items():
        page_size = sizes.get(page_idx) or markers[0][2]
        height = page_size[1]
        blocks = blocks_by_page.get(page_idx, [])
        headers = [item for item in markers if item[1] in _HEADER_TYPES]
        footers = [item for item in markers if item[1] in _FOOTER_TYPES]
        if headers:
            bottom = max(bbox[3] for bbox, _, _ in headers)
            explicit_headers = [item for item in headers if item[1] == "page_header"]
            if len(explicit_headers) < 2:
                neighbour_anchor = (
                    max(item[0][3] for item in explicit_headers)
                    if explicit_headers
                    else bottom
                )
                neighbour_limit = (
                    neighbour_anchor + height * _FURNITURE_BAND_NEIGHBOUR_RATIO
                )
                maximum_bottom = (
                    neighbour_anchor + height * _FURNITURE_BAND_MAX_EXTENSION_RATIO
                )
                bottom = max(
                    [bottom]
                    + [
                        bbox[3]
                        for bbox, block_type in blocks
                        if block_type not in excluded_markers
                        and bbox[1] <= neighbour_limit
                        and bbox[3] <= maximum_bottom
                    ]
                )
            next_content_starts = [
                bbox[1]
                for bbox, block_type in blocks
                if block_type not in excluded_markers and bbox[1] >= bottom + 0.5
            ]
            if next_content_starts:
                bottom = max(
                    bottom,
                    min(min(next_content_starts) - 2.0, bottom + height * 0.01),
                )
            header_bottoms[page_idx] = bottom
        if footers:
            top = min(bbox[1] for bbox, _, _ in footers)
            explicit_footers = [item for item in footers if item[1] == "page_footer"]
            if len(explicit_footers) < 2:
                neighbour_anchor = (
                    min(item[0][1] for item in explicit_footers)
                    if explicit_footers
                    else top
                )
                neighbour_limit = (
                    neighbour_anchor - height * _FURNITURE_BAND_NEIGHBOUR_RATIO
                )
                minimum_top = (
                    neighbour_anchor - height * _FURNITURE_BAND_MAX_EXTENSION_RATIO
                )
                top = min(
                    [top]
                    + [
                        bbox[1]
                        for bbox, block_type in blocks
                        if block_type not in excluded_markers
                        and bbox[3] >= neighbour_limit
                        and bbox[1] >= minimum_top
                    ]
                )
            footer_tops[page_idx] = top

    def inside_furniture(
        page_idx: int, bbox: tuple[float, float, float, float]
    ) -> bool:
        return bbox[3] <= header_bottoms.get(page_idx, 0.0) or bbox[1] >= footer_tops.get(
            page_idx, math.inf
        )

    blocks: list[_HeadingBlock] = []
    for index, raw in enumerate(ocr_result.get("blocks") or []):
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        block_type = _semantic_block_type(raw)
        if block_type not in _HEADING_TYPES:
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        if inside_furniture(page_idx, bbox):
            continue
        blocks.append(_HeadingBlock(index, page_idx, bbox, page_size, block_type))

    raw_regions: list[tuple[int, tuple[float, float, float, float], tuple[float, float], str]] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        block_type = _semantic_block_type(raw)
        if block_type not in _HEADING_TYPES:
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        if inside_furniture(page_idx, bbox):
            continue
        raw_regions.append((page_idx, bbox, page_size, block_type))

    members_by_region: dict[int, list[_HeadingBlock]] = {index: [] for index in range(len(raw_regions))}
    matched_blocks: set[int] = set()
    for block in blocks:
        candidates: list[tuple[float, int]] = []
        for region_idx, (page_idx, bbox, _, _) in enumerate(raw_regions):
            if page_idx != block.page_idx:
                continue
            score = _bbox_overlap_score(block.bbox, bbox)
            if score > 0:
                candidates.append((score, region_idx))
        if not candidates:
            continue
        _, region_idx = max(candidates)
        members_by_region[region_idx].append(block)
        matched_blocks.add(block.index)

    headings: list[_HeadingRegion] = []
    for region_idx, (page_idx, bbox, page_size, block_type) in enumerate(raw_regions):
        members = members_by_region[region_idx]
        if not members:
            continue
        member_boxes = [item.bbox for item in members]
        headings.append(
            _HeadingRegion(
                page_idx=page_idx,
                bbox=_bbox_union([bbox, *member_boxes]),
                page_size=page_size,
                block_type=block_type,
                member_bboxes=tuple(member_boxes),
            )
        )

    for block in blocks:
        if block.index in matched_blocks:
            continue
        headings.append(
            _HeadingRegion(
                page_idx=block.page_idx,
                bbox=block.bbox,
                page_size=block.page_size,
                block_type=block.block_type,
                member_bboxes=(block.bbox,),
            )
        )
    return sorted(
        (
            heading
            for heading in _merge_adjacent_heading_regions(headings)
            if heading.page_idx not in preserved_pages
        ),
        key=lambda item: (item.page_idx, item.bbox[1], item.bbox[0]),
    )


def _heading_region_key(
    page_idx: int,
    bbox: tuple[float, float, float, float],
) -> tuple[int, tuple[float, float, float, float]]:
    return page_idx, tuple(round(value, 2) for value in bbox)


def _heading_source_texts(
    ocr_result: dict[str, Any],
    headings: list[_HeadingRegion],
) -> list[str]:
    blocks: list[
        tuple[int, tuple[float, float, float, float], str]
    ] = []
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        block_type = _semantic_block_type(raw)
        if block_type not in _HEADING_TYPES:
            continue
        text = str(raw.get("text") or "").strip()
        bbox = _positive_bbox(raw.get("bbox"))
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if page_idx >= 0 and bbox is not None and text:
            blocks.append((page_idx, bbox, text))

    sources: list[str] = []
    for heading in headings:
        parts = [
            (bbox, text)
            for page_idx, bbox, text in blocks
            if page_idx == heading.page_idx
            and _bbox_overlap_score(heading.bbox, bbox) > 0
        ]
        parts.sort(key=lambda item: (item[0][1], item[0][0]))
        texts = [text for _, text in parts]
        if not texts:
            raise OcrHeadingTranslationError(
                f"第 {heading.page_idx + 1} 页标题区域没有可翻译的 OCR 原文"
            )
        joined = "".join(texts) if _CJK_RE.search("".join(texts)) else " ".join(texts)
        sources.append(_CJK_GAP_RE.sub("", " ".join(joined.split())).strip())
    return sources


def _mapped_rect(
    segment: fitz.Rect,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    return map_ocr_rect_to_visual(segment, bbox, page_size)


def _clean_pdf_text(value: Any) -> str:
    raw = str(value or "")
    cleaned = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in raw
    )
    cleaned = " ".join(cleaned.split())
    return _CJK_GAP_RE.sub("", cleaned).strip()


def _normalize_heading_order(value: str) -> str:
    """Restore section-number order and separation around a heading label."""
    match = _TRAILING_SECTION_NUMBER_RE.search(value)
    normalized = value
    if match is not None and match.start() != 0:
        label = value[: match.start()].strip()
        if label:
            normalized = f"{match.group(1)} {label}"
    return _JOINED_SECTION_LABEL_RE.sub(r"\1 ", normalized)


def _pdf_text_blocks(page: fitz.Page, segment: fitz.Rect) -> list[_PdfTextBlock]:
    blocks: list[_PdfTextBlock] = []
    for raw in page.get_text("dict").get("blocks") or []:
        if raw.get("type") != 0:
            continue
        rect = pdf_rect_to_visual(
            page,
            fitz.Rect(raw.get("bbox") or (0, 0, 0, 0)),
        )
        if rect.is_empty or (rect & segment).is_empty:
            continue
        lines: list[str] = []
        fonts: set[str] = set()
        horizontal = True
        for line in raw.get("lines") or []:
            line_parts: list[str] = []
            for span in line.get("spans") or []:
                line_parts.append(str(span.get("text") or ""))
                font = str(span.get("font") or "").strip()
                if font:
                    fonts.add(font)
            lines.append("".join(line_parts))
            if line_parts:
                horizontal = horizontal and is_visually_horizontal(
                    page,
                    tuple(line.get("dir") or (1.0, 0.0)),
                )
        text = _clean_pdf_text("\n".join(lines))
        if text:
            blocks.append(
                _PdfTextBlock(
                    rect,
                    text,
                    tuple(sorted(fonts)),
                    horizontal,
                )
            )
    return blocks


def _source_items(
    ocr_result: dict[str, Any],
    page_idx: int,
    segment: fitz.Rect,
    fallback_page_size: tuple[float, float],
) -> list[_SourceItem]:
    items: list[_SourceItem] = []
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if raw_page_idx != page_idx:
            continue
        block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if not should_inject_source_text(block_type):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or fallback_page_size
        if bbox is None:
            continue
        rect = _mapped_rect(segment, bbox, page_size)
        if not rect.is_empty:
            items.append(_SourceItem(rect, block_type))
    return items


def _rect_overlap_ratio(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(1.0, min(first.get_area(), second.get_area()))


def _vertical_overlap_ratio(first: fitz.Rect, second: fitz.Rect) -> float:
    overlap = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return overlap / max(1.0, min(first.height, second.height))


def _horizontal_overlap_ratio(first: fitz.Rect, second: fitz.Rect) -> float:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    return overlap / max(1.0, min(first.width, second.width))


def _font_candidates(*, serif: bool, needs_cjk: bool) -> list[Path]:
    cache_root = Path(os.getenv("XDG_CACHE_HOME") or "/root/.cache") / "babeldoc" / "fonts"
    candidates: list[Path] = []
    if needs_cjk:
        candidates.extend(
            [
                cache_root / "SourceHanSerifCN-Bold.ttf",
                cache_root / "SourceHanSansCN-Bold.ttf",
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                cache_root / "GoNotoKurrent-Bold.ttf",
            ]
        )
    if serif:
        candidates.extend(
            [
                cache_root / "NotoSerif-Bold.ttf",
                Path("/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                cache_root / "NotoSans-Bold.ttf",
                Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ]
        )
    return candidates


def _resolve_bold_font(font_names: tuple[str, ...], text: str) -> Path:
    lowered = " ".join(font_names).casefold()
    # Use one serif family for every translated heading unless the document is
    # explicitly sans-serif. The body restorer resolves the matching regular
    # face from the same family.
    serif = "sans" not in lowered
    needs_cjk = bool(_CJK_RE.search(text))
    for candidate in _font_candidates(serif=serif, needs_cjk=needs_cjk):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No bold font is available for OCR heading restoration")


def _textbox_spare(
    text: str,
    rect: fitz.Rect,
    *,
    font_path: Path,
    font_size: float,
    align: int,
) -> float:
    scratch = fitz.open()
    try:
        page = scratch.new_page(width=max(1.0, rect.width), height=max(1.0, rect.height))
        return float(
            page.insert_textbox(
                page.rect,
                text,
                fontname="headingfit",
                fontfile=str(font_path),
                fontsize=font_size,
                lineheight=1.05,
                align=align,
            )
        )
    finally:
        scratch.close()


def _fit_heading_font_size(
    text: str,
    rect: fitz.Rect,
    *,
    font_path: Path,
    desired: float,
    minimum: float,
    align: int,
) -> float | None:
    size = desired
    while size >= minimum - 0.001:
        if _textbox_spare(text, rect, font_path=font_path, font_size=size, align=align) >= 0:
            return round(size, 2)
        size -= 0.25
    return None


def _is_document_title(
    heading: _HeadingRegion,
    source_rect: fitz.Rect,
    segment: fitz.Rect,
    content_width: float,
) -> bool:
    return heading.block_type == "title" or (
        source_rect.y0 < segment.y0 + segment.height * 0.25
        and source_rect.width >= content_width * 0.60
        and len(heading.member_bboxes) >= 2
    )


def _heading_alignment(
    heading: _HeadingRegion,
    segment: fitz.Rect,
    *,
    content_left: float,
    content_right: float,
) -> int:
    if heading.block_type != "title":
        return fitz.TEXT_ALIGN_LEFT
    member_rects = [
        _mapped_rect(segment, bbox, heading.page_size) for bbox in heading.member_bboxes
    ]
    content_width = max(1.0, content_right - content_left)
    widest = max((rect.width for rect in member_rects), default=content_width)
    if widest >= content_width * 0.78:
        return fitz.TEXT_ALIGN_LEFT
    content_center = (content_left + content_right) / 2
    center_offset = statistics.median(
        abs((rect.x0 + rect.x1) / 2 - content_center) for rect in member_rects
    )
    return (
        fitz.TEXT_ALIGN_CENTER
        if center_offset <= content_width * 0.06
        else fitz.TEXT_ALIGN_LEFT
    )


def _page_repair_plans(
    page: fitz.Page,
    *,
    page_idx: int,
    segment: fitz.Rect,
    headings: list[_HeadingRegion],
    ocr_result: dict[str, Any],
    heading_font: Path,
    heading_translations: dict[
        tuple[int, tuple[float, float, float, float]], str
    ] | None = None,
) -> list[_RepairPlan]:
    page_headings = [heading for heading in headings if heading.page_idx == page_idx]
    if not page_headings:
        return []
    fallback_size = page_headings[0].page_size
    source_items = _source_items(ocr_result, page_idx, segment, fallback_size)
    column_items = [
        item for item in source_items if item.block_type not in _COLUMN_EXCLUDED_TYPES
    ]
    if not column_items:
        logger.warning(
            "Page %d has headings but no reliable text column; using page-safe margins",
            page_idx + 1,
        )
    content_left = min(
        (item.rect.x0 for item in column_items),
        default=segment.x0 + segment.width * 0.08,
    )
    content_right = max(
        (item.rect.x1 for item in column_items),
        default=segment.x1 - segment.width * 0.08,
    )
    content_left = max(segment.x0 + segment.width * 0.025, content_left)
    content_right = min(segment.x1 - segment.width * 0.025, content_right)
    if content_right - content_left < segment.width * 0.25:
        content_left = segment.x0 + segment.width * 0.08
        content_right = segment.x1 - segment.width * 0.08
    content_width = content_right - content_left

    mapped_headings = [
        (heading, _mapped_rect(segment, heading.bbox, heading.page_size))
        for heading in page_headings
    ]
    target_blocks = _pdf_text_blocks(page, segment)
    candidate_pairs: list[tuple[float, int, int]] = []
    for heading_idx, (_, source_rect) in enumerate(mapped_headings):
        for target_idx, target in enumerate(target_blocks):
            score = _rect_overlap_ratio(source_rect, target.rect)
            vertical_overlap = _vertical_overlap_ratio(source_rect, target.rect)
            if score > 0:
                candidate_pairs.append(
                    (score + vertical_overlap, heading_idx, target_idx)
                )
            elif not target.horizontal and vertical_overlap >= 0.45:
                # A layout engine may ignore PDF /Rotate and move the target
                # title sideways while keeping it in the correct visual band.
                # Match that residue so it is redacted and redrawn horizontally.
                candidate_pairs.append(
                    (0.40 + vertical_overlap, heading_idx, target_idx)
                )

    assignments: dict[int, list[int]] = {}
    used_targets: set[int] = set()
    for _, heading_idx, target_idx in sorted(candidate_pairs, reverse=True):
        if target_idx in used_targets:
            continue
        assignments.setdefault(heading_idx, []).append(target_idx)
        used_targets.add(target_idx)
    if len(assignments) != len(mapped_headings):
        logger.warning(
            "Page %d matched %d/%d headings; trusted unmatched translations "
            "will be handled by the page layout solver",
            page_idx + 1,
            len(assignments),
            len(mapped_headings),
        )

    plans: list[_RepairPlan] = []
    for heading_idx, (heading, source_rect) in enumerate(mapped_headings):
        target_indices = assignments.get(heading_idx, [])
        target_indices.sort(
            key=lambda index: (
                target_blocks[index].rect.y0,
                target_blocks[index].rect.x0,
            )
        )
        assigned_targets = [target_blocks[index] for index in target_indices]
        target = assigned_targets[0] if assigned_targets else None
        target_rect = (
            fitz.Rect(assigned_targets[0].rect) if assigned_targets else None
        )
        for assigned in assigned_targets[1:]:
            assert target_rect is not None
            target_rect.include_rect(assigned.rect)
        translation = (heading_translations or {}).get(
            _heading_region_key(heading.page_idx, heading.bbox)
        )
        if target is None and translation is None:
            logger.warning(
                "Page %d unmatched heading has no trusted serial translation; "
                "preserving the layout-engine output",
                page_idx + 1,
            )
            continue
        translated_text = (
            translation
            if translation is not None
            else " ".join(dict.fromkeys(item.text for item in assigned_targets))
        )
        heading_text = _normalize_heading_order(_clean_pdf_text(translated_text))
        if not heading_text:
            raise RuntimeError(f"Page {page_idx + 1} heading translation is empty")
        unsafe_target = any(
            item.rect.height > max(40.0, source_rect.height * 3.2)
            or len(item.text) > 1000
            for item in assigned_targets
        )
        targets_are_horizontal = not assigned_targets or all(
            item.horizontal for item in assigned_targets
        )
        document_title = _is_document_title(heading, source_rect, segment, content_width)
        desired = 12.0 if document_title else 10.5
        align = _heading_alignment(
            heading,
            segment,
            content_left=content_left,
            content_right=content_right,
        )

        available_right = content_right
        for item in column_items:
            if item.rect.x0 <= source_rect.x1 + 2:
                continue
            if _vertical_overlap_ratio(source_rect, item.rect) >= 0.30:
                available_right = min(available_right, item.rect.x0 - 3)
        draw_left = max(
            content_left,
            min(source_rect.x0, target_rect.x0)
            if target_rect is not None and targets_are_horizontal
            else source_rect.x0,
        )
        safe_width = available_right - draw_left
        # PDFMathTranslate's target glyph box may extend slightly beyond the
        # source content column, especially after shrink-to-fit rendering.  It
        # is a redaction/matching box, not the minimum width needed to redraw
        # the heading: the fixed-size textbox preflight below is the authority
        # on whether the translated text can wrap safely in this column.
        unsafe_width = safe_width < segment.width * 0.18

        next_y_candidates = [
            item.rect.y0
            for item in column_items
            if item.rect.y0 > source_rect.y1 + 0.5
            and _horizontal_overlap_ratio(
                fitz.Rect(draw_left, source_rect.y0, available_right, source_rect.y1),
                item.rect,
            )
            >= 0.08
        ]
        next_y = min(next_y_candidates) if next_y_candidates else min(
            segment.y1, source_rect.y1 + max(36.0, source_rect.height * 2.5)
        )
        base_top = (
            min(source_rect.y0, target_rect.y0)
            if target_rect is not None and targets_are_horizontal
            else source_rect.y0
        )
        draw_top = max(segment.y0, base_top - 0.5)
        draw_bottom = min(segment.y1, next_y - 1.25)
        unsafe_height = draw_bottom - draw_top < 8
        draw_rect = (
            fitz.Rect(draw_left, draw_top, available_right, draw_bottom)
            if not unsafe_width and not unsafe_height
            else fitz.Rect(source_rect)
        )
        fitted = (
            _fit_heading_font_size(
                heading_text,
                draw_rect,
                font_path=heading_font,
                desired=desired,
                minimum=desired,
                align=align,
            )
            if not unsafe_width and not unsafe_height and not unsafe_target
            else None
        )
        deferred = fitted is None
        if deferred:
            logger.warning(
                "Page %d heading cannot use its source-page box at the fixed "
                "%.1f pt level; scheduling it for the page layout solver",
                page_idx + 1,
                desired,
            )
        plans.append(
            _RepairPlan(
                source_rect=source_rect,
                target_rect=fitz.Rect(target_rect) if target_rect is not None else None,
                draw_rect=draw_rect,
                text=heading_text,
                font_path=heading_font,
                font_size=fitted if fitted is not None else desired,
                align=align,
                deferred=deferred,
                target_rects=tuple(
                    fitz.Rect(item.rect) for item in assigned_targets
                ),
            )
        )
    return plans


def _font_resource_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", path.stem)[:24]
    return f"ocrheading{cleaned or 'bold'}"


def _apply_page_plans(page: fitz.Page, plans: list[_RepairPlan]) -> None:
    immediate = [plan for plan in plans if not plan.deferred]
    for plan in immediate:
        redaction_rects = plan.target_rects or (
            (plan.target_rect,) if plan.target_rect is not None else ()
        )
        for raw_rect in redaction_rects:
            redact_rect = fitz.Rect(raw_rect)
            redact_rect.x0 -= 0.7
            redact_rect.y0 -= 0.7
            redact_rect.x1 += 0.7
            redact_rect.y1 += 0.7
            add_visual_redaction(page, redact_rect, fill=None, cross_out=False)
    if any(plan.target_rects or plan.target_rect is not None for plan in immediate):
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )
    for plan in immediate:
        spare = insert_visual_textbox(
            page,
            plan.draw_rect,
            plan.text,
            fontname=_font_resource_name(plan.font_path),
            fontfile=str(plan.font_path),
            fontsize=plan.font_size,
            lineheight=1.05,
            align=plan.align,
            color=(0, 0, 0),
            overlay=True,
        )
        if spare < 0:
            raise RuntimeError("OCR heading unexpectedly overflowed after preflight fitting")


def _save_replacement(document: fitz.Document, destination: Path) -> None:
    original_mode = destination.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-heading-",
        suffix=".pdf",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        document.save(str(temporary), garbage=4, deflate=True)
        temporary.chmod(original_mode)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _repair_document(
    pdf_path: Path,
    *,
    headings: list[_HeadingRegion],
    ocr_result: dict[str, Any],
    segment_factory: Any,
    logical_page_factory: Any,
    heading_font: Path,
    heading_translations: dict[
        tuple[int, tuple[float, float, float, float]], str
    ] | None = None,
) -> _DocumentRepair:
    document = fitz.open(str(pdf_path))
    repaired_headings = 0
    repaired_pages = 0
    deferred_headings: list[OcrDeferredHeading] = []
    try:
        for page_idx in range(document.page_count):
            logical_page_idx = logical_page_factory(page_idx)
            if logical_page_idx is None:
                continue
            segment = segment_factory(document[page_idx], page_idx)
            if segment is None:
                continue
            plans = _page_repair_plans(
                document[page_idx],
                page_idx=logical_page_idx,
                segment=segment,
                headings=headings,
                ocr_result=ocr_result,
                heading_font=heading_font,
                heading_translations=heading_translations,
            )
            if not plans:
                continue
            deferred_headings.extend(
                OcrDeferredHeading(
                    page_idx=logical_page_idx,
                    source_rect=tuple(float(value) for value in plan.source_rect),
                    target_rect=(
                        tuple(float(value) for value in plan.target_rect)
                        if plan.target_rect is not None
                        else None
                    ),
                    text=plan.text,
                    font_path=plan.font_path,
                    font_size=plan.font_size,
                    align=plan.align,
                    target_rects=tuple(
                        tuple(float(value) for value in rect)
                        for rect in plan.target_rects
                    ),
                )
                for plan in plans
                if plan.deferred
            )
            immediate_count = sum(not plan.deferred for plan in plans)
            if not immediate_count:
                continue
            _apply_page_plans(document[page_idx], plans)
            repaired_headings += immediate_count
            repaired_pages += 1
        if repaired_headings:
            _save_replacement(document, pdf_path)
    finally:
        document.close()
    return _DocumentRepair(
        repaired_headings,
        repaired_pages,
        tuple(deferred_headings),
    )


def restore_ocr_heading_typography(
    *,
    ocr_result: dict[str, Any],
    translated_pdf: str | Path,
    bilingual_pdf: str | Path | None = None,
    heading_translation_plan: OcrHeadingTranslationPlan | None = None,
) -> OcrHeadingTypographyResult:
    """Restore semantic headings in mono and supported bilingual PDF layouts."""
    translated_path = Path(translated_pdf)
    if not translated_path.is_file():
        raise FileNotFoundError(f"Translated PDF does not exist: {translated_path}")
    headings = _normalise_heading_regions(ocr_result)
    if not headings:
        return OcrHeadingTypographyResult(
            translated_path,
            Path(bilingual_pdf) if bilingual_pdf else None,
            None,
            0,
            0,
            0,
            0,
            0,
        )

    heading_translations: dict[
        tuple[int, tuple[float, float, float, float]], str
    ] = {}
    if heading_translation_plan is not None:
        if (
            heading_translation_plan.region_count != len(headings)
            or len(heading_translation_plan.regions) != len(headings)
        ):
            raise OcrHeadingTranslationError(
                "标题翻译计划与 OCR 标题区域数量不一致"
            )
        heading_translations = {
            _heading_region_key(item.page_idx, item.bbox): item.target_text
            for item in heading_translation_plan.regions
        }
        expected_keys = {
            _heading_region_key(item.page_idx, item.bbox) for item in headings
        }
        if set(heading_translations) != expected_keys:
            raise OcrHeadingTranslationError(
                "标题翻译计划与 OCR 标题区域坐标不一致"
            )

    mono_document = fitz.open(str(translated_path))
    try:
        needs_cjk = bool(_CJK_RE.search("\n".join(page.get_text("text") for page in mono_document)))
    finally:
        mono_document.close()
    heading_font = _resolve_bold_font((), "中" if needs_cjk else "")

    mono = _repair_document(
        translated_path,
        headings=headings,
        ocr_result=ocr_result,
        segment_factory=lambda page, _page_idx: fitz.Rect(page.rect),
        logical_page_factory=lambda page_idx: page_idx,
        heading_font=heading_font,
        heading_translations=heading_translations,
    )

    bilingual_path = Path(bilingual_pdf) if bilingual_pdf else None
    bilingual = _DocumentRepair(0, 0)
    if bilingual_path is not None and bilingual_path.is_file():
        layout = detect_ocr_bilingual_layout(translated_path, bilingual_path)
        if layout is not None:
            bilingual = _repair_document(
                bilingual_path,
                headings=headings,
                ocr_result=ocr_result,
                segment_factory=layout.target_segment,
                logical_page_factory=layout.logical_page_index,
                heading_font=heading_font,
                heading_translations=heading_translations,
            )

    return OcrHeadingTypographyResult(
        translated_pdf_path=translated_path,
        bilingual_pdf_path=bilingual_path,
        heading_font_path=heading_font,
        heading_count=len(headings),
        repaired_headings=mono.repaired_headings,
        repaired_pages=mono.repaired_pages,
        bilingual_repaired_headings=bilingual.repaired_headings,
        bilingual_repaired_pages=bilingual.repaired_pages,
        deferred_headings=mono.deferred_headings,
        bilingual_deferred_headings=bilingual.deferred_headings,
    )


__all__ = [
    "OcrDeferredHeading",
    "OcrHeadingRegionTranslation",
    "OcrHeadingTranslationError",
    "OcrHeadingTranslationPlan",
    "OcrHeadingTypographyResult",
    "restore_ocr_heading_typography",
]
