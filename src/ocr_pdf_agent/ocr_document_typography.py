"""Normalize scanned-PDF body typography and restore numeric page numbers.

The OCR bridge intentionally preserves page geometry. Target-language prose
can nevertheless be shrunk unevenly when a short Chinese line expands in
English. This module redraws semantic prose regions with one document-wide
regular face and 9 pt size, removes header/footer bands, and restores page
numbers without translating them. It first restores tables as clean source
pixels so the dedicated final pass can replace each complete table with a
translated vector grid; figures, images, charts and seals remain untouched.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .config import Settings
from .ocr_bilingual_layout import detect_ocr_bilingual_layout
from .ocr_heading_typography import (
    OcrHeadingTranslationPlan,
    OcrHeadingTypographyResult,
    restore_ocr_heading_typography,
)
from .ocr_pdf_coordinates import (
    add_visual_redaction,
    insert_visual_textbox,
    is_visually_horizontal,
    map_ocr_rect_to_visual,
    pdf_rect_to_visual,
    show_pdf_page_visual,
)
from .ocr_semantics import (
    canonical_ocr_type,
    is_numbered_heading_text,
    should_inject_source_text,
    visually_preserved_page_indices,
)
from .ocr_table_redraw import (
    _expose_values,
    _needs_translation,
    _protect_values,
    _restore_values,
    _translation_is_valid,
)
from .translator import _build_client, translate_chunk

logger = logging.getLogger(__name__)

_BODY_FONT_SIZE = 9.0
_BODY_LINE_HEIGHT = 1.25
_VERTICAL_RHYTHM_GAP = _BODY_FONT_SIZE * _BODY_LINE_HEIGHT
_MIN_VERTICAL_RHYTHM_GAP = _BODY_FONT_SIZE * 0.50
_CONTINUATION_MARGIN_X_RATIO = 0.075
_CONTINUATION_MARGIN_TOP = 42.0
_CONTINUATION_MARGIN_BOTTOM = 48.0
_PAGE_NUMBER_FONT_SIZE = 8.0
_PAGE_NUMBER_CONTENT_GAP = 4.0
_BODY_REGION_TYPE = "text"
_VISUAL_REGION_TYPES = frozenset({"table", "image", "chart", "seal"})
_PIXEL_PRESERVED_REGION_TYPES = _VISUAL_REGION_TYPES | {
    "figure_caption",
    "table_caption",
}
_COLUMN_EXCLUDED_TYPES = frozenset(
    {"figure_caption", "page_aside_text", "page_footnote", "table_caption", "table_text"}
)
_HEADER_TYPES = frozenset({"page_header", "header_image"})
_FOOTER_TYPES = frozenset({"page_footer", "footer_image"})
_FURNITURE_BAND_NEIGHBOUR_RATIO = 0.008
_FURNITURE_BAND_MAX_EXTENSION_RATIO = 0.03
_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_CJK_GAP_RE = re.compile(
    r"(?<=[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af])\s+"
    r"(?=[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af])"
)
_ASCII_SHORT_TAIL_RE = re.compile(r"\b([A-Za-z]{2,})\s+([a-z]{1,3})\b")
_ASCII_SHORT_HEAD_RE = re.compile(r"\b([b-z])\s+([a-z]{3,})\b")
_SHORT_DOCUMENT_LABEL_RE = re.compile(
    r"^\s*(?:附件|附录|appendix|annex|attachment)\s*"
    r"(?:\d+|[IVXLCDM]+|[一二三四五六七八九十]+)\s*[:：]?\s*$",
    flags=re.IGNORECASE,
)
_REPAIRABLE_WORD_SUFFIXES = (
    "able",
    "ally",
    "ation",
    "ed",
    "edly",
    "ible",
    "ing",
    "ion",
    "ly",
    "ment",
    "ness",
    "ody",
    "sion",
    "tion",
)


@dataclass(frozen=True)
class OcrDocumentTypographyResult:
    translated_pdf_path: Path
    bilingual_pdf_path: Path | None
    body_font_path: Path
    body_font_size: float
    body_line_height: float
    paragraph_gap: float
    body_region_count: int
    repaired_body_blocks: int
    repaired_body_pages: int
    restored_page_numbers: int
    removed_header_footer_bands: int
    protected_visual_regions: int
    bilingual_repaired_body_blocks: int
    bilingual_restored_page_numbers: int
    bilingual_protected_visual_regions: int
    rhythm_blocks: int
    bilingual_rhythm_blocks: int
    headings: OcrHeadingTypographyResult
    continuation_pages: int = 0
    source_page_indices: tuple[int, ...] = ()
    continuation_page_indices: tuple[int, ...] = ()
    continuation_source_pages: tuple[int, ...] = ()


class OcrBodyTranslationError(ValueError):
    """Raised when a clean OCR body translation cannot be produced safely."""


@dataclass(frozen=True)
class OcrBodyRegionTranslation:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    source_text: str
    target_text: str
    protected_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class OcrBodyTranslationPlan:
    regions: tuple[OcrBodyRegionTranslation, ...]
    region_count: int
    translated_regions: int
    protected_values: int


@dataclass(frozen=True)
class _BodyRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]


@dataclass(frozen=True)
class _PageNumber:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    text: str


@dataclass(frozen=True)
class _FurnitureBand:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]


@dataclass(frozen=True)
class _PixelRegion:
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    region_type: str


@dataclass(frozen=True)
class _SourceItem:
    rect: fitz.Rect
    block_type: str


@dataclass(frozen=True)
class _PdfBlock:
    rect: fitz.Rect
    text: str
    line_x0s: tuple[float, ...]
    horizontal: bool = True


@dataclass(frozen=True)
class _BodyPlan:
    target_rects: tuple[fitz.Rect, ...]
    draw_rect: fitz.Rect
    text: str
    font_size: float
    line_height: float
    anchor_rect: fitz.Rect
    deferred: bool = False


@dataclass(frozen=True)
class _RhythmItem:
    rect: fitz.Rect
    text: str
    font_path: Path
    font_size: float
    line_height: float
    align: int = fitz.TEXT_ALIGN_LEFT
    redaction_rects: tuple[fitz.Rect, ...] = ()
    deferred: bool = False


@dataclass(frozen=True)
class _RhythmPlacement:
    item: _RhythmItem
    draw_rect: fitz.Rect


@dataclass(frozen=True)
class _RhythmPageLayout:
    placements: tuple[_RhythmPlacement, ...]
    overflow: tuple[_RhythmItem, ...]


@dataclass(frozen=True)
class _RhythmRepair:
    blocks: int
    pages: int
    source_page_indices: tuple[int, ...] = ()
    continuation_page_indices: tuple[int, ...] = ()
    continuation_source_pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ContinuationLineage:
    source_page_indices: tuple[int, ...]
    continuation_page_indices: tuple[int, ...]
    continuation_source_pages: tuple[int, ...]


@dataclass(frozen=True)
class _DeferredBodyBlock:
    page_idx: int
    item: _RhythmItem


@dataclass(frozen=True)
class _DocumentRepair:
    repaired_body_blocks: int
    repaired_body_pages: int
    restored_page_numbers: int
    removed_header_footer_bands: int
    protected_visual_regions: int
    deferred_body_blocks: tuple[_DeferredBodyBlock, ...] = ()


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
    result: dict[int, tuple[float, float]] = {}
    for raw in ocr_result.get("pages") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        page_size = _positive_pair(raw.get("page_size"))
        if page_idx >= 0 and page_size is not None:
            result[page_idx] = page_size
    return result


def _mapped_rect(
    segment: fitz.Rect,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    return map_ocr_rect_to_visual(segment, bbox, page_size)


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


def _bbox_union(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _is_table_caption_candidate(
    bbox: tuple[float, float, float, float],
    table_boxes: list[tuple[float, float, float, float]],
    page_size: tuple[float, float],
) -> bool:
    """Recognize a short text line immediately above a table as its caption."""
    _, page_height = page_size
    for table in table_boxes:
        horizontal_overlap = max(
            0.0,
            min(bbox[2], table[2]) - max(bbox[0], table[0]),
        )
        if horizontal_overlap < min(bbox[2] - bbox[0], table[2] - table[0]) * 0.25:
            continue
        if (
            bbox[1] < table[1]
            and table[1] - page_height * 0.04 <= bbox[3]
            and bbox[3] <= table[1] + page_height * 0.015
        ):
            return True
    return False


def _deduplicate_body_regions(regions: list[_BodyRegion]) -> list[_BodyRegion]:
    """Remove duplicated PP-Structure regions and parent wrappers.

    Some pages contain both one parent paragraph region and two child regions,
    or the same region twice. Keeping all of them creates a false mismatch
    against the PDF's single text block even though every paragraph is covered.
    """
    unique: list[_BodyRegion] = []
    for region in regions:
        area = _bbox_area(region.bbox)
        duplicate = False
        for existing in unique:
            if existing.page_idx != region.page_idx:
                continue
            shared = _bbox_intersection_area(existing.bbox, region.bbox)
            if shared / max(1.0, max(area, _bbox_area(existing.bbox))) >= 0.90:
                duplicate = True
                break
        if not duplicate:
            unique.append(region)

    redundant: set[int] = set()
    for outer_idx, outer in enumerate(unique):
        outer_area = _bbox_area(outer.bbox)
        children: list[_BodyRegion] = []
        for inner_idx, inner in enumerate(unique):
            if inner_idx == outer_idx or inner.page_idx != outer.page_idx:
                continue
            inner_area = _bbox_area(inner.bbox)
            center_x = (inner.bbox[0] + inner.bbox[2]) / 2
            center_y = (inner.bbox[1] + inner.bbox[3]) / 2
            if (
                inner_area < outer_area * 0.75
                and outer.bbox[0] <= center_x <= outer.bbox[2]
                and outer.bbox[1] <= center_y <= outer.bbox[3]
            ):
                children.append(inner)
        covered = sum(
            _bbox_intersection_area(outer.bbox, child.bbox) for child in children
        )
        if len(children) >= 2 and covered / max(1.0, outer_area) >= 0.60:
            redundant.add(outer_idx)
    return [region for index, region in enumerate(unique) if index not in redundant]


def _normalise_body_regions(ocr_result: dict[str, Any]) -> list[_BodyRegion]:
    preserved_pages = visually_preserved_page_indices(ocr_result)
    sizes = _page_sizes(ocr_result)
    blocks_by_page: dict[
        int,
        list[tuple[tuple[float, float, float, float], tuple[float, float]]],
    ] = {}
    heading_boxes_by_page: dict[
        int, list[tuple[float, float, float, float]]
    ] = {}
    short_document_labels_by_page: dict[
        int, list[tuple[float, float, float, float]]
    ] = {}
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        promoted_heading = block_type == "text" and is_numbered_heading_text(
            raw.get("text")
        )
        if block_type in {"title", "section_heading"} or promoted_heading:
            heading_boxes_by_page.setdefault(page_idx, []).append(bbox)
            continue
        if block_type == _BODY_REGION_TYPE:
            blocks_by_page.setdefault(page_idx, []).append((bbox, page_size))
            normalized_text = unicodedata.normalize(
                "NFKC", str(raw.get("text") or "")
            )
            if _SHORT_DOCUMENT_LABEL_RE.fullmatch(normalized_text):
                short_document_labels_by_page.setdefault(page_idx, []).append(bbox)

    def follows_heading(
        page_idx: int,
        bbox: tuple[float, float, float, float],
        page_size: tuple[float, float],
    ) -> bool:
        width, height = page_size
        for heading_bbox in heading_boxes_by_page.get(page_idx, []):
            gap = bbox[1] - heading_bbox[3]
            if not 0.0 <= gap <= height * 0.04:
                continue
            horizontal_gap = max(
                0.0,
                max(heading_bbox[0], bbox[0])
                - min(heading_bbox[2], bbox[2]),
            )
            if horizontal_gap <= width * 0.08:
                return True
        return False

    visuals_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    tables_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    header_bottoms: dict[int, float] = {}
    footer_tops: dict[int, float] = {}
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        if page_idx < 0 or bbox is None:
            continue
        if region_type in _VISUAL_REGION_TYPES:
            visuals_by_page.setdefault(page_idx, []).append(bbox)
            if region_type == "table":
                tables_by_page.setdefault(page_idx, []).append(bbox)
        elif region_type in _HEADER_TYPES:
            header_bottoms[page_idx] = max(
                header_bottoms.get(page_idx, 0.0),
                bbox[3],
            )
        elif region_type in _FOOTER_TYPES:
            footer_tops[page_idx] = min(
                footer_tops.get(page_idx, math.inf),
                bbox[1],
            )

    regions: list[_BodyRegion] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        if canonical_ocr_type(raw.get("type") or raw.get("sub_type")) != _BODY_REGION_TYPE:
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        members = [
            block_bbox
            for block_bbox, _ in blocks_by_page.get(page_idx, [])
            if _bbox_overlap_score(bbox, block_bbox) > 0
        ]
        if not members:
            continue
        combined = _bbox_union([bbox, *members])
        width, height = page_size
        short_document_label = any(
            _bbox_overlap_score(combined, label_bbox) > 0
            for label_bbox in short_document_labels_by_page.get(page_idx, [])
        )
        table_caption = _is_table_caption_candidate(
            combined,
            tables_by_page.get(page_idx, []),
            page_size,
        )
        # Judge the top margin from PP-Structure's semantic region rather than
        # the union. Its text-line boxes can extend a little above a valid body
        # region (or only graze its top edge). Rejecting the expanded union here
        # drops the paragraph and lets the orphan fallback promote each line as
        # a separate body region, while the PDF layout engine commonly keeps
        # those lines in one text block.
        if bbox[1] < height * 0.10 and not (
            short_document_label
            and bbox[1] > header_bottoms.get(page_idx, 0.0) + 2.0
        ):
            continue
        if (
            combined[2] - combined[0] < width * 0.22
            and len(members) < 2
            and not table_caption
            and not follows_heading(page_idx, combined, page_size)
            and not short_document_label
        ):
            continue
        if not table_caption and any(
            _bbox_overlap_score(combined, visual) >= 0.08
            for visual in visuals_by_page.get(page_idx, [])
        ):
            continue
        regions.append(_BodyRegion(page_idx, combined, page_size))

    # Layout detection occasionally omits a region for a wide continuation or
    # title line even though OCR returns a reliable ``text`` block. Leaving such
    # a block to the layout engine allowed model preambles, NUL glyphs and
    # truncated protected identifiers to survive at the top of a page. Promote
    # uncovered, wide blocks below the detected header into first-class body
    # regions so the serial translator and deterministic redraw own them too.
    for page_idx, blocks in blocks_by_page.items():
        for bbox, page_size in blocks:
            if any(
                region.page_idx == page_idx
                and _bbox_overlap_score(region.bbox, bbox) > 0
                for region in regions
            ):
                continue
            width, _ = page_size
            table_caption = _is_table_caption_candidate(
                bbox,
                tables_by_page.get(page_idx, []),
                page_size,
            )
            if (
                bbox[2] - bbox[0] < width * 0.22
                and not table_caption
                and not follows_heading(page_idx, bbox, page_size)
            ):
                continue
            if bbox[1] <= header_bottoms.get(page_idx, 0.0) + 2.0:
                continue
            if bbox[3] >= footer_tops.get(page_idx, math.inf) - 2.0:
                continue
            if not table_caption and any(
                _bbox_overlap_score(bbox, visual) >= 0.08
                for visual in visuals_by_page.get(page_idx, [])
            ):
                continue
            regions.append(_BodyRegion(page_idx, bbox, page_size))
    return sorted(
        (
            region
            for region in _deduplicate_body_regions(regions)
            if region.page_idx not in preserved_pages
        ),
        key=lambda item: (item.page_idx, item.bbox[1], item.bbox[0]),
    )


def _body_region_key(
    page_idx: int,
    bbox: tuple[float, float, float, float],
) -> tuple[int, tuple[float, float, float, float]]:
    return page_idx, tuple(round(value, 2) for value in bbox)


def _body_source_texts(
    ocr_result: dict[str, Any],
    regions: list[_BodyRegion],
) -> list[str]:
    blocks: list[
        tuple[int, tuple[float, float, float, float], str]
    ] = []
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        if canonical_ocr_type(raw.get("type") or raw.get("sub_type")) != _BODY_REGION_TYPE:
            continue
        if is_numbered_heading_text(raw.get("text")):
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
    for region in regions:
        parts = [
            (bbox, text)
            for page_idx, bbox, text in blocks
            if page_idx == region.page_idx and _bbox_overlap_score(region.bbox, bbox) > 0
        ]
        parts.sort(key=lambda item: (item[0][1], item[0][0]))
        texts = [text for _, text in parts]
        if not texts:
            raise OcrBodyTranslationError(
                f"第 {region.page_idx + 1} 页正文区域没有可翻译的 OCR 原文"
            )
        joined = "".join(texts) if _CJK_RE.search("".join(texts)) else " ".join(texts)
        sources.append(_CJK_GAP_RE.sub("", " ".join(joined.split())).strip())
    return sources


async def translate_ocr_body_regions(
    *,
    ocr_result: dict[str, Any],
    settings: Settings,
    on_progress: Callable[[str, float, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> OcrBodyTranslationPlan:
    """Translate OCR prose directly so tight PDF boxes cannot corrupt words."""
    regions = _normalise_body_regions(ocr_result)
    if not regions:
        return OcrBodyTranslationPlan((), 0, 0, 0)
    sources = _body_source_texts(ocr_result, regions)
    results: list[OcrBodyRegionTranslation | None] = [None] * len(regions)
    candidates = [
        index
        for index, source in enumerate(sources)
        if _needs_translation(source, settings.target_language)
    ]
    for index, (region, source) in enumerate(zip(regions, sources, strict=True)):
        if index not in candidates:
            results[index] = OcrBodyRegionTranslation(
                region.page_idx,
                region.bbox,
                region.page_size,
                source,
                source,
            )
    if not candidates:
        return OcrBodyTranslationPlan(
            tuple(item for item in results if item is not None),
            len(regions),
            0,
            0,
        )

    if on_progress:
        on_progress("body-translate", 0.0, f"翻译正文段落：0/{len(candidates)}")
    client = _build_client(settings)
    concurrency = min(6, max(1, int(getattr(settings, "max_workers", 4) or 4)))
    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    done = 0

    async def translate_one(index: int) -> None:
        nonlocal done
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        source = sources[index]
        protected_text, values = _protect_values(source)
        protected_text = _expose_values(protected_text, values)
        previous = sources[index - 1] if index else ""
        following = sources[index + 1] if index + 1 < len(sources) else ""
        async with semaphore:
            last_error: Exception | None = None
            rejected_translation: str | None = None
            translated = ""
            attempt_count = 3 if values else 2
            for attempt in range(attempt_count):
                translated = ""
                try:
                    translated = await translate_chunk(
                        client,
                        protected_text,
                        previous if attempt < 2 else "",
                        following if attempt < 2 else "",
                        settings,
                        seg_type="para",
                        source_kind="pdf",
                        has_layout=True,
                        layout_retry_reason=(
                            "ocr_body_placeholder_integrity" if attempt else None
                        ),
                        required_literals=tuple(value for _, value in values),
                        rejected_translation=rejected_translation,
                    )
                    translated = _restore_values(translated, values)
                    if not _translation_is_valid(
                        source,
                        translated,
                        settings.target_language,
                    ):
                        raise OcrBodyTranslationError(
                            "正文未完整翻译到目标语言"
                        )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 - strict integrity retries
                    last_error = exc
                    if translated.strip():
                        rejected_translation = translated
            if last_error is not None:
                raise OcrBodyTranslationError(
                    f"第 {regions[index].page_idx + 1} 页正文段落翻译失败：{last_error}"
                ) from last_error
            region = regions[index]
            results[index] = OcrBodyRegionTranslation(
                region.page_idx,
                region.bbox,
                region.page_size,
                source,
                translated,
                tuple(value for _, value in values),
            )
        async with progress_lock:
            done += 1
            if on_progress:
                on_progress(
                    "body-translate",
                    100.0 * done / len(candidates),
                    f"翻译正文段落：{done}/{len(candidates)}",
                )

    try:
        await asyncio.gather(*(translate_one(index) for index in candidates))
    finally:
        await client.aclose()

    completed = tuple(item for item in results if item is not None)
    if len(completed) != len(regions):
        raise OcrBodyTranslationError("正文翻译计划不完整")
    return OcrBodyTranslationPlan(
        completed,
        len(regions),
        len(candidates),
        sum(len(item.protected_values) for item in completed),
    )


def _normalise_page_number(value: str, page_idx: int) -> str | None:
    normalized = unicodedata.normalize("NFKC", value or "")
    compact = " ".join(normalized.split())
    fraction = re.fullmatch(r"(\d+)\s*[/／]\s*(\d+)", compact)
    if fraction:
        return f"{fraction.group(1)}/{fraction.group(2)}"
    single = re.fullmatch(
        r"(?:第\s*)?(\d+)(?:\s*页|\s*page)?",
        compact,
        flags=re.IGNORECASE,
    )
    if single:
        return single.group(1)
    # PP-Structure occasionally classifies the B3 company logo as a page
    # number.  Extracting any digit from arbitrary alphanumeric text turned
    # that logo into a stray "3" on every translated page.
    return None


def _furniture_and_page_numbers(
    ocr_result: dict[str, Any],
) -> tuple[list[_FurnitureBand], list[_PageNumber]]:
    preserved_pages = visually_preserved_page_indices(ocr_result)
    sizes = _page_sizes(ocr_result)
    regions_by_page: dict[
        int, list[tuple[tuple[float, float, float, float], str, tuple[float, float]]]
    ] = {}
    blocks_by_page: dict[int, list[tuple[tuple[float, float, float, float], str, str]]] = {}
    page_numbers: list[_PageNumber] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if page_idx in preserved_pages:
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        regions_by_page.setdefault(page_idx, []).append((bbox, region_type, page_size))
    represented_marker_types = {
        (page_idx, region_type)
        for page_idx, regions in regions_by_page.items()
        for _, region_type, _ in regions
        if region_type in _HEADER_TYPES | _FOOTER_TYPES
    }
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if page_idx in preserved_pages:
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if page_idx < 0 or bbox is None or page_size is None:
            continue
        block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        text = str(raw.get("text") or "").strip()
        blocks_by_page.setdefault(page_idx, []).append((bbox, block_type, text))
        if block_type in _HEADER_TYPES | _FOOTER_TYPES and (
            page_idx,
            block_type,
        ) not in represented_marker_types:
            regions_by_page.setdefault(page_idx, []).append((bbox, block_type, page_size))
        if block_type == "page_number":
            page_number = _normalise_page_number(text, page_idx)
            if page_number is not None:
                page_numbers.append(
                    _PageNumber(
                        page_idx,
                        bbox,
                        page_size,
                        page_number,
                    )
                )

    bands: list[_FurnitureBand] = []
    for page_idx, regions in regions_by_page.items():
        page_size = sizes.get(page_idx) or regions[0][2]
        width, height = page_size
        headers = [bbox for bbox, region_type, _ in regions if region_type in _HEADER_TYPES]
        footers = [bbox for bbox, region_type, _ in regions if region_type in _FOOTER_TYPES]
        blocks = blocks_by_page.get(page_idx, [])
        if headers:
            bottom = max(item[3] for item in headers)
            explicit_headers = [
                bbox for bbox, region_type, _ in regions if region_type == "page_header"
            ]
            if len(explicit_headers) < 2:
                neighbour_anchor = (
                    max(item[3] for item in explicit_headers)
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
                        for bbox, block_type, _ in blocks
                        if block_type
                        not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                        and bbox[1] <= neighbour_limit
                        and bbox[3] <= maximum_bottom
                    ]
                )
            next_content_starts = [
                bbox[1]
                for bbox, block_type, _ in blocks
                if block_type
                not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                and bbox[1] >= bottom + 0.5
            ]
            if next_content_starts:
                bottom = max(
                    bottom,
                    min(min(next_content_starts) - 2.0, bottom + height * 0.01),
                )
            bands.append(_FurnitureBand(page_idx, (0.0, 0.0, width, bottom), page_size))
        if footers:
            top = min(item[1] for item in footers)
            explicit_footers = [
                bbox for bbox, region_type, _ in regions if region_type == "page_footer"
            ]
            if len(explicit_footers) < 2:
                neighbour_anchor = (
                    min(item[1] for item in explicit_footers)
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
                        for bbox, block_type, _ in blocks
                        if block_type
                        not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                        and bbox[3] >= neighbour_limit
                        and bbox[1] >= minimum_top
                    ]
                )
            bands.append(_FurnitureBand(page_idx, (0.0, top, width, height), page_size))
    return bands, page_numbers


def _repair_broken_ascii_words(value: str) -> str:
    """Join only high-confidence word fragments introduced by tight PDF boxes."""

    def replace(match: re.Match[str]) -> str:
        first, second = match.group(1), match.group(2)
        candidate = (first + second).casefold()
        if candidate.endswith(_REPAIRABLE_WORD_SUFFIXES):
            return first + second
        return match.group(0)

    previous = value
    for _ in range(3):
        repaired = _ASCII_SHORT_TAIL_RE.sub(replace, previous)
        repaired = _ASCII_SHORT_HEAD_RE.sub(replace, repaired)
        if repaired == previous:
            break
        previous = repaired
    return previous


def _clean_pdf_text(value: Any) -> str:
    raw = str(value or "")
    cleaned = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in raw
    )
    compact = _CJK_GAP_RE.sub("", " ".join(cleaned.split())).strip()
    return _repair_broken_ascii_words(compact)


def _pdf_blocks(page: fitz.Page, segment: fitz.Rect) -> list[_PdfBlock]:
    result: list[_PdfBlock] = []
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
        line_x0s: list[float] = []
        horizontal = True
        for line in raw.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(str(span.get("text") or "") for span in spans)
            if text.strip():
                lines.append(text)
                line_rect = pdf_rect_to_visual(
                    page,
                    fitz.Rect(line.get("bbox") or raw.get("bbox") or (0, 0, 0, 0)),
                )
                line_x0s.append(float(line_rect.x0))
                horizontal = horizontal and is_visually_horizontal(
                    page,
                    tuple(line.get("dir") or (1.0, 0.0)),
                )
        text = _clean_pdf_text("\n".join(lines))
        if text:
            result.append(_PdfBlock(rect, text, tuple(line_x0s), horizontal))
    return result


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


def _source_items(
    ocr_result: dict[str, Any],
    page_idx: int,
    segment: fitz.Rect,
    fallback_size: tuple[float, float],
) -> tuple[list[_SourceItem], list[_SourceItem]]:
    text_items: list[_SourceItem] = []
    layout_items: list[_SourceItem] = []
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if raw_page_idx != page_idx:
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or fallback_size
        if bbox is None:
            continue
        block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        rect = _mapped_rect(segment, bbox, page_size)
        if rect.is_empty:
            continue
        layout_items.append(_SourceItem(rect, block_type))
        if should_inject_source_text(block_type):
            text_items.append(_SourceItem(rect, block_type))
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if raw_page_idx != page_idx:
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if region_type not in _VISUAL_REGION_TYPES:
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or fallback_size
        if bbox is not None:
            layout_items.append(_SourceItem(_mapped_rect(segment, bbox, page_size), region_type))
    return text_items, layout_items


def _regular_font(needs_cjk: bool) -> Path:
    cache = Path(os.getenv("XDG_CACHE_HOME") or "/root/.cache") / "babeldoc" / "fonts"
    candidates = (
        [
            cache / "SourceHanSerifCN-Regular.ttf",
            cache / "SourceHanSansCN-Regular.ttf",
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            cache / "GoNotoKurrent-Regular.ttf",
        ]
        if needs_cjk
        else []
    )
    candidates.extend(
        [
            # BabelDOC's compact Latin Noto faces and GoNotoKurrent do not
            # contain U+2264/U+2265.  Source Han and DejaVu do, so prefer them
            # even for an otherwise English target whenever protected
            # comparison limits may be present.
            cache / "SourceHanSerifCN-Regular.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
            cache / "NotoSerif-Regular.ttf",
            Path("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No regular font is available for OCR body normalization")


def _textbox_spare(
    text: str,
    rect: fitz.Rect,
    font: Path,
    size: float,
    line_height: float = _BODY_LINE_HEIGHT,
    align: int = fitz.TEXT_ALIGN_LEFT,
) -> float:
    scratch = fitz.open()
    try:
        page = scratch.new_page(width=max(1.0, rect.width), height=max(1.0, rect.height))
        return float(
            page.insert_textbox(
                page.rect,
                text,
                fontname="bodyfit",
                fontfile=str(font),
                fontsize=size,
                lineheight=line_height,
                align=align,
            )
        )
    finally:
        scratch.close()


def _body_plans(
    page: fitz.Page,
    *,
    page_idx: int,
    segment: fitz.Rect,
    body_regions: list[_BodyRegion],
    body_translations: dict[
        tuple[int, tuple[float, float, float, float]], str
    ],
    ocr_result: dict[str, Any],
    body_font: Path,
) -> list[_BodyPlan]:
    regions = [item for item in body_regions if item.page_idx == page_idx]
    if not regions:
        return []
    text_items, layout_items = _source_items(ocr_result, page_idx, segment, regions[0].page_size)
    column_items = [
        item for item in text_items if item.block_type not in _COLUMN_EXCLUDED_TYPES
    ]
    if not column_items:
        logger.warning(
            "Page %d has body regions but no reliable text column; using the "
            "page-safe margins and deferring paragraphs that do not fit",
            page_idx + 1,
        )
    content_left = max(
        segment.x0 + segment.width * 0.025,
        min((item.rect.x0 for item in column_items), default=segment.x0),
    )
    content_right = min(
        segment.x1 - segment.width * 0.025,
        max((item.rect.x1 for item in column_items), default=segment.x1),
    )
    if content_right - content_left < segment.width * 0.25:
        content_left = segment.x0 + segment.width * 0.08
        content_right = segment.x1 - segment.width * 0.08

    mapped = [(item, _mapped_rect(segment, item.bbox, item.page_size)) for item in regions]
    targets = _pdf_blocks(page, segment)
    pairs: list[tuple[float, int, int]] = []
    for region_idx, (_, source_rect) in enumerate(mapped):
        for target_idx, target in enumerate(targets):
            score = _rect_overlap_ratio(source_rect, target.rect)
            if score > 0:
                pairs.append(
                    (
                        score + _vertical_overlap_ratio(source_rect, target.rect),
                        region_idx,
                        target_idx,
                    )
                )
            elif (
                target.rect.width <= 6.0
                and _vertical_overlap_ratio(source_rect, target.rect) >= 0.50
                and target.rect.x1 >= source_rect.x0 - segment.width * 0.08
                and target.rect.x0 <= source_rect.x1 + segment.width * 0.08
            ):
                # Layout engines occasionally emit a stray punctuation glyph
                # just outside the paragraph's horizontal source box.
                pairs.append((0.25, region_idx, target_idx))
            elif (
                not target.horizontal
                and _vertical_overlap_ratio(source_rect, target.rect) >= 0.45
            ):
                # Some layout engines ignore PDF /Rotate and place an otherwise
                # matching line in the correct visual band but at a displaced
                # x-coordinate. Pair it by band so deterministic OCR redraw can
                # remove the rotated residue and restore the complete translation.
                pairs.append(
                    (
                        0.40
                        + _vertical_overlap_ratio(source_rect, target.rect),
                        region_idx,
                        target_idx,
                    )
                )
    assignments: dict[int, list[int]] = {}
    used_targets: set[int] = set()
    for _, region_idx, target_idx in sorted(pairs, reverse=True):
        if target_idx in used_targets:
            continue
        assignments.setdefault(region_idx, []).append(target_idx)
        used_targets.add(target_idx)
    unmatched_regions = [
        region_idx for region_idx in range(len(mapped)) if region_idx not in assignments
    ]
    if unmatched_regions:
        drawable_unmatched = sum(
            _body_region_key(mapped[index][0].page_idx, mapped[index][0].bbox)
            in body_translations
            for index in unmatched_regions
        )
        logger.warning(
            "Page %d matched %d/%d body regions; redrawing %d unmatched "
            "regions directly from the OCR translation plan",
            page_idx + 1,
            len(assignments),
            len(mapped),
            drawable_unmatched,
        )

    font = fitz.Font(fontfile=str(body_font))
    space_width = max(1.0, float(font.text_length("\u00a0", fontsize=_BODY_FONT_SIZE)))
    plans: list[_BodyPlan] = []
    for region_idx, (region, source_rect) in enumerate(mapped):
        target_indices = assignments.get(region_idx, [])
        translated_text = body_translations.get(
            _body_region_key(region.page_idx, region.bbox)
        )
        # A layout engine can omit a translated paragraph entirely even though
        # the serial OCR translation plan is complete. In that case there is
        # no PDF text object to redact or use as a geometry anchor; draw the
        # trusted translation at its OCR source rectangle instead. If this
        # helper is called without a translation plan, leave the unmatched
        # engine output untouched because there is no safe target text to draw.
        if not target_indices and translated_text is None:
            continue
        target_indices.sort(
            key=lambda index: (targets[index].rect.y0, targets[index].rect.x0)
        )
        assigned_targets = [targets[index] for index in target_indices]
        targets_are_horizontal = not assigned_targets or all(
            target.horizontal for target in assigned_targets
        )
        target_rect = (
            fitz.Rect(assigned_targets[0].rect)
            if assigned_targets
            else fitz.Rect(source_rect)
        )
        for target in assigned_targets[1:]:
            target_rect.include_rect(target.rect)
        available_right = content_right
        for item in layout_items:
            if (
                item.rect.x0 > source_rect.x1 + 2
                and _vertical_overlap_ratio(source_rect, item.rect) >= 0.30
            ):
                available_right = min(available_right, item.rect.x0 - 3)
        draw_left = max(
            content_left,
            min(source_rect.x0, target_rect.x0)
            if targets_are_horizontal
            else source_rect.x0,
        )
        next_candidates = [
            item.rect.y0
            for item in layout_items
            if (
                item.rect.y0 > source_rect.y1 + 0.5
                or (
                    item.block_type == "table"
                    and item.rect.y0 > source_rect.y0 + 0.5
                )
            )
            and _horizontal_overlap_ratio(
                fitz.Rect(draw_left, source_rect.y0, available_right, source_rect.y1),
                item.rect,
            )
            >= 0.08
        ]
        next_y = min(next_candidates) if next_candidates else min(
            segment.y1, source_rect.y1 + max(36.0, source_rect.height * 1.5)
        )
        base_top = (
            min(source_rect.y0, target_rect.y0)
            if targets_are_horizontal
            else source_rect.y0
        )
        previous_target_bottoms = [
            candidate.rect.y1
            for candidate_idx, candidate in enumerate(targets)
            if candidate_idx not in target_indices
            and candidate.rect.y1 <= base_top
            and _horizontal_overlap_ratio(
                fitz.Rect(draw_left, base_top, available_right, base_top + 1.0),
                candidate.rect,
            )
            >= 0.08
        ]
        # Matched engine text already supplies a reliable anchor and keeps a
        # small safety gap from the preceding block. Direct OCR fallback boxes
        # are often only one line high; allow them to use that gap too so a
        # two-line 9 pt translation can fit. The final rhythm pass restores the
        # normal document-wide paragraph gap after insertion.
        previous_gap = 1.5 if assigned_targets else 0.0
        previous_limit = (
            max(previous_target_bottoms) + previous_gap
            if previous_target_bottoms
            else segment.y0
        )
        top_leading = 5.5 if assigned_targets else 6.5
        draw_rect = fitz.Rect(
            draw_left,
            max(segment.y0, base_top - top_leading, previous_limit),
            available_right,
            min(segment.y1, next_y - 0.2),
        )
        unsafe_layout_area = (
            draw_rect.width < segment.width * 0.18 or draw_rect.height < 8
        )
        line_x0s = [x0 for target in assigned_targets for x0 in target.line_x0s]
        first_x0 = (
            assigned_targets[0].line_x0s[0]
            if assigned_targets and assigned_targets[0].line_x0s
            else target_rect.x0
        )
        indent = 0.0
        if targets_are_horizontal:
            indent = (
                max(0.0, first_x0 - min(line_x0s))
                if len(line_x0s) > 1
                else max(0.0, target_rect.x0 - source_rect.x0)
            )
        prefix = "\u00a0" * min(24, int(round(indent / space_width)))
        if translated_text is None:
            seen_text: set[str] = set()
            text_parts: list[str] = []
            for target in assigned_targets:
                if target.text not in seen_text:
                    text_parts.append(target.text)
                    seen_text.add(target.text)
            translated_text = " ".join(text_parts)
        text = prefix + translated_text
        font_size = _BODY_FONT_SIZE
        line_height = _BODY_LINE_HEIGHT
        deferred = unsafe_layout_area or (
            _textbox_spare(text, draw_rect, body_font, font_size, line_height) < 0
        )
        if deferred:
            logger.warning(
                "Page %d body paragraph cannot use its source-page box at the "
                "fixed %.1f pt / %.2f line-height style; scheduling it for the "
                "page layout solver instead of shrinking the font",
                page_idx + 1,
                font_size,
                line_height,
            )
        plans.append(
            _BodyPlan(
                tuple(fitz.Rect(target.rect) for target in assigned_targets),
                draw_rect,
                text,
                font_size,
                line_height,
                fitz.Rect(source_rect),
                deferred,
            )
        )
    return plans


def _font_resource_name(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", path.stem)[:24]
    return f"ocrbody{cleaned or 'regular'}"


def _font_matches(font_name: str, font_path: Path) -> bool:
    embedded = re.sub(r"[^a-z0-9]", "", font_name.casefold())
    expected = re.sub(r"[^a-z0-9]", "", font_path.stem.casefold())
    return bool(embedded and expected and (embedded in expected or expected in embedded))


def _visual_rects(
    ocr_result: dict[str, Any],
    page_idx: int,
    segment: fitz.Rect,
) -> list[fitz.Rect]:
    result: list[fitz.Rect] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if raw_page_idx != page_idx:
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if region_type not in _VISUAL_REGION_TYPES:
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size"))
        if bbox is not None and page_size is not None:
            rect = _mapped_rect(segment, bbox, page_size)
            if not rect.is_empty:
                result.append(rect)
    return result


def _preserved_pixel_regions(
    ocr_result: dict[str, Any],
    page_idx: int,
) -> list[_PixelRegion]:
    result: list[_PixelRegion] = []
    seen: set[tuple[tuple[float, float, float, float], str]] = set()
    represented_types: set[str] = set()
    sources: list[dict[str, Any]] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if raw_page_idx == page_idx and region_type in _PIXEL_PRESERVED_REGION_TYPES:
            sources.append(raw)
            represented_types.add(region_type)
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        try:
            raw_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if (
            raw_page_idx == page_idx
            and region_type in _PIXEL_PRESERVED_REGION_TYPES
            and region_type not in represented_types
        ):
            sources.append(raw)

    for raw in sources:
        if not isinstance(raw, dict):
            continue
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size"))
        if bbox is None or page_size is None:
            continue
        key = (tuple(round(value, 2) for value in bbox), region_type)
        if key not in seen:
            result.append(_PixelRegion(bbox, page_size, region_type))
            seen.add(key)
    return result


def _rhythm_items(
    page: fitz.Page,
    *,
    segment: fitz.Rect,
    body_font: Path,
    heading_font: Path | None,
    visual_rects: list[fitz.Rect],
) -> list[_RhythmItem]:
    result: list[_RhythmItem] = []
    for raw in page.get_text("dict").get("blocks") or []:
        if raw.get("type") != 0:
            continue
        rect = pdf_rect_to_visual(
            page,
            fitz.Rect(raw.get("bbox") or (0, 0, 0, 0)),
        )
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if rect.is_empty or not segment.contains(center):
            continue
        lines: list[str] = []
        spans: list[dict[str, Any]] = []
        for line in raw.get("lines") or []:
            line_spans = line.get("spans") or []
            line_text = "".join(str(span.get("text") or "") for span in line_spans)
            line_text = "".join(
                " " if unicodedata.category(character).startswith("C") else character
                for character in line_text
            ).rstrip()
            if line_text.strip():
                lines.append(line_text)
                spans.extend(line_spans)
        if not lines or not spans:
            continue
        body = all(
            abs(float(span.get("size") or 0.0) - _BODY_FONT_SIZE) <= 0.15
            and _font_matches(str(span.get("font") or ""), body_font)
            for span in spans
        )
        heading_size = float(spans[0].get("size") or 0.0)
        heading = heading_font is not None and all(
            abs(float(span.get("size") or 0.0) - heading_size) <= 0.15
            and any(abs(heading_size - expected) <= 0.15 for expected in (10.5, 12.0))
            and _font_matches(str(span.get("font") or ""), heading_font)
            for span in spans
        )
        if not body and not heading:
            continue
        result.append(
            _RhythmItem(
                rect=rect,
                text="\n".join(lines),
                font_path=body_font if body else heading_font,
                font_size=_BODY_FONT_SIZE if body else heading_size,
                line_height=_BODY_LINE_HEIGHT if body else 1.05,
            )
        )
    return sorted(result, key=lambda item: (item.rect.y0, item.rect.x0))


def _immutable_text_rects(
    page: fitz.Page,
    *,
    segment: fitz.Rect,
    owned_items: list[_RhythmItem],
    replaceable_rects: list[fitz.Rect],
) -> list[fitz.Rect]:
    """Return engine text that the fixed-style compositor does not own.

    These rectangles must remain stationary, but they still participate in
    collision avoidance.  Treating them as invisible was the direct cause of
    SourceHan headings and body text being moved over Times-Roman residue.
    """
    obstacles: list[fitz.Rect] = []
    for raw in page.get_text("dict").get("blocks") or []:
        if raw.get("type") != 0:
            continue
        rect = pdf_rect_to_visual(
            page,
            fitz.Rect(raw.get("bbox") or (0, 0, 0, 0)),
        )
        if rect.is_empty:
            continue
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if not segment.contains(center):
            continue
        if any(_rect_overlap_ratio(rect, item.rect) >= 0.95 for item in owned_items):
            continue
        if any(_rect_overlap_ratio(rect, target) >= 0.50 for target in replaceable_rects):
            continue
        obstacles.append(rect)
    return obstacles


def _visual_barrier(
    first: _RhythmItem,
    second: _RhythmItem,
    visual_rects: list[fitz.Rect],
    segment: fitz.Rect,
) -> tuple[float, float] | None:
    candidates = [
        rect
        for rect in visual_rects
        if rect.width >= segment.width * 0.20
        and rect.y0 >= first.rect.y1 - 1.0
        and rect.y1 <= second.rect.y0 + 1.0
        and max(0.0, min(rect.x1, segment.x1) - max(rect.x0, segment.x0))
        >= min(rect.width, segment.width) * 0.25
    ]
    if not candidates:
        return None
    return min(rect.y0 for rect in candidates), max(rect.y1 for rect in candidates)


def _rhythm_item_height(item: _RhythmItem, width: float) -> float:
    if not item.deferred:
        return item.rect.height
    probe_height = max(1000.0, len(item.text) * item.font_size * 2.0)
    probe = fitz.Rect(0.0, 0.0, max(1.0, width), probe_height)
    spare = _textbox_spare(
        item.text,
        probe,
        item.font_path,
        item.font_size,
        item.line_height,
        item.align,
    )
    if spare < 0:
        return probe_height
    return max(item.font_size * item.line_height, probe_height - spare)


def _plan_rhythm_placements(
    items: list[_RhythmItem],
    *,
    segment: fitz.Rect,
    visual_rects: list[fitz.Rect],
    obstacle_rects: list[fitz.Rect] | None = None,
    logical_page_number: int | None = None,
) -> _RhythmPageLayout:
    if not items:
        return _RhythmPageLayout((), ())
    ordered = sorted(items, key=lambda item: (item.rect.y0, item.rect.x0))
    text_obstacles = list(obstacle_rects or [])
    overlapping_visuals = [
        item
        for item in ordered
        if any(_rect_overlap_ratio(item.rect, visual) >= 0.05 for visual in visual_rects)
    ]
    candidates = [item for item in ordered if item not in overlapping_visuals]
    if not candidates:
        return _RhythmPageLayout((), tuple(ordered))
    barrier_ranges = sorted(
        (
            max(segment.y0, rect.y0),
            min(segment.y1, rect.y1),
        )
        for rect in visual_rects
        if rect.width >= segment.width * 0.20
        and max(0.0, min(rect.x1, segment.x1) - max(rect.x0, segment.x0))
        >= min(rect.width, segment.width) * 0.25
        and rect.y1 > segment.y0
        and rect.y0 < segment.y1
    )
    merged_barriers: list[tuple[float, float]] = []
    for start, end in barrier_ranges:
        if not merged_barriers or start > merged_barriers[-1][1] + 1.0:
            merged_barriers.append((start, end))
        else:
            merged_barriers[-1] = (
                merged_barriers[-1][0],
                max(merged_barriers[-1][1], end),
            )
    bands: list[tuple[float, float, bool]] = []
    lower = segment.y0
    for start, end in merged_barriers:
        if start > lower + 0.5:
            bands.append((lower, start, lower > segment.y0 + 0.5))
        lower = max(lower, end)
    if lower < segment.y1 - 0.5:
        bands.append((lower, segment.y1, lower > segment.y0 + 0.5))
    items_by_band: list[list[_RhythmItem]] = [[] for _ in bands]
    for item in candidates:
        center = (item.rect.y0 + item.rect.y1) / 2
        matching = [
            index
            for index, (lower, upper, _follows_visual) in enumerate(bands)
            if lower - 0.5 <= center <= upper + 0.5
        ]
        if matching:
            items_by_band[matching[0]].append(item)
        elif bands:
            nearest = min(
                range(len(bands)),
                key=lambda index: min(
                    abs(center - bands[index][0]),
                    abs(center - bands[index][1]),
                ),
            )
            items_by_band[nearest].append(item)
    clusters: list[tuple[list[_RhythmItem], float, float, bool]] = []
    assigned_ids: set[int] = set()
    for (lower, upper, follows_visual), band_items in zip(
        bands,
        items_by_band,
        strict=True,
    ):
        if band_items:
            clusters.append((band_items, lower, upper, follows_visual))
            assigned_ids.update(id(item) for item in band_items)
    overflow_candidates = [
        item for item in candidates if id(item) not in assigned_ids
    ]

    placements: list[_RhythmPlacement] = []
    overflow: list[_RhythmItem] = [*overlapping_visuals, *overflow_candidates]
    content_right = segment.x1 - segment.width * 0.025
    for cluster, lower, upper, after_visual in clusters:
        working = list(cluster)
        before_visual = upper < segment.y1 - 0.5
        available_height = max(0.0, upper - lower)
        page_context = (
            f"Page {logical_page_number}"
            if logical_page_number is not None
            else "Page"
        )
        heights = [
            _rhythm_item_height(
                item,
                max(1.0, content_right - max(segment.x0, item.rect.x0)),
            )
            for item in working
        ]
        while working:
            gap_count = len(working) - 1 + int(after_visual) + int(before_visual)
            text_height = sum(heights)
            minimum_height = text_height + _MIN_VERTICAL_RHYTHM_GAP * gap_count
            if minimum_height <= available_height + 0.5:
                break
            overflow.insert(0, working.pop())
            heights.pop()
        if not working:
            logger.warning(
                "%s has no room for the next fixed-style text block inside a "
                "visual-bounded band; moving the trailing content to continuation pages",
                page_context,
            )
            continue
        gap_count = len(working) - 1 + int(after_visual) + int(before_visual)
        text_height = sum(heights)
        preferred_height = text_height + _VERTICAL_RHYTHM_GAP * gap_count
        gap = _VERTICAL_RHYTHM_GAP
        if preferred_height > available_height + 0.5 and gap_count:
            gap = max(
                _MIN_VERTICAL_RHYTHM_GAP,
                (available_height - text_height) / gap_count,
            )
            logger.warning(
                "%s fixed %.2f pt paragraph gap needs %.1f pt inside a %.1f pt "
                "visual-bounded band; compressing the gap to %.2f pt while "
                "preserving the fixed font size and line height",
                page_context,
                _VERTICAL_RHYTHM_GAP,
                preferred_height,
                available_height,
                gap,
            )
        total_height = text_height + gap * (len(working) - 1)
        minimum_start = lower + (gap if after_visual else 0.0)
        maximum_end = upper - (gap if before_visual else 0.0)
        start = max(working[0].rect.y0, minimum_start)
        if start + total_height > maximum_end:
            start = max(minimum_start, maximum_end - total_height)
        if start + total_height > maximum_end + 0.5:
            logger.warning(
                "%s could not place a fixed-style typography cluster after gap "
                "fitting; moving it to continuation pages",
                page_context,
            )
            overflow.extend(working)
            continue
        cursor = start
        placed_all = True
        for item_index, (item, height) in enumerate(
            zip(working, heights, strict=True)
        ):
            while True:
                occupied = fitz.Rect(
                    item.rect.x0,
                    cursor,
                    max(item.rect.x1, content_right),
                    cursor + height,
                )
                collisions = [
                    obstacle
                    for obstacle in text_obstacles
                    if _vertical_overlap_ratio(occupied, obstacle) >= 0.10
                    and _horizontal_overlap_ratio(occupied, obstacle) >= 0.08
                ]
                if not collisions:
                    break
                cursor = max(obstacle.y1 for obstacle in collisions) + gap
            remaining_height = sum(heights[item_index:]) + gap * (
                len(working) - item_index - 1
            )
            if cursor + remaining_height > maximum_end + 0.5:
                overflow.extend(working[item_index:])
                placed_all = False
                break
            draw_rect = fitz.Rect(
                item.rect.x0,
                cursor,
                max(item.rect.x1 + 2.0, content_right),
                cursor + height + item.font_size * 2.0,
            )
            placements.append(_RhythmPlacement(item, draw_rect))
            cursor += height + gap
        if not placed_all:
            logger.warning(
                "%s immutable PDF text leaves no collision-free room for the "
                "remaining fixed-style blocks; moving the trailing content to "
                "continuation pages",
                page_context,
            )

    if overflow:
        cutoff = min((item.rect.y0, item.rect.x0) for item in overflow)
        overflow = [
            item for item in ordered if (item.rect.y0, item.rect.x0) >= cutoff
        ]
        overflow_ids = {id(item) for item in overflow}
        placements = [
            placement
            for placement in placements
            if id(placement.item) not in overflow_ids
        ]
    return _RhythmPageLayout(tuple(placements), tuple(overflow))


def _rhythm_placements(
    items: list[_RhythmItem],
    *,
    segment: fitz.Rect,
    visual_rects: list[fitz.Rect],
    obstacle_rects: list[fitz.Rect] | None = None,
    logical_page_number: int | None = None,
) -> list[_RhythmPlacement]:
    return list(
        _plan_rhythm_placements(
            items,
            segment=segment,
            visual_rects=visual_rects,
            obstacle_rects=obstacle_rects,
            logical_page_number=logical_page_number,
        ).placements
    )


def _rhythm_font_resource_name(path: Path, size: float) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", path.stem)[:18]
    return f"ocrrhythm{cleaned}{int(round(size * 10))}"


def _text_prefix_for_rect(
    item: _RhythmItem,
    text: str,
    rect: fitz.Rect,
) -> tuple[str, str]:
    if _textbox_spare(
        text,
        rect,
        item.font_path,
        item.font_size,
        item.line_height,
        item.align,
    ) >= 0:
        return text, ""
    units = re.findall(r"\S+\s*", text) if re.search(r"\s", text) else list(text)
    if not units:
        return "", text

    def fitting_prefix(parts: list[str]) -> int:
        low = 1
        high = len(parts)
        fitted = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = "".join(parts[:middle]).rstrip()
            if candidate and _textbox_spare(
                candidate,
                rect,
                item.font_path,
                item.font_size,
                item.line_height,
                item.align,
            ) >= 0:
                fitted = middle
                low = middle + 1
            else:
                high = middle - 1
        return fitted

    count = fitting_prefix(units)
    if count == 0 and len(units) == 1 and len(text) > 1:
        units = list(text)
        count = fitting_prefix(units)
    if count == 0:
        return "", text
    prefix = "".join(units[:count]).rstrip()
    remainder = "".join(units[count:]).lstrip()
    return prefix, remainder


def _continuation_target_page(
    document: fitz.Document,
    *,
    reference_rect: fitz.Rect,
    mode: str,
    target_offset: int,
) -> tuple[fitz.Page, tuple[int, ...]]:
    if mode == "interleaved":
        pair_start = document.page_count
        document.new_page(width=reference_rect.width, height=reference_rect.height)
        document.new_page(width=reference_rect.width, height=reference_rect.height)
        if target_offset not in {0, 1}:
            raise RuntimeError("Interleaved continuation has no reliable target offset")
        return (
            document[pair_start + target_offset],
            (pair_start, pair_start + 1),
        )
    page = document.new_page(width=reference_rect.width, height=reference_rect.height)
    return page, (page.number,)


def _continuation_segment(page: fitz.Page, mode: str) -> fitz.Rect:
    segment = fitz.Rect(page.rect)
    if mode == "side_by_side":
        segment.x0 += segment.width / 2
    return segment


def _append_rhythm_continuations(
    document: fitz.Document,
    *,
    overflow_by_page: dict[int, list[_RhythmItem]],
    reference_rects: dict[int, fitz.Rect],
    reference_page_indices: dict[int, int],
    mode: str,
) -> _ContinuationLineage:
    original_page_count = document.page_count
    appended_groups: dict[int, list[tuple[int, ...]]] = {}
    for source_page_idx in sorted(overflow_by_page):
        queued = list(overflow_by_page[source_page_idx])
        if not queued:
            continue
        reference = reference_rects[source_page_idx]
        target_offset = 0
        if mode == "interleaved":
            target_offset = (
                reference_page_indices[source_page_idx] - source_page_idx * 2
            )
        page: fitz.Page | None = None
        content_rect = fitz.Rect()
        cursor = 0.0
        page_has_text = False

        def next_page(
            *,
            reference_rect: fitz.Rect,
            source_index: int,
            continuation_target_offset: int,
        ) -> None:
            nonlocal page, content_rect, cursor, page_has_text
            page, physical_group = _continuation_target_page(
                document,
                reference_rect=reference_rect,
                mode=mode,
                target_offset=continuation_target_offset,
            )
            appended_groups.setdefault(source_index, []).append(physical_group)
            segment = _continuation_segment(page, mode)
            content_rect = fitz.Rect(
                segment.x0 + segment.width * _CONTINUATION_MARGIN_X_RATIO,
                segment.y0 + _CONTINUATION_MARGIN_TOP,
                segment.x1 - segment.width * _CONTINUATION_MARGIN_X_RATIO,
                segment.y1 - _CONTINUATION_MARGIN_BOTTOM,
            )
            cursor = content_rect.y0
            page_has_text = False

        for item in queued:
            remaining = item.text
            first_chunk = True
            while remaining:
                if page is None:
                    next_page(
                        reference_rect=reference,
                        source_index=source_page_idx,
                        continuation_target_offset=target_offset,
                    )
                assert page is not None
                gap = _VERTICAL_RHYTHM_GAP if page_has_text and first_chunk else 0.0
                draw_top = cursor + gap
                available = fitz.Rect(
                    content_rect.x0,
                    draw_top,
                    content_rect.x1,
                    content_rect.y1,
                )
                if available.height < item.font_size * item.line_height:
                    next_page(
                        reference_rect=reference,
                        source_index=source_page_idx,
                        continuation_target_offset=target_offset,
                    )
                    continue
                prefix, remainder = _text_prefix_for_rect(item, remaining, available)
                if remainder and page_has_text:
                    # Keep complete paragraphs together whenever a fresh page
                    # can accommodate them. Split only an item that is taller
                    # than an otherwise empty continuation page.
                    next_page(
                        reference_rect=reference,
                        source_index=source_page_idx,
                        continuation_target_offset=target_offset,
                    )
                    continue
                if not prefix:
                    raise RuntimeError(
                        "A fixed-style paragraph cannot fit on an empty continuation page"
                    )
                spare = insert_visual_textbox(
                    page,
                    available,
                    prefix,
                    fontname=_rhythm_font_resource_name(item.font_path, item.font_size),
                    fontfile=str(item.font_path),
                    fontsize=item.font_size,
                    lineheight=item.line_height,
                    align=item.align,
                    color=(0, 0, 0),
                    overlay=True,
                )
                if spare < 0:
                    raise RuntimeError(
                        "Continuation paragraph overflowed after fixed-style preflight"
                    )
                cursor = available.y1 - spare
                page_has_text = True
                remaining = remainder
                first_chunk = False
                if remaining:
                    next_page(
                        reference_rect=reference,
                        source_index=source_page_idx,
                        continuation_target_offset=target_offset,
                    )

    logical_pages = sorted(reference_page_indices)
    if logical_pages != list(range(len(logical_pages))):
        raise RuntimeError("Fixed-style pagination produced a sparse source page map")
    if mode == "interleaved" and original_page_count != len(logical_pages) * 2:
        raise RuntimeError("Interleaved pagination source page count changed unexpectedly")
    if mode != "interleaved" and original_page_count != len(logical_pages):
        raise RuntimeError("Pagination source page count changed unexpectedly")

    page_order: list[int] = []
    source_page_indices: list[int] = []
    continuation_page_indices: list[int] = []
    continuation_source_pages: list[int] = []
    for logical_page_idx in logical_pages:
        if mode == "interleaved":
            original_group = (logical_page_idx * 2, logical_page_idx * 2 + 1)
            target_offset = (
                reference_page_indices[logical_page_idx] - logical_page_idx * 2
            )
        else:
            original_group = (reference_page_indices[logical_page_idx],)
            target_offset = 0
        source_page_indices.append(len(page_order) + target_offset)
        page_order.extend(original_group)
        for physical_group in appended_groups.get(logical_page_idx, ()):
            continuation_page_indices.append(len(page_order) + target_offset)
            continuation_source_pages.append(logical_page_idx)
            page_order.extend(physical_group)

    if sorted(page_order) != list(range(document.page_count)):
        raise RuntimeError("Fixed-style pagination page lineage is incomplete")
    if page_order != list(range(document.page_count)):
        document.select(page_order)
    return _ContinuationLineage(
        tuple(source_page_indices),
        tuple(continuation_page_indices),
        tuple(continuation_source_pages),
    )


def _repair_vertical_rhythm(
    pdf_path: Path,
    *,
    ocr_result: dict[str, Any],
    body_font: Path,
    heading_font: Path | None,
    page_numbers: list[_PageNumber],
    segment_factory: Callable[[fitz.Page, int], fitz.Rect | None],
    logical_page_factory: Callable[[int], int | None],
    deferred_body_blocks: tuple[_DeferredBodyBlock, ...] = (),
    continuation_mode: str = "mono",
) -> _RhythmRepair:
    document = fitz.open(str(pdf_path))
    repaired_blocks = 0
    repaired_pages = 0
    overflow_by_page: dict[int, list[_RhythmItem]] = {}
    reference_rects: dict[int, fitz.Rect] = {}
    reference_page_indices: dict[int, int] = {}
    changed = False
    try:
        original_page_count = document.page_count
        for page_idx in range(original_page_count):
            page = document[page_idx]
            logical_page_idx = logical_page_factory(page_idx)
            if logical_page_idx is None:
                continue
            segment = segment_factory(page, page_idx)
            if segment is None:
                continue
            reference_rects[logical_page_idx] = fitz.Rect(page.rect)
            reference_page_indices[logical_page_idx] = page_idx
            visuals = _visual_rects(ocr_result, logical_page_idx, segment)
            items = _rhythm_items(
                page,
                segment=segment,
                body_font=body_font,
                heading_font=heading_font,
                visual_rects=visuals,
            )
            deferred_here = [
                block.item
                for block in deferred_body_blocks
                if block.page_idx == logical_page_idx
            ]
            deferred_targets = [
                rect
                for item in deferred_here
                for rect in item.redaction_rects
            ]
            if deferred_targets:
                items = [
                    item
                    for item in items
                    if not any(
                        _rect_overlap_ratio(item.rect, target) >= 0.50
                        for target in deferred_targets
                    )
                ]
            immutable_text = _immutable_text_rects(
                page,
                segment=segment,
                owned_items=items,
                replaceable_rects=deferred_targets,
            )
            items.extend(deferred_here)
            items.sort(key=lambda item: (item.rect.y0, item.rect.x0))
            page_numbers_here = [
                number
                for number in page_numbers
                if number.page_idx == logical_page_idx
            ]
            rhythm_segment = fitz.Rect(segment)
            if page_numbers_here:
                rhythm_segment.y1 = min(
                    rhythm_segment.y1,
                    min(
                        _page_number_rect(number, segment).y0
                        for number in page_numbers_here
                    )
                    - _PAGE_NUMBER_CONTENT_GAP,
                )
                if rhythm_segment.y1 <= rhythm_segment.y0:
                    logger.warning(
                        "Page %d page-number geometry leaves no content area; "
                        "ignoring that clearance instead of failing pagination",
                        logical_page_idx + 1,
                    )
                    rhythm_segment = fitz.Rect(segment)
            layout = _plan_rhythm_placements(
                items,
                segment=rhythm_segment,
                visual_rects=visuals,
                obstacle_rects=immutable_text,
                logical_page_number=logical_page_idx + 1,
            )
            placement_list = list(layout.placements)
            draw_placements = (
                placement_list
                if any(
                    placement.item.deferred
                    or abs(placement.item.rect.y0 - placement.draw_rect.y0) > 0.25
                    for placement in placement_list
                )
                else []
            )
            overflow_items = list(layout.overflow)
            if overflow_items:
                overflow_by_page.setdefault(logical_page_idx, []).extend(
                    overflow_items
                )
            if not draw_placements and not overflow_items and not page_numbers_here:
                continue
            redaction_items = [
                *(placement.item for placement in draw_placements),
                *overflow_items,
            ]
            seen_redactions: set[tuple[float, float, float, float]] = set()
            for item in redaction_items:
                redaction_rects = item.redaction_rects or (item.rect,)
                for raw_rect in redaction_rects:
                    rect = fitz.Rect(raw_rect)
                    key = tuple(round(value, 2) for value in rect)
                    if rect.is_empty or key in seen_redactions:
                        continue
                    seen_redactions.add(key)
                    add_visual_redaction(
                        page,
                        rect,
                        fill=None,
                        cross_out=False,
                    )
            for number in page_numbers_here:
                add_visual_redaction(
                    page,
                    _page_number_rect(number, segment),
                    fill=(1, 1, 1),
                    cross_out=False,
                )
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )
            for placement in draw_placements:
                item = placement.item
                spare = insert_visual_textbox(
                    page,
                    placement.draw_rect,
                    item.text,
                    fontname=_rhythm_font_resource_name(item.font_path, item.font_size),
                    fontfile=str(item.font_path),
                    fontsize=item.font_size,
                    lineheight=item.line_height,
                    align=item.align,
                    color=(0, 0, 0),
                    overlay=True,
                )
                if spare < 0:
                    raise RuntimeError("Typography block overflowed during vertical rhythm repair")
            for number in page_numbers_here:
                _draw_page_number(page, number, segment, body_font)
            if draw_placements or overflow_items:
                repaired_blocks += len(draw_placements) + len(overflow_items)
                repaired_pages += 1
            changed = True
        lineage = (
            _append_rhythm_continuations(
                document,
                overflow_by_page=overflow_by_page,
                reference_rects=reference_rects,
                reference_page_indices=reference_page_indices,
                mode=continuation_mode,
            )
            if overflow_by_page
            else _ContinuationLineage(
                tuple(
                    reference_page_indices[index]
                    for index in sorted(reference_page_indices)
                ),
                (),
                (),
            )
        )
        if lineage.continuation_page_indices:
            changed = True
        if changed:
            _save_replacement(document, pdf_path)
    finally:
        document.close()
    return _RhythmRepair(
        repaired_blocks,
        repaired_pages,
        source_page_indices=lineage.source_page_indices,
        continuation_page_indices=lineage.continuation_page_indices,
        continuation_source_pages=lineage.continuation_source_pages,
    )


def _save_replacement(document: fitz.Document, destination: Path) -> None:
    mode = destination.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-typography-",
        suffix=".pdf",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        document.save(str(temporary), garbage=4, deflate=True)
        temporary.chmod(mode)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _draw_page_number(
    page: fitz.Page,
    number: _PageNumber,
    segment: fitz.Rect,
    font: Path,
) -> None:

    rect = _page_number_rect(number, segment)
    spare = insert_visual_textbox(
        page,
        rect,
        number.text,
        fontname=_font_resource_name(font),
        fontfile=str(font),
        fontsize=_PAGE_NUMBER_FONT_SIZE,
        lineheight=1.0,
        align=fitz.TEXT_ALIGN_CENTER,
        color=(0, 0, 0),
        overlay=True,
    )
    if spare < 0:
        raise RuntimeError(f"Page {number.page_idx + 1} numeric page number overflowed")


def _page_number_rect(number: _PageNumber, segment: fitz.Rect) -> fitz.Rect:
    source_rect = _mapped_rect(segment, number.bbox, number.page_size)
    width = max(source_rect.width + 18.0, 54.0)
    return fitz.Rect(
        max(segment.x0, (source_rect.x0 + source_rect.x1 - width) / 2),
        max(segment.y0, source_rect.y0 - 2.0),
        min(segment.x1, (source_rect.x0 + source_rect.x1 + width) / 2),
        min(segment.y1, source_rect.y1 + 4.0),
    )


def _repair_document(
    pdf_path: Path,
    *,
    source_pdf: Path | None,
    ocr_result: dict[str, Any],
    body_regions: list[_BodyRegion],
    body_translations: dict[
        tuple[int, tuple[float, float, float, float]], str
    ],
    bands: list[_FurnitureBand],
    page_numbers: list[_PageNumber],
    body_font: Path,
    target_segment: Callable[[fitz.Page, int], fitz.Rect | None],
    furniture_segments: Callable[[fitz.Page, int], list[fitz.Rect]],
    logical_page_factory: Callable[[int], int | None],
) -> _DocumentRepair:
    document = fitz.open(str(pdf_path))
    source_document = (
        fitz.open(str(source_pdf)) if source_pdf is not None and source_pdf.is_file() else None
    )
    repaired_blocks = 0
    repaired_pages = 0
    restored_numbers = 0
    removed_bands = 0
    protected_visuals = 0
    deferred_body_blocks: list[_DeferredBodyBlock] = []
    changed = False
    try:
        for page_idx in range(document.page_count):
            page = document[page_idx]
            logical_page_idx = logical_page_factory(page_idx)
            if logical_page_idx is None:
                continue
            target = target_segment(page, page_idx)
            segments = furniture_segments(page, page_idx)
            redaction_count = 0
            plans = (
                _body_plans(
                    page,
                    page_idx=logical_page_idx,
                    segment=target,
                    body_regions=body_regions,
                    body_translations=body_translations,
                    ocr_result=ocr_result,
                    body_font=body_font,
                )
                if target is not None
                else []
            )
            page_bands = [
                item for item in bands if item.page_idx == logical_page_idx
            ]
            page_numbers_here = [
                item for item in page_numbers if item.page_idx == logical_page_idx
            ]
            pixel_regions = (
                _preserved_pixel_regions(ocr_result, logical_page_idx)
                if target is not None
                and source_document is not None
                and logical_page_idx < source_document.page_count
                else []
            )
            for plan in plans:
                if plan.deferred:
                    deferred_body_blocks.append(
                        _DeferredBodyBlock(
                            logical_page_idx,
                            _RhythmItem(
                                rect=fitz.Rect(plan.anchor_rect),
                                text=plan.text,
                                font_path=body_font,
                                font_size=plan.font_size,
                                line_height=plan.line_height,
                                redaction_rects=tuple(
                                    fitz.Rect(rect) for rect in plan.target_rects
                                ),
                                deferred=True,
                            ),
                        )
                    )
                    continue
                for target_rect in plan.target_rects:
                    rect = fitz.Rect(target_rect)
                    rect.x0 -= 0.7
                    rect.y0 -= 0.7
                    rect.x1 += 0.7
                    rect.y1 += 0.7
                    add_visual_redaction(page, rect, fill=None, cross_out=False)
                    redaction_count += 1
            if target is not None:
                for region in pixel_regions:
                    rect = _mapped_rect(target, region.bbox, region.page_size)
                    add_visual_redaction(page, rect, fill=None, cross_out=False)
                    redaction_count += 1
                    protected_visuals += 1
            for segment in segments:
                for band in page_bands:
                    add_visual_redaction(
                        page,
                        _mapped_rect(segment, band.bbox, band.page_size),
                        fill=(1, 1, 1),
                        cross_out=False,
                    )
                    redaction_count += 1
                    removed_bands += 1
                for number in page_numbers_here:
                    add_visual_redaction(
                        page,
                        _mapped_rect(segment, number.bbox, number.page_size),
                        fill=(1, 1, 1),
                        cross_out=False,
                    )
                    redaction_count += 1
            if redaction_count:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )
                changed = True
            for plan in plans:
                if plan.deferred:
                    continue
                spare = insert_visual_textbox(
                    page,
                    plan.draw_rect,
                    plan.text,
                    fontname=_font_resource_name(body_font),
                    fontfile=str(body_font),
                    fontsize=plan.font_size,
                    lineheight=plan.line_height,
                    align=fitz.TEXT_ALIGN_LEFT,
                    color=(0, 0, 0),
                    overlay=True,
                )
                if spare < 0:
                    raise RuntimeError("OCR body unexpectedly overflowed after preflight fitting")
                changed = True
            for segment in segments:
                for number in page_numbers_here:
                    _draw_page_number(page, number, segment, body_font)
                    restored_numbers += 1
            if target is not None and source_document is not None:
                source_page = source_document[logical_page_idx]
                source_segment = fitz.Rect(source_page.rect)
                for region in pixel_regions:
                    source_rect = _mapped_rect(
                        source_segment,
                        region.bbox,
                        region.page_size,
                    )
                    target_rect = _mapped_rect(target, region.bbox, region.page_size)
                    if source_rect.is_empty or target_rect.is_empty:
                        continue
                    show_pdf_page_visual(
                        page,
                        target_rect,
                        source_document,
                        logical_page_idx,
                        clip=source_rect,
                        keep_proportion=False,
                        overlay=True,
                    )
            if plans:
                repaired_pages += 1
                repaired_blocks += len(plans)
        if changed:
            _save_replacement(document, pdf_path)
    finally:
        document.close()
        if source_document is not None:
            source_document.close()
    return _DocumentRepair(
        repaired_blocks,
        repaired_pages,
        restored_numbers,
        removed_bands,
        protected_visuals,
        tuple(deferred_body_blocks),
    )


def restore_ocr_document_typography(
    *,
    ocr_result: dict[str, Any],
    translated_pdf: str | Path,
    bilingual_pdf: str | Path | None = None,
    source_pdf: str | Path | None = None,
    body_translation_plan: OcrBodyTranslationPlan | None = None,
    heading_translation_plan: OcrHeadingTranslationPlan | None = None,
) -> OcrDocumentTypographyResult:
    """Apply uniform body/title styles and numeric pagination to OCR PDFs."""
    translated_path = Path(translated_pdf)
    if not translated_path.is_file():
        raise FileNotFoundError(f"Translated PDF does not exist: {translated_path}")
    source_path = Path(source_pdf) if source_pdf else None
    if source_path is not None and not source_path.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {source_path}")
    body_regions = _normalise_body_regions(ocr_result)
    body_translations = {
        _body_region_key(item.page_idx, item.bbox): item.target_text
        for item in (body_translation_plan.regions if body_translation_plan else ())
    }
    if body_translation_plan is not None and len(body_translations) != len(body_regions):
        raise OcrBodyTranslationError(
            "正文翻译计划与 OCR 正文区域数量不一致"
        )
    bands, page_numbers = _furniture_and_page_numbers(ocr_result)

    # Repair headings first. Some PDF engines give a shrunken multi-line
    # heading a bounding box that reaches into the following paragraph. Body
    # text is therefore redrawn last so heading cleanup cannot clip its first
    # line.
    headings = restore_ocr_heading_typography(
        ocr_result=ocr_result,
        translated_pdf=translated_path,
        bilingual_pdf=bilingual_pdf,
        heading_translation_plan=heading_translation_plan,
    )

    def deferred_heading_blocks(raw_items: tuple[Any, ...]) -> tuple[_DeferredBodyBlock, ...]:
        return tuple(
            _DeferredBodyBlock(
                item.page_idx,
                _RhythmItem(
                    rect=fitz.Rect(item.source_rect),
                    text=item.text,
                    font_path=item.font_path,
                    font_size=item.font_size,
                    line_height=1.05,
                    align=item.align,
                    redaction_rects=tuple(
                        fitz.Rect(rect) for rect in item.target_rects
                    )
                    or (
                        (fitz.Rect(item.target_rect),)
                        if item.target_rect is not None
                        else ()
                    ),
                    deferred=True,
                ),
            )
            for item in raw_items
        )

    mono_document = fitz.open(str(translated_path))
    try:
        source_page_count = mono_document.page_count
        target_text = "\n".join(page.get_text("text") for page in mono_document)
    finally:
        mono_document.close()
    body_font = _regular_font(bool(_CJK_RE.search(target_text)))

    mono = _repair_document(
        translated_path,
        source_pdf=source_path,
        ocr_result=ocr_result,
        body_regions=body_regions,
        body_translations=body_translations,
        bands=bands,
        page_numbers=page_numbers,
        body_font=body_font,
        target_segment=lambda page, _page_idx: fitz.Rect(page.rect),
        furniture_segments=lambda page, _page_idx: [fitz.Rect(page.rect)],
        logical_page_factory=lambda page_idx: page_idx,
    )

    bilingual_path = Path(bilingual_pdf) if bilingual_pdf else None
    bilingual = _DocumentRepair(0, 0, 0, 0, 0)
    bilingual_rhythm = _RhythmRepair(0, 0)
    layout = None
    if bilingual_path is not None and bilingual_path.is_file():
        layout = detect_ocr_bilingual_layout(translated_path, bilingual_path)
        if layout is not None:
            bilingual = _repair_document(
                bilingual_path,
                source_pdf=source_path,
                ocr_result=ocr_result,
                body_regions=body_regions,
                body_translations=body_translations,
                bands=bands,
                page_numbers=page_numbers,
                body_font=body_font,
                target_segment=layout.target_segment,
                furniture_segments=layout.furniture_segments,
                logical_page_factory=layout.logical_page_index,
            )

    mono_rhythm = _repair_vertical_rhythm(
        translated_path,
        ocr_result=ocr_result,
        body_font=body_font,
        heading_font=headings.heading_font_path,
        page_numbers=page_numbers,
        segment_factory=lambda page, _page_idx: fitz.Rect(page.rect),
        logical_page_factory=lambda page_idx: page_idx,
        deferred_body_blocks=(
            *mono.deferred_body_blocks,
            *deferred_heading_blocks(headings.deferred_headings),
        ),
    )
    if layout is not None and bilingual_path is not None:
        bilingual_rhythm = _repair_vertical_rhythm(
            bilingual_path,
            ocr_result=ocr_result,
            body_font=body_font,
            heading_font=headings.heading_font_path,
            page_numbers=page_numbers,
            segment_factory=layout.target_segment,
            logical_page_factory=layout.logical_page_index,
            deferred_body_blocks=(
                *bilingual.deferred_body_blocks,
                *deferred_heading_blocks(headings.bilingual_deferred_headings),
            ),
            continuation_mode=layout.mode,
        )
        if (
            bilingual_rhythm.continuation_source_pages
            != mono_rhythm.continuation_source_pages
        ):
            logger.warning(
                "Mono and bilingual fixed-style pagination produced different "
                "continuation lineage: mono=%s bilingual=%s",
                mono_rhythm.continuation_source_pages,
                bilingual_rhythm.continuation_source_pages,
            )

    return OcrDocumentTypographyResult(
        translated_pdf_path=translated_path,
        bilingual_pdf_path=bilingual_path,
        body_font_path=body_font,
        body_font_size=_BODY_FONT_SIZE,
        body_line_height=_BODY_LINE_HEIGHT,
        paragraph_gap=_VERTICAL_RHYTHM_GAP,
        body_region_count=len(body_regions),
        repaired_body_blocks=mono.repaired_body_blocks,
        repaired_body_pages=mono.repaired_body_pages,
        restored_page_numbers=mono.restored_page_numbers,
        removed_header_footer_bands=mono.removed_header_footer_bands,
        protected_visual_regions=mono.protected_visual_regions,
        bilingual_repaired_body_blocks=bilingual.repaired_body_blocks,
        bilingual_restored_page_numbers=bilingual.restored_page_numbers,
        bilingual_protected_visual_regions=bilingual.protected_visual_regions,
        rhythm_blocks=mono_rhythm.blocks,
        bilingual_rhythm_blocks=bilingual_rhythm.blocks,
        headings=headings,
        continuation_pages=len(mono_rhythm.continuation_page_indices),
        source_page_indices=(
            mono_rhythm.source_page_indices or tuple(range(source_page_count))
        ),
        continuation_page_indices=mono_rhythm.continuation_page_indices,
        continuation_source_pages=mono_rhythm.continuation_source_pages,
    )


__all__ = [
    "OcrBodyRegionTranslation",
    "OcrBodyTranslationError",
    "OcrBodyTranslationPlan",
    "OcrDocumentTypographyResult",
    "restore_ocr_document_typography",
    "translate_ocr_body_regions",
]
