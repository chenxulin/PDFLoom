"""Translate OCR table descriptions and rebuild complete vector tables.

PP-StructureV3 supplies a table region, structured HTML and positioned table
text for scanned documents.  This module reconciles the HTML grid with the
visible source coordinates, reconstructs row/column spans, protects values and
identifiers before model translation, removes the complete source table on the
translated page, and draws a fresh searchable vector table.  It never places
translated strings on top of the source table image.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from .config import Settings
from .ocr_bilingual_layout import OcrBilingualLayout, detect_ocr_bilingual_layout
from .ocr_pdf_coordinates import (
    add_visual_redaction,
    draw_visual_rect,
    insert_visual_textbox,
    map_ocr_rect_to_visual,
    pdf_rect_to_visual,
)
from .ocr_semantics import canonical_ocr_type, is_furniture_region_type
from .translator import _build_client, translate_chunk

_TABLE_FONT_SIZE = 9.0
_TABLE_LINE_HEIGHT = 1.25
_CELL_PADDING_X = 2.5
# Keep the requested 9 pt / 1.25 line box intact.  Tight vertical cell insets
# let dense source grids fit without shrinking the text itself.
_CELL_PADDING_Y = 0.7
_GLYPH_BOUNDARY_SAFETY = 1.0
_CELL_BOUNDARY_TOLERANCE = 1.0
_BORDER_WIDTH = 0.55
_OUTER_BORDER_WIDTH = 0.85
_PAGE_MARGIN = 36.0
_CONTENT_CLEARANCE = 5.0
_HEADING_CLEARANCE = 13.0
_MIN_COLUMN_WIDTH = 34.0
_HEADING_REGION_TYPES = frozenset({"title", "section_heading", "table_caption", "figure_caption"})
_TARGET_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_FULL_DATE_RE = re.compile(r"^\s*\d{2,4}[./-]\d{1,2}[./-]\d{1,2}\s*$")
_FULL_NUMERIC_RE = re.compile(r"^\s*[<>≤≥±]?\s*[\d.,]+(?:\s*(?:%|‰|℃|°C|[A-Za-zμµ]+(?:/[A-Za-zμµ]+)?))?\s*$")
_FULL_CODE_RE = re.compile(r"^\s*(?=.*(?:\d|[-/.()]))[A-Za-z][A-Za-z0-9_.()/-]*\s*$")
_FULL_FORMULA_RE = re.compile(r"^\s*(?=[A-Za-z0-9]*[a-z0-9])(?:[A-Z][a-z]?\d*){2,}\s*$")
_LOCKED_ABBREVIATIONS = frozenset(
    {
        "API",
        "CAPA",
        "GMP",
        "HPLC",
        "ICH",
        "N/A",
        "NA",
        "ND",
        "OOT",
        "OOS",
        "QA",
        "QC",
        "RRT",
        "RT",
        "SOP",
        "TOC",
        "USP",
    }
)

# One pass over the original cell prevents a later regex from matching digits
# inside placeholders created by an earlier substitution.  Longer/specific
# alternatives intentionally precede plain numbers.
_PROTECTED_VALUE_RE = re.compile(
    r"\d{2,4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|[<>≤≥±]?\d+(?:\.\d+)?\s*(?:%|‰|℃|°C|mol/L|mmol/L|μmol/L|mg/mL|μg/mL|µg/mL|ug/mL|ng/mL|mg/L|μg/L|µg/L|ug/L|ng/L|ppm|ppb|mg|kg|μg|µg|ug|ng|mL|μL|µL|uL|L|g|h|min|s)(?:\s*/\s*(?:kg|mL|L|μL|µL|uL|h|min|s))?"
    r"|(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9_.()/-]*\d[A-Za-z0-9_.()/-]*|[A-Z]{2,}(?:[-/][A-Z0-9.()]+)+)(?![A-Za-z0-9])"
    r"|(?<![A-Za-z])(?:API|CAPA|GMP|HPLC|ICH|N/A|ND|OOT|OOS|QA|QC|RRT|RT|SOP|TOC|USP)(?:\([A-Za-z0-9-]+\))?(?![A-Za-z])"
    r"|(?<![A-Za-z])(?=[A-Za-z0-9]*[a-z0-9])(?:[A-Z][a-z]?\d*){2,}(?![A-Za-z])"
    r"|(?<=[\u3400-\u9fff])[A-Z](?![A-Za-z])"
    r"|[<>≤≥±]?\d+(?:\.\d+)?"
)
_PLACEHOLDER_RE = re.compile(r"\[\[JTBL\d{3}\]\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SIGNATURE_DATE_TABLE_RE = re.compile(
    r"(?:签名|签字|手签).{0,12}(?:日期|date)"
    r"|signature.{0,12}(?:date|日期)",
    flags=re.IGNORECASE | re.DOTALL,
)


class OcrTableError(ValueError):
    """Raised when a complete, readable table rebuild cannot be guaranteed."""


@dataclass(frozen=True)
class OcrTableCell:
    row: int
    column: int
    row_span: int
    column_span: int
    source_text: str
    target_text: str
    is_header: bool
    translated: bool
    protected_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class OcrTable:
    index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    row_count: int
    column_count: int
    cells: tuple[OcrTableCell, ...]
    preserve_as_image: bool = False


@dataclass(frozen=True)
class OcrTableTranslationPlan:
    tables: tuple[OcrTable, ...]
    table_count: int
    cell_count: int
    translated_cells: int
    preserved_cells: int
    protected_values: int

    @property
    def image_preserved_tables(self) -> int:
        return sum(table.preserve_as_image for table in self.tables)


@dataclass(frozen=True)
class OcrTableRedrawResult:
    translated_pdf_path: Path
    bilingual_pdf_path: Path | None
    table_font_size: float
    table_line_height: float
    redrawn_tables: int
    redrawn_cells: int
    bilingual_redrawn_tables: int
    bilingual_redrawn_cells: int
    continuation_pages: int
    source_page_indices: tuple[int, ...]
    continuation_page_indices: tuple[int, ...]
    continuation_page_groups: tuple[tuple[int, ...], ...]
    translated_overlay_regions: tuple[OcrTableOverlayRegion, ...]
    repeated_header_texts: tuple[str, ...]


@dataclass(frozen=True)
class OcrTableOverlayRegion:
    """A visual-coordinate area intentionally changed by table rendering."""

    page_idx: int
    bbox: tuple[float, float, float, float]
    continuation: bool


@dataclass(frozen=True)
class _RawCell:
    text: str
    row_span: int
    column_span: int
    header: bool


@dataclass(frozen=True)
class _PositionedTableText:
    text: str
    bbox: tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) * 0.5

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class _DrawLayout:
    table_rect: fitz.Rect
    column_widths: tuple[float, ...]
    row_heights: tuple[float, ...]


@dataclass(frozen=True)
class _DrawStats:
    tables: int
    cells: int
    inserted_pages: int = 0
    source_page_indices: tuple[int, ...] = ()
    continuation_page_indices: tuple[int, ...] = ()
    continuation_page_groups: tuple[tuple[int, ...], ...] = ()
    overlay_regions: tuple[OcrTableOverlayRegion, ...] = ()


@dataclass(frozen=True)
class _TableFragmentSpec:
    data_start: int
    data_end: int
    on_source_page: bool
    continuation_slot: int | None = None


@dataclass(frozen=True)
class _TablePagination:
    table: OcrTable
    header_rows: int
    fragments: tuple[_TableFragmentSpec, ...]
    paginated: bool


@dataclass(frozen=True)
class _PaginationPlan:
    tables: tuple[_TablePagination, ...]
    continuation_counts: tuple[int, ...]
    repeated_header_texts: tuple[str, ...]


@dataclass(frozen=True)
class _PageMap:
    source_target_pages: tuple[int, ...]
    continuation_target_pages: tuple[tuple[int, ...], ...]

    @property
    def inserted_target_pages(self) -> tuple[int, ...]:
        return tuple(page_idx for page_indexes in self.continuation_target_pages for page_idx in page_indexes)


@dataclass(frozen=True)
class _MeasuredTable:
    source_rect: fitz.Rect
    safe_top: float
    safe_bottom: float
    left: float
    column_widths: tuple[float, ...]
    row_heights: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedFragment:
    page_idx: int
    table: OcrTable
    layout: _DrawLayout


class _StructuredTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_RawCell]] = []
        self._row: list[_RawCell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_attrs: dict[str, str] = {}
        self._cell_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered == "tr":
            self._row = []
            return
        if lowered in {"td", "th"} and self._row is not None:
            self._cell_parts = []
            self._cell_attrs = {key.casefold(): str(value or "") for key, value in attrs}
            self._cell_header = lowered == "th"
            return
        if lowered == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            self._row.append(
                _RawCell(
                    text=_clean_cell_text("".join(self._cell_parts)),
                    row_span=_span(self._cell_attrs.get("rowspan")),
                    column_span=_span(self._cell_attrs.get("colspan")),
                    header=self._cell_header,
                )
            )
            self._cell_parts = None
            self._cell_attrs = {}
            self._cell_header = False
            return
        if lowered == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _clean_cell_text(value: Any) -> str:
    lines = [" ".join(part.split()) for part in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def should_preserve_table_as_image(structured_content: Any) -> bool:
    """Keep approval/signature tables pixel-exact because handwriting is OCR-unsafe."""
    plain = unescape(_HTML_TAG_RE.sub(" ", str(structured_content or "")))
    plain = " ".join(plain.split())
    return bool(_SIGNATURE_DATE_TABLE_RE.search(plain))


def _span(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return min(100, max(1, parsed))


def _positive_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        result = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and item > 0 for item in result):
        return None
    return result


def _positive_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        result = tuple(float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result) or result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _markdown_rows(value: str) -> list[list[_RawCell]]:
    rows: list[list[_RawCell]] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        values = [part.strip().replace("\\|", "|") for part in line[1:-1].split("|")]
        if values and all(re.fullmatch(r":?-{3,}:?", item) for item in values):
            continue
        rows.append([_RawCell(item, 1, 1, len(rows) == 0) for item in values])
    return rows


def _structured_rows(value: Any) -> list[list[_RawCell]]:
    content = str(value or "").strip()
    if not content:
        return []
    if "<table" in content.casefold():
        parser = _StructuredTableParser()
        parser.feed(content)
        parser.close()
        rows = parser.rows
    else:
        rows = _markdown_rows(content)
    # PP-Structure can absorb the small gap between a caption and a table as
    # one empty, fully merged first row.  It is not part of the source grid and
    # wastes valuable vertical space in the rebuilt table.
    while len(rows) > 1 and len(rows[0]) == 1 and not rows[0][0].text and rows[0][0].column_span > 1:
        rows = rows[1:]
    return rows


def _logical_column_hint(rows: list[list[_RawCell]]) -> int:
    """Return the consensus grid width, ignoring empty OCR overflow spans."""
    totals = [sum(cell.column_span for cell in row) for row in rows if row]
    if not totals:
        return 0
    counts = Counter(totals)
    expected = max(counts, key=lambda width: (counts[width], width))
    # Never discard a populated overflow column.  Empty trailing merged cells
    # are the known PP-Structure failure mode and may safely be clamped.
    for row in rows:
        total = 0
        for cell in row:
            total += cell.column_span
            if total > expected and cell.text:
                expected = total
    return expected


def _positioned_table_texts(
    blocks: list[Any],
    *,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> tuple[_PositionedTableText, ...]:
    """Return OCR table text blocks normalised to the table region coordinate space."""
    positioned: list[_PositionedTableText] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        try:
            block_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if block_page_idx != page_idx:
            continue
        if canonical_ocr_type(raw.get("type") or raw.get("sub_type")) != "table_text":
            continue
        block_bbox = _positive_bbox(raw.get("bbox"))
        block_page_size = _positive_pair(raw.get("page_size")) or page_size
        text = _clean_cell_text(raw.get("text"))
        if block_bbox is None or not text:
            continue
        scale_x = page_size[0] / block_page_size[0]
        scale_y = page_size[1] / block_page_size[1]
        normalised_bbox = (
            block_bbox[0] * scale_x,
            block_bbox[1] * scale_y,
            block_bbox[2] * scale_x,
            block_bbox[3] * scale_y,
        )
        center_x = (normalised_bbox[0] + normalised_bbox[2]) * 0.5
        center_y = (normalised_bbox[1] + normalised_bbox[3]) * 0.5
        if not (bbox[0] - 1.0 <= center_x <= bbox[2] + 1.0 and bbox[1] - 1.0 <= center_y <= bbox[3] + 1.0):
            continue
        key = text, normalised_bbox
        if key in seen:
            continue
        seen.add(key)
        positioned.append(_PositionedTableText(text=text, bbox=normalised_bbox))
    return tuple(sorted(positioned, key=lambda item: (item.center_y, item.center_x)))


def _group_positioned_text_lines(
    positioned: tuple[_PositionedTableText, ...],
) -> list[list[_PositionedTableText]]:
    """Cluster OCR blocks that share one visual text baseline."""
    groups: list[list[_PositionedTableText]] = []
    for item in positioned:
        if not groups:
            groups.append([item])
            continue
        current = groups[-1]
        center = median(member.center_y for member in current)
        height = median(member.height for member in current)
        tolerance = max(4.0, min(14.0, max(height, item.height) * 0.55))
        if abs(item.center_y - center) <= tolerance:
            current.append(item)
        else:
            groups.append([item])
    return groups


def _infer_column_anchors(
    positioned: tuple[_PositionedTableText, ...],
    *,
    column_count: int,
    table_width: float,
) -> tuple[float, ...]:
    """Infer stable source-column centres from complete value rows."""
    if column_count < 2:
        return ()
    samples: list[tuple[float, ...]] = []
    for group in _group_positioned_text_lines(positioned):
        if len(group) != column_count or not all(_is_locked_value(item.text) for item in group):
            continue
        samples.append(tuple(sorted(item.center_x for item in group)))
    # Two agreeing rows are enough to reject a coincidental numeric header and
    # still recover short tables.  A single row remains too ambiguous to alter
    # merge topology automatically.
    if len(samples) < 2:
        return ()
    anchors = tuple(float(median(sample[column] for sample in samples)) for column in range(column_count))
    gaps = [right - left for left, right in zip(anchors, anchors[1:], strict=False)]
    if not gaps or min(gaps) < max(8.0, table_width / (column_count * 5.0)):
        return ()
    tolerance = min(gaps) * 0.38
    if any(
        abs(sample[column] - anchors[column]) > tolerance
        for sample in samples
        for column in range(column_count)
    ):
        return ()
    return anchors


def _nearest_anchor_column(value: float, anchors: tuple[float, ...]) -> int:
    return min(range(len(anchors)), key=lambda column: abs(value - anchors[column]))


def _visual_locked_rows(
    positioned: tuple[_PositionedTableText, ...],
    anchors: tuple[float, ...],
) -> list[tuple[float, tuple[str, ...]]]:
    """Read complete code/measurement rows directly from positioned OCR text."""
    if not anchors:
        return []
    result: list[tuple[float, tuple[str, ...]]] = []
    for group in _group_positioned_text_lines(positioned):
        if len(group) != len(anchors) or not all(_is_locked_value(item.text) for item in group):
            continue
        by_column: dict[int, _PositionedTableText] = {}
        valid = True
        for item in group:
            column = _nearest_anchor_column(item.center_x, anchors)
            if column in by_column:
                valid = False
                break
            neighbour_gaps = []
            if column:
                neighbour_gaps.append(anchors[column] - anchors[column - 1])
            if column + 1 < len(anchors):
                neighbour_gaps.append(anchors[column + 1] - anchors[column])
            if neighbour_gaps and abs(item.center_x - anchors[column]) > min(neighbour_gaps) * 0.45:
                valid = False
                break
            by_column[column] = item
        if not valid or set(by_column) != set(range(len(anchors))):
            continue
        result.append(
            (
                float(median(item.center_y for item in group)),
                tuple(by_column[column].text for column in range(len(anchors))),
            )
        )
    return result


def _cell_text_key(text: str) -> str:
    return re.sub(r"\s+", "", _clean_cell_text(text)).casefold()


def _header_cell_column_candidates(
    text: str,
    *,
    positioned: tuple[_PositionedTableText, ...],
    anchors: tuple[float, ...],
) -> list[tuple[int, int]]:
    """Locate one HTML header cell against its source OCR text coordinates."""
    cell_key = _cell_text_key(text)
    if not cell_key or not anchors:
        return []
    matches = [
        item for item in positioned if (block_key := _cell_text_key(item.text)) and block_key in cell_key
    ]
    if not matches:
        return []
    gaps = [right - left for left, right in zip(anchors, anchors[1:], strict=False)]
    cluster_gap = max(16.0, (min(gaps) if gaps else 32.0) * 0.55)
    clusters: list[list[_PositionedTableText]] = []
    for item in sorted(matches, key=lambda member: member.center_x):
        if not clusters or item.center_x - clusters[-1][-1].center_x > cluster_gap:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    candidates: list[tuple[int, int]] = []
    anchor_tolerance = (min(gaps) if gaps else 24.0) * 0.15
    for cluster in clusters:
        x0 = min(item.bbox[0] for item in cluster)
        x1 = max(item.bbox[2] for item in cluster)
        columns = [
            column
            for column, anchor in enumerate(anchors)
            if x0 - anchor_tolerance <= anchor <= x1 + anchor_tolerance
        ]
        if not columns:
            center = median(item.center_x for item in cluster)
            columns = [_nearest_anchor_column(float(center), anchors)]
        start, end = min(columns), max(columns)
        if (start, end) not in candidates:
            candidates.append((start, end))
    return sorted(candidates)


def _map_header_row(
    row: list[_RawCell],
    *,
    positioned: tuple[_PositionedTableText, ...],
    anchors: tuple[float, ...],
) -> list[tuple[_RawCell, int, int]] | None:
    mapped: list[tuple[_RawCell, int, int]] = []
    previous_end = -1
    for cell in (item for item in row if item.text):
        candidates = _header_cell_column_candidates(
            cell.text,
            positioned=positioned,
            anchors=anchors,
        )
        selected = next(
            ((start, end) for start, end in candidates if start > previous_end),
            None,
        )
        if selected is None:
            return None
        start, end = selected
        mapped.append((cell, start, end))
        previous_end = end
    return mapped or None


def _repair_phantom_multilevel_header(
    rows: list[list[_RawCell]],
    *,
    positioned: tuple[_PositionedTableText, ...],
    anchors: tuple[float, ...],
    visual_rows: list[tuple[float, tuple[str, ...]]],
) -> list[list[_RawCell]]:
    """Replace PP-Structure's empty header scaffold with the visible source grid."""
    column_count = len(anchors)
    if len(rows) < 3 or column_count < 2:
        return rows
    scaffold = rows[0]
    if (
        not scaffold
        or any(cell.text for cell in scaffold)
        or sum(cell.column_span for cell in scaffold) != column_count
        or not any(cell.row_span > 1 or cell.column_span > 1 for cell in scaffold)
    ):
        return rows

    first_value_y = visual_rows[0][0] if visual_rows else float("inf")
    typical_height = median(item.height for item in positioned) if positioned else 0.0
    header_cutoff = first_value_y - max(2.0, typical_height * 0.25)
    header_texts = tuple(item for item in positioned if item.center_y < header_cutoff)
    top = _map_header_row(rows[1], positioned=header_texts, anchors=anchors)
    lower = _map_header_row(rows[2], positioned=header_texts, anchors=anchors)
    if top is None or lower is None:
        return rows

    expected_columns = set(range(column_count))
    top_coverage: set[int] = set()
    for _cell, start, end in top:
        covered = set(range(start, end + 1))
        if top_coverage & covered:
            return rows
        top_coverage.update(covered)
    if top_coverage != expected_columns:
        return rows

    lower_coverage: set[int] = set()
    for _cell, start, end in lower:
        covered = set(range(start, end + 1))
        if lower_coverage & covered:
            return rows
        lower_coverage.update(covered)
    standalone_coverage = expected_columns - lower_coverage
    if not standalone_coverage or lower_coverage | standalone_coverage != expected_columns:
        return rows

    rebuilt_top: list[_RawCell] = []
    for cell, start, end in top:
        covered = set(range(start, end + 1))
        if covered & lower_coverage and not covered <= lower_coverage:
            return rows
        rebuilt_top.append(
            _RawCell(
                text=cell.text,
                row_span=2 if covered <= standalone_coverage else 1,
                column_span=end - start + 1,
                header=True,
            )
        )
    rebuilt_lower = [
        _RawCell(
            text=cell.text,
            row_span=1,
            column_span=end - start + 1,
            header=True,
        )
        for cell, start, end in lower
    ]
    return [rebuilt_top, rebuilt_lower, *rows[3:]]


def _simple_raw_row_values(row: list[_RawCell], column_count: int) -> tuple[str, ...] | None:
    if len(row) != column_count or any(cell.row_span != 1 or cell.column_span != 1 for cell in row):
        return None
    return tuple(cell.text for cell in row)


def _repair_positioned_value_rows(
    rows: list[list[_RawCell]],
    *,
    visual_rows: list[tuple[float, tuple[str, ...]]],
    column_count: int,
) -> list[list[_RawCell]]:
    """Correct concatenated values and append confidently observed trailing rows."""
    if len(visual_rows) < 2 or column_count < 2:
        return rows
    visual_keys = [_cell_text_key(values[0]) for _y, values in visual_rows]
    if any(not key for key in visual_keys) or len(set(visual_keys)) != len(visual_keys):
        return rows
    visual_by_key = {key: index for index, key in enumerate(visual_keys)}

    matches: list[tuple[int, int]] = []
    for row_idx, row in enumerate(rows):
        values = _simple_raw_row_values(row, column_count)
        if values is None or not values[0]:
            continue
        key = _cell_text_key(values[0])
        visual_idx = visual_by_key.get(key)
        locked = sum(_is_locked_value(value) for value in values)
        if visual_idx is not None and locked >= max(2, column_count - 1):
            matches.append((row_idx, visual_idx))
    if len(matches) < 2:
        return rows
    raw_indexes = [row_idx for row_idx, _visual_idx in matches]
    visual_indexes = [visual_idx for _row_idx, visual_idx in matches]
    if raw_indexes != sorted(raw_indexes) or visual_indexes != sorted(visual_indexes):
        return rows
    if len(set(visual_indexes)) != len(visual_indexes):
        return rows

    repaired = list(rows)
    for row_idx, visual_idx in matches:
        repaired[row_idx] = [
            _RawCell(text=value, row_span=1, column_span=1, header=False)
            for value in visual_rows[visual_idx][1]
        ]

    last_raw_idx, last_visual_idx = matches[-1]
    if last_raw_idx == len(rows) - 1 and last_visual_idx + 1 < len(visual_rows):
        existing_keys = {
            _cell_text_key(values[0])
            for row in repaired
            if (values := _simple_raw_row_values(row, column_count)) is not None and values[0]
        }
        for _y, values in visual_rows[last_visual_idx + 1 :]:
            if _cell_text_key(values[0]) in existing_keys:
                continue
            repaired.append(
                [_RawCell(text=value, row_span=1, column_span=1, header=False) for value in values]
            )
    return repaired


def _repair_rows_from_positioned_text(
    rows: list[list[_RawCell]],
    *,
    blocks: list[Any],
    page_idx: int,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> list[list[_RawCell]]:
    """Use visible source coordinates when PP-Structure HTML loses table geometry."""
    column_count = _logical_column_hint(rows)
    positioned = _positioned_table_texts(
        blocks,
        page_idx=page_idx,
        bbox=bbox,
        page_size=page_size,
    )
    anchors = _infer_column_anchors(
        positioned,
        column_count=column_count,
        table_width=bbox[2] - bbox[0],
    )
    if not anchors:
        return rows
    visual_rows = _visual_locked_rows(positioned, anchors)
    repaired = _repair_phantom_multilevel_header(
        rows,
        positioned=positioned,
        anchors=anchors,
        visual_rows=visual_rows,
    )
    return _repair_positioned_value_rows(
        repaired,
        visual_rows=visual_rows,
        column_count=column_count,
    )


def _infer_multilevel_headers(
    cells: tuple[OcrTableCell, ...],
) -> tuple[OcrTableCell, ...]:
    """Mark descriptive continuation rows beneath merged first-row headers."""
    header_rows = {cell.row for cell in cells if cell.is_header}
    for row_idx in range(1, 3):
        row_cells = [cell for cell in cells if cell.row == row_idx]
        if not row_cells:
            break
        continuation = any(
            cell.row < row_idx < cell.row + cell.row_span for cell in cells if cell.is_header
        ) or any(cell.row == row_idx - 1 and cell.column_span > 1 for cell in cells if cell.is_header)
        if not continuation:
            break
        populated = [cell for cell in row_cells if cell.source_text]
        data_like = [
            cell
            for cell in populated
            if _is_locked_value(cell.source_text)
            and (
                bool(re.search(r"\d", cell.source_text))
                or cell.source_text.strip().upper() in {"N/A", "NA", "ND"}
            )
        ]
        if populated and len(data_like) / len(populated) >= 0.50:
            break
        header_rows.add(row_idx)
    return tuple(replace(cell, is_header=True) if cell.row in header_rows else cell for cell in cells)


def _logical_cells(rows: list[list[_RawCell]]) -> tuple[tuple[OcrTableCell, ...], int, int]:
    occupied: set[tuple[int, int]] = set()
    cells: list[OcrTableCell] = []
    row_count = 0
    expected_columns = _logical_column_hint(rows)
    column_count = expected_columns
    for row_idx, raw_row in enumerate(rows):
        row = list(raw_row)
        occupied_here = sum(1 for column in range(expected_columns) if (row_idx, column) in occupied)
        supplied_slots = sum(cell.column_span for cell in row)
        # PP-Structure occasionally represents an omitted blank cell as an
        # empty trailing <td>.  A long procedure then shifts into the short
        # category column and makes the rebuilt table both semantically wrong
        # and unnecessarily tall.  Rotate that explicit blank to the first
        # available position only when the row otherwise exactly fills the
        # logical grid and starts with unmistakably narrative text.
        if (
            row
            and expected_columns - occupied_here == supplied_slots
            and not row[-1].text
            and len(re.sub(r"\s+", "", row[0].text)) >= 24
            and row[-1].column_span == 1
        ):
            row = [row[-1], *row[:-1]]
        column_idx = 0
        for raw in row:
            while (row_idx, column_idx) in occupied:
                column_idx += 1
            available_span = expected_columns - column_idx
            if available_span <= 0:
                if raw.text:
                    expected_columns = column_idx + raw.column_span
                    available_span = raw.column_span
                else:
                    continue
            column_span = min(raw.column_span, available_span)
            if column_span < raw.column_span and raw.text:
                expected_columns = column_idx + raw.column_span
                column_span = raw.column_span
            for covered_row in range(row_idx, row_idx + raw.row_span):
                for covered_column in range(column_idx, column_idx + column_span):
                    occupied.add((covered_row, covered_column))
            cells.append(
                OcrTableCell(
                    row=row_idx,
                    column=column_idx,
                    row_span=raw.row_span,
                    column_span=column_span,
                    source_text=raw.text,
                    target_text=raw.text,
                    is_header=raw.header or row_idx == 0,
                    translated=False,
                )
            )
            row_count = max(row_count, row_idx + raw.row_span)
            column_count = max(column_count, column_idx + column_span)
            column_idx += column_span
    return _infer_multilevel_headers(tuple(cells)), row_count, column_count


def _recover_trailing_blank_row(
    cells: tuple[OcrTableCell, ...],
    *,
    row_count: int,
    column_count: int,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    blocks: list[Any],
) -> tuple[OcrTableCell, ...]:
    """Fill a blank final HTML row from OCR table-text blocks when available."""
    trailing_indexes = [index for index, cell in enumerate(cells) if cell.row == row_count - 1]
    if (
        not trailing_indexes
        or any(cells[index].source_text for index in trailing_indexes)
        or column_count < 1
    ):
        return cells

    table_width = bbox[2] - bbox[0]
    table_height = bbox[3] - bbox[1]
    band_top = bbox[1] + table_height * 0.82
    candidates: list[tuple[float, float, str]] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        try:
            block_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if block_page_idx != page_idx:
            continue
        if canonical_ocr_type(raw.get("type") or raw.get("sub_type")) != "table_text":
            continue
        block_bbox = _positive_bbox(raw.get("bbox"))
        block_page_size = _positive_pair(raw.get("page_size")) or page_size
        text = _clean_cell_text(raw.get("text"))
        if block_bbox is None or not text:
            continue
        scale_x = page_size[0] / block_page_size[0]
        scale_y = page_size[1] / block_page_size[1]
        center_x = (block_bbox[0] + block_bbox[2]) * 0.5 * scale_x
        center_y = (block_bbox[1] + block_bbox[3]) * 0.5 * scale_y
        if bbox[0] <= center_x <= bbox[2] and band_top <= center_y <= bbox[3]:
            candidates.append((center_y, center_x, text))
    if not candidates:
        return cells

    grouped: dict[int, list[tuple[float, float, str]]] = {index: [] for index in trailing_indexes}
    for candidate in candidates:
        center_x = candidate[1]
        containing = [
            index
            for index in trailing_indexes
            if bbox[0] + table_width * cells[index].column / column_count
            <= center_x
            <= bbox[0] + table_width * (cells[index].column + cells[index].column_span) / column_count
        ]
        best_index = (
            containing[0]
            if containing
            else min(
                trailing_indexes,
                key=lambda index: abs(
                    center_x
                    - (
                        bbox[0]
                        + table_width * (cells[index].column + cells[index].column_span / 2) / column_count
                    )
                ),
            )
        )
        grouped[best_index].append(candidate)

    recovered = list(cells)
    for index, items in grouped.items():
        if not items:
            continue
        text = " ".join(item[2] for item in sorted(items))
        recovered[index] = replace(
            recovered[index],
            source_text=text,
            target_text=text,
        )
    return tuple(recovered)


def _refine_table_bbox(
    bbox: tuple[float, float, float, float],
    *,
    page_idx: int,
    page_size: tuple[float, float],
    blocks: list[Any],
) -> tuple[float, float, float, float]:
    """Exclude a caption that PP-Structure accidentally overlaps with a table."""
    obstacle_bottom = bbox[1]
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        try:
            block_page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if block_page_idx != page_idx:
            continue
        item_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if item_type in {"table", "table_text"} or is_furniture_region_type(item_type):
            continue
        block_bbox = _positive_bbox(raw.get("bbox"))
        block_page_size = _positive_pair(raw.get("page_size")) or page_size
        if block_bbox is None:
            continue
        scale_x = page_size[0] / block_page_size[0]
        scale_y = page_size[1] / block_page_size[1]
        normalized = (
            block_bbox[0] * scale_x,
            block_bbox[1] * scale_y,
            block_bbox[2] * scale_x,
            block_bbox[3] * scale_y,
        )
        horizontal_overlap = max(
            0.0,
            min(bbox[2], normalized[2]) - max(bbox[0], normalized[0]),
        )
        if horizontal_overlap > 0 and normalized[1] < bbox[1] < normalized[3]:
            obstacle_bottom = max(obstacle_bottom, normalized[3])
    if obstacle_bottom <= bbox[1]:
        return bbox
    # Roughly 3–4 PDF points in the usual OCR coordinate space: enough to
    # preserve a readable caption-to-table gap without shrinking the text.
    clearance = max(4.0, page_size[1] * 0.004)
    refined_top = min(bbox[3] - 1.0, obstacle_bottom + clearance)
    return bbox[0], refined_top, bbox[2], bbox[3]


def extract_ocr_tables(ocr_result: dict[str, Any]) -> tuple[OcrTable, ...]:
    """Read every PP-Structure table into a validated logical cell grid."""
    tables: list[OcrTable] = []
    blocks = list(ocr_result.get("blocks") or [])
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        if canonical_ocr_type(raw.get("type") or raw.get("sub_type")) != "table":
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            page_idx = -1
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size"))
        if page_idx < 0 or bbox is None or page_size is None:
            raise OcrTableError("OCR 表格缺少可靠页码、坐标或页面尺寸")
        bbox = _refine_table_bbox(
            bbox,
            page_idx=page_idx,
            page_size=page_size,
            blocks=blocks,
        )
        rows = _structured_rows(raw.get("structured_content"))
        rows = _repair_rows_from_positioned_text(
            rows,
            blocks=blocks,
            page_idx=page_idx,
            bbox=bbox,
            page_size=page_size,
        )
        cells, row_count, column_count = _logical_cells(rows)
        if not cells or row_count < 1 or column_count < 1:
            raise OcrTableError(f"第 {page_idx + 1} 页表格没有可重建的单元格结构，不能安全翻译")
        cells = _recover_trailing_blank_row(
            cells,
            row_count=row_count,
            column_count=column_count,
            page_idx=page_idx,
            bbox=bbox,
            page_size=page_size,
            blocks=blocks,
        )
        tables.append(
            OcrTable(
                index=len(tables),
                page_idx=page_idx,
                bbox=bbox,
                page_size=page_size,
                row_count=row_count,
                column_count=column_count,
                cells=cells,
                preserve_as_image=should_preserve_table_as_image(raw.get("structured_content")),
            )
        )
    return tuple(tables)


def _is_locked_value(text: str) -> bool:
    value = _clean_cell_text(text)
    if not value:
        return True
    compact = value.replace(" ", "")
    return bool(
        _FULL_DATE_RE.fullmatch(value)
        or _FULL_NUMERIC_RE.fullmatch(value)
        or _FULL_CODE_RE.fullmatch(value)
        or _FULL_FORMULA_RE.fullmatch(value)
        or compact.upper() in _LOCKED_ABBREVIATIONS
    )


def _needs_translation(text: str, target_language: str) -> bool:
    if _is_locked_value(text):
        return False
    target = (target_language or "").replace("_", "-").casefold()
    has_cjk = bool(_TARGET_CJK_RE.search(text))
    if target.startswith("en"):
        return has_cjk
    if target.startswith(("zh", "ja", "ko")):
        return bool(re.search(r"[A-Za-z]", text)) and not has_cjk
    return has_cjk or bool(re.search(r"[A-Za-z]", text))


def _protect_values(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    parts: list[str] = []
    protected: list[tuple[str, str]] = []
    cursor = 0
    for match in _PROTECTED_VALUE_RE.finditer(text):
        if match.start() < cursor:
            continue
        parts.append(text[cursor : match.start()])
        placeholder = f"[[JTBL{len(protected):03d}]]"
        protected.append((placeholder, match.group(0)))
        parts.append(placeholder)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), tuple(protected)


def _expose_values(
    text: str,
    protected: tuple[tuple[str, str], ...],
) -> str:
    """Attach each exact value to its model-visible slot without changing IDs."""
    result = text
    for placeholder, value in protected:
        if result.count(placeholder) != 1:
            raise OcrTableError(f"原文数据占位符不唯一 {placeholder}")
        result = result.replace(
            placeholder,
            f"[[{placeholder[2:-2]}|{value}]]",
        )
    return result


def _restore_values(text: str, protected: tuple[tuple[str, str], ...]) -> str:
    result = _clean_cell_text(text)
    for placeholder, value in protected:
        token = placeholder[2:-2]
        prefix = re.escape(token[:-3])
        suffix = re.escape(token[-3:])
        token_pattern = rf"{prefix}[\s_-]*{suffix}"
        # The JTBL id, rather than the model-echoed payload, is the authoritative
        # mapping key.  Models commonly keep the id but localise a visible value
        # (for example ``2025`` -> ``2025 year``) or adjust unit spacing.  Requiring
        # that redundant payload to remain byte-identical turns a recoverable
        # formatting drift into a fatal document error.  Consume any single-line
        # payload inside a recognised wrapper and restore the exact OCR value
        # below.  Duplicate ids are still rejected after normalisation, and an
        # unknown/renumbered JTBL token is still rejected by the final check.
        wrapped_value_payload = r"(?:\s*[|=:]\s*[^\[\]\r\n]*?)?"
        bare_value_payload = rf"(?:\s*[|=:]\s*{re.escape(value)})?"
        wrapper_patterns = (
            rf"\[\s*\[\s*{token_pattern}{wrapped_value_payload}\s*\]\s*\]",
            rf"［\s*［\s*{token_pattern}{wrapped_value_payload}\s*］\s*］",
            rf"【\s*{token_pattern}{wrapped_value_payload}\s*】",
            rf"(?<!\[)\[\s*{token_pattern}{wrapped_value_payload}\s*\](?!\])",
        )
        for pattern in wrapper_patterns:
            result = re.sub(pattern, placeholder, result, flags=re.IGNORECASE)
        bare_pattern = (
            rf"(?<![A-Za-z0-9\[［【]){token_pattern}{bare_value_payload}"
            rf"(?![A-Za-z0-9\]］】])"
        )
        result = re.sub(bare_pattern, placeholder, result, flags=re.IGNORECASE)

    value_counts = Counter(value for _, value in protected)
    raw_value_fallbacks: set[str] = set()
    for placeholder, value in protected:
        placeholder_count = result.count(placeholder)
        token = placeholder[2:-2]
        token_pattern = re.compile(
            rf"{re.escape(token[:-3])}[\s_-]*{re.escape(token[-3:])}",
            flags=re.IGNORECASE,
        )
        raw_value_is_unambiguous = (
            placeholder_count == 0
            and token_pattern.search(result) is None
            and value_counts[value] == 1
            and result.count(value) == 1
        )
        if raw_value_is_unambiguous:
            raw_value_fallbacks.add(placeholder)
            continue
        if placeholder_count != 1:
            raise OcrTableError(f"译文未逐字保留数据占位符 {placeholder}")

    for placeholder, value in protected:
        if placeholder not in raw_value_fallbacks:
            result = result.replace(placeholder, value)
    if _PLACEHOLDER_RE.search(result) or re.search(r"(?i)JTBL[\s_-]*\d{3}", result):
        raise OcrTableError("译文含有未恢复的数据占位符")
    return result


def _translation_is_valid(source: str, target: str, target_language: str) -> bool:
    if not target.strip():
        return False
    normalized_target = (target_language or "").replace("_", "-").casefold()
    if normalized_target.startswith("en") and _TARGET_CJK_RE.search(target):
        return False
    return not (source.strip() == target.strip() and _needs_translation(source, target_language))


async def translate_ocr_tables(
    *,
    ocr_result: dict[str, Any],
    settings: Settings,
    on_progress: Callable[[str, float, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> OcrTableTranslationPlan:
    """Translate description cells while preserving all factual value tokens."""
    tables = extract_ocr_tables(ocr_result)
    candidates: list[tuple[int, int, str, tuple[tuple[str, str], ...]]] = []
    mutable_cells = [list(table.cells) for table in tables]
    preserved_cells = 0
    locked_value_cells = 0
    for table_idx, table in enumerate(tables):
        for cell_idx, cell in enumerate(table.cells):
            if (
                table.preserve_as_image
                or not cell.source_text
                or not _needs_translation(cell.source_text, settings.target_language)
            ):
                preserved_cells += 1
                if not table.preserve_as_image and cell.source_text and _is_locked_value(cell.source_text):
                    locked_value_cells += 1
                continue
            protected_text, values = _protect_values(cell.source_text)
            protected_text = _expose_values(protected_text, values)
            candidates.append((table_idx, cell_idx, protected_text, values))

    if not candidates:
        return OcrTableTranslationPlan(
            tables=tables,
            table_count=len(tables),
            cell_count=sum(len(table.cells) for table in tables),
            translated_cells=0,
            preserved_cells=preserved_cells,
            protected_values=locked_value_cells,
        )

    if on_progress:
        on_progress("table-translate", 0.0, f"翻译表格说明文字：0/{len(candidates)}")
    client = _build_client(settings)
    concurrency = min(6, max(1, int(getattr(settings, "max_workers", 4) or 4)))
    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    done = 0

    async def translate_one(
        table_idx: int,
        cell_idx: int,
        protected_text: str,
        values: tuple[tuple[str, str], ...],
    ) -> None:
        nonlocal done
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        cell = tables[table_idx].cells[cell_idx]
        table = tables[table_idx]
        async with semaphore:
            last_error: Exception | None = None
            rejected_translation: str | None = None
            translated = ""
            for attempt in range(2):
                translated = ""
                try:
                    translated = await translate_chunk(
                        client,
                        protected_text,
                        (
                            f"Table {table_idx + 1}, row {cell.row + 1}, "
                            f"column {cell.column + 1}. Translate description text only; "
                            "copy every [[JTBLnnn]] placeholder exactly."
                        ),
                        "",
                        settings,
                        seg_type="table_cell",
                        source_kind="pdf",
                        has_layout=True,
                        layout_retry_reason=("table_value_placeholder_integrity" if attempt else None),
                        required_literals=tuple(value for _, value in values),
                        rejected_translation=rejected_translation,
                    )
                    translated = _restore_values(translated, values)
                    if not _translation_is_valid(cell.source_text, translated, settings.target_language):
                        raise OcrTableError("表格说明文字未完整翻译到目标语言")
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 - strict second attempt
                    last_error = exc
                    if translated.strip():
                        rejected_translation = translated
            if last_error is not None:
                raise OcrTableError(
                    f"第 {table.page_idx + 1} 页表格第 {cell.row + 1} 行"
                    f"第 {cell.column + 1} 列翻译失败：{last_error}"
                ) from last_error
            mutable_cells[table_idx][cell_idx] = replace(
                cell,
                target_text=translated,
                translated=True,
                protected_values=tuple(value for _, value in values),
            )
        async with progress_lock:
            done += 1
            if on_progress:
                on_progress(
                    "table-translate",
                    100.0 * done / len(candidates),
                    f"翻译表格说明文字：{done}/{len(candidates)}",
                )

    try:
        await asyncio.gather(
            *(
                translate_one(table_idx, cell_idx, protected_text, values)
                for table_idx, cell_idx, protected_text, values in candidates
            )
        )
    finally:
        await client.aclose()

    translated_tables = tuple(
        replace(table, cells=tuple(mutable_cells[index])) for index, table in enumerate(tables)
    )
    return OcrTableTranslationPlan(
        tables=translated_tables,
        table_count=len(translated_tables),
        cell_count=sum(len(table.cells) for table in translated_tables),
        translated_cells=len(candidates),
        preserved_cells=preserved_cells,
        protected_values=(locked_value_cells + sum(len(values) for *_, values in candidates)),
    )


def _mapped_rect(
    segment: fitz.Rect,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    return map_ocr_rect_to_visual(segment, bbox, page_size)


def _horizontal_overlap(first: fitz.Rect, second: fitz.Rect) -> float:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    return overlap / max(1.0, min(first.width, second.width))


def _safe_vertical_bounds(
    table: OcrTable,
    source_rect: fitz.Rect,
    segment: fitz.Rect,
    ocr_result: dict[str, Any],
) -> tuple[float, float]:
    previous = segment.y0 + _PAGE_MARGIN
    following = segment.y1 - _PAGE_MARGIN
    for raw in [*(ocr_result.get("regions") or []), *(ocr_result.get("blocks") or [])]:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        if page_idx != table.page_idx:
            continue
        item_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if item_type == "table_text" or is_furniture_region_type(item_type):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size")) or table.page_size
        if bbox is None:
            continue
        rect = _mapped_rect(segment, bbox, page_size)
        if rect.is_empty or _horizontal_overlap(rect, source_rect) < 0.12:
            continue
        # Ignore the table region itself and any duplicated layout wrapper.
        intersection = rect & source_rect
        if (
            not intersection.is_empty
            and intersection.get_area() >= min(rect.get_area(), source_rect.get_area()) * 0.50
        ):
            continue
        clearance = _HEADING_CLEARANCE if item_type in _HEADING_REGION_TYPES else _CONTENT_CLEARANCE
        if rect.y1 <= source_rect.y0 + 0.5:
            previous = max(previous, rect.y1 + clearance)
        elif rect.y0 >= source_rect.y1 - 0.5:
            following = min(following, rect.y0 - clearance)
    return min(previous, source_rect.y0), max(following, source_rect.y1)


def _font_resource(path: Path, *, bold: bool) -> str:
    stem = re.sub(r"[^A-Za-z0-9]", "", path.stem)[:20]
    return f"ocrtable{'bold' if bold else 'regular'}{stem}"


def _builtin_cjk_font(text: str) -> str | None:
    """Use PyMuPDF CJK fonts so extracted text retains canonical Unicode."""
    if re.search(r"[\uac00-\ud7af]", text):
        return "korea-s"
    if re.search(r"[\u3040-\u30ff]", text):
        return "japan-s"
    if re.search(r"[\u3400-\u9fff]", text):
        return "china-s"
    return None


def _textbox_font_args(text: str, font: Path, *, bold: bool) -> dict[str, str]:
    builtin = _builtin_cjk_font(text)
    if builtin is not None:
        return {"fontname": builtin}
    return {
        "fontname": _font_resource(font, bold=bold),
        "fontfile": str(font),
    }


def _measurement_font(text: str, font: Path) -> fitz.Font:
    builtin = _builtin_cjk_font(text)
    return fitz.Font(fontname=builtin) if builtin is not None else fitz.Font(fontfile=str(font))


def _resolve_bold_font(regular_font: Path) -> Path:
    cache = Path(os.getenv("XDG_CACHE_HOME") or "/root/.cache") / "babeldoc" / "fonts"
    names = (
        regular_font.name.replace("Regular", "Bold"),
        regular_font.name.replace("-Roman", "-Bold"),
    )
    candidates = [regular_font.with_name(name) for name in names if name != regular_font.name]
    candidates.extend(
        [
            cache / "GoNotoKurrent-Bold.ttf",
            cache / "SourceHanSerifCN-Bold.ttf",
            cache / "NotoSerif-Bold.ttf",
            Path("/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No bold font is available for OCR table headers")


class _TextMeasurer:
    def __init__(self) -> None:
        self.document = fitz.open()
        self.page = self.document.new_page(width=2000, height=2000)
        self.cache: dict[tuple[str, float, str, float, bool], float] = {}

    def close(self) -> None:
        self.document.close()

    def height(self, text: str, width: float, font: Path, *, bold: bool) -> float:
        if not text:
            return 0.0
        usable_width = max(1.0, width - 2 * _CELL_PADDING_X)
        key = (text, round(usable_width, 2), str(font), _TABLE_FONT_SIZE, bold)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        low = _TABLE_FONT_SIZE * _TABLE_LINE_HEIGHT
        high = 1200.0
        for _ in range(14):
            middle = (low + high) / 2
            shape = self.page.new_shape()
            spare = shape.insert_textbox(
                fitz.Rect(0, 0, usable_width, middle),
                text,
                **_textbox_font_args(text, font, bold=bold),
                fontsize=_TABLE_FONT_SIZE,
                lineheight=_TABLE_LINE_HEIGHT,
                align=fitz.TEXT_ALIGN_CENTER if bold else fitz.TEXT_ALIGN_LEFT,
            )
            if spare >= 0:
                high = middle
            else:
                low = middle

        # ``insert_textbox`` can report a non-negative theoretical spare while
        # the embedded font's actual glyph metrics consume more vertical space
        # (notably Source Han on /Rotate pages). Render once into a tall scratch
        # box and include the real text-trace boundary in the required height.
        rendered = fitz.open()
        try:
            rendered_page = rendered.new_page(
                width=usable_width,
                height=1200.0,
            )
            rendered_spare = rendered_page.insert_textbox(
                rendered_page.rect,
                text,
                **_textbox_font_args(text, font, bold=bold),
                fontsize=_TABLE_FONT_SIZE,
                lineheight=_TABLE_LINE_HEIGHT,
                align=fitz.TEXT_ALIGN_CENTER if bold else fitz.TEXT_ALIGN_LEFT,
            )
            traces = rendered_page.get_texttrace()
            if rendered_spare < 0 or not traces:
                raise OcrTableError("无法可靠测量 9 pt 表格文字的真实边界")
            rendered_bottom = max(float((trace.get("bbox") or (0, 0, 0, 0))[3]) for trace in traces)
        finally:
            rendered.close()
        result = max(high, rendered_bottom + _GLYPH_BOUNDARY_SAFETY) + 2 * _CELL_PADDING_Y
        self.cache[key] = result
        return result


def _intrinsic_column_widths(table: OcrTable, regular_font: Path, bold_font: Path) -> list[float]:
    widths = [_MIN_COLUMN_WIDTH] * table.column_count
    flow_lengths = [0] * table.column_count
    translated_load = [0.0] * table.column_count
    value_cells = [0] * table.column_count
    populated_cells = [0] * table.column_count
    for cell in table.cells:
        if not cell.target_text:
            continue
        font = _measurement_font(
            cell.target_text,
            bold_font if cell.is_header else regular_font,
        )
        tokens = re.split(r"[\s\n]+", cell.target_text)
        longest = max(
            (float(font.text_length(token, fontsize=_TABLE_FONT_SIZE)) for token in tokens if token),
            default=0.0,
        )
        total_text_width = sum(
            float(font.text_length(line, fontsize=_TABLE_FONT_SIZE))
            for line in cell.target_text.splitlines() or [cell.target_text]
        )
        # A longest-token-only heuristic makes narrative columns far too
        # narrow beside many compact numeric columns.  The square-root term
        # gives prose a larger share of page width without letting one long
        # cell starve the rest of the grid.
        desired = min(
            190.0 * cell.column_span,
            max(
                _MIN_COLUMN_WIDTH * cell.column_span,
                longest + 2 * _CELL_PADDING_X + 3.0,
                math.sqrt(max(1.0, total_text_width) * 72.0) + 2 * _CELL_PADDING_X,
            ),
        )
        per_column = desired / cell.column_span
        for column in range(cell.column, cell.column + cell.column_span):
            widths[column] = max(widths[column], per_column)
            populated_cells[column] += 1
            if cell.translated:
                translated_load[column] += len(cell.target_text) / cell.column_span
            elif _is_locked_value(cell.target_text):
                value_cells[column] += 1
            if cell.column_span == 1:
                flow_lengths[column] = max(flow_lengths[column], len(cell.target_text))
    for column, length in enumerate(flow_lengths):
        if length >= 80:
            widths[column] *= min(3.0, 1.0 + length / 80.0)
        elif translated_load[column] >= 80:
            widths[column] *= min(2.5, 1.0 + translated_load[column] / 160.0)
        if populated_cells[column] >= 3 and value_cells[column] / populated_cells[column] >= 0.60:
            widths[column] *= 0.62
    return widths


def _minimum_column_widths(
    table: OcrTable,
    regular_font: Path,
    bold_font: Path,
) -> list[float]:
    minimums = [_MIN_COLUMN_WIDTH] * table.column_count
    spanning: list[tuple[OcrTableCell, float]] = []
    for cell in table.cells:
        if not cell.target_text:
            continue
        font = _measurement_font(
            cell.target_text,
            bold_font if cell.is_header else regular_font,
        )
        longest = max(
            (
                float(font.text_length(token, fontsize=_TABLE_FONT_SIZE))
                for token in re.split(r"[\s\n]+", cell.target_text)
                if token
            ),
            default=0.0,
        )
        required = longest + 2 * _CELL_PADDING_X + 3.0
        if cell.column_span == 1:
            minimums[cell.column] = max(minimums[cell.column], required)
        else:
            spanning.append((cell, required))
    for cell, required in spanning:
        indexes = range(cell.column, cell.column + cell.column_span)
        current = sum(minimums[index] for index in indexes)
        if current + 0.01 >= required:
            continue
        addition = (required - current) / cell.column_span
        for index in indexes:
            minimums[index] += addition
    return minimums


def _allocate_column_widths(
    intrinsic: list[float],
    total_width: float,
    minimums: list[float] | None = None,
) -> tuple[float, ...]:
    minimums = minimums or [_MIN_COLUMN_WIDTH] * len(intrinsic)
    minimum_total = sum(minimums)
    if total_width < minimum_total - 0.5:
        raise OcrTableError("页面可用宽度不足以按 9 pt 重绘表格")
    remaining = total_width - minimum_total
    demands = [max(1.0, desired - minimum) for desired, minimum in zip(intrinsic, minimums, strict=True)]
    demand_total = sum(demands)
    widths = [
        minimum + remaining * demand / demand_total for minimum, demand in zip(minimums, demands, strict=True)
    ]
    correction = total_width - sum(widths)
    widths[-1] += correction
    return tuple(widths)


def _required_row_heights(
    table: OcrTable,
    column_widths: tuple[float, ...],
    regular_font: Path,
    bold_font: Path,
    measurer: _TextMeasurer,
) -> tuple[float, ...]:
    minimum = _TABLE_FONT_SIZE * _TABLE_LINE_HEIGHT + 2 * _CELL_PADDING_Y
    heights = [minimum] * table.row_count
    spanning: list[tuple[OcrTableCell, float]] = []
    for cell in table.cells:
        width = sum(column_widths[cell.column : cell.column + cell.column_span])
        needed = measurer.height(
            cell.target_text,
            width,
            bold_font if cell.is_header else regular_font,
            bold=cell.is_header,
        )
        if cell.row_span == 1:
            heights[cell.row] = max(heights[cell.row], needed)
        else:
            spanning.append((cell, needed))
    for cell, needed in spanning:
        indexes = range(cell.row, cell.row + cell.row_span)
        current = sum(heights[index] for index in indexes)
        if current + 0.01 >= needed:
            continue
        addition = (needed - current) / cell.row_span
        for index in indexes:
            heights[index] += addition
    return tuple(heights)


def _measure_table(
    table: OcrTable,
    *,
    segment: fitz.Rect,
    ocr_result: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
    measurer: _TextMeasurer,
) -> _MeasuredTable:
    source_rect = _mapped_rect(segment, table.bbox, table.page_size)
    maximum_width = max(1.0, segment.width - 2 * _PAGE_MARGIN)
    intrinsic = _intrinsic_column_widths(table, regular_font, bold_font)
    minimums = _minimum_column_widths(table, regular_font, bold_font)
    desired_width = min(
        maximum_width,
        max(source_rect.width, sum(intrinsic), table.column_count * _MIN_COLUMN_WIDTH),
    )
    column_widths = _allocate_column_widths(intrinsic, desired_width, minimums)
    row_heights = _required_row_heights(
        table,
        column_widths,
        regular_font,
        bold_font,
        measurer,
    )
    safe_top, safe_bottom = _safe_vertical_bounds(table, source_rect, segment, ocr_result)
    left = min(
        max(segment.x0 + _PAGE_MARGIN, source_rect.x0),
        segment.x1 - _PAGE_MARGIN - desired_width,
    )
    left = max(segment.x0 + _PAGE_MARGIN, left)
    return _MeasuredTable(
        source_rect=source_rect,
        safe_top=safe_top,
        safe_bottom=safe_bottom,
        left=left,
        column_widths=column_widths,
        row_heights=row_heights,
    )


def _layout_table(
    table: OcrTable,
    *,
    segment: fitz.Rect,
    ocr_result: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
    measurer: _TextMeasurer,
) -> _DrawLayout:
    measured = _measure_table(
        table,
        segment=segment,
        ocr_result=ocr_result,
        regular_font=regular_font,
        bold_font=bold_font,
        measurer=measurer,
    )
    required_height = sum(measured.row_heights)
    safe_height = measured.safe_bottom - measured.safe_top
    if required_height > safe_height + 0.5:
        raise OcrTableError(
            f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表在 9 pt / 1.25 倍行距下无法放入页面安全区域"
        )
    # Preserve the source table height when possible.  A source scan may touch
    # the physical page margin; in that case the natural 9 pt table is safely
    # recomposed inside the content area rather than rejected for its old box.
    desired_height = max(
        required_height,
        min(measured.source_rect.height, safe_height),
    )
    top = measured.source_rect.y0
    bottom = top + desired_height
    if bottom > measured.safe_bottom:
        top -= bottom - measured.safe_bottom
        bottom = measured.safe_bottom
    if top < measured.safe_top:
        top = measured.safe_top
        bottom = top + desired_height
    if bottom > measured.safe_bottom + 0.5:
        raise OcrTableError(f"第 {table.page_idx + 1} 页表格垂直布局溢出")
    table_rect = fitz.Rect(
        measured.left,
        top,
        measured.left + sum(measured.column_widths),
        bottom,
    )

    extra_height = table_rect.height - required_height
    expanded_rows = list(measured.row_heights)
    if extra_height > 0.01:
        addition = extra_height / len(expanded_rows)
        expanded_rows = [value + addition for value in expanded_rows]
    return _DrawLayout(table_rect, measured.column_widths, tuple(expanded_rows))


def _header_row_count(table: OcrTable) -> int:
    first_row = [cell for cell in table.cells if cell.row == 0]
    if not first_row:
        return min(1, table.row_count)
    header_end = max(cell.row + cell.row_span for cell in first_row)
    while header_end < table.row_count:
        next_row = [cell for cell in table.cells if cell.row == header_end]
        if not next_row or not all(cell.is_header for cell in next_row):
            break
        header_end = max(cell.row + cell.row_span for cell in next_row)
    if header_end > table.row_count:
        raise OcrTableError(f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表的表头跨越表格边界")
    return max(1, header_end)


def _validate_fragment_coverage(
    table: OcrTable,
    *,
    header_rows: int,
    fragments: tuple[_TableFragmentSpec, ...],
) -> None:
    coverage = [0] * table.row_count
    for fragment in fragments:
        if not header_rows <= fragment.data_start <= fragment.data_end <= table.row_count:
            raise OcrTableError("表格分页行范围无效")
        for row in range(fragment.data_start, fragment.data_end):
            coverage[row] += 1
    if any(coverage[row] != 1 for row in range(header_rows, table.row_count)):
        raise OcrTableError(f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表分页后存在漏行或重复行")


def _atomic_data_row_groups(
    table: OcrTable,
    header_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Return indivisible row ranges so no rowspan is cut by pagination."""
    groups: list[tuple[int, int]] = []
    row = header_rows
    while row < table.row_count:
        end = row + 1
        while True:
            spanning_end = max(
                (
                    cell.row + cell.row_span
                    for cell in table.cells
                    if cell.row < end and cell.row + cell.row_span > row
                ),
                default=end,
            )
            spanning_end = min(table.row_count, spanning_end)
            if spanning_end <= end:
                break
            end = spanning_end
        groups.append((row, end))
        row = end
    return tuple(groups)


def _split_table_fragments(
    table: OcrTable,
    row_heights: tuple[float, ...],
    *,
    first_capacity: float,
    continuation_capacity: float,
) -> tuple[int, tuple[_TableFragmentSpec, ...]]:
    header_rows = _header_row_count(table)
    header_height = sum(row_heights[:header_rows])
    tolerance = 0.5
    if header_height > continuation_capacity + tolerance:
        raise OcrTableError(
            f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表的表头在 9 pt / 1.25 倍行距下无法放入续页"
        )
    groups = _atomic_data_row_groups(table, header_rows)
    for start, end in groups:
        if header_height + sum(row_heights[start:end]) > continuation_capacity + tolerance:
            raise OcrTableError(
                f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表第 "
                f"{start + 1}–{end} 行与表头在 9 pt / 1.25 倍行距下无法共同放入一页"
            )

    fragments: list[_TableFragmentSpec] = []
    group_idx = 0
    if not groups:
        if header_height <= first_capacity + tolerance:
            return header_rows, (_TableFragmentSpec(header_rows, header_rows, True),)
        return header_rows, (_TableFragmentSpec(header_rows, header_rows, False),)

    first_start = groups[0][0]
    first_end = first_start
    used = header_height
    while group_idx < len(groups):
        start, end = groups[group_idx]
        group_height = sum(row_heights[start:end])
        if used + group_height > first_capacity + tolerance:
            break
        first_end = end
        used += group_height
        group_idx += 1
    if first_end > first_start:
        fragments.append(_TableFragmentSpec(first_start, first_end, True))

    while group_idx < len(groups):
        start = groups[group_idx][0]
        end = start
        used = header_height
        while group_idx < len(groups):
            group_start, group_end = groups[group_idx]
            group_height = sum(row_heights[group_start:group_end])
            if used + group_height > continuation_capacity + tolerance:
                break
            end = group_end
            used += group_height
            group_idx += 1
        if end <= start:
            raise OcrTableError(f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表无法按完整行分页")
        fragments.append(_TableFragmentSpec(start, end, False))
    return header_rows, tuple(fragments)


def _table_header_text(table: OcrTable, header_rows: int) -> str:
    return "\n".join(
        cell.target_text
        for cell in sorted(table.cells, key=lambda item: (item.row, item.column))
        if cell.row < header_rows and cell.target_text
    )


def _build_pagination_plan(
    pdf_path: Path,
    *,
    plan: OcrTableTranslationPlan,
    ocr_result: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
) -> _PaginationPlan:
    document = fitz.open(str(pdf_path))
    measurer = _TextMeasurer()
    continuation_counts = [0] * document.page_count
    table_plans: list[_TablePagination] = []
    repeated_header_texts: list[str] = []
    try:
        for table in plan.tables:
            if table.preserve_as_image:
                continue
            if not 0 <= table.page_idx < document.page_count:
                raise OcrTableError(f"表格引用了不存在的第 {table.page_idx + 1} 页")
            segment = fitz.Rect(document[table.page_idx].rect)
            measured = _measure_table(
                table,
                segment=segment,
                ocr_result=ocr_result,
                regular_font=regular_font,
                bold_font=bold_font,
                measurer=measurer,
            )
            first_capacity = measured.safe_bottom - measured.safe_top
            required_height = sum(measured.row_heights)
            if required_height <= first_capacity + 0.5:
                header_rows = _header_row_count(table)
                fragments = (_TableFragmentSpec(header_rows, table.row_count, True),)
                _validate_fragment_coverage(
                    table,
                    header_rows=header_rows,
                    fragments=fragments,
                )
                table_plans.append(_TablePagination(table, header_rows, fragments, False))
                continue

            continuation_capacity = max(1.0, segment.height - 2 * _PAGE_MARGIN)
            header_rows, fragments = _split_table_fragments(
                table,
                measured.row_heights,
                first_capacity=first_capacity,
                continuation_capacity=continuation_capacity,
            )
            assigned: list[_TableFragmentSpec] = []
            header_text = _table_header_text(table, header_rows)
            continuation_fragment_count = 0
            for fragment in fragments:
                if fragment.on_source_page:
                    assigned.append(fragment)
                    continue
                slot = continuation_counts[table.page_idx]
                continuation_counts[table.page_idx] += 1
                continuation_fragment_count += 1
                assigned.append(replace(fragment, continuation_slot=slot))
            repeated_copies = continuation_fragment_count
            if not any(fragment.on_source_page for fragment in assigned):
                repeated_copies = max(0, repeated_copies - 1)
            repeated_header_texts.extend(header_text for _ in range(repeated_copies))
            _validate_fragment_coverage(
                table,
                header_rows=header_rows,
                fragments=tuple(assigned),
            )
            table_plans.append(_TablePagination(table, header_rows, tuple(assigned), True))
    finally:
        measurer.close()
        document.close()
    return _PaginationPlan(
        tables=tuple(table_plans),
        continuation_counts=tuple(continuation_counts),
        repeated_header_texts=tuple(repeated_header_texts),
    )


def _fragment_table(
    table: OcrTable,
    *,
    header_rows: int,
    data_start: int,
    data_end: int,
) -> OcrTable:
    header_cells = [cell for cell in table.cells if cell.row < header_rows]
    data_cells = [
        replace(cell, row=header_rows + cell.row - data_start)
        for cell in table.cells
        if data_start <= cell.row and cell.row + cell.row_span <= data_end
    ]
    fragment = replace(
        table,
        row_count=header_rows + data_end - data_start,
        cells=tuple([*header_cells, *data_cells]),
    )
    if any(
        cell.column < 0
        or cell.column + cell.column_span > table.column_count
        or cell.row < 0
        or cell.row + cell.row_span > fragment.row_count
        for cell in fragment.cells
    ):
        raise OcrTableError(f"第 {table.page_idx + 1} 页第 {table.index + 1} 张表分页后列映射无效")
    return fragment


def _fragment_layout(
    measured: _MeasuredTable,
    *,
    header_rows: int,
    data_start: int,
    data_end: int,
    segment: fitz.Rect,
    on_source_page: bool,
) -> _DrawLayout:
    row_heights = (
        *measured.row_heights[:header_rows],
        *measured.row_heights[data_start:data_end],
    )
    height = sum(row_heights)
    if on_source_page:
        safe_top, safe_bottom = measured.safe_top, measured.safe_bottom
        top = min(
            max(measured.source_rect.y0, safe_top),
            safe_bottom - height,
        )
    else:
        safe_top = segment.y0 + _PAGE_MARGIN
        safe_bottom = segment.y1 - _PAGE_MARGIN
        top = safe_top
    if top < safe_top - 0.5 or top + height > safe_bottom + 0.5:
        raise OcrTableError("分页表格片段超出页面安全区域")
    table_rect = fitz.Rect(
        measured.left,
        top,
        measured.left + sum(measured.column_widths),
        top + height,
    )
    return _DrawLayout(table_rect, measured.column_widths, tuple(row_heights))


def _cell_rect(
    layout: _DrawLayout,
    cell: OcrTableCell,
) -> fitz.Rect:
    x0 = layout.table_rect.x0 + sum(layout.column_widths[: cell.column])
    x1 = x0 + sum(layout.column_widths[cell.column : cell.column + cell.column_span])
    y0 = layout.table_rect.y0 + sum(layout.row_heights[: cell.row])
    y1 = y0 + sum(layout.row_heights[cell.row : cell.row + cell.row_span])
    return fitz.Rect(x0, y0, x1, y1)


def _draw_cell_text(
    page: fitz.Page,
    cell: OcrTableCell,
    rect: fitz.Rect,
    *,
    regular_font: Path,
    bold_font: Path,
    measurer: _TextMeasurer,
) -> None:
    if not cell.target_text:
        return
    font = bold_font if cell.is_header else regular_font
    value_only = _is_locked_value(cell.target_text)
    align = (
        fitz.TEXT_ALIGN_CENTER
        if cell.is_header or value_only or len(cell.target_text) <= 18
        else fitz.TEXT_ALIGN_LEFT
    )
    text_height = (
        measurer.height(
            cell.target_text,
            rect.width,
            font,
            bold=cell.is_header,
        )
        - 2 * _CELL_PADDING_Y
    )
    inner = fitz.Rect(
        rect.x0 + _CELL_PADDING_X,
        max(rect.y0 + _CELL_PADDING_Y, (rect.y0 + rect.y1 - text_height) / 2),
        rect.x1 - _CELL_PADDING_X,
        rect.y1 - _CELL_PADDING_Y,
    )
    existing_traces = page.get_texttrace()
    previous_sequence = max(
        (int(trace.get("seqno", -1)) for trace in existing_traces),
        default=-1,
    )
    spare = insert_visual_textbox(
        page,
        inner,
        cell.target_text,
        **_textbox_font_args(cell.target_text, font, bold=cell.is_header),
        fontsize=_TABLE_FONT_SIZE,
        lineheight=_TABLE_LINE_HEIGHT,
        align=align,
        color=(0, 0, 0),
        overlay=True,
    )
    if spare < -0.05:
        raise OcrTableError(f"表格第 {cell.row + 1} 行第 {cell.column + 1} 列在 9 pt 下溢出")
    inserted_traces = [
        trace for trace in page.get_texttrace() if int(trace.get("seqno", -1)) > previous_sequence
    ]
    if not inserted_traces:
        raise OcrTableError(f"表格第 {cell.row + 1} 行第 {cell.column + 1} 列未生成可验证文字")
    visible_bounds = [
        pdf_rect_to_visual(
            page,
            fitz.Rect(trace.get("bbox") or (0, 0, 0, 0)),
            clip=False,
        )
        for trace in inserted_traces
    ]
    tolerance = _CELL_BOUNDARY_TOLERANCE
    if any(
        bound.x0 < rect.x0 - tolerance
        or bound.y0 < rect.y0 - tolerance
        or bound.x1 > rect.x1 + tolerance
        or bound.y1 > rect.y1 + tolerance
        for bound in visible_bounds
    ):
        raise OcrTableError(f"表格第 {cell.row + 1} 行第 {cell.column + 1} 列文字超过单元格边界")


def _draw_table(
    page: fitz.Page,
    table: OcrTable,
    layout: _DrawLayout,
    *,
    regular_font: Path,
    bold_font: Path,
    measurer: _TextMeasurer,
) -> None:
    for cell in table.cells:
        rect = _cell_rect(layout, cell)
        fill = (0.94, 0.95, 0.96) if cell.is_header else (1, 1, 1)
        draw_visual_rect(
            page,
            rect,
            color=(0.18, 0.18, 0.18),
            fill=fill,
            width=_BORDER_WIDTH,
            overlay=True,
        )
    draw_visual_rect(
        page,
        layout.table_rect,
        color=(0.08, 0.08, 0.08),
        fill=None,
        width=_OUTER_BORDER_WIDTH,
        overlay=True,
    )
    for cell in table.cells:
        _draw_cell_text(
            page,
            cell,
            _cell_rect(layout, cell),
            regular_font=regular_font,
            bold_font=bold_font,
            measurer=measurer,
        )


def _save_replacement(document: fitz.Document, destination: Path) -> None:
    mode = destination.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-tables-",
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


def _target_segment(
    page: fitz.Page,
    bilingual_layout: OcrBilingualLayout | None,
) -> fitz.Rect:
    if bilingual_layout is not None and bilingual_layout.mode == "side_by_side":
        midpoint = page.rect.x0 + page.rect.width / 2
        return fitz.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1)
    return fitz.Rect(page.rect)


def _insert_continuation_pages(
    document: fitz.Document,
    *,
    continuation_counts: tuple[int, ...],
    bilingual_layout: OcrBilingualLayout | None,
) -> _PageMap:
    logical_count = len(continuation_counts)
    mode = bilingual_layout.mode if bilingual_layout is not None else "mono"
    expected_pages = logical_count * 2 if mode == "interleaved" else logical_count
    if document.page_count != expected_pages:
        raise OcrTableError(f"分页前 PDF 页数 {document.page_count} 与逻辑页数 {logical_count} 不一致")

    target_offsets: list[int] = []
    for logical_idx in range(logical_count):
        if mode != "interleaved":
            target_offsets.append(0)
            continue
        assert bilingual_layout is not None
        target_idx = bilingual_layout.target_page_index(logical_idx)
        offset = None if target_idx is None else target_idx - logical_idx * 2
        if offset not in {0, 1}:
            raise OcrTableError(f"第 {logical_idx + 1} 页没有可靠的双语译文页位置")
        target_offsets.append(offset)

    # Insert from the back so original physical page indexes remain stable
    # while page sizes are copied.  Interleaved bilingual output receives a
    # blank source/target pair for every translated continuation page.
    for logical_idx in range(logical_count - 1, -1, -1):
        count = continuation_counts[logical_idx]
        if count <= 0:
            continue
        if mode == "interleaved":
            reference = fitz.Rect(document[logical_idx * 2].rect)
            insert_at = logical_idx * 2 + 2
            for _ in range(count):
                document.new_page(
                    pno=insert_at,
                    width=reference.width,
                    height=reference.height,
                )
                document.new_page(
                    pno=insert_at + 1,
                    width=reference.width,
                    height=reference.height,
                )
                insert_at += 2
        else:
            reference = fitz.Rect(document[logical_idx].rect)
            insert_at = logical_idx + 1
            for _ in range(count):
                document.new_page(
                    pno=insert_at,
                    width=reference.width,
                    height=reference.height,
                )
                insert_at += 1

    source_pages: list[int] = []
    continuation_pages: list[tuple[int, ...]] = []
    preceding = 0
    for logical_idx, count in enumerate(continuation_counts):
        if mode == "interleaved":
            pair_start = logical_idx * 2 + preceding * 2
            offset = target_offsets[logical_idx]
            source_pages.append(pair_start + offset)
            continuation_pages.append(tuple(pair_start + 2 + slot * 2 + offset for slot in range(count)))
        else:
            source_page = logical_idx + preceding
            source_pages.append(source_page)
            continuation_pages.append(tuple(source_page + 1 + slot for slot in range(count)))
        preceding += count
    return _PageMap(tuple(source_pages), tuple(continuation_pages))


def _redraw_document(
    pdf_path: Path,
    *,
    pagination: _PaginationPlan,
    ocr_result: dict[str, Any],
    regular_font: Path,
    bold_font: Path,
    bilingual_layout: OcrBilingualLayout | None = None,
) -> _DrawStats:
    document = fitz.open(str(pdf_path))
    measurer = _TextMeasurer()
    try:
        page_map = _insert_continuation_pages(
            document,
            continuation_counts=pagination.continuation_counts,
            bilingual_layout=bilingual_layout,
        )
        prepared: list[_PreparedFragment] = []
        for table_plan in pagination.tables:
            table = table_plan.table
            source_page_idx = page_map.source_target_pages[table.page_idx]
            source_page = document[source_page_idx]
            source_segment = _target_segment(source_page, bilingual_layout)
            measured = _measure_table(
                table,
                segment=source_segment,
                ocr_result=ocr_result,
                regular_font=regular_font,
                bold_font=bold_font,
                measurer=measurer,
            )
            for fragment_spec in table_plan.fragments:
                if fragment_spec.on_source_page:
                    page_idx = source_page_idx
                    segment = source_segment
                else:
                    if fragment_spec.continuation_slot is None:
                        raise OcrTableError("分页表格缺少续页位置")
                    page_idx = page_map.continuation_target_pages[table.page_idx][
                        fragment_spec.continuation_slot
                    ]
                    segment = _target_segment(document[page_idx], bilingual_layout)
                fragment = _fragment_table(
                    table,
                    header_rows=table_plan.header_rows,
                    data_start=fragment_spec.data_start,
                    data_end=fragment_spec.data_end,
                )
                if table_plan.paginated:
                    layout = _fragment_layout(
                        measured,
                        header_rows=table_plan.header_rows,
                        data_start=fragment_spec.data_start,
                        data_end=fragment_spec.data_end,
                        segment=segment,
                        on_source_page=fragment_spec.on_source_page,
                    )
                else:
                    layout = _layout_table(
                        table,
                        segment=segment,
                        ocr_result=ocr_result,
                        regular_font=regular_font,
                        bold_font=bold_font,
                        measurer=measurer,
                    )
                prepared.append(_PreparedFragment(page_idx, fragment, layout))

        # Remove each complete original table once.  Continuation pages are
        # blank and therefore require no redaction.
        pages_with_redactions: set[int] = set()
        overlay_regions: list[OcrTableOverlayRegion] = []
        for table_plan in pagination.tables:
            table = table_plan.table
            page_idx = page_map.source_target_pages[table.page_idx]
            page = document[page_idx]
            segment = _target_segment(page, bilingual_layout)
            source_rect = _mapped_rect(segment, table.bbox, table.page_size)
            redaction_rect = fitz.Rect(source_rect)
            for fragment in prepared:
                if fragment.page_idx == page_idx and fragment.table.index == table.index:
                    redaction_rect.include_rect(fragment.layout.table_rect)
            add_visual_redaction(
                page,
                redaction_rect,
                fill=(1, 1, 1),
                cross_out=False,
            )
            pages_with_redactions.add(page_idx)
            overlay_regions.append(
                OcrTableOverlayRegion(
                    page_idx=page_idx,
                    bbox=tuple(float(value) for value in redaction_rect),
                    continuation=False,
                )
            )
        for page_idx in pages_with_redactions:
            document[page_idx].apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

        for fragment in prepared:
            _draw_table(
                document[fragment.page_idx],
                fragment.table,
                fragment.layout,
                regular_font=regular_font,
                bold_font=bold_font,
                measurer=measurer,
            )
            if fragment.page_idx in page_map.inserted_target_pages:
                overlay_regions.append(
                    OcrTableOverlayRegion(
                        page_idx=fragment.page_idx,
                        bbox=tuple(float(value) for value in fragment.layout.table_rect),
                        continuation=True,
                    )
                )
        if pagination.tables:
            _save_replacement(document, pdf_path)
        return _DrawStats(
            tables=len(pagination.tables),
            cells=sum(len(item.table.cells) for item in pagination.tables),
            inserted_pages=sum(pagination.continuation_counts),
            source_page_indices=page_map.source_target_pages,
            continuation_page_indices=page_map.inserted_target_pages,
            continuation_page_groups=page_map.continuation_target_pages,
            overlay_regions=tuple(overlay_regions),
        )
    finally:
        measurer.close()
        document.close()


def redraw_ocr_tables(
    *,
    ocr_result: dict[str, Any],
    plan: OcrTableTranslationPlan,
    translated_pdf: str | Path,
    bilingual_pdf: str | Path | None = None,
    body_font_path: str | Path,
    bold_font_path: str | Path | None = None,
) -> OcrTableRedrawResult:
    """Replace complete source tables with fitted 9 pt vector tables."""
    translated_path = Path(translated_pdf)
    bilingual_path = Path(bilingual_pdf) if bilingual_pdf else None
    regular_font = Path(body_font_path)
    bold_font = Path(bold_font_path) if bold_font_path else _resolve_bold_font(regular_font)
    for path, label in (
        (translated_path, "translated PDF"),
        (regular_font, "body font"),
        (bold_font, "bold font"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not plan.tables:
        with fitz.open(str(translated_path)) as document:
            source_pages = tuple(range(document.page_count))
        return OcrTableRedrawResult(
            translated_path,
            bilingual_path,
            _TABLE_FONT_SIZE,
            _TABLE_LINE_HEIGHT,
            0,
            0,
            0,
            0,
            0,
            source_pages,
            (),
            (),
            (),
            (),
        )

    bilingual_layout: OcrBilingualLayout | None = None
    if bilingual_path is not None and bilingual_path.is_file():
        bilingual_layout = detect_ocr_bilingual_layout(translated_path, bilingual_path)
        if bilingual_layout is None:
            raise OcrTableError("无法识别双语 PDF 的原文/译文页面布局")

    pagination = _build_pagination_plan(
        translated_path,
        plan=plan,
        ocr_result=ocr_result,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    mono = _redraw_document(
        translated_path,
        pagination=pagination,
        ocr_result=ocr_result,
        regular_font=regular_font,
        bold_font=bold_font,
    )
    bilingual = _DrawStats(0, 0)
    if bilingual_path is not None and bilingual_path.is_file() and bilingual_layout:
        bilingual = _redraw_document(
            bilingual_path,
            pagination=pagination,
            ocr_result=ocr_result,
            regular_font=regular_font,
            bold_font=bold_font,
            bilingual_layout=bilingual_layout,
        )

    return OcrTableRedrawResult(
        translated_pdf_path=translated_path,
        bilingual_pdf_path=bilingual_path,
        table_font_size=_TABLE_FONT_SIZE,
        table_line_height=_TABLE_LINE_HEIGHT,
        redrawn_tables=mono.tables,
        redrawn_cells=mono.cells,
        bilingual_redrawn_tables=bilingual.tables,
        bilingual_redrawn_cells=bilingual.cells,
        continuation_pages=mono.inserted_pages,
        source_page_indices=mono.source_page_indices,
        continuation_page_indices=mono.continuation_page_indices,
        continuation_page_groups=mono.continuation_page_groups,
        translated_overlay_regions=mono.overlay_regions,
        repeated_header_texts=pagination.repeated_header_texts,
    )


__all__ = [
    "OcrTable",
    "OcrTableCell",
    "OcrTableError",
    "OcrTableOverlayRegion",
    "OcrTableRedrawResult",
    "OcrTableTranslationPlan",
    "extract_ocr_tables",
    "redraw_ocr_tables",
    "should_preserve_table_as_image",
    "translate_ocr_tables",
]
