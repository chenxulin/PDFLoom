"""Translate PP-StructureV3 tables and redraw complete searchable vector grids."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, replace
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz

from .config import Settings
from .coordinates import (
    add_visual_redaction,
    draw_visual_rect,
    insert_visual_textbox,
    map_ocr_rect,
    visual_page_rect,
)
from .llm import Translator
from .models import TableRedrawStats
from .terminology import (
    TerminologyRequirement,
    exact_preferred_target,
    missing_requirements,
    requirement_instruction,
    requirements_for,
)
from .translation_context import context_excerpt


class TableRedrawError(RuntimeError):
    pass


@dataclass(frozen=True)
class TableCell:
    row: int
    column: int
    row_span: int
    column_span: int
    source_text: str
    target_text: str
    is_header: bool
    translated: bool = False
    protected_literals: tuple[str, ...] = ()


@dataclass(frozen=True)
class OcrTable:
    index: int
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    row_count: int
    column_count: int
    cells: tuple[TableCell, ...]
    preserve_as_image: bool = False


@dataclass(frozen=True)
class TableTranslationPlan:
    tables: tuple[OcrTable, ...]
    translated_cells: int
    protected_literals: int


@dataclass(frozen=True)
class _RawCell:
    text: str
    row_span: int = 1
    column_span: int = 1
    header: bool = False


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_RawCell]] = []
        self._row: list[_RawCell] | None = None
        self._parts: list[str] | None = None
        self._attrs: dict[str, str] = {}
        self._header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered == "tr":
            self._row = []
        elif lowered in {"td", "th"} and self._row is not None:
            self._parts = []
            self._attrs = {key.casefold(): str(value or "") for key, value in attrs}
            self._header = lowered == "th"
        elif lowered == "br" and self._parts is not None:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"td", "th"} and self._row is not None and self._parts is not None:
            self._row.append(
                _RawCell(
                    text=_clean("".join(self._parts)),
                    row_span=_span(self._attrs.get("rowspan")),
                    column_span=_span(self._attrs.get("colspan")),
                    header=self._header,
                )
            )
            self._parts = None
            self._attrs = {}
            self._header = False
        elif lowered == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


_CJK = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_HTML_TAG = re.compile(r"<[^>]+>")
_SIGNATURE_TABLE = re.compile(
    r"(?:签名|签字|手签).{0,16}(?:日期|date)|signature.{0,16}(?:date|日期)",
    re.IGNORECASE | re.DOTALL,
)
_LOCKED_ABBREVIATIONS = {
    "API",
    "CAPA",
    "GMP",
    "HPLC",
    "ICH",
    "N/A",
    "NA",
    "ND",
    "OOS",
    "OOT",
    "QA",
    "QC",
    "SOP",
    "USP",
}
_FULL_DATE = re.compile(r"^\s*\d{2,4}[./-]\d{1,2}[./-]\d{1,2}\s*$")
_FULL_NUMBER = re.compile(r"^\s*[<>≤≥±]?\s*[\d.,]+(?:\s*(?:%|‰|℃|°C|[A-Za-zμµ]+(?:/[A-Za-zμµ]+)?))?\s*$")
_FULL_CODE = re.compile(r"^\s*(?=.*(?:\d|[-/.()]))[A-Za-z][A-Za-z0-9_.()/-]*\s*$")
_FULL_FORMULA = re.compile(r"^\s*(?:[A-Z][a-z]?\d*){2,}\s*$")
_PROTECTED = re.compile(
    r"\d{2,4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|[<>≤≥±]?\d+(?:\.\d+)?\s*(?:%|‰|℃|°C|mol/L|mmol/L|μmol/L|mg/mL|μg/mL|µg/mL|ug/mL|ng/mL|mg/L|μg/L|µg/L|ug/L|ng/L|ppm|ppb|mg|kg|μg|µg|ug|ng|mL|μL|µL|uL|L|g|h|min|s)(?:\s*/\s*(?:kg|mL|L|μL|µL|uL|h|min|s))?"
    r"|(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9]*\d[A-Za-z0-9_.()/-]*|[A-Z]{2,}(?:[-/][A-Z0-9.()]+)+)(?![A-Za-z0-9])"
    r"|(?<![A-Za-z])(?:API|CAPA|GMP|HPLC|ICH|N/A|ND|OOT|OOS|QA|QC|SOP|USP)(?![A-Za-z])"
    r"|(?<![A-Za-z])(?:[A-Z][a-z]?\d*){2,}(?![A-Za-z])"
    r"|[<>≤≥±]?\d+(?:\.\d+)?"
)


def _clean(value: Any) -> str:
    lines = [" ".join(line.split()) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _span(value: Any) -> int:
    try:
        return min(100, max(1, int(value)))
    except (TypeError, ValueError):
        return 1


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        box = tuple(float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in box):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def _markdown_rows(content: str) -> list[list[_RawCell]]:
    rows: list[list[_RawCell]] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        values = [part.strip().replace("\\|", "|") for part in line[1:-1].split("|")]
        if values and all(re.fullmatch(r":?-{3,}:?", value) for value in values):
            continue
        rows.append([_RawCell(value, header=not rows) for value in values])
    return rows


def _structured_rows(content: Any) -> list[list[_RawCell]]:
    text = str(content or "").strip()
    if not text:
        return []
    if "<table" in text.casefold():
        parser = _HtmlTableParser()
        parser.feed(text)
        parser.close()
        rows = parser.rows
    else:
        rows = _markdown_rows(text)
    while len(rows) > 1 and len(rows[0]) == 1 and not rows[0][0].text:
        rows.pop(0)
    return rows


def _logical_cells(rows: list[list[_RawCell]]) -> tuple[tuple[TableCell, ...], int, int]:
    occupied: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    row_count = 0
    column_count = max((sum(cell.column_span for cell in row) for row in rows), default=0)
    for row_index, row in enumerate(rows):
        column = 0
        for raw in row:
            while (row_index, column) in occupied:
                column += 1
            for covered_row in range(row_index, row_index + raw.row_span):
                for covered_column in range(column, column + raw.column_span):
                    occupied.add((covered_row, covered_column))
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    row_span=raw.row_span,
                    column_span=raw.column_span,
                    source_text=raw.text,
                    target_text=raw.text,
                    is_header=raw.header or row_index == 0,
                )
            )
            row_count = max(row_count, row_index + raw.row_span)
            column_count = max(column_count, column + raw.column_span)
            column += raw.column_span
    return tuple(cells), row_count, column_count


def extract_tables(ocr_result: dict[str, Any]) -> tuple[OcrTable, ...]:
    tables: list[OcrTable] = []
    page_sizes: dict[int, tuple[float, float]] = {}
    for page in ocr_result.get("pages") or []:
        if isinstance(page, dict):
            try:
                page_idx = int(page.get("page_idx"))
            except (TypeError, ValueError):
                continue
            size = _pair(page.get("page_size"))
            if size:
                page_sizes[page_idx] = size
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or raw.get("sub_type") or "").strip().lower()
        if kind != "table":
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            page_idx = -1
        box = _bbox(raw.get("bbox"))
        size = _pair(raw.get("page_size")) or page_sizes.get(page_idx)
        rows = _structured_rows(raw.get("structured_content"))
        cells, row_count, column_count = _logical_cells(rows)
        if page_idx < 0 or box is None or size is None:
            raise TableRedrawError("OCR table is missing a reliable page, box, or page size")
        if not cells or row_count < 1 or column_count < 1:
            raise TableRedrawError(f"Page {page_idx + 1} table has no reconstructable row/column structure")
        plain = " ".join(unescape(_HTML_TAG.sub(" ", str(raw.get("structured_content") or ""))).split())
        tables.append(
            OcrTable(
                index=len(tables),
                page_idx=page_idx,
                bbox=box,
                page_size=size,
                row_count=row_count,
                column_count=column_count,
                cells=cells,
                preserve_as_image=bool(_SIGNATURE_TABLE.search(plain)),
            )
        )
    return tuple(tables)


def _locked(text: str) -> bool:
    value = _clean(text)
    if not value:
        return True
    return bool(
        _FULL_DATE.fullmatch(value)
        or _FULL_NUMBER.fullmatch(value)
        or _FULL_CODE.fullmatch(value)
        or _FULL_FORMULA.fullmatch(value)
        or value.replace(" ", "").upper() in _LOCKED_ABBREVIATIONS
    )


def _needs_translation(text: str, target_language: str) -> bool:
    if _locked(text):
        return False
    target = target_language.lower().replace("_", "-")
    has_cjk = bool(_CJK.search(text))
    if target.startswith("en"):
        return has_cjk
    if target.startswith(("zh", "ja", "ko")):
        return bool(re.search(r"[A-Za-z]", text)) and not has_cjk
    return has_cjk or bool(re.search(r"[A-Za-z]", text))


def _protect(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    parts: list[str] = []
    values: list[tuple[str, str]] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        if match.start() < cursor:
            continue
        parts.append(text[cursor : match.start()])
        token = f"JTBL{len(values):03d}"
        wrapper = f"[[{token}|{match.group(0)}]]"
        values.append((wrapper, match.group(0)))
        parts.append(wrapper)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), tuple(values)


def _restore(text: str, values: tuple[tuple[str, str], ...]) -> str:
    result = _clean(text)
    for wrapper, value in values:
        if result.count(wrapper) != 1:
            raise TableRedrawError(f"Translation did not preserve protected wrapper {wrapper[:11]}…")
        result = result.replace(wrapper, value)
    if re.search(r"JTBL\s*\d{3}", result, re.IGNORECASE):
        raise TableRedrawError("Translation contains an unrecovered table placeholder")
    return result


def _translation_context(
    ocr_result: dict[str, Any],
    table: OcrTable,
    cell: TableCell,
    requirements: tuple[TerminologyRequirement, ...],
    *,
    retry: bool,
) -> str:
    headers = [candidate.source_text for candidate in table.cells if candidate.is_header]
    column_headers = [
        candidate.source_text
        for candidate in table.cells
        if candidate.is_header and candidate.column <= cell.column < candidate.column + candidate.column_span
    ]
    row = [
        candidate.source_text
        for candidate in sorted(table.cells, key=lambda item: item.column)
        if candidate.row <= cell.row < candidate.row + candidate.row_span
    ]
    parts = [
        f"PDF table {table.index + 1} on page {table.page_idx + 1}; current cell row "
        f"{cell.row + 1}, column {cell.column + 1}.",
        "Translate only the current Text input as a concise one-line table cell.",
    ]
    if headers:
        parts.append("Table headers: " + " | ".join(headers))
    if column_headers:
        parts.append("Current column header: " + " | ".join(column_headers))
    if row:
        parts.append("Current source row: " + " | ".join(row))
    excerpt = context_excerpt(ocr_result.get("markdown"), cell.source_text, limit=1000)
    if excerpt:
        parts.append("Document excerpt: " + excerpt)
    instruction = requirement_instruction(requirements)
    if instruction:
        parts.append(instruction)
    if retry:
        parts.append(
            "The previous answer violated mandatory domain terminology. Retry and include every required "
            "target term exactly, without commentary."
        )
    return "\n".join(parts)


async def translate_tables(
    ocr_result: dict[str, Any],
    translator: Translator,
    settings: Settings,
    *,
    progress: Any = None,
) -> TableTranslationPlan:
    tables = extract_tables(ocr_result)
    mutable = [list(table.cells) for table in tables]
    candidates: list[tuple[int, int, str, tuple[tuple[str, str], ...]]] = []
    locked_count = 0
    for table_index, table in enumerate(tables):
        for cell_index, cell in enumerate(table.cells):
            if table.preserve_as_image or not _needs_translation(cell.source_text, settings.target_language):
                locked_count += int(bool(cell.source_text) and _locked(cell.source_text))
                continue
            protected, values = _protect(cell.source_text)
            candidates.append((table_index, cell_index, protected, values))

    semaphore = asyncio.Semaphore(min(6, settings.max_workers))
    progress_lock = asyncio.Lock()
    completed = 0

    async def translate_one(
        table_index: int,
        cell_index: int,
        protected: str,
        values: tuple[tuple[str, str], ...],
    ) -> None:
        nonlocal completed
        table = tables[table_index]
        cell = table.cells[cell_index]
        async with semaphore:
            requirements = (
                requirements_for(cell.source_text, settings.target_language)
                if settings.enforce_cmc_terminology
                else ()
            )
            missing = requirements
            restored = ""
            for attempt in range(2):
                translated = await translator.translate(
                    protected,
                    context=_translation_context(
                        ocr_result,
                        table,
                        cell,
                        requirements,
                        retry=attempt > 0,
                    ),
                    required_literals=tuple(wrapper for wrapper, _ in values),
                )
                restored = _restore(translated, values)
                preferred = exact_preferred_target(cell.source_text, settings.target_language)
                if settings.enforce_cmc_terminology and preferred is not None:
                    restored = preferred
                missing = missing_requirements(restored, requirements)
                if not missing:
                    break
            if missing:
                mappings = ", ".join(
                    f"{requirement.source_term}->{requirement.required_target}" for requirement in missing
                )
                raise TableRedrawError(
                    f"Page {table.page_idx + 1} table cell ({cell.row + 1}, "
                    f"{cell.column + 1}) violates mandatory terminology: {mappings}"
                )
            if not restored or (
                restored == cell.source_text and _needs_translation(restored, settings.target_language)
            ):
                raise TableRedrawError(f"Page {table.page_idx + 1} table cell was not translated")
            mutable[table_index][cell_index] = replace(
                cell,
                target_text=restored.replace("\n", " "),
                translated=True,
                protected_literals=tuple(value for _, value in values),
            )
        async with progress_lock:
            completed += 1
            if progress:
                progress(
                    "table-translate",
                    100.0 * completed / max(1, len(candidates)),
                    f"Translated table cells {completed}/{len(candidates)}",
                )

    await asyncio.gather(
        *(
            translate_one(table_index, cell_index, protected, values)
            for table_index, cell_index, protected, values in candidates
        )
    )
    return TableTranslationPlan(
        tables=tuple(replace(table, cells=tuple(mutable[index])) for index, table in enumerate(tables)),
        translated_cells=len(candidates),
        protected_literals=locked_count + sum(len(values) for *_, values in candidates),
    )


def _font(path: Path, fallback: str) -> fitz.Font:
    if path.is_file():
        try:
            return fitz.Font(fontfile=str(path))
        except Exception:
            pass
    return fitz.Font(fallback)


def _measure(font: fitz.Font, text: str, size: float) -> float:
    try:
        return float(font.text_length(text, fontsize=size))
    except Exception:
        # Conservative fallback for unusual TTC faces.
        return sum(size if _CJK.match(char) else size * 0.58 for char in text)


def _wrap(text: str, width: float, font: fitz.Font, size: float) -> list[str]:
    if not text:
        return [""]
    width = max(width, size)
    lines: list[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        current = ""
        # Preserve word boundaries for Latin text while allowing CJK to wrap
        # character-by-character.
        tokens = (
            list(paragraph) if _CJK.search(paragraph) else re.findall(r"\S+\s*", paragraph) or [paragraph]
        )
        for token in tokens:
            candidate = current + token
            if current and _measure(font, candidate.rstrip(), size) > width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
            while current and _measure(font, current.rstrip(), size) > width:
                cut = max(1, len(current) - 1)
                while cut > 1 and _measure(font, current[:cut], size) > width:
                    cut -= 1
                lines.append(current[:cut].rstrip())
                current = current[cut:].lstrip()
        lines.append(current.rstrip())
    return lines or [""]


def _column_widths(table: OcrTable, total_width: float) -> tuple[float, ...]:
    weights = [1.0] * table.column_count
    for cell in table.cells:
        if cell.column_span != 1:
            continue
        visual_length = sum(1.0 if _CJK.match(char) else 0.55 for char in cell.target_text)
        weights[cell.column] = max(weights[cell.column], min(5.0, math.sqrt(max(1.0, visual_length))))
    floor = min(34.0, total_width / max(1, table.column_count))
    flexible = max(0.0, total_width - floor * table.column_count)
    weight_sum = sum(weights)
    return tuple(floor + flexible * weight / weight_sum for weight in weights)


def _row_heights(
    table: OcrTable,
    widths: tuple[float, ...],
    regular: fitz.Font,
    bold: fitz.Font,
    font_size: float,
    line_height: float,
) -> tuple[tuple[float, ...], dict[tuple[int, int], list[str]]]:
    padding_x = 2.5
    padding_y = 1.2
    heights = [font_size * line_height + 2 * padding_y for _ in range(table.row_count)]
    wrapped: dict[tuple[int, int], list[str]] = {}
    for cell in table.cells:
        width = sum(widths[cell.column : cell.column + cell.column_span]) - 2 * padding_x
        lines = _wrap(cell.target_text, width, bold if cell.is_header else regular, font_size)
        wrapped[(cell.row, cell.column)] = lines
        required = max(1, len(lines)) * font_size * line_height + 2 * padding_y
        current = sum(heights[cell.row : cell.row + cell.row_span])
        if required > current:
            extra = (required - current) / cell.row_span
            for row in range(cell.row, cell.row + cell.row_span):
                heights[row] += extra
    return tuple(heights), wrapped


def _fit_layout(
    table: OcrTable,
    rect: fitz.Rect,
    regular: fitz.Font,
    bold: fitz.Font,
    settings: Settings,
) -> tuple[float, tuple[float, ...], tuple[float, ...], dict[tuple[int, int], list[str]]]:
    widths = _column_widths(table, rect.width)
    size = settings.table_font_size
    while size + 1e-6 >= settings.table_min_font_size:
        heights, wrapped = _row_heights(
            table,
            widths,
            regular,
            bold,
            size,
            settings.table_line_height,
        )
        if sum(heights) <= rect.height + 0.1:
            # Use the source table's full vertical extent. Besides preserving
            # its visual proportions, this gives font ascenders/descenders the
            # breathing room that PyMuPDF's textbox fitter requires.
            spare = max(0.0, rect.height - sum(heights))
            if heights and spare:
                expanded = tuple(height + spare / len(heights) for height in heights)
            else:
                expanded = heights
            return size, widths, expanded, wrapped
        size -= 0.5
    raise TableRedrawError(
        f"Page {table.page_idx + 1} table cannot fit safely even at "
        f"{settings.table_min_font_size:g} pt; source is left untouched"
    )


def _cumulative(origin: float, values: tuple[float, ...]) -> tuple[float, ...]:
    result = [origin]
    for value in values:
        result.append(result[-1] + value)
    return tuple(result)


def _font_args(path: Path, text: str, *, bold: bool) -> tuple[str, dict[str, Any]]:
    if path.is_file():
        return ("ocr-agent-bold" if bold else "ocr-agent-regular"), {"fontfile": str(path)}
    return ("china-s" if _CJK.search(text) else ("hebo" if bold else "helv")), {}


def _draw_table(
    page: fitz.Page,
    table: OcrTable,
    segment: fitz.Rect,
    settings: Settings,
    regular: fitz.Font,
    bold: fitz.Font,
) -> int:
    table_rect = map_ocr_rect(segment, table.bbox, table.page_size)
    if table_rect.is_empty:
        raise TableRedrawError(f"Page {table.page_idx + 1} table maps outside the target page")
    font_size, widths, heights, wrapped = _fit_layout(table, table_rect, regular, bold, settings)
    xs = _cumulative(table_rect.x0, widths)
    ys = _cumulative(table_rect.y0, heights)

    add_visual_redaction(page, table_rect, fill=(1, 1, 1))
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    # Draw fills first so merged-cell backgrounds never cover a neighbour's
    # border or text.
    for cell in table.cells:
        rect = fitz.Rect(
            xs[cell.column],
            ys[cell.row],
            xs[cell.column + cell.column_span],
            ys[cell.row + cell.row_span],
        )
        if cell.is_header:
            draw_visual_rect(page, rect, color=None, fill=(0.92, 0.94, 0.96), overlay=True)
    for cell in table.cells:
        rect = fitz.Rect(
            xs[cell.column],
            ys[cell.row],
            xs[cell.column + cell.column_span],
            ys[cell.row + cell.row_span],
        )
        draw_visual_rect(
            page,
            rect,
            color=(0.15, 0.15, 0.15),
            width=0.65 if not cell.is_header else 0.85,
            overlay=True,
        )
        text_rect = fitz.Rect(rect.x0 + 2.5, rect.y0 + 1.2, rect.x1 - 2.5, rect.y1 - 1.0)
        text = "\n".join(wrapped[(cell.row, cell.column)])
        font_path = settings.bold_font_path if cell.is_header else settings.regular_font_path
        font_name, extra = _font_args(font_path, text, bold=cell.is_header)
        spare = insert_visual_textbox(
            page,
            text_rect,
            text,
            fontname=font_name,
            fontsize=font_size,
            color=(0, 0, 0),
            lineheight=settings.table_line_height,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
            **extra,
        )
        if spare < -0.5:
            raise TableRedrawError(
                f"Page {table.page_idx + 1} table cell ({cell.row + 1}, {cell.column + 1}) overflowed"
            )
    return len(table.cells)


def _redraw_document(
    pdf_path: Path,
    plan: TableTranslationPlan,
    settings: Settings,
    *,
    source_page_count: int,
    bilingual_mode: str | None = None,
) -> tuple[int, int]:
    document = fitz.open(str(pdf_path))
    regular = _font(settings.regular_font_path, "china-s")
    bold = _font(settings.bold_font_path, "china-s")
    redrawn = 0
    cells = 0
    try:
        for table in plan.tables:
            if table.preserve_as_image:
                continue
            target_page_idx = table.page_idx * 2 + 1 if bilingual_mode == "interleaved" else table.page_idx
            if target_page_idx >= document.page_count:
                raise TableRedrawError("Translated PDF page count does not match the OCR source")
            page = document[target_page_idx]
            if bilingual_mode == "side_by_side":
                midpoint = page.rect.x0 + page.rect.width / 2
                segment = fitz.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1)
            else:
                segment = visual_page_rect(page)
            cells += _draw_table(page, table, segment, settings, regular, bold)
            redrawn += 1
        temporary = pdf_path.with_name(f".{pdf_path.name}.table-redraw.tmp.pdf")
        document.save(str(temporary), garbage=4, deflate=True)
        document.close()
        temporary.replace(pdf_path)
        return redrawn, cells
    except Exception:
        document.close()
        raise


def _bilingual_mode(mono_pdf: Path, bilingual_pdf: Path) -> str | None:
    with fitz.open(str(mono_pdf)) as mono, fitz.open(str(bilingual_pdf)) as bilingual:
        if bilingual.page_count == mono.page_count * 2:
            return "interleaved"
        if bilingual.page_count == mono.page_count and all(
            bilingual[index].rect.width >= mono[index].rect.width * 1.8
            and abs(bilingual[index].rect.height - mono[index].rect.height) <= 2
            for index in range(mono.page_count)
        ):
            return "side_by_side"
    return None


def redraw_tables(
    translated_pdf: str | Path,
    plan: TableTranslationPlan,
    settings: Settings,
    *,
    source_page_count: int,
    bilingual_pdf: str | Path | None = None,
) -> TableRedrawStats:
    mono = Path(translated_pdf).resolve()
    if not mono.is_file():
        raise FileNotFoundError(mono)
    if not plan.tables:
        return TableRedrawStats()
    redrawn, cells = _redraw_document(
        mono,
        plan,
        settings,
        source_page_count=source_page_count,
    )
    bilingual = Path(bilingual_pdf).resolve() if bilingual_pdf else None
    if bilingual is not None and bilingual.is_file():
        mode = _bilingual_mode(mono, bilingual)
        if mode is None:
            raise TableRedrawError("Cannot identify PDFMathTranslate bilingual page layout")
        _redraw_document(
            bilingual,
            plan,
            settings,
            source_page_count=source_page_count,
            bilingual_mode=mode,
        )
    return TableRedrawStats(
        tables_detected=len(plan.tables),
        tables_redrawn=redrawn,
        cells_redrawn=cells,
        cells_translated=plan.translated_cells,
        protected_literals=plan.protected_literals,
        preserved_image_tables=sum(table.preserve_as_image for table in plan.tables),
    )


__all__ = [
    "OcrTable",
    "TableCell",
    "TableRedrawError",
    "TableTranslationPlan",
    "extract_tables",
    "redraw_tables",
    "translate_tables",
]
