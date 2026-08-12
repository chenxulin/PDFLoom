"""Page-ordered serial translation for scanned-PDF prose and tables.

The scanned-PDF pipeline used to translate every body region concurrently and
then start a second concurrent batch for table cells.  Besides losing reading
order, that made a data-heavy paragraph with many protected values one large
model request.  This module builds one page ledger, walks it top-to-bottom with
one in-flight request, and splits only data-dense prose at punctuation before
assembling the final body and table plans.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .config import Settings
from .ocr_document_typography import (
    OcrBodyRegionTranslation,
    OcrBodyTranslationError,
    OcrBodyTranslationPlan,
    _body_source_texts,
    _normalise_body_regions,
)
from .ocr_heading_typography import (
    OcrHeadingRegionTranslation,
    OcrHeadingTranslationError,
    OcrHeadingTranslationPlan,
    _heading_source_texts,
    _normalise_heading_regions,
)
from .ocr_table_redraw import (
    OcrTableError,
    OcrTableTranslationPlan,
    _expose_values,
    _is_locked_value,
    _needs_translation,
    _protect_values,
    _restore_values,
    _translation_is_valid,
    extract_ocr_tables,
)
from .terminology import exact_preferred_target, normalize_target_output
from .translator import _build_client, translate_chunk

_VISIBLE_SLOT_RE = re.compile(r"\[\[JTBL\d{3}\|[^\[\]\r\n]+\]\]")
_CLAUSE_RE = re.compile(r".+?(?:[，,；;。！？!?：:、]|$)", flags=re.DOTALL)
_MAX_BODY_SLOTS_PER_REQUEST = 8
_MAX_BODY_CHARS_PER_REQUEST = 240
_MAX_INTEGRITY_RETRY_CHARS = 120
_WORKSHOP_DIGITS = str.maketrans("〇零一二三四五六七八九", "00123456789")
_WORKSHOP_NUMBER_RE = re.compile(r"[〇零一二三四五六七八九](?=车间)")


@dataclass(frozen=True)
class OcrSerialTranslationResult:
    body_plan: OcrBodyTranslationPlan
    heading_plan: OcrHeadingTranslationPlan
    table_plan: OcrTableTranslationPlan
    translated_items: int
    model_requests: int


@dataclass(frozen=True)
class _ContentItem:
    kind: str
    page_idx: int
    y: float
    x: float
    source: str
    translatable: bool
    body_idx: int | None = None
    heading_idx: int | None = None
    table_idx: int | None = None
    cell_idx: int | None = None


def _normalize_workshop_numbers(text: str) -> str:
    """Expose Chinese workshop ordinals as factual digits before protection."""
    return _WORKSHOP_NUMBER_RE.sub(
        lambda match: match.group(0).translate(_WORKSHOP_DIGITS),
        text,
    )


def _hard_split_slot_dense_piece(piece: str) -> list[str]:
    matches = list(_VISIBLE_SLOT_RE.finditer(piece))
    if len(matches) <= _MAX_BODY_SLOTS_PER_REQUEST:
        return [piece]
    parts: list[str] = []
    start = 0
    for index in range(
        _MAX_BODY_SLOTS_PER_REQUEST - 1,
        len(matches),
        _MAX_BODY_SLOTS_PER_REQUEST,
    ):
        split_at = matches[index].end()
        parts.append(piece[start:split_at])
        start = split_at
    if start < len(piece):
        parts.append(piece[start:])
    return [part for part in parts if part]


def _split_data_dense_body(text: str) -> tuple[str, ...]:
    """Keep ordinary paragraphs whole; split only requests with many data slots."""
    if len(_VISIBLE_SLOT_RE.findall(text)) <= _MAX_BODY_SLOTS_PER_REQUEST:
        return (text,)
    raw_pieces = _CLAUSE_RE.findall(text)
    if not raw_pieces or "".join(raw_pieces) != text:
        raw_pieces = [text]
    pieces = [part for piece in raw_pieces for part in _hard_split_slot_dense_piece(piece)]
    fragments: list[str] = []
    current = ""
    current_slots = 0
    for piece in pieces:
        piece_slots = len(_VISIBLE_SLOT_RE.findall(piece))
        exceeds_limit = current and (
            current_slots + piece_slots > _MAX_BODY_SLOTS_PER_REQUEST
            or len(current) + len(piece) > _MAX_BODY_CHARS_PER_REQUEST
        )
        if exceeds_limit:
            fragments.append(current)
            current = ""
            current_slots = 0
        current += piece
        current_slots += piece_slots
        if current_slots >= _MAX_BODY_SLOTS_PER_REQUEST or len(current) >= _MAX_BODY_CHARS_PER_REQUEST:
            fragments.append(current)
            current = ""
            current_slots = 0
    if current:
        fragments.append(current)
    return tuple(fragment for fragment in fragments if fragment.strip()) or (text,)


def _fragment_values(
    fragment: str,
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((placeholder, value) for placeholder, value in values if placeholder[2:-2] in fragment)


def _join_fragments(parts: list[str], target_language: str) -> str:
    target = (target_language or "").replace("_", "-").casefold()
    separator = "" if target.startswith(("zh", "ja", "ko")) else " "
    return separator.join(part.strip() for part in parts if part.strip()).strip()


def _split_placeholder_integrity_retry(text: str) -> tuple[str, ...]:
    """Split a failed fragment at clauses without cutting through a slot."""
    slots = list(_VISIBLE_SLOT_RE.finditer(text))
    if len(slots) <= 1:
        clauses = _CLAUSE_RE.findall(text)
        if len(clauses) <= 1 or "".join(clauses) != text:
            return (text,)
        packed: list[str] = []
        current = ""
        for clause in clauses:
            contains_slot = bool(_VISIBLE_SLOT_RE.search(clause))
            if contains_slot:
                if current.strip():
                    packed.append(current)
                    current = ""
                packed.append(clause)
                continue
            if current and len(current) + len(clause) > _MAX_INTEGRITY_RETRY_CHARS:
                packed.append(current)
                current = ""
            current += clause
        if current.strip():
            packed.append(current)
        if len(packed) == 1:
            midpoint = max(1, len(clauses) // 2)
            packed = ["".join(clauses[:midpoint]), "".join(clauses[midpoint:])]
        compact = tuple(piece for piece in packed if piece.strip())
        return compact if len(compact) > 1 else (text,)

    candidates: list[tuple[int, int, int]] = []
    for clause in _CLAUSE_RE.finditer(text):
        boundary = clause.end()
        if boundary <= 0 or boundary >= len(text):
            continue
        left_slots = sum(match.end() <= boundary for match in slots)
        right_slots = len(slots) - left_slots
        if left_slots and right_slots:
            candidates.append(
                (
                    abs(left_slots - right_slots),
                    abs(boundary * 2 - len(text)),
                    boundary,
                )
            )
    split_at = min(candidates)[2] if candidates else slots[(len(slots) - 1) // 2].end()
    left, right = text[:split_at], text[split_at:]
    if not left.strip() or not right.strip():
        return (text,)
    return left, right


async def _translate_checked(
    *,
    client: Any,
    text: str,
    source_for_validation: str,
    values: tuple[tuple[str, str], ...],
    context_prev: str,
    context_next: str,
    settings: Settings,
    seg_type: str,
    retry_reason: str,
) -> str:
    last_error: Exception | None = None
    rejected_translation: str | None = None
    attempt_count = 3 if values else 2
    for attempt in range(attempt_count):
        translated = ""
        try:
            translated = await translate_chunk(
                client,
                text,
                context_prev if attempt < 2 else "",
                context_next if attempt < 2 else "",
                settings,
                seg_type=seg_type,
                source_kind="pdf",
                has_layout=True,
                layout_retry_reason=retry_reason if attempt else None,
                required_literals=tuple(value for _, value in values),
                rejected_translation=rejected_translation,
            )
            restored = _restore_values(translated, values)
            if not _translation_is_valid(
                source_for_validation,
                restored,
                settings.target_language,
            ):
                raise ValueError("译文未完整翻译到目标语言")
            return restored
        except Exception as exc:  # noqa: BLE001 - strict integrity retries
            last_error = exc
            if translated.strip():
                rejected_translation = translated
    if last_error is None:  # pragma: no cover - attempt_count is always positive
        raise RuntimeError("串行翻译没有执行")
    raise last_error


async def _translate_single_raw_value(
    *,
    client: Any,
    text: str,
    values: tuple[tuple[str, str], ...],
    context_prev: str,
    context_next: str,
    settings: Settings,
    seg_type: str,
) -> str:
    """Last-resort retry with one authoritative OCR value visible as plain text."""
    if len(values) != 1:
        raise OcrTableError("原值重译只允许一个数据槽")
    raw_source = _restore_values(text, values)
    value = values[0][1]
    last_error: Exception | None = None
    rejected_translation: str | None = None
    for attempt in range(2):
        translated = ""
        try:
            translated = await translate_chunk(
                client,
                raw_source,
                context_prev if attempt == 0 else "",
                context_next if attempt == 0 else "",
                settings,
                seg_type=seg_type,
                source_kind="pdf",
                has_layout=True,
                layout_retry_reason="ocr_raw_value_integrity",
                required_literals=(value,),
                rejected_translation=rejected_translation,
            )
            if translated.count(value) != raw_source.count(value):
                raise OcrTableError(f"译文未逐字保留 OCR 原值 {value}")
            if not _translation_is_valid(
                raw_source,
                translated,
                settings.target_language,
            ):
                raise ValueError("译文未完整翻译到目标语言")
            return translated
        except Exception as exc:  # noqa: BLE001 - strict integrity retries
            last_error = exc
            if translated.strip():
                rejected_translation = translated
    if last_error is None:  # pragma: no cover - loop is always entered
        raise RuntimeError("OCR 原值重译没有执行")
    raise last_error


async def _translate_with_integrity_fallback(
    *,
    client: Any,
    text: str,
    source_for_validation: str,
    values: tuple[tuple[str, str], ...],
    context_prev: str,
    context_next: str,
    settings: Settings,
    seg_type: str,
    retry_reason: str,
    depth: int = 0,
) -> str:
    """Retry normally, then isolate failed literal clauses until every fact is safe."""
    try:
        return await _translate_checked(
            client=client,
            text=text,
            source_for_validation=source_for_validation,
            values=values,
            context_prev=context_prev,
            context_next=context_next,
            settings=settings,
            seg_type=seg_type,
            retry_reason=retry_reason,
        )
    except (OcrTableError, ValueError) as integrity_error:
        pieces = _split_placeholder_integrity_retry(text)
        if len(pieces) > 1 and depth < 4:
            translated_pieces: list[str] = []
            for piece_idx, piece in enumerate(pieces):
                piece_values = _fragment_values(piece, values)
                piece_source = _restore_values(piece, piece_values) if piece_values else piece
                translated_pieces.append(
                    await _translate_with_integrity_fallback(
                        client=client,
                        text=piece,
                        source_for_validation=piece_source,
                        values=piece_values,
                        context_prev=(pieces[piece_idx - 1] if piece_idx else context_prev),
                        context_next=(pieces[piece_idx + 1] if piece_idx + 1 < len(pieces) else context_next),
                        settings=settings,
                        seg_type=seg_type,
                        retry_reason=retry_reason,
                        depth=depth + 1,
                    )
                )
            translated = _join_fragments(
                translated_pieces,
                settings.target_language,
            )
            if not _translation_is_valid(
                source_for_validation,
                translated,
                settings.target_language,
            ):
                raise ValueError("分治译文未完整翻译到目标语言") from integrity_error
            return translated
        if len(values) == 1:
            return await _translate_single_raw_value(
                client=client,
                text=text,
                values=values,
                context_prev=context_prev,
                context_next=context_next,
                settings=settings,
                seg_type=seg_type,
            )
        raise


async def translate_ocr_content_serially(
    *,
    ocr_result: dict[str, Any],
    settings: Settings,
    translator: Any | None = None,
    on_progress: Callable[[str, float, str], None] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> OcrSerialTranslationResult:
    """Translate body regions, headings and table cells in page order."""
    body_regions = _normalise_body_regions(ocr_result)
    body_sources = _body_source_texts(ocr_result, body_regions)
    body_results: list[OcrBodyRegionTranslation | None] = [None] * len(body_regions)
    heading_regions = _normalise_heading_regions(ocr_result)
    heading_sources = _heading_source_texts(ocr_result, heading_regions)
    heading_results: list[OcrHeadingRegionTranslation | None] = [None] * len(heading_regions)

    tables = extract_ocr_tables(ocr_result)
    mutable_cells = [list(table.cells) for table in tables]
    preserved_table_cells = 0
    locked_table_values = 0
    items: list[_ContentItem] = []

    for body_idx, (region, source) in enumerate(zip(body_regions, body_sources, strict=True)):
        translatable = _needs_translation(source, settings.target_language)
        if not translatable:
            body_results[body_idx] = OcrBodyRegionTranslation(
                region.page_idx,
                region.bbox,
                region.page_size,
                source,
                source,
            )
        items.append(
            _ContentItem(
                kind="body",
                page_idx=region.page_idx,
                y=region.bbox[1],
                x=region.bbox[0],
                source=source,
                translatable=translatable,
                body_idx=body_idx,
            )
        )

    for heading_idx, (region, source) in enumerate(zip(heading_regions, heading_sources, strict=True)):
        translatable = _needs_translation(source, settings.target_language)
        if not translatable:
            heading_results[heading_idx] = OcrHeadingRegionTranslation(
                region.page_idx,
                region.bbox,
                region.page_size,
                source,
                source,
            )
        items.append(
            _ContentItem(
                kind="heading",
                page_idx=region.page_idx,
                y=region.bbox[1],
                x=region.bbox[0],
                source=source,
                translatable=translatable,
                heading_idx=heading_idx,
            )
        )

    for table_idx, table in enumerate(tables):
        row_height = (table.bbox[3] - table.bbox[1]) / max(1, table.row_count)
        column_width = (table.bbox[2] - table.bbox[0]) / max(1, table.column_count)
        for cell_idx, cell in enumerate(table.cells):
            translatable = (
                not table.preserve_as_image
                and bool(cell.source_text)
                and _needs_translation(
                    cell.source_text,
                    settings.target_language,
                )
            )
            if not translatable:
                preserved_table_cells += 1
                if not table.preserve_as_image and cell.source_text and _is_locked_value(cell.source_text):
                    locked_table_values += 1
            items.append(
                _ContentItem(
                    kind="table",
                    page_idx=table.page_idx,
                    y=table.bbox[1] + row_height * (cell.row + 0.5),
                    x=table.bbox[0] + column_width * (cell.column + 0.5),
                    source=cell.source_text,
                    translatable=translatable,
                    table_idx=table_idx,
                    cell_idx=cell_idx,
                )
            )

    ordered = sorted(
        items,
        key=lambda item: (
            item.page_idx,
            item.y,
            item.x,
            {"heading": 0, "body": 1, "table": 2}[item.kind],
        ),
    )
    candidate_count = sum(item.translatable for item in ordered)
    if on_progress:
        on_progress(
            "content-translate-serial",
            0.0,
            f"按页面顺序串行翻译正文、标题和表格：0/{candidate_count}",
        )

    translated_items = 0
    model_requests = 0
    translated_table_cells = 0
    protected_table_values = locked_table_values
    client = (translator or _build_client(settings)) if candidate_count else None
    owns_client = candidate_count > 0 and translator is None
    try:
        for position, item in enumerate(ordered):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            if not item.translatable:
                continue
            previous = ordered[position - 1].source if position else ""
            following = ordered[position + 1].source if position + 1 < len(ordered) else ""
            if item.kind in {"body", "heading"}:
                if item.kind == "body":
                    assert item.body_idx is not None
                    region = body_regions[item.body_idx]
                    error_type = OcrBodyTranslationError
                    content_label = "正文段落"
                    retry_reason = "ocr_body_placeholder_integrity"
                else:
                    assert item.heading_idx is not None
                    region = heading_regions[item.heading_idx]
                    error_type = OcrHeadingTranslationError
                    content_label = "标题"
                    retry_reason = "ocr_heading_placeholder_integrity"
                normalized_source = _normalize_workshop_numbers(item.source)
                protected_text, values = _protect_values(normalized_source)
                exposed_text = _expose_values(protected_text, values)
                fragments = _split_data_dense_body(exposed_text)
                translated_fragments: list[str] = []
                for fragment_idx, fragment in enumerate(fragments):
                    fragment_values = _fragment_values(fragment, values)
                    fragment_previous = fragments[fragment_idx - 1] if fragment_idx else previous
                    fragment_following = (
                        fragments[fragment_idx + 1] if fragment_idx + 1 < len(fragments) else following
                    )
                    try:
                        translated_fragment = await _translate_with_integrity_fallback(
                            client=client,
                            text=fragment,
                            source_for_validation=fragment,
                            values=fragment_values,
                            context_prev=fragment_previous,
                            context_next=fragment_following,
                            settings=settings,
                            seg_type="para",
                            retry_reason=retry_reason,
                        )
                    except Exception as exc:
                        raise error_type(
                            f"第 {region.page_idx + 1} 页{content_label}"
                            f"第 {fragment_idx + 1}/{len(fragments)} 片翻译失败：{exc}"
                        ) from exc
                    translated_fragments.append(translated_fragment)
                    model_requests += 1
                translated = _join_fragments(
                    translated_fragments,
                    settings.target_language,
                )
                if not _translation_is_valid(
                    normalized_source,
                    translated,
                    settings.target_language,
                ):
                    raise error_type(f"第 {region.page_idx + 1} 页{content_label}未完整翻译到目标语言")
                preferred = exact_preferred_target(
                    item.source,
                    settings.target_language,
                )
                if preferred is not None:
                    translated = preferred
                translated = normalize_target_output(
                    translated,
                    settings.target_language,
                )
                if item.kind == "body":
                    assert item.body_idx is not None
                    body_results[item.body_idx] = OcrBodyRegionTranslation(
                        region.page_idx,
                        region.bbox,
                        region.page_size,
                        item.source,
                        translated,
                        tuple(value for _, value in values),
                    )
                else:
                    assert item.heading_idx is not None
                    heading_results[item.heading_idx] = OcrHeadingRegionTranslation(
                        region.page_idx,
                        region.bbox,
                        region.page_size,
                        item.source,
                        translated,
                        tuple(value for _, value in values),
                    )
            else:
                assert item.table_idx is not None and item.cell_idx is not None
                table = tables[item.table_idx]
                cell = table.cells[item.cell_idx]
                normalized_source = _normalize_workshop_numbers(item.source)
                protected_text, values = _protect_values(normalized_source)
                exposed_text = _expose_values(protected_text, values)
                location = (
                    f"Table {item.table_idx + 1}, row {cell.row + 1}, "
                    f"column {cell.column + 1}. Translate this cell only."
                )
                try:
                    translated = await _translate_with_integrity_fallback(
                        client=client,
                        text=exposed_text,
                        source_for_validation=normalized_source,
                        values=values,
                        context_prev=f"{location}\nPrevious item: {previous}",
                        context_next=f"Next item: {following}" if following else "",
                        settings=settings,
                        seg_type="table_cell",
                        retry_reason="table_value_placeholder_integrity",
                    )
                except Exception as exc:
                    raise OcrTableError(
                        f"第 {table.page_idx + 1} 页表格第 {cell.row + 1} 行"
                        f"第 {cell.column + 1} 列翻译失败：{exc}"
                    ) from exc
                preferred = exact_preferred_target(
                    item.source,
                    settings.target_language,
                )
                if preferred is not None:
                    translated = preferred
                translated = normalize_target_output(
                    translated,
                    settings.target_language,
                )
                mutable_cells[item.table_idx][item.cell_idx] = replace(
                    cell,
                    target_text=translated,
                    translated=True,
                    protected_values=tuple(value for _, value in values),
                )
                translated_table_cells += 1
                protected_table_values += len(values)
                model_requests += 1

            translated_items += 1
            if on_progress:
                on_progress(
                    "content-translate-serial",
                    100.0 * translated_items / max(1, candidate_count),
                    f"按页面顺序串行翻译正文、标题和表格："
                    f"{translated_items}/{candidate_count}（第 {item.page_idx + 1} 页）",
                )
    finally:
        if client is not None and owns_client:
            await client.aclose()

    completed_body = tuple(item for item in body_results if item is not None)
    if len(completed_body) != len(body_regions):
        raise OcrBodyTranslationError("串行正文翻译计划不完整")
    body_plan = OcrBodyTranslationPlan(
        completed_body,
        len(body_regions),
        sum(_needs_translation(source, settings.target_language) for source in body_sources),
        sum(len(item.protected_values) for item in completed_body),
    )
    completed_headings = tuple(item for item in heading_results if item is not None)
    if len(completed_headings) != len(heading_regions):
        raise OcrHeadingTranslationError("串行标题翻译计划不完整")
    heading_plan = OcrHeadingTranslationPlan(
        completed_headings,
        len(heading_regions),
        sum(_needs_translation(source, settings.target_language) for source in heading_sources),
        sum(len(item.protected_values) for item in completed_headings),
    )
    translated_tables = tuple(
        replace(table, cells=tuple(mutable_cells[index])) for index, table in enumerate(tables)
    )
    table_plan = OcrTableTranslationPlan(
        tables=translated_tables,
        table_count=len(translated_tables),
        cell_count=sum(len(table.cells) for table in translated_tables),
        translated_cells=translated_table_cells,
        preserved_cells=preserved_table_cells,
        protected_values=protected_table_values,
    )
    return OcrSerialTranslationResult(
        body_plan=body_plan,
        heading_plan=heading_plan,
        table_plan=table_plan,
        translated_items=translated_items,
        model_requests=model_requests,
    )


__all__ = [
    "OcrSerialTranslationResult",
    "translate_ocr_content_serially",
]
