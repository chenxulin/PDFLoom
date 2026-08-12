"""Translate OCR prose in reading order and backfill it at OCR coordinates.

PDFMathTranslate remains the layout engine, but a final OCR-coordinate pass is
needed for short marginal paragraphs that a scientific layout model may
classify as furniture or otherwise leave untranslated.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fitz

from .config import Settings
from .coordinates import (
    add_visual_redaction,
    insert_visual_textbox,
    map_ocr_rect,
    visual_page_rect,
    visual_rect_to_pdf,
)
from .llm import Translator
from .models import BodyRedrawStats
from .ocr_filter import VISUAL_KINDS, is_low_confidence_speck
from .terminology import (
    exact_preferred_target,
    missing_requirements,
    requirement_instruction,
    requirements_for,
)
from .translation_context import context_excerpt


class BodyRedrawError(RuntimeError):
    pass


@dataclass(frozen=True)
class BodyBlock:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    source_text: str
    target_text: str
    kind: str
    translated: bool = False
    engine_rendered: bool = False
    redraw: bool = False
    protected_literals: tuple[str, ...] = ()


@dataclass(frozen=True)
class BodyTranslationPlan:
    blocks: tuple[BodyBlock, ...]
    translated_blocks: int
    protected_literals: int


_CJK = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_FURNITURE = {
    "header",
    "page_header",
    "footer",
    "page_footer",
    "page_number",
    "number",
    "page_footnote",
    "footnote",
    "page_aside_text",
}
_TABLE = {"table", "table_text"}
_HEADINGS = {"title", "doc_title", "section_heading", "paragraph_title"}
_LITERAL = re.compile(
    r"\d{2,4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|https?://\S+|\b[\w.+-]+@[\w.-]+\.\w+\b"
    r"|(?<![A-Za-z0-9])(?:[A-Z]{1,}[A-Z0-9]*[-/.][A-Z0-9_.()/-]*[A-Z0-9)])(?![A-Za-z0-9])"
    r"|[<>≤≥±]?\d+(?:\.\d+)?\s*(?:%|‰|℃|°C|mol/L|mmol/L|μmol/L|mg/mL|μg/mL|µg/mL|ug/mL|ng/mL|mg/L|ppm|ppb|mg|kg|μg|µg|ug|ng|mL|μL|µL|uL|L|g|h|min|s)?"
)


def _kind(value: Any) -> str:
    return str(value or "text").strip().lower().replace("-", "_").replace(" ", "_")


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
    return box if box[2] > box[0] and box[3] > box[1] else None


def _page_sizes(ocr_result: dict[str, Any]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for raw in ocr_result.get("pages") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        size = _pair(raw.get("page_size"))
        if size:
            result[page_idx] = size
    return result


@dataclass(frozen=True)
class _Region:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    kind: str


def _excluded_regions(ocr_result: dict[str, Any]) -> tuple[_Region, ...]:
    sizes = _page_sizes(ocr_result)
    regions: list[_Region] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        box = _bbox(raw.get("bbox"))
        size = _pair(raw.get("page_size")) or sizes.get(page_idx)
        kind = _kind(raw.get("type") or raw.get("sub_type"))
        if box and size and kind in (_FURNITURE | VISUAL_KINDS | _TABLE):
            regions.append(_Region(page_idx, box, size, kind))
    return tuple(regions)


def _inside_excluded(
    page_idx: int,
    box: tuple[float, float, float, float],
    size: tuple[float, float],
    regions: tuple[_Region, ...],
) -> bool:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    for region in regions:
        if region.page_idx != page_idx:
            continue
        scale_x = size[0] / region.page_size[0]
        scale_y = size[1] / region.page_size[1]
        mapped = (
            region.bbox[0] * scale_x,
            region.bbox[1] * scale_y,
            region.bbox[2] * scale_x,
            region.bbox[3] * scale_y,
        )
        if mapped[0] <= cx <= mapped[2] and mapped[1] <= cy <= mapped[3]:
            return True
    return False


def extract_body_blocks(ocr_result: dict[str, Any]) -> tuple[BodyBlock, ...]:
    sizes = _page_sizes(ocr_result)
    excluded = _excluded_regions(ocr_result)
    result: list[BodyBlock] = []
    seen: set[tuple[int, tuple[float, float, float, float], str]] = set()
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        box = _bbox(raw.get("bbox"))
        size = _pair(raw.get("page_size")) or sizes.get(page_idx)
        text = " ".join(str(raw.get("text") or "").split()).strip()
        kind = _kind(raw.get("type") or raw.get("sub_type"))
        if not box or not size or not text:
            continue
        if is_low_confidence_speck(text, box, size, raw.get("confidence")):
            continue
        if kind in (_FURNITURE | VISUAL_KINDS | _TABLE) or _inside_excluded(
            page_idx,
            box,
            size,
            excluded,
        ):
            continue
        key = (page_idx, box, text)
        if key in seen:
            continue
        seen.add(key)
        result.append(BodyBlock(page_idx, box, size, text, text, kind))
    return tuple(sorted(result, key=lambda block: (block.page_idx, block.bbox[1], block.bbox[0])))


def _needs_translation(text: str, target_language: str) -> bool:
    target = target_language.lower().replace("_", "-")
    has_cjk = bool(_CJK.search(text))
    if target.startswith("en"):
        return has_cjk
    if target.startswith(("zh", "ja", "ko")):
        return bool(re.search(r"[A-Za-z]", text)) and not has_cjk
    return has_cjk or bool(re.search(r"[A-Za-z]", text))


def _contains_rendered_target(text: str, target_language: str) -> bool:
    if not text.strip():
        return False
    target = target_language.lower().replace("_", "-")
    has_cjk = bool(_CJK.search(text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if target.startswith("en"):
        return has_latin and not has_cjk
    if target.startswith(("zh", "ja", "ko")):
        return has_cjk
    return not _needs_translation(text, target_language)


def _rendered_region_texts(
    pdf_path: str | Path | None,
    blocks: tuple[BodyBlock, ...],
) -> tuple[str, ...]:
    if pdf_path is None:
        return tuple("" for _ in blocks)
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result: list[str] = []
    with fitz.open(str(path)) as document:
        for block in blocks:
            if block.page_idx >= document.page_count:
                result.append("")
                continue
            page = document[block.page_idx]
            rect = map_ocr_rect(visual_page_rect(page), block.bbox, block.page_size)
            if rect.is_empty:
                result.append("")
                continue
            result.append(page.get_textbox(visual_rect_to_pdf(page, rect)).strip())
    return tuple(result)


def _protect(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    parts: list[str] = []
    values: list[tuple[str, str]] = []
    cursor = 0
    for match in _LITERAL.finditer(text):
        if match.start() < cursor:
            continue
        parts.append(text[cursor : match.start()])
        wrapper = f"[[JBODY{len(values):03d}|{match.group(0)}]]"
        values.append((wrapper, match.group(0)))
        parts.append(wrapper)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), tuple(values)


def _restore(text: str, values: tuple[tuple[str, str], ...]) -> str:
    result = " ".join(text.split()).strip()
    for wrapper, value in values:
        if result.count(wrapper) != 1:
            raise BodyRedrawError(f"Translation did not preserve {wrapper[:12]}…")
        result = result.replace(wrapper, value)
    if re.search(r"JBODY\s*\d{3}", result, re.IGNORECASE):
        raise BodyRedrawError("Translation contains an unrecovered body placeholder")
    return result


def _translation_context(
    ocr_result: dict[str, Any],
    blocks: tuple[BodyBlock, ...],
    block_index: int,
    requirements: tuple[Any, ...],
    *,
    retry: bool,
) -> str:
    block = blocks[block_index]
    neighbours = [
        candidate.source_text
        for candidate in blocks[max(0, block_index - 2) : block_index + 3]
        if candidate.page_idx == block.page_idx
    ]
    parts = [
        f"OCR PDF page {block.page_idx + 1}, type {block.kind}.",
        "Translate only the current Text input; the following source material is context.",
    ]
    if neighbours:
        parts.append("Neighbouring OCR blocks: " + " | ".join(neighbours))
    excerpt = context_excerpt(ocr_result.get("markdown"), block.source_text)
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


async def translate_body(
    ocr_result: dict[str, Any],
    translator: Translator,
    settings: Settings,
    *,
    existing_pdf: str | Path | None = None,
    progress: Any = None,
) -> BodyTranslationPlan:
    blocks = extract_body_blocks(ocr_result)
    rendered_texts = _rendered_region_texts(existing_pdf, blocks)
    output: list[BodyBlock] = []
    candidates = sum(
        _needs_translation(block.source_text, settings.target_language)
        and not _contains_rendered_target(rendered, settings.target_language)
        for block, rendered in zip(blocks, rendered_texts, strict=True)
    )
    translated = 0
    protected_count = 0
    for block_index, (block, rendered) in enumerate(zip(blocks, rendered_texts, strict=True)):
        if _contains_rendered_target(rendered, settings.target_language):
            output.append(
                replace(
                    block,
                    target_text=" ".join(rendered.split()),
                    engine_rendered=True,
                )
            )
            continue
        if not _needs_translation(block.source_text, settings.target_language):
            output.append(block)
            continue
        protected, values = _protect(block.source_text)
        requirements = (
            requirements_for(block.source_text, settings.target_language)
            if settings.enforce_cmc_terminology
            else ()
        )
        missing = requirements
        restored = ""
        for attempt in range(2):
            target = await translator.translate(
                protected,
                context=_translation_context(
                    ocr_result,
                    blocks,
                    block_index,
                    requirements,
                    retry=attempt > 0,
                ),
                required_literals=tuple(wrapper for wrapper, _ in values),
            )
            restored = _restore(target, values)
            preferred = exact_preferred_target(block.source_text, settings.target_language)
            if settings.enforce_cmc_terminology and preferred is not None:
                restored = preferred
            missing = missing_requirements(restored, requirements)
            if not missing:
                break
        if missing:
            mappings = ", ".join(
                f"{requirement.source_term}->{requirement.required_target}" for requirement in missing
            )
            raise BodyRedrawError(
                f"Page {block.page_idx + 1} OCR block violates mandatory terminology: {mappings}"
            )
        if not restored or restored == block.source_text:
            raise BodyRedrawError(
                f"Page {block.page_idx + 1} OCR block was not translated: {block.source_text[:60]}"
            )
        output.append(
            replace(
                block,
                target_text=restored,
                translated=True,
                redraw=True,
                protected_literals=tuple(value for _, value in values),
            )
        )
        translated += 1
        protected_count += len(values)
        if progress:
            progress(
                "body-translate",
                100.0 * translated / max(1, candidates),
                f"Translated OCR body blocks {translated}/{candidates}",
            )
    return BodyTranslationPlan(tuple(output), translated, protected_count)


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


def _font_args(path: Path, text: str, *, bold: bool) -> tuple[str, dict[str, Any]]:
    if path.is_file():
        return ("ocr-agent-bold" if bold else "ocr-agent-regular"), {"fontfile": str(path)}
    return ("china-s" if _CJK.search(text) else ("hebo" if bold else "helv")), {}


def _insert(page: fitz.Page, rect: fitz.Rect, block: BodyBlock, settings: Settings) -> None:
    heading = block.kind in _HEADINGS
    font_path = settings.bold_font_path if heading else settings.regular_font_path
    font_name, extra = _font_args(font_path, block.target_text, bold=heading)
    start_size = max(10.0, min(16.0, rect.height * 0.60)) if heading else settings.table_font_size
    for step in range(21):
        size = start_size - 0.5 * step
        if size < min(5.5, settings.table_min_font_size):
            break
        spare = insert_visual_textbox(
            page,
            rect,
            block.target_text,
            fontname=font_name,
            fontsize=size,
            color=(0, 0, 0),
            lineheight=settings.table_line_height,
            align=fitz.TEXT_ALIGN_LEFT,
            overlay=True,
            **extra,
        )
        if spare >= -0.1:
            return
    raise BodyRedrawError(
        f"Page {block.page_idx + 1} translated OCR block does not fit its source region: "
        f"{block.source_text[:80]} -> {block.target_text[:120]}"
    )


def _expand_into_following_whitespace(
    rect: fitz.Rect,
    peer_rects: list[fitz.Rect],
    segment: fitz.Rect,
) -> fitz.Rect:
    """Use bounded whitespace below a block without crossing the next OCR row."""
    next_tops: list[float] = []
    for peer in peer_rects:
        if peer.y0 < rect.y1 - 0.1:
            continue
        overlap = max(0.0, min(rect.x1, peer.x1) - max(rect.x0, peer.x0))
        if overlap < min(rect.width, peer.width) * 0.25:
            continue
        next_tops.append(peer.y0)
    maximum_bottom = min(segment.y1, rect.y1 + max(6.0, rect.height * 1.5))
    if next_tops:
        maximum_bottom = min(maximum_bottom, min(next_tops) - 0.8)
    if maximum_bottom <= rect.y1 + 0.1:
        return rect
    return fitz.Rect(rect.x0, rect.y0, rect.x1, maximum_bottom)


def _redraw_document(
    path: Path,
    plan: BodyTranslationPlan,
    settings: Settings,
    *,
    bilingual_mode: str | None = None,
) -> int:
    document = fitz.open(str(path))
    by_page: dict[int, list[tuple[BodyBlock, fitz.Rect]]] = defaultdict(list)
    try:
        mapped: list[tuple[int, BodyBlock, fitz.Rect, fitz.Rect]] = []
        for block in plan.blocks:
            target_page_idx = block.page_idx * 2 + 1 if bilingual_mode == "interleaved" else block.page_idx
            if target_page_idx >= document.page_count:
                raise BodyRedrawError("Translated PDF page count does not match OCR pages")
            page = document[target_page_idx]
            if bilingual_mode == "side_by_side":
                midpoint = page.rect.x0 + page.rect.width / 2
                segment = fitz.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1)
            else:
                segment = visual_page_rect(page)
            rect = map_ocr_rect(segment, block.bbox, block.page_size)
            # Cover tiny layout-engine glyph excursions without entering an
            # adjacent table/figure region.
            rect = fitz.Rect(rect.x0 - 0.8, rect.y0 - 0.8, rect.x1 + 0.8, rect.y1 + 1.5) & segment
            mapped.append((target_page_idx, block, rect, segment))

        peer_rects_by_page: dict[int, list[fitz.Rect]] = defaultdict(list)
        for page_idx, _block, rect, _segment in mapped:
            peer_rects_by_page[page_idx].append(rect)
        for page_idx, block, rect, segment in mapped:
            if not block.redraw:
                continue
            expanded = _expand_into_following_whitespace(
                rect,
                [peer for peer in peer_rects_by_page[page_idx] if peer != rect],
                segment,
            )
            by_page[page_idx].append((block, expanded))

        for page_idx, items in by_page.items():
            page = document[page_idx]
            for _block, rect in items:
                add_visual_redaction(page, rect, fill=(1, 1, 1))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )
            for block, rect in items:
                _insert(
                    page,
                    fitz.Rect(rect.x0 + 0.6, rect.y0 + 0.3, rect.x1 - 0.6, rect.y1 - 0.3),
                    block,
                    settings,
                )

        temporary = path.with_name(f".{path.name}.body-redraw.tmp.pdf")
        document.save(str(temporary), garbage=4, deflate=True)
        document.close()
        temporary.replace(path)
        return sum(block.redraw for block in plan.blocks)
    except Exception:
        document.close()
        raise


def redraw_body(
    translated_pdf: str | Path,
    plan: BodyTranslationPlan,
    settings: Settings,
    *,
    bilingual_pdf: str | Path | None = None,
) -> BodyRedrawStats:
    mono = Path(translated_pdf).resolve()
    if not mono.is_file():
        raise FileNotFoundError(mono)
    if not plan.blocks:
        return BodyRedrawStats()
    redraw_count = sum(block.redraw for block in plan.blocks)
    redrawn = _redraw_document(mono, plan, settings) if redraw_count else 0
    bilingual = Path(bilingual_pdf).resolve() if bilingual_pdf else None
    if redraw_count and bilingual is not None and bilingual.is_file():
        mode = _bilingual_mode(mono, bilingual)
        if mode is None:
            raise BodyRedrawError("Cannot identify PDFMathTranslate bilingual page layout")
        _redraw_document(bilingual, plan, settings, bilingual_mode=mode)
    return BodyRedrawStats(
        blocks_detected=len(plan.blocks),
        blocks_translated=plan.translated_blocks,
        blocks_redrawn=redrawn,
        protected_literals=plan.protected_literals,
    )


__all__ = [
    "BodyBlock",
    "BodyRedrawError",
    "BodyTranslationPlan",
    "extract_body_blocks",
    "redraw_body",
    "translate_body",
]
