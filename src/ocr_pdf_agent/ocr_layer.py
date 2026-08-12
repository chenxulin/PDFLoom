"""Build a clean visible OCR source layer for PDFMathTranslate v1.

Tables, formulas, charts, figures and stamps are deliberately not injected.
PDFMathTranslate handles prose and headings; recognized tables are translated
and redrawn as vectors in the final pass.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .coordinates import (
    add_visual_redaction,
    insert_visual_textbox,
    map_ocr_rect,
    pdf_rect_to_visual,
    visual_page_rect,
)
from .models import OcrLayerArtifact
from .ocr_filter import VISUAL_KINDS, is_low_confidence_speck


class OcrLayerError(RuntimeError):
    pass


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
_CJK = re.compile(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass(frozen=True)
class _Block:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    text: str
    kind: str


@dataclass(frozen=True)
class _Region:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    kind: str


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
    for item in ocr_result.get("pages") or []:
        if not isinstance(item, dict):
            continue
        try:
            page_idx = int(item.get("page_idx"))
        except (TypeError, ValueError):
            continue
        size = _pair(item.get("page_size"))
        if size:
            result[page_idx] = size
    return result


def _regions(ocr_result: dict[str, Any]) -> tuple[_Region, ...]:
    sizes = _page_sizes(ocr_result)
    result: list[_Region] = []
    for raw in ocr_result.get("regions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        box = _bbox(raw.get("bbox"))
        size = _pair(raw.get("page_size")) or sizes.get(page_idx)
        if box and size:
            result.append(_Region(page_idx, box, size, _kind(raw.get("type") or raw.get("sub_type"))))
    return tuple(result)


def _inside_region(block: _Block, regions: tuple[_Region, ...], kinds: set[str]) -> bool:
    cx = (block.bbox[0] + block.bbox[2]) * 0.5
    cy = (block.bbox[1] + block.bbox[3]) * 0.5
    for region in regions:
        if region.page_idx != block.page_idx or region.kind not in kinds:
            continue
        scale_x = block.page_size[0] / region.page_size[0]
        scale_y = block.page_size[1] / region.page_size[1]
        box = (
            region.bbox[0] * scale_x,
            region.bbox[1] * scale_y,
            region.bbox[2] * scale_x,
            region.bbox[3] * scale_y,
        )
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            return True
    return False


def _blocks(
    ocr_result: dict[str, Any],
    page_count: int,
) -> tuple[tuple[_Block, ...], int, int, int]:
    sizes = _page_sizes(ocr_result)
    regions = _regions(ocr_result)
    result: list[_Block] = []
    skipped_tables = 0
    skipped_visuals = 0
    skipped_noise = 0
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
        if page_idx < 0 or page_idx >= page_count or not box or not size or not text:
            continue
        if is_low_confidence_speck(text, box, size, raw.get("confidence")):
            skipped_noise += 1
            continue
        block = _Block(page_idx, box, size, text, kind)
        if kind in _FURNITURE or _inside_region(block, regions, _FURNITURE):
            continue
        if kind in _TABLE or _inside_region(block, regions, _TABLE):
            skipped_tables += 1
            continue
        if kind in VISUAL_KINDS or _inside_region(block, regions, VISUAL_KINDS):
            skipped_visuals += 1
            continue
        key = (page_idx, box, text)
        if key not in seen:
            seen.add(key)
            result.append(block)
    return tuple(result), skipped_tables, skipped_visuals, skipped_noise


def _remove_existing_text(page: fitz.Page) -> bool:
    rectangles: list[fitz.Rect] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                box = fitz.Rect(span.get("bbox") or ())
                if not box.is_empty:
                    rectangles.append(pdf_rect_to_visual(page, box))
    for rect in rectangles:
        add_visual_redaction(page, rect, fill=None)
    return bool(rectangles)


def _insert_block(page: fitz.Page, block: _Block, font_path: Path | None) -> bool:
    rect = map_ocr_rect(visual_page_rect(page), block.bbox, block.page_size)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return False
    # A small inset avoids touching neighbouring layout regions. The initial
    # size follows the recognized line height and is then fitted downwards.
    target = fitz.Rect(rect.x0 + 0.4, rect.y0 + 0.2, rect.x1 - 0.4, rect.y1 - 0.2)
    start_size = max(5.0, min(14.0, rect.height * 0.72))
    font_name = "ocragent"
    font_kwargs: dict[str, Any] = {}
    if font_path is not None and font_path.is_file():
        font_kwargs["fontfile"] = str(font_path)
    else:
        font_name = "china-s" if _CJK.search(block.text) else "helv"
    for step in range(19):
        size = start_size - step * 0.5
        if size < 4.0:
            break
        result = insert_visual_textbox(
            page,
            target,
            block.text,
            fontname=font_name,
            fontsize=size,
            color=(0, 0, 0),
            lineheight=1.05,
            overlay=True,
            **font_kwargs,
        )
        if result >= -0.01:
            return True
    return False


def build_ocr_source_pdf(
    source_pdf: str | Path,
    output_pdf: str | Path,
    ocr_result: dict[str, Any],
    *,
    font_path: str | Path | None = None,
) -> OcrLayerArtifact:
    source_path = Path(source_pdf).resolve()
    output_path = Path(output_pdf).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise OcrLayerError("OCR intermediate PDF may not overwrite the uploaded PDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regular_font = Path(font_path) if font_path else None

    source = fitz.open(str(source_path))
    output = fitz.open()
    inserted = 0
    removed_pages = 0
    masked = 0
    try:
        blocks, skipped_tables, skipped_visuals, skipped_noise = _blocks(
            ocr_result,
            source.page_count,
        )
        by_page: dict[int, list[_Block]] = defaultdict(list)
        for block in blocks:
            by_page[block.page_idx].append(block)
        furniture = tuple(region for region in _regions(ocr_result) if region.kind in _FURNITURE)
        furniture_by_page: dict[int, list[_Region]] = defaultdict(list)
        for region in furniture:
            furniture_by_page[region.page_idx].append(region)

        for page_idx in range(source.page_count):
            output.insert_pdf(source, from_page=page_idx, to_page=page_idx)
            page = output[-1]
            if _remove_existing_text(page):
                removed_pages += 1
            # Cover source pixels for prose and repeated page furniture while
            # leaving tables, formulas and figures untouched.
            for region in furniture_by_page.get(page_idx, []):
                rect = map_ocr_rect(visual_page_rect(page), region.bbox, region.page_size)
                if not rect.is_empty:
                    add_visual_redaction(page, rect, fill=(1, 1, 1))
                    masked += 1
            for block in by_page.get(page_idx, []):
                rect = map_ocr_rect(visual_page_rect(page), block.bbox, block.page_size)
                if not rect.is_empty:
                    add_visual_redaction(page, rect, fill=(1, 1, 1))
                    masked += 1
            if page.first_annot is not None:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )
            failed: list[str] = []
            for block in by_page.get(page_idx, []):
                if _insert_block(page, block, regular_font):
                    inserted += 1
                else:
                    failed.append(block.text[:60])
            if failed:
                raise OcrLayerError(
                    f"Page {page_idx + 1} has {len(failed)} OCR block(s) that do not fit; first: {failed[0]}"
                )

        metadata = dict(source.metadata or {})
        metadata["producer"] = "ocr_pdf_agent PP-StructureV3 bridge"
        metadata["subject"] = "Visible OCR source layer for PDFMathTranslate"
        output.set_metadata(metadata)
        output.save(str(output_path), garbage=4, deflate=True)
    except Exception:
        output.close()
        source.close()
        output_path.unlink(missing_ok=True)
        raise
    else:
        output.close()
        source.close()

    if inserted == 0:
        output_path.unlink(missing_ok=True)
        raise OcrLayerError("PaddleOCR returned no usable prose/title blocks")
    with fitz.open(str(output_path)) as verification:
        extracted = "".join(page.get_text("text") for page in verification).strip()
    if not extracted:
        output_path.unlink(missing_ok=True)
        raise OcrLayerError("Generated OCR source PDF has no searchable text layer")
    return OcrLayerArtifact(
        pdf_path=output_path,
        inserted_blocks=inserted,
        skipped_table_blocks=skipped_tables,
        skipped_visual_blocks=skipped_visuals,
        skipped_noise_blocks=skipped_noise,
        removed_text_pages=removed_pages,
        masked_regions=masked,
    )
