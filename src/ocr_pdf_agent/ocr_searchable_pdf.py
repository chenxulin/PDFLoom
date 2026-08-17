"""Build a clean OCR source-text PDF for either PDFMathTranslate engine.

Every scanned page keeps its original visual content. Existing PDF text
objects are removed, page furniture is masked according to PP-StructureV3
semantics, and exactly one fresh source-text layer is written from OCR
coordinates. The v2 bridge keeps that layer invisible and lets BabelDOC's OCR
workaround cover the scanned glyphs. The stable v1 bridge masks the source
text regions first and writes a visible OCR layer because v1 has no equivalent
OCR-workaround switch. Formulas and visual regions are intentionally not
emitted as ordinary text so either layout engine can detect/protect them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import fitz

from .ocr_pdf_coordinates import (
    add_visual_redaction,
    draw_visual_rect,
    insert_visual_textbox,
    map_ocr_rect_to_visual,
    visual_page_rect,
)
from .ocr_semantics import (
    canonical_ocr_type,
    is_formula_type,
    is_furniture_region_type,
    is_furniture_text_type,
    is_preserved_visual_type,
    should_inject_source_text,
    visually_preserved_page_indices,
)
from .fonts import resolve_cjk_font

_MIN_SOURCE_FONT_SIZE = 1.0
_MIN_TEXT_LAYER_COVERAGE = 0.78
_MAX_TEXT_LAYER_DUPLICATION_RATIO = 1.25
_FURNITURE_MASK_PADDING_POINTS = 1.0
_FURNITURE_BAND_NEIGHBOUR_RATIO = 0.008
_FURNITURE_BAND_MAX_EXTENSION_RATIO = 0.03
_SOURCE_LAYER_MODES = {"hidden", "visible_masked"}
SourceLayerMode = Literal["hidden", "visible_masked"]


class OcrSearchablePdfError(ValueError):
    """Raised when a trustworthy single-layer OCR PDF cannot be produced."""


@dataclass(frozen=True)
class OcrSearchablePdfResult:
    pdf_path: Path
    page_count: int
    block_count: int
    injected_pages: int
    # Kept for compatibility with historical callers. The clean pipeline never
    # reuses an old layer and never raster-flattens a page.
    reused_text_pages: int
    flattened_pages: int
    removed_text_layer_pages: int
    masked_furniture_regions: int
    skipped_furniture_blocks: int
    skipped_formula_blocks: int
    preserved_visual_blocks: int
    source_layer_mode: SourceLayerMode
    masked_source_regions: int
    text_layer_coverage: float
    text_layer_duplication_ratio: float


@dataclass(frozen=True)
class _OcrBlock:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    text: str
    block_type: str


@dataclass(frozen=True)
class _FurnitureRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    region_type: str


@dataclass(frozen=True)
class _LayerVerification:
    coverage: float
    duplication_ratio: float


def _positive_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _positive_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _page_sizes(ocr_result: dict[str, Any]) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    for raw in ocr_result.get("pages") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        size = _positive_pair(raw.get("page_size"))
        if page_idx >= 0 and size is not None:
            sizes[page_idx] = size
    for raw in [*(ocr_result.get("blocks") or []), *(ocr_result.get("regions") or [])]:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            continue
        size = _positive_pair(raw.get("page_size"))
        if page_idx >= 0 and size is not None:
            sizes.setdefault(page_idx, size)
    return sizes


def _page_angles(ocr_result: dict[str, Any]) -> dict[int, int]:
    angles: dict[int, int] = {}
    for raw in ocr_result.get("pages") or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_idx = int(raw.get("page_idx"))
            angle = int(raw.get("angle") or 0) % 360
        except (TypeError, ValueError):
            continue
        if page_idx >= 0 and angle in {0, 90, 180, 270}:
            angles[page_idx] = angle
    return angles


def _normalise_blocks(
    ocr_result: dict[str, Any], page_count: int
) -> tuple[list[_OcrBlock], dict[str, int]]:
    preserved_pages = visually_preserved_page_indices(ocr_result)
    blocks: list[_OcrBlock] = []
    stats = {
        "skipped_furniture": 0,
        "skipped_formula": 0,
        "preserved_visual": 0,
    }
    rejected = 0
    for raw in ocr_result.get("blocks") or []:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        block_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            page_idx = -1
        if page_idx in preserved_pages:
            stats["preserved_visual"] += 1
            continue
        if is_furniture_text_type(block_type):
            stats["skipped_furniture"] += 1
            continue
        if is_formula_type(block_type):
            stats["skipped_formula"] += 1
            continue
        if is_preserved_visual_type(block_type) or block_type == "table":
            stats["preserved_visual"] += 1
            continue
        if not should_inject_source_text(block_type):
            continue

        bbox = _positive_bbox(raw.get("bbox"))
        page_size = _positive_pair(raw.get("page_size"))
        if bbox is None or page_size is None or not 0 <= page_idx < page_count:
            rejected += 1
            continue
        blocks.append(
            _OcrBlock(
                page_idx=page_idx,
                bbox=bbox,
                page_size=page_size,
                text=text,
                block_type=block_type,
            )
        )

    if rejected:
        raise OcrSearchablePdfError(
            f"版面 OCR 有 {rejected} 个源文块缺少有效页码、bbox 或 page_size，"
            "无法生成可靠文字层"
        )
    if not blocks and not preserved_pages:
        raise OcrSearchablePdfError("版面 OCR 未返回可用于 PDFMathTranslate 的坐标源文块")
    return (
        sorted(blocks, key=lambda block: (block.page_idx, block.bbox[1], block.bbox[0])),
        stats,
    )


def _normalise_furniture_regions(
    ocr_result: dict[str, Any], page_count: int
) -> list[_FurnitureRegion]:
    sizes = _page_sizes(ocr_result)
    canonical_regions = [
        item for item in (ocr_result.get("regions") or []) if isinstance(item, dict)
    ]
    # Typed text blocks are only a compatibility fallback. Layout regions are
    # preferred because they cover the complete visible header/footer glyphs.
    candidates = list(canonical_regions)
    represented = {
        (
            int(item.get("page_idx", -1)),
            canonical_ocr_type(item.get("type") or item.get("sub_type")),
        )
        for item in canonical_regions
        if is_furniture_region_type(item.get("type") or item.get("sub_type"))
    }
    for item in ocr_result.get("blocks") or []:
        if not isinstance(item, dict):
            continue
        region_type = canonical_ocr_type(item.get("type") or item.get("sub_type"))
        try:
            key = (int(item.get("page_idx")), region_type)
        except (TypeError, ValueError):
            key = (-1, region_type)
        if is_furniture_region_type(region_type) and key not in represented:
            candidates.append(item)
    regions: list[_FurnitureRegion] = []
    seen: set[tuple[int, tuple[float, float, float, float], str]] = set()
    rejected = 0
    for raw in candidates:
        region_type = canonical_ocr_type(raw.get("type") or raw.get("sub_type"))
        if not is_furniture_region_type(region_type):
            continue
        bbox = _positive_bbox(raw.get("bbox"))
        try:
            page_idx = int(raw.get("page_idx"))
        except (TypeError, ValueError):
            page_idx = -1
        page_size = _positive_pair(raw.get("page_size")) or sizes.get(page_idx)
        if bbox is None or page_size is None or not 0 <= page_idx < page_count:
            rejected += 1
            continue
        key = (page_idx, bbox, region_type)
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            _FurnitureRegion(
                page_idx=page_idx,
                bbox=bbox,
                page_size=page_size,
                region_type=region_type,
            )
        )
    if rejected:
        raise OcrSearchablePdfError(
            f"版面 OCR 有 {rejected} 个页眉、页脚或页码区域缺少可靠坐标，"
            "无法安全遮除"
        )
    return _collapse_furniture_bands(ocr_result, regions, sizes)


def _collapse_furniture_bands(
    ocr_result: dict[str, Any],
    regions: list[_FurnitureRegion],
    sizes: dict[int, tuple[float, float]],
) -> list[_FurnitureRegion]:
    """Expand semantic header/footer markers to their complete page bands."""
    by_page: dict[int, list[_FurnitureRegion]] = defaultdict(list)
    for region in regions:
        by_page[region.page_idx].append(region)

    raw_blocks = [
        item for item in (ocr_result.get("blocks") or []) if isinstance(item, dict)
    ]
    collapsed: list[_FurnitureRegion] = []
    for page_idx, page_regions in by_page.items():
        page_size = sizes.get(page_idx) or page_regions[0].page_size
        width, height = page_size
        page_blocks: list[tuple[tuple[float, float, float, float], str]] = []
        for raw in raw_blocks:
            try:
                raw_page_idx = int(raw.get("page_idx"))
            except (TypeError, ValueError):
                continue
            bbox = _positive_bbox(raw.get("bbox"))
            if raw_page_idx == page_idx and bbox is not None:
                page_blocks.append(
                    (bbox, canonical_ocr_type(raw.get("type") or raw.get("sub_type")))
                )

        header_markers = [
            item
            for item in page_regions
            if item.region_type in {"page_header", "header_image"}
        ]
        footer_markers = [
            item
            for item in page_regions
            if item.region_type in {"page_footer", "footer_image"}
        ]
        if header_markers:
            header_bottom = max(item.bbox[3] for item in header_markers)
            explicit_headers = [
                item for item in header_markers if item.region_type == "page_header"
            ]
            if len(explicit_headers) < 2:
                neighbour_anchor = (
                    max(item.bbox[3] for item in explicit_headers)
                    if explicit_headers
                    else header_bottom
                )
                neighbour_limit = (
                    neighbour_anchor + height * _FURNITURE_BAND_NEIGHBOUR_RATIO
                )
                maximum_bottom = (
                    neighbour_anchor + height * _FURNITURE_BAND_MAX_EXTENSION_RATIO
                )
                header_bottom = max(
                    [header_bottom]
                    + [
                        bbox[3]
                        for bbox, block_type in page_blocks
                        if block_type
                        not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                        and bbox[1] <= neighbour_limit
                        and bbox[3] <= maximum_bottom
                    ]
                )
            next_content_starts = [
                bbox[1]
                for bbox, block_type in page_blocks
                if block_type
                not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                and bbox[1] >= header_bottom + 0.5
            ]
            if next_content_starts:
                header_bottom = max(
                    header_bottom,
                    min(
                        min(next_content_starts) - 2.0,
                        header_bottom + height * 0.01,
                    ),
                )
            collapsed.append(
                _FurnitureRegion(
                    page_idx,
                    (0.0, 0.0, width, min(height, header_bottom)),
                    page_size,
                    "page_header",
                )
            )
        if footer_markers:
            footer_top = min(item.bbox[1] for item in footer_markers)
            explicit_footers = [
                item for item in footer_markers if item.region_type == "page_footer"
            ]
            if len(explicit_footers) < 2:
                neighbour_anchor = (
                    min(item.bbox[1] for item in explicit_footers)
                    if explicit_footers
                    else footer_top
                )
                neighbour_limit = (
                    neighbour_anchor - height * _FURNITURE_BAND_NEIGHBOUR_RATIO
                )
                minimum_top = (
                    neighbour_anchor - height * _FURNITURE_BAND_MAX_EXTENSION_RATIO
                )
                footer_top = min(
                    [footer_top]
                    + [
                        bbox[1]
                        for bbox, block_type in page_blocks
                        if block_type
                        not in {"page_header", "header_image", "page_footer", "footer_image", "page_number"}
                        and bbox[3] >= neighbour_limit
                        and bbox[1] >= minimum_top
                    ]
                )
            collapsed.append(
                _FurnitureRegion(
                    page_idx,
                    (0.0, max(0.0, footer_top), width, height),
                    page_size,
                    "page_footer",
                )
            )
        collapsed.extend(
            item
            for item in page_regions
            if item.region_type == "page_number"
            or item.region_type not in {
                "page_header",
                "header_image",
                "page_footer",
                "footer_image",
            }
        )
    return sorted(
        collapsed,
        key=lambda item: (item.page_idx, item.bbox[1], item.bbox[0], item.region_type),
    )


def _compact_text(text: str, *, limit: int = 100_000) -> str:
    compact = "".join(character.casefold() for character in text if character.isalnum())
    return compact[:limit]


def _ocr_page_text(blocks: list[_OcrBlock]) -> str:
    return "\n".join(block.text for block in blocks)


def _mapped_rect(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    return map_ocr_rect_to_visual(visual_page_rect(page), bbox, page_size)


def _source_font_sizes(rect: fitz.Rect, text: str) -> list[float]:
    source_lines = max(1, text.count("\n") + 1)
    initial = min(24.0, max(_MIN_SOURCE_FONT_SIZE, rect.height * 0.78 / source_lines))
    sizes: list[float] = []
    current = initial
    while current > _MIN_SOURCE_FONT_SIZE:
        sizes.append(round(current, 2))
        current *= 0.78
    sizes.append(_MIN_SOURCE_FONT_SIZE)
    return list(dict.fromkeys(sizes))


def _insert_source_block(
    page: fitz.Page,
    block: _OcrBlock,
    *,
    font_path: Path,
    source_layer_mode: SourceLayerMode,
) -> bool:
    rect = _mapped_rect(page, block.bbox, block.page_size)
    if rect.width <= 1 or rect.height <= 1:
        return False
    for font_size in _source_font_sizes(rect, block.text):
        spare = insert_visual_textbox(
            page,
            rect,
            block.text,
            fontname="ocrhidden",
            fontfile=str(font_path),
            fontsize=font_size,
            lineheight=1.0,
            color=(0, 0, 0),
            render_mode=3 if source_layer_mode == "hidden" else 0,
            overlay=True,
        )
        if spare >= 0:
            return True
    return False


def _remove_existing_text_layer(page: fitz.Page) -> bool:
    if not str(page.get_text("text") or "").strip():
        return False
    add_visual_redaction(page, visual_page_rect(page), fill=None, cross_out=False)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )
    return True


def _mask_furniture(page: fitz.Page, region: _FurnitureRegion) -> bool:
    rect = _mapped_rect(page, region.bbox, region.page_size)
    if rect.width <= 0 or rect.height <= 0:
        return False
    rect = fitz.Rect(
        max(page.rect.x0, rect.x0 - _FURNITURE_MASK_PADDING_POINTS),
        max(page.rect.y0, rect.y0 - _FURNITURE_MASK_PADDING_POINTS),
        min(page.rect.x1, rect.x1 + _FURNITURE_MASK_PADDING_POINTS),
        min(page.rect.y1, rect.y1 + _FURNITURE_MASK_PADDING_POINTS),
    )
    draw_visual_rect(
        page,
        rect,
        color=(1, 1, 1),
        fill=(1, 1, 1),
        overlay=True,
    )
    return True


def _block_overlaps_furniture(
    page: fitz.Page,
    block: _OcrBlock,
    regions: list[_FurnitureRegion],
) -> bool:
    block_rect = _mapped_rect(page, block.bbox, block.page_size)
    if block_rect.is_empty:
        return False
    center = fitz.Point(
        (block_rect.x0 + block_rect.x1) / 2,
        (block_rect.y0 + block_rect.y1) / 2,
    )
    for region in regions:
        region_rect = _mapped_rect(page, region.bbox, region.page_size)
        intersection = block_rect & region_rect
        if region_rect.contains(center) or (
            not intersection.is_empty
            and intersection.get_area() / max(1.0, block_rect.get_area()) >= 0.50
        ):
            return True
    return False


def _mask_source_block(page: fitz.Page, block: _OcrBlock) -> bool:
    """Hide scanned source glyphs before building a v1-compatible text layer."""
    rect = _mapped_rect(page, block.bbox, block.page_size)
    if rect.width <= 0 or rect.height <= 0:
        return False
    draw_visual_rect(
        page,
        rect,
        color=(1, 1, 1),
        fill=(1, 1, 1),
        overlay=True,
    )
    return True


def _verify_searchable_layer(
    output_pdf: Path,
    blocks_by_page: dict[int, list[_OcrBlock]],
) -> _LayerVerification:
    document = fitz.open(str(output_pdf))
    try:
        failures: list[str] = []
        total_expected = 0
        total_extracted = 0
        total_matched = 0
        for page_idx in range(document.page_count):
            expected = _compact_text(_ocr_page_text(blocks_by_page.get(page_idx, [])))
            extracted = _compact_text(document[page_idx].get_text("text") or "")
            total_expected += len(expected)
            total_extracted += len(extracted)
            if not expected:
                if extracted:
                    failures.append(f"第 {page_idx + 1} 页仍有非 OCR 文字层")
                continue
            matcher = SequenceMatcher(None, expected, extracted, autojunk=False)
            matched = sum(item.size for item in matcher.get_matching_blocks())
            total_matched += matched
            coverage = min(1.0, matched / len(expected))
            duplication_ratio = len(extracted) / len(expected)
            if coverage < _MIN_TEXT_LAYER_COVERAGE:
                failures.append(f"第 {page_idx + 1} 页源文覆盖率仅 {coverage:.1%}")
            if duplication_ratio > _MAX_TEXT_LAYER_DUPLICATION_RATIO:
                failures.append(
                    f"第 {page_idx + 1} 页文字量为 OCR 的 {duplication_ratio:.2f} 倍，疑似重复层"
                )
        if failures:
            raise OcrSearchablePdfError("OCR 单一源文层校验失败：" + "；".join(failures[:12]))
        coverage = min(1.0, total_matched / total_expected) if total_expected else 1.0
        duplication_ratio = total_extracted / total_expected if total_expected else 0.0
        return _LayerVerification(
            coverage=coverage,
            duplication_ratio=duplication_ratio,
        )
    finally:
        document.close()


def build_ocr_searchable_pdf(
    input_pdf: str | Path,
    output_pdf: str | Path,
    ocr_result: dict[str, Any],
    *,
    input_profile: str,
    source_layer_mode: SourceLayerMode = "hidden",
) -> OcrSearchablePdfResult:
    """Create and validate a single-source-layer PDF for a layout engine.

    Existing text is never trusted or reused, including on ``searchable_scan``
    input. Any malformed coordinate or failed insertion aborts the build rather
    than sending a partial source layer to translation.
    """
    source_path = Path(input_pdf)
    destination = Path(output_pdf)
    if not source_path.is_file():
        raise FileNotFoundError(f"扫描 PDF 不存在：{source_path}")
    if source_path.resolve() == destination.resolve():
        raise OcrSearchablePdfError("OCR 中间 PDF 不能覆盖原始扫描文件")
    if input_profile not in {"image_scan", "searchable_scan"}:
        raise OcrSearchablePdfError(f"OCR 中间 PDF 收到了非扫描输入类型：{input_profile}")
    if source_layer_mode not in _SOURCE_LAYER_MODES:
        raise OcrSearchablePdfError(f"不支持的 OCR 源文层模式：{source_layer_mode}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    source = fitz.open(str(source_path))
    output = fitz.open()
    page_count = source.page_count
    preserved_pages = visually_preserved_page_indices(ocr_result)
    blocks_by_page: dict[int, list[_OcrBlock]] = defaultdict(list)
    try:
        blocks, semantic_stats = _normalise_blocks(ocr_result, page_count)
        page_angles = _page_angles(ocr_result)
        furniture_regions = _normalise_furniture_regions(ocr_result, page_count)
        furniture_by_page: dict[int, list[_FurnitureRegion]] = defaultdict(list)
        for block in blocks:
            blocks_by_page[block.page_idx].append(block)
        for region in furniture_regions:
            furniture_by_page[region.page_idx].append(region)

        font_path = resolve_cjk_font()
        injected_pages = 0
        removed_text_layer_pages = 0
        masked_furniture_regions = 0
        masked_source_regions = 0
        failed_blocks: list[tuple[int, str]] = []
        injected_blocks_by_page: dict[int, list[_OcrBlock]] = defaultdict(list)

        for page_idx in range(page_count):
            output.insert_pdf(source, from_page=page_idx, to_page=page_idx)
            target_page = output[-1]
            correction = page_angles.get(page_idx, 0)
            if correction:
                target_page.set_rotation(
                    (int(target_page.rotation or 0) + correction) % 360
                )
            if _remove_existing_text_layer(target_page):
                removed_text_layer_pages += 1

            if page_idx in preserved_pages:
                continue

            for region in furniture_by_page.get(page_idx, []):
                if _mask_furniture(target_page, region):
                    masked_furniture_regions += 1

            page_blocks: list[_OcrBlock] = []
            for block in blocks_by_page.get(page_idx, []):
                if _block_overlaps_furniture(
                    target_page, block, furniture_by_page.get(page_idx, [])
                ):
                    semantic_stats["skipped_furniture"] += 1
                    continue
                page_blocks.append(block)
            injected_blocks_by_page[page_idx].extend(page_blocks)
            if source_layer_mode == "visible_masked":
                for block in page_blocks:
                    if _mask_source_block(target_page, block):
                        masked_source_regions += 1
            for block in page_blocks:
                if not _insert_source_block(
                    target_page,
                    block,
                    font_path=font_path,
                    source_layer_mode=source_layer_mode,
                ):
                    failed_blocks.append((page_idx + 1, block.text[:60]))
            if page_blocks:
                injected_pages += 1

        if failed_blocks:
            first_page, first_text = failed_blocks[0]
            raise OcrSearchablePdfError(
                f"有 {len(failed_blocks)} 个 OCR 源文块无法写入源文层；"
                f"首个失败位于第 {first_page} 页：{first_text}"
            )

        try:
            metadata = dict(source.metadata or {})
            metadata["producer"] = "OCR PDF Agent PP-StructureV3 single-source-layer bridge"
            metadata["subject"] = "Intermediate OCR source PDF for PDFMathTranslate"
            output.set_metadata(metadata)
        except Exception:
            pass
        try:
            toc = source.get_toc()
            if toc:
                output.set_toc(toc)
        except Exception:
            pass

        output.save(str(destination), garbage=4, deflate=True)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        output.close()
        source.close()

    try:
        verification = _verify_searchable_layer(destination, dict(injected_blocks_by_page))
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    return OcrSearchablePdfResult(
        pdf_path=destination,
        page_count=page_count,
        block_count=sum(len(items) for items in injected_blocks_by_page.values()),
        injected_pages=injected_pages,
        reused_text_pages=0,
        flattened_pages=0,
        removed_text_layer_pages=removed_text_layer_pages,
        masked_furniture_regions=masked_furniture_regions,
        skipped_furniture_blocks=semantic_stats["skipped_furniture"],
        skipped_formula_blocks=semantic_stats["skipped_formula"],
        preserved_visual_blocks=semantic_stats["preserved_visual"],
        source_layer_mode=source_layer_mode,
        masked_source_regions=masked_source_regions,
        text_layer_coverage=verification.coverage,
        text_layer_duplication_ratio=verification.duplication_ratio,
    )


__all__ = [
    "OcrSearchablePdfError",
    "OcrSearchablePdfResult",
    "build_ocr_searchable_pdf",
]
