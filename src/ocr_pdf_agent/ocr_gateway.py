"""HTTP client and result adapter for the dedicated PP-StructureV3 service.

The public contract is deliberately independent of Paddle's result classes::

    {
        "provider": "paddleocr-ppstructurev3",
        "pages": [{"page_idx": 0, "page_size": [width, height]}],
        "regions": [{"page_idx": 0, "bbox": [...], "type": "page_header"}],
        "blocks": [{"page_idx": 0, "bbox": [...], "text": "...", "type": "text"}],
        "markdown": "...",
    }

``regions`` retains document semantics such as header/footer/page number,
title, table, image and equation.  OCR lines inherit the type of their layout
region so the redraw layer can remove repeated furniture and avoid translating
figures or formulas as prose.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import statistics
import tempfile
import threading
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://192.168.1.88:18093"
_DEFAULT_API_PATH = "/api/v1/structure"
_DEFAULT_LAYOUT_MODEL = "PP-DocLayout-M"
_DEFAULT_TEXT_DETECTION_MODEL = "PP-OCRv5_server_det"
_DEFAULT_TEXT_RECOGNITION_MODEL = "PP-OCRv5_server_rec"
_OCR_CACHE_SCHEMA_VERSION = 3
_DEFAULT_OCR_MAX_CONCURRENT_REQUESTS = 2
_DEFAULT_OCR_TARGET_DPI = 600
_DEFAULT_OCR_DOWNSAMPLE_THRESHOLD_DPI = 600
_OOS_OCR_CONFUSION_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:0OS|O0S|00S)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

_REGION_TYPE_MAP = {
    "header": "page_header",
    "page_header": "page_header",
    "header_image": "header_image",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "footer_image": "footer_image",
    "number": "page_number",
    "page_number": "page_number",
    "footnote": "page_footnote",
    "aside_text": "page_aside_text",
    "doc_title": "title",
    "paragraph_title": "section_heading",
    "title": "title",
    "table": "table",
    "table_title": "table_caption",
    "table_caption": "table_caption",
    "figure": "image",
    "figure_title": "figure_caption",
    "figure_caption": "figure_caption",
    "image_title": "figure_caption",
    "image": "image",
    "chart": "chart",
    "seal": "seal",
    "stamp": "seal",
    "display_formula": "equation",
    "formula": "equation",
    "inline_formula": "inline_equation",
    "abstract": "text",
    "reference": "text",
    "references": "text",
    "content": "text",
    "text": "text",
}
_BLOCK_TYPE_MAP = {**_REGION_TYPE_MAP, "table": "table_text"}
_MARKDOWN_EXCLUDED = {
    "page_header",
    "page_footer",
    "page_number",
    "page_footnote",
    "page_aside_text",
    "header_image",
    "footer_image",
    "image",
    "chart",
    "equation",
    "inline_equation",
    "interline_equation",
    "seal",
}

OcrProgressCallback = Callable[[int, int, str], Awaitable[None] | None]


@dataclass(frozen=True)
class _PageProxy:
    pdf_bytes: bytes
    downsampled: bool
    source_dpi: float | None


@dataclass
class _OcrJob:
    key: str
    task: asyncio.Task[dict[str, Any]] | None = None
    subscribers: int = 0
    callbacks: dict[object, OcrProgressCallback] = field(default_factory=dict)
    latest_progress: tuple[int, int, str] | None = None


_OCR_JOBS: dict[str, _OcrJob] = {}
_OCR_JOBS_GUARD = threading.Lock()
_OCR_SEMAPHORE: asyncio.Semaphore | None = None
_OCR_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None
_OCR_SEMAPHORE_LIMIT: int | None = None
_ACTIVE_SETTINGS: ContextVar[Settings | None] = ContextVar(
    "ocr_gateway_settings",
    default=None,
)
_SETTING_ATTRIBUTES = {
    "PADDLEOCR_ENABLED": "paddleocr_enabled",
    "PADDLEOCR_API_URL": "paddleocr_api_url",
    "PADDLEOCR_API_PATH": "paddleocr_api_path",
    "PADDLEOCR_TIMEOUT_SECONDS": "paddleocr_timeout_seconds",
    "PADDLEOCR_MAX_CONCURRENT_REQUESTS": "paddleocr_max_concurrent_requests",
    "PADDLEOCR_PROXY_TARGET_DPI": "paddleocr_proxy_target_dpi",
    "PADDLEOCR_DOWNSAMPLE_THRESHOLD_DPI": "paddleocr_downsample_threshold_dpi",
    "PADDLEOCR_CACHE_ENABLED": "paddleocr_cache_enabled",
    "PADDLEOCR_CACHE_DIR": "paddleocr_cache_dir",
    "PADDLEOCR_ORIENTATION_RETRY": "paddleocr_orientation_retry",
}


def _configured_value(name: str) -> str | None:
    """Read standalone settings first, then an explicit process environment."""
    settings = _ACTIVE_SETTINGS.get()
    attribute = _SETTING_ATTRIBUTES.get(name)
    if settings is not None and attribute is not None:
        value = getattr(settings, attribute, None)
        return None if value is None else str(value)
    return os.getenv(name)


def _correct_common_ocr_confusions(payload: dict[str, Any]) -> dict[str, Any]:
    """Correct high-confidence domain acronyms before they reach translation."""
    correction_count = 0

    def corrected(value: str) -> str:
        nonlocal correction_count
        result, count = _OOS_OCR_CONFUSION_RE.subn("OOS", value)
        correction_count += count
        return result

    for block in payload.get("blocks") or []:
        if isinstance(block, dict):
            block["text"] = corrected(str(block.get("text") or ""))
    for region in payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        structured = region.get("structured_content")
        if isinstance(structured, str):
            region["structured_content"] = corrected(structured)
    payload["markdown"] = corrected(str(payload.get("markdown") or ""))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    previous_count = int(metadata.get("common_text_corrections") or 0)
    metadata["common_text_corrections"] = previous_count + correction_count
    return payload


_STRUCTURED_REGION_TYPES = {"table", "chart", "equation", "inline_equation", "image"}


def _env_bool(name: str, default: bool) -> bool:
    raw = _configured_value(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def api_url() -> str:
    settings = _ACTIVE_SETTINGS.get()
    if settings is not None:
        return settings.ocr_endpoint
    base = (_configured_value("PADDLEOCR_API_URL") or _DEFAULT_API_URL).strip().rstrip("/")
    path = (_configured_value("PADDLEOCR_API_PATH") or _DEFAULT_API_PATH).strip()
    if not base:
        return ""
    normalized_path = "/" + path.lstrip("/") if path else ""
    if normalized_path and not base.endswith(normalized_path):
        return base + normalized_path
    return base


def _service_token() -> str:
    # ATTACHMENT_OCR_SERVICE_TOKEN is a migration fallback so the existing
    # production secret can be reused without printing or rewriting .env.
    settings = _ACTIVE_SETTINGS.get()
    token = (
        settings.ocr_token
        if settings is not None
        else (
            os.getenv("PADDLEOCR_SERVICE_TOKEN")
            or os.getenv("ATTACHMENT_OCR_SERVICE_TOKEN")
            or ""
        )
    ).strip()
    if "\r" in token or "\n" in token:
        raise ValueError("PADDLEOCR_SERVICE_TOKEN 配置无效")
    return token


def _auth_headers() -> dict[str, str]:
    token = _service_token()
    return {"X-OCR-Service-Token": token} if token else {}


def is_enabled() -> bool:
    return _env_bool("PADDLEOCR_ENABLED", True) and bool(api_url() and _service_token())


def _positive_float(name: str, default: float) -> float:
    raw = (_configured_value(name) or str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = (_configured_value(name) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _ocr_target_dpi() -> int:
    return _positive_int("PADDLEOCR_PROXY_TARGET_DPI", _DEFAULT_OCR_TARGET_DPI)


def _ocr_downsample_threshold_dpi() -> int:
    configured = _positive_int(
        "PADDLEOCR_DOWNSAMPLE_THRESHOLD_DPI",
        _DEFAULT_OCR_DOWNSAMPLE_THRESHOLD_DPI,
    )
    return max(configured, _ocr_target_dpi())


def _ocr_request_semaphore() -> asyncio.Semaphore:
    """Return one process-global OCR request gate for the active event loop."""
    global _OCR_SEMAPHORE, _OCR_SEMAPHORE_LOOP, _OCR_SEMAPHORE_LIMIT
    loop = asyncio.get_running_loop()
    limit = _positive_int(
        "PADDLEOCR_MAX_CONCURRENT_REQUESTS",
        _DEFAULT_OCR_MAX_CONCURRENT_REQUESTS,
    )
    if (
        _OCR_SEMAPHORE is None
        or _OCR_SEMAPHORE_LOOP is not loop
        or limit != _OCR_SEMAPHORE_LIMIT
    ):
        _OCR_SEMAPHORE = asyncio.Semaphore(limit)
        _OCR_SEMAPHORE_LOOP = loop
        _OCR_SEMAPHORE_LIMIT = limit
    return _OCR_SEMAPHORE


def _model_name(env_name: str, default: str) -> str:
    return (_configured_value(env_name) or default).strip() or default


def _http_timeout() -> httpx.Timeout:
    seconds = _positive_float("PADDLEOCR_TIMEOUT_SECONDS", 3600)
    return httpx.Timeout(connect=10.0, read=seconds, write=120.0, pool=10.0)


def _error_excerpt(response: httpx.Response) -> str:
    return re.sub(r"\s+", " ", response.text or "").strip()[:600]


def _pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (width, height)) or width <= 0 or height <= 0:
        return None
    return width, height


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _polygon_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if not points:
        return None
    return _bbox([min(x for x, _ in points), min(y for _, y in points),
                  max(x for x, _ in points), max(y for _, y in points)])


def _number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _right_angle(value: Any) -> int:
    try:
        angle = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return angle if angle in {0, 90, 180, 270} else 0


def _core_result(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    core = raw.get("res")
    return core if isinstance(core, dict) else raw


def _nested_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("res")
    return nested if isinstance(nested, dict) else raw


def _canonical_type(label: Any, *, for_block: bool = False) -> tuple[str, str]:
    source = str(label or "text").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = _BLOCK_TYPE_MAP if for_block else _REGION_TYPE_MAP
    return mapping.get(source, source or "text"), source or "text"


def _intersection_ratio(
    line: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> tuple[bool, float]:
    x0, y0, x1, y1 = line
    rx0, ry0, rx1, ry1 = region
    center_inside = rx0 <= (x0 + x1) / 2 <= rx1 and ry0 <= (y0 + y1) / 2 <= ry1
    intersection = max(0.0, min(x1, rx1) - max(x0, rx0)) * max(
        0.0, min(y1, ry1) - max(y0, ry0)
    )
    line_area = max((x1 - x0) * (y1 - y0), 1.0)
    return center_inside, intersection / line_area


def _line_type(
    rect: tuple[float, float, float, float], regions: list[dict[str, Any]]
) -> tuple[str, str]:
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for region in regions:
        region_rect = _bbox(region.get("bbox"))
        if region_rect is None:
            continue
        center_inside, overlap = _intersection_ratio(rect, region_rect)
        if not center_inside and overlap < 0.5:
            continue
        region_area = (region_rect[2] - region_rect[0]) * (region_rect[3] - region_rect[1])
        score = _confidence(region.get("confidence")) or 0.0
        candidates.append(((float(center_inside), overlap, score, -region_area), region))
    if not candidates:
        return "text", "text"
    region = max(candidates, key=lambda item: item[0])[1]
    return _canonical_type(region.get("source_type") or region.get("type"), for_block=True)


def _enrich_structured_regions(
    core: dict[str, Any],
    page_regions: list[dict[str, Any]],
    *,
    page_idx: int,
    page_size: list[int | float],
) -> None:
    """Attach formula/chart/table parser output to its semantic layout region."""
    parsing_results = core.get("parsing_res_list") or []
    for raw in parsing_results:
        if not isinstance(raw, dict):
            continue
        region_type, source_type = _canonical_type(
            raw.get("block_label") or raw.get("label") or raw.get("type")
        )
        if region_type not in _STRUCTURED_REGION_TYPES:
            continue
        rect = _bbox(
            raw.get("block_bbox") or raw.get("coordinate") or raw.get("bbox")
        )
        if rect is None:
            continue
        content = raw.get("block_content")
        if content is None:
            content = raw.get("content")
        if content is None:
            content = raw.get("markdown")

        compatible = [
            region
            for region in page_regions
            if region.get("type") == region_type and _bbox(region.get("bbox")) is not None
        ]
        target: dict[str, Any] | None = None
        if compatible:
            target = max(
                compatible,
                key=lambda region: _intersection_ratio(
                    rect, _bbox(region["bbox"]) or rect
                )[1],
            )
        if target is None:
            target = {
                "page_idx": page_idx,
                "bbox": [_number(value) for value in rect],
                "page_size": page_size,
                "type": region_type,
                "source_type": source_type,
            }
            page_regions.append(target)
        if content not in (None, "", [], {}):
            target["structured_content"] = content


def _markdown_from_blocks(blocks: list[dict[str, Any]]) -> str:
    by_page: dict[int, list[str]] = defaultdict(list)
    for block in blocks:
        text = str(block.get("text") or "").strip()
        block_type = str(block.get("type") or "text")
        if not text or block_type in _MARKDOWN_EXCLUDED:
            continue
        if block_type == "title":
            text = f"# {text}"
        elif block_type == "section_heading":
            text = f"### {text}"
        by_page[int(block["page_idx"])].append(text)
    return "\n\n".join(
        f"## 第 {page_idx + 1} 页\n\n" + "\n\n".join(by_page[page_idx])
        for page_idx in sorted(by_page)
    )


def normalize_results(payload: dict[str, Any], *, filename: str) -> dict[str, Any]:
    """Normalize PP-StructureV3 result JSON into the redraw contract."""
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raw_results = [payload]

    pages: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for fallback_page_idx, raw_result in enumerate(raw_results):
        core = _core_result(raw_result)
        if core is None:
            continue
        try:
            page_idx = int(core.get("page_index", fallback_page_idx))
        except (TypeError, ValueError):
            page_idx = fallback_page_idx
        size = _pair(core.get("_page_size")) or _pair(
            [core.get("width"), core.get("height")]
        )
        if page_idx < 0 or size is None:
            raise ValueError(
                f"PP-StructureV3 第 {fallback_page_idx + 1} 页未返回可靠的图像尺寸"
            )
        page_size = [_number(size[0]), _number(size[1])]
        preprocessor = _nested_result(core.get("doc_preprocessor_res"))
        classifier_angle = _right_angle(preprocessor.get("angle"))
        # PaddleX applies the classifier angle counter-clockwise through
        # OpenCV. PDF /Rotate uses clockwise degrees, so expose the equivalent
        # page rotation required by downstream PyMuPDF writers.
        angle = (-classifier_angle) % 360
        pages.append(
            {
                "page_idx": page_idx,
                "page_size": page_size,
                "angle": angle,
            }
        )

        layout = _nested_result(core.get("layout_det_res"))
        page_regions: list[dict[str, Any]] = []
        for raw_box in layout.get("boxes") or []:
            if not isinstance(raw_box, dict):
                continue
            rect = _bbox(raw_box.get("coordinate") or raw_box.get("bbox"))
            if rect is None:
                continue
            region_type, source_type = _canonical_type(raw_box.get("label"))
            region: dict[str, Any] = {
                "page_idx": page_idx,
                "bbox": [_number(value) for value in rect],
                "page_size": page_size,
                "type": region_type,
                "source_type": source_type,
            }
            score = _confidence(raw_box.get("score"))
            if score is not None:
                region["confidence"] = score
            page_regions.append(region)
        _enrich_structured_regions(
            core,
            page_regions,
            page_idx=page_idx,
            page_size=page_size,
        )
        regions.extend(page_regions)

        overall = _nested_result(core.get("overall_ocr_res"))
        texts = overall.get("rec_texts") or []
        scores = overall.get("rec_scores") or []
        boxes = overall.get("rec_boxes") or []
        polygons = overall.get("rec_polys") or overall.get("dt_polys") or []
        for index, raw_text in enumerate(texts):
            text = str(raw_text or "").strip()
            rect = _bbox(boxes[index]) if index < len(boxes) else None
            if rect is None and index < len(polygons):
                rect = _polygon_bbox(polygons[index])
            if not text or rect is None:
                continue
            block_type, source_type = _line_type(rect, page_regions)
            block: dict[str, Any] = {
                "page_idx": page_idx,
                "bbox": [_number(value) for value in rect],
                "page_size": page_size,
                "text": text,
                "type": block_type,
                "source_type": source_type,
            }
            score = _confidence(scores[index]) if index < len(scores) else None
            if score is not None:
                block["confidence"] = score
            blocks.append(block)

    pages.sort(key=lambda item: int(item["page_idx"]))
    regions.sort(key=lambda item: (int(item["page_idx"]), item["bbox"][1], item["bbox"][0]))
    blocks.sort(key=lambda item: (int(item["page_idx"]), item["bbox"][1], item["bbox"][0]))
    if not pages:
        raise ValueError("PP-StructureV3 未返回可用页面")
    if len({int(page["page_idx"]) for page in pages}) != len(pages):
        raise ValueError("PP-StructureV3 返回了重复页码")

    normalized = {
        "provider": "paddleocr-ppstructurev3",
        "markdown": _markdown_from_blocks(blocks),
        "blocks": blocks,
        "regions": regions,
        "pages": pages,
        "metadata": {
            "engine": "PaddleOCR PP-StructureV3",
            "layout_model": _model_name("PADDLEOCR_LAYOUT_MODEL", _DEFAULT_LAYOUT_MODEL),
            "text_detection_model": _model_name(
                "PADDLEOCR_TEXT_DETECTION_MODEL", _DEFAULT_TEXT_DETECTION_MODEL
            ),
            "text_recognition_model": _model_name(
                "PADDLEOCR_TEXT_RECOGNITION_MODEL", _DEFAULT_TEXT_RECOGNITION_MODEL
            ),
            "filename": filename,
            "page_count": len(pages),
            "region_count": len(regions),
            "block_count": len(blocks),
            "enabled_modules": {
                "doc_orientation_classify": _env_bool(
                    "PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY", True
                ),
                "doc_unwarping": _env_bool("PADDLEOCR_USE_DOC_UNWARPING", False),
                "textline_orientation": _env_bool(
                    "PADDLEOCR_USE_TEXTLINE_ORIENTATION", True
                ),
                "table_recognition": _env_bool("PADDLEOCR_USE_TABLE_RECOGNITION", True),
                "formula_recognition": _env_bool(
                    "PADDLEOCR_USE_FORMULA_RECOGNITION", True
                ),
                "chart_recognition": _env_bool("PADDLEOCR_USE_CHART_RECOGNITION", True),
                "seal_recognition": _env_bool("PADDLEOCR_USE_SEAL_RECOGNITION", True),
                "region_detection": _env_bool("PADDLEOCR_USE_REGION_DETECTION", True),
            },
            "furniture_min_confidence": _positive_float(
                "PADDLEOCR_FURNITURE_MIN_CONFIDENCE", 0.65
            ),
        },
    }
    return _correct_common_ocr_confusions(normalized)


def validate_layout_response(payload: Any, *, filename: str) -> dict[str, Any]:
    """Validate the dedicated service's normalized coordinate contract."""
    if not isinstance(payload, dict):
        raise ValueError("PP-StructureV3 服务返回格式无效：顶层必须是 JSON 对象")
    if payload.get("provider") != "paddleocr-ppstructurev3":
        raise ValueError("PP-StructureV3 服务返回了错误的 provider")
    pages = payload.get("pages")
    blocks = payload.get("blocks")
    regions = payload.get("regions")
    if not isinstance(pages, list) or not pages:
        raise ValueError("PP-StructureV3 服务未返回页面尺寸")
    if not isinstance(blocks, list) or not isinstance(regions, list):
        raise ValueError("PP-StructureV3 服务未返回 blocks/regions 数组")

    page_indexes: set[int] = set()
    for page in pages:
        if not isinstance(page, dict) or _pair(page.get("page_size")) is None:
            raise ValueError("PP-StructureV3 服务返回了无效页面尺寸")
        try:
            page_idx = int(page.get("page_idx"))
        except (TypeError, ValueError) as exc:
            raise ValueError("PP-StructureV3 服务返回了无效页码") from exc
        if page_idx < 0 or page_idx in page_indexes:
            raise ValueError("PP-StructureV3 服务返回了重复或无效页码")
        if _right_angle(page.get("angle")) != page.get("angle", 0):
            raise ValueError("PP-StructureV3 服务返回了无效页面方向")
        page_indexes.add(page_idx)

    for collection_name, collection in (("blocks", blocks), ("regions", regions)):
        for item in collection:
            if not isinstance(item, dict) or _bbox(item.get("bbox")) is None:
                raise ValueError(f"PP-StructureV3 服务返回了无效 {collection_name} 坐标")
            try:
                page_idx = int(item.get("page_idx"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"PP-StructureV3 服务返回了无效 {collection_name} 页码"
                ) from exc
            if page_idx not in page_indexes:
                raise ValueError(
                    f"PP-StructureV3 服务返回的 {collection_name} 引用了未知页面"
                )
            if collection_name == "blocks" and not str(item.get("text") or "").strip():
                raise ValueError("PP-StructureV3 服务返回了空文本 block")

    payload["markdown"] = str(payload.get("markdown") or "")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata
    metadata.setdefault("filename", filename)
    return _correct_common_ocr_confusions(payload)


def _pdf_page_count(pdf_bytes: bytes) -> int:
    import fitz

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if document.page_count < 1:
            raise ValueError("扫描 PDF 没有页面")
        return document.page_count
    finally:
        document.close()


def _page_image_dpi(page: Any) -> float | None:
    """Estimate DPI from images covering most of a page.

    Small high-resolution logos must not trigger whole-page rasterization, so
    only placements covering at least half of the visible page are considered.
    """
    page_rect = page.rect
    page_area = max(float(page_rect.get_area()), 1.0)
    candidates: list[float] = []
    for image in page.get_images(full=True):
        if len(image) < 4:
            continue
        xref, width, height = int(image[0]), int(image[2]), int(image[3])
        try:
            placements = page.get_image_rects(xref)
        except Exception:
            placements = []
        for placement in placements:
            rect = placement & page_rect
            if rect.is_empty or rect.get_area() / page_area < 0.50:
                continue
            if rect.width <= 0 or rect.height <= 0:
                continue
            candidates.append(
                max(width * 72.0 / rect.width, height * 72.0 / rect.height)
            )
    return max(candidates) if candidates else None


def _build_page_proxy(
    pdf_bytes: bytes,
    page_index: int,
    *,
    extra_rotation: int = 0,
) -> _PageProxy:
    """Build a one-page OCR input without modifying the source document."""
    import fitz

    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    proxy = fitz.open()
    try:
        if not 0 <= page_index < source.page_count:
            raise ValueError(f"扫描 PDF 页码超出范围：{page_index + 1}")
        page = source[page_index]
        source_dpi = _page_image_dpi(page)
        target_dpi = _ocr_target_dpi()
        should_downsample = bool(
            source_dpi is not None
            and source_dpi > _ocr_downsample_threshold_dpi()
        )
        if should_downsample:
            # Render only the OCR proxy.  The original PDF remains the visual
            # source for masking, translation layout, and final delivery.
            scale = target_dpi / 72.0
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csGRAY,
                alpha=False,
                annots=True,
            )
            target_page = proxy.new_page(width=page.rect.width, height=page.rect.height)
            target_page.insert_image(
                target_page.rect,
                stream=pixmap.tobytes("png"),
                keep_proportion=False,
                overlay=True,
            )
        else:
            proxy.insert_pdf(source, from_page=page_index, to_page=page_index)
        rotation = _right_angle(extra_rotation)
        if rotation:
            proxy[0].set_rotation((int(proxy[0].rotation or 0) + rotation) % 360)
        return _PageProxy(
            pdf_bytes=proxy.tobytes(garbage=4, deflate=True),
            downsampled=should_downsample,
            source_dpi=source_dpi,
        )
    finally:
        proxy.close()
        source.close()


def _cache_fingerprint(pdf_bytes: bytes) -> str:
    source_digest = hashlib.sha256(pdf_bytes).hexdigest()
    config = {
        "schema": _OCR_CACHE_SCHEMA_VERSION,
        "source_sha256": source_digest,
        "endpoint": api_url(),
        "target_dpi": _ocr_target_dpi(),
        "downsample_threshold_dpi": _ocr_downsample_threshold_dpi(),
        "layout_model": _model_name("PADDLEOCR_LAYOUT_MODEL", _DEFAULT_LAYOUT_MODEL),
        "text_detection_model": _model_name(
            "PADDLEOCR_TEXT_DETECTION_MODEL", _DEFAULT_TEXT_DETECTION_MODEL
        ),
        "text_recognition_model": _model_name(
            "PADDLEOCR_TEXT_RECOGNITION_MODEL", _DEFAULT_TEXT_RECOGNITION_MODEL
        ),
        "doc_orientation_classify": _env_bool(
            "PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY", True
        ),
        "doc_unwarping": _env_bool("PADDLEOCR_USE_DOC_UNWARPING", False),
        "textline_orientation": _env_bool(
            "PADDLEOCR_USE_TEXTLINE_ORIENTATION", True
        ),
        "orientation_retry": _env_bool("PADDLEOCR_ORIENTATION_RETRY", True),
        "table_recognition": _env_bool("PADDLEOCR_USE_TABLE_RECOGNITION", True),
        "formula_recognition": _env_bool("PADDLEOCR_USE_FORMULA_RECOGNITION", True),
        "chart_recognition": _env_bool("PADDLEOCR_USE_CHART_RECOGNITION", True),
        "seal_recognition": _env_bool("PADDLEOCR_USE_SEAL_RECOGNITION", True),
        "region_detection": _env_bool("PADDLEOCR_USE_REGION_DETECTION", True),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_key: str) -> Path:
    configured = (_configured_value("PADDLEOCR_CACHE_DIR") or "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        settings = _ACTIVE_SETTINGS.get()
        root = (
            settings.storage_dir if settings is not None else Path.cwd() / "storage"
        ) / "ocr-cache" / "ppstructurev3"
    return root / f"{cache_key}.json"


def _load_cached_layout(cache_key: str, *, filename: str) -> dict[str, Any] | None:
    path = _cache_path(cache_key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validated = validate_layout_response(payload, filename=filename)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - a bad cache must never block OCR
        logger.warning("Ignoring invalid PP-StructureV3 cache entry %s: %s", cache_key[:12], exc)
        return None
    result = copy.deepcopy(validated)
    result.setdefault("metadata", {})["filename"] = filename
    result["metadata"]["cache_hit"] = True
    return result


def _store_cached_layout(cache_key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{cache_key}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


async def _invoke_progress(
    callback: OcrProgressCallback | None,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        pending = callback(completed, total, message)
        if inspect.isawaitable(pending):
            await pending
    except Exception as exc:  # noqa: BLE001 - observability must not break OCR
        logger.debug("PP-StructureV3 progress callback failed: %s", exc)


async def _broadcast_progress(
    job: _OcrJob,
    completed: int,
    total: int,
    message: str,
) -> None:
    with _OCR_JOBS_GUARD:
        job.latest_progress = (completed, total, message)
        callbacks = list(job.callbacks.values())
    for callback in callbacks:
        await _invoke_progress(callback, completed, total, message)


async def _post_page_pdf(
    client: httpx.AsyncClient,
    pdf_bytes: bytes,
    *,
    filename: str,
) -> dict[str, Any]:
    endpoint = api_url()
    try:
        response = await client.post(
            endpoint,
            headers=_auth_headers(),
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=_http_timeout(),
        )
    except httpx.TimeoutException as exc:
        raise ValueError("PP-StructureV3 OCR 服务调用超时") from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"无法连接 PP-StructureV3 OCR 服务（{endpoint}）：{exc}") from exc

    if response.status_code in {401, 403}:
        raise ValueError(
            f"PP-StructureV3 OCR 服务鉴权失败（HTTP {response.status_code}）"
        )
    if response.status_code == 413:
        raise ValueError("扫描 PDF 超过 PP-StructureV3 OCR 服务上传限制")
    if response.status_code >= 400:
        detail = _error_excerpt(response)
        suffix = f"：{detail}" if detail else ""
        raise ValueError(
            f"PP-StructureV3 OCR 服务调用失败（HTTP {response.status_code}）{suffix}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("PP-StructureV3 OCR 服务未返回 JSON") from exc
    return validate_layout_response(payload, filename=filename)


def _remap_single_page(payload: dict[str, Any], page_index: int) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    pages = result.get("pages") or []
    if len(pages) != 1:
        raise ValueError("PP-StructureV3 单页请求返回了非单页结果")
    pages[0]["page_idx"] = page_index
    for collection_name in ("blocks", "regions"):
        for item in result.get(collection_name) or []:
            item["page_idx"] = page_index
    return result


def _page_layout_quality(payload: dict[str, Any]) -> dict[str, float | int | bool]:
    blocks = [item for item in payload.get("blocks") or [] if isinstance(item, dict)]
    confidences = [
        score
        for item in blocks
        if (score := _confidence(item.get("confidence"))) is not None
    ]
    vertical = 0
    horizontal = 0
    valid_boxes = 0
    for item in blocks:
        rect = _bbox(item.get("bbox"))
        if rect is None:
            continue
        valid_boxes += 1
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        vertical += int(height >= width * 1.25)
        horizontal += int(width >= height * 1.25)
    median_confidence = statistics.median(confidences) if confidences else 0.0
    vertical_ratio = vertical / valid_boxes if valid_boxes else 0.0
    horizontal_ratio = horizontal / valid_boxes if valid_boxes else 0.0
    suspicious = bool(
        valid_boxes >= 6
        and vertical_ratio >= 0.55
        and (median_confidence < 0.80 or horizontal_ratio < 0.35)
    )
    score = median_confidence * 0.75 + horizontal_ratio * 0.25
    return {
        "block_count": valid_boxes,
        "median_confidence": round(median_confidence, 4),
        "vertical_ratio": round(vertical_ratio, 4),
        "horizontal_ratio": round(horizontal_ratio, 4),
        "score": round(score, 4),
        "suspicious": suspicious,
    }


def _set_page_orientation(
    payload: dict[str, Any],
    *,
    proxy_rotation: int,
    quality: dict[str, float | int | bool],
    retried: bool,
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    pages = result.get("pages") or []
    if len(pages) != 1:
        raise ValueError("PP-StructureV3 单页方向归一化收到了非单页结果")
    service_rotation = _right_angle(pages[0].get("angle"))
    pages[0]["angle"] = (_right_angle(proxy_rotation) + service_rotation) % 360
    metadata = result.setdefault("metadata", {})
    metadata["orientation_retry"] = retried
    metadata["orientation_proxy_rotation"] = _right_angle(proxy_rotation)
    metadata["orientation_service_rotation"] = service_rotation
    metadata["page_ocr_quality"] = quality
    return result


def _merge_page_layouts(
    page_results: list[dict[str, Any]],
    *,
    filename: str,
    downsampled_pages: int,
    highest_source_dpi: float | None,
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    corrections = 0
    base_metadata: dict[str, Any] = {}
    orientation_retried_pages: list[int] = []
    page_ocr_quality: dict[str, Any] = {}
    for result in page_results:
        pages.extend(copy.deepcopy(result.get("pages") or []))
        blocks.extend(copy.deepcopy(result.get("blocks") or []))
        regions.extend(copy.deepcopy(result.get("regions") or []))
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            if not base_metadata:
                base_metadata = copy.deepcopy(metadata)
            corrections += int(metadata.get("common_text_corrections") or 0)
            result_pages = result.get("pages") or []
            result_page_idx = (
                int(result_pages[0].get("page_idx", -1)) if result_pages else -1
            )
            if metadata.get("orientation_retry") and result_page_idx >= 0:
                orientation_retried_pages.append(result_page_idx)
            if metadata.get("page_ocr_quality") and result_page_idx >= 0:
                page_ocr_quality[str(result_page_idx)] = copy.deepcopy(
                    metadata["page_ocr_quality"]
                )
    pages.sort(key=lambda item: int(item["page_idx"]))
    blocks.sort(key=lambda item: (int(item["page_idx"]), item["bbox"][1], item["bbox"][0]))
    regions.sort(key=lambda item: (int(item["page_idx"]), item["bbox"][1], item["bbox"][0]))
    base_metadata.update(
        {
            "filename": filename,
            "page_count": len(pages),
            "block_count": len(blocks),
            "region_count": len(regions),
            "common_text_corrections": corrections,
            "proxy_target_dpi": _ocr_target_dpi(),
            "proxy_downsample_threshold_dpi": _ocr_downsample_threshold_dpi(),
            "proxy_downsampled_pages": downsampled_pages,
            "highest_source_dpi": (
                round(highest_source_dpi, 1) if highest_source_dpi is not None else None
            ),
            "orientation_retried_pages": orientation_retried_pages,
            "page_ocr_quality": page_ocr_quality,
            "cache_hit": False,
        }
    )
    merged = {
        "provider": "paddleocr-ppstructurev3",
        "markdown": _markdown_from_blocks(blocks),
        "pages": pages,
        "blocks": blocks,
        "regions": regions,
        "metadata": base_metadata,
    }
    return validate_layout_response(merged, filename=filename)


async def _run_ocr_job(
    job: _OcrJob,
    pdf_bytes: bytes,
    *,
    filename: str,
    client: httpx.AsyncClient | None,
    cache_key: str,
    cache_enabled: bool,
) -> dict[str, Any]:
    page_count = await asyncio.to_thread(_pdf_page_count, pdf_bytes)
    await _broadcast_progress(job, 0, page_count, f"准备逐页 OCR，共 {page_count} 页")
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=_http_timeout(), trust_env=False)
    page_results: list[dict[str, Any]] = []
    downsampled_pages = 0
    source_dpis: list[float] = []
    try:
        for page_index in range(page_count):
            await _broadcast_progress(
                job,
                page_index,
                page_count,
                f"第 {page_index + 1}/{page_count} 页等待 OCR 资源",
            )
            async with _ocr_request_semaphore():
                proxy = await asyncio.to_thread(_build_page_proxy, pdf_bytes, page_index)
                if proxy.downsampled:
                    downsampled_pages += 1
                if proxy.source_dpi is not None:
                    source_dpis.append(proxy.source_dpi)
                dpi_note = (
                    f"（{proxy.source_dpi:.0f}→{_ocr_target_dpi()} DPI）"
                    if proxy.downsampled and proxy.source_dpi is not None
                    else ""
                )
                await _broadcast_progress(
                    job,
                    page_index,
                    page_count,
                    f"正在识别第 {page_index + 1}/{page_count} 页{dpi_note}",
                )
                page_filename = f"{Path(filename).stem}.page-{page_index + 1:04d}.pdf"
                page_payload = await _post_page_pdf(
                    client,
                    proxy.pdf_bytes,
                    filename=page_filename,
                )
                initial_quality = _page_layout_quality(page_payload)
                candidates = [(page_payload, 0, initial_quality)]
                if (
                    bool(initial_quality["suspicious"])
                    and _env_bool("PADDLEOCR_ORIENTATION_RETRY", True)
                ):
                    await _broadcast_progress(
                        job,
                        page_index,
                        page_count,
                        f"第 {page_index + 1}/{page_count} 页方向异常，正在旋转复核",
                    )
                    for rotation in (90, 270):
                        rotated_proxy = await asyncio.to_thread(
                            _build_page_proxy,
                            pdf_bytes,
                            page_index,
                            extra_rotation=rotation,
                        )
                        rotated_payload = await _post_page_pdf(
                            client,
                            rotated_proxy.pdf_bytes,
                            filename=(
                                f"{Path(filename).stem}.page-{page_index + 1:04d}"
                                f".rot-{rotation}.pdf"
                            ),
                        )
                        candidates.append(
                            (
                                rotated_payload,
                                rotation,
                                _page_layout_quality(rotated_payload),
                            )
                        )
                page_payload, proxy_rotation, selected_quality = max(
                    candidates,
                    key=lambda item: float(item[2]["score"]),
                )
                if bool(selected_quality["suspicious"]):
                    raise ValueError(
                        f"第 {page_index + 1} 页 OCR 方向/质量不可靠，已停止生成，"
                        "避免低置信度文字覆盖原扫描页"
                    )
                page_payload = _set_page_orientation(
                    page_payload,
                    proxy_rotation=proxy_rotation,
                    quality=selected_quality,
                    retried=len(candidates) > 1,
                )
            page_results.append(_remap_single_page(page_payload, page_index))
            await _broadcast_progress(
                job,
                page_index + 1,
                page_count,
                f"已完成第 {page_index + 1}/{page_count} 页 OCR",
            )
    finally:
        if owns_client:
            await client.aclose()

    merged = _merge_page_layouts(
        page_results,
        filename=filename,
        downsampled_pages=downsampled_pages,
        highest_source_dpi=max(source_dpis) if source_dpis else None,
    )
    if cache_enabled:
        try:
            await asyncio.to_thread(_store_cached_layout, cache_key, merged)
        except Exception as exc:  # noqa: BLE001 - cache writes are optional
            logger.warning("Unable to store PP-StructureV3 cache %s: %s", cache_key[:12], exc)
    return merged


async def _wait_for_shared_job(
    job: _OcrJob,
    cancel_event: asyncio.Event | None,
) -> dict[str, Any]:
    assert job.task is not None
    shared = asyncio.shield(job.task)
    if cancel_event is None:
        try:
            return await shared
        finally:
            if not shared.done():
                shared.cancel()
            await asyncio.gather(shared, return_exceptions=True)
    if cancel_event.is_set():
        shared.cancel()
        raise asyncio.CancelledError
    cancellation = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {shared, cancellation},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation in done and cancel_event.is_set():
            shared.cancel()
            raise asyncio.CancelledError
        return await shared
    finally:
        cancellation.cancel()
        if not shared.done():
            # Cancel only this shield wrapper.  The underlying shared OCR job
            # remains alive while another subscriber still needs it.
            shared.cancel()
        await asyncio.gather(cancellation, shared, return_exceptions=True)


async def _pdf_to_layout(
    pdf_bytes: bytes,
    *,
    filename: str,
    client: httpx.AsyncClient | None = None,
    cancel_event: asyncio.Event | None = None,
    on_progress: OcrProgressCallback | None = None,
) -> dict[str, Any]:
    """OCR a scanned PDF page-by-page with cancellation and content deduplication."""
    if not is_enabled():
        raise ValueError(
            "PP-StructureV3 OCR 服务未配置；请设置 PADDLEOCR_API_URL 和服务令牌"
        )
    if not pdf_bytes:
        raise ValueError("扫描 PDF 内容为空")
    cache_key = _cache_fingerprint(pdf_bytes)
    cache_enabled = client is None and _env_bool("PADDLEOCR_CACHE_ENABLED", True)
    if cache_enabled:
        cached = await asyncio.to_thread(
            _load_cached_layout,
            cache_key,
            filename=filename,
        )
        if cached is not None:
            total = len(cached.get("pages") or [])
            await _invoke_progress(on_progress, total, total, f"命中 OCR 内容缓存，共 {total} 页")
            return cached

    subscriber_id = object()
    joined_existing = False
    with _OCR_JOBS_GUARD:
        job = _OCR_JOBS.get(cache_key)
        if job is None:
            job = _OcrJob(key=cache_key)
            _OCR_JOBS[cache_key] = job
            job.task = asyncio.create_task(
                _run_ocr_job(
                    job,
                    pdf_bytes,
                    filename=filename,
                    client=client,
                    cache_key=cache_key,
                    cache_enabled=cache_enabled,
                ),
                name=f"ppstructure-{cache_key[:12]}",
            )
        else:
            joined_existing = True
        job.subscribers += 1
        if on_progress is not None:
            job.callbacks[subscriber_id] = on_progress
        latest_progress = job.latest_progress

    if joined_existing:
        await _invoke_progress(on_progress, 0, 0, "检测到相同扫描件，复用正在进行的 OCR")
    if latest_progress is not None:
        await _invoke_progress(on_progress, *latest_progress)

    cancel_job = False
    try:
        result = await _wait_for_shared_job(job, cancel_event)
        returned = copy.deepcopy(result)
        returned.setdefault("metadata", {})["filename"] = filename
        returned["metadata"]["deduplicated_inflight"] = joined_existing
        return returned
    finally:
        with _OCR_JOBS_GUARD:
            job.callbacks.pop(subscriber_id, None)
            job.subscribers = max(0, job.subscribers - 1)
            if job.subscribers == 0:
                if _OCR_JOBS.get(cache_key) is job:
                    _OCR_JOBS.pop(cache_key, None)
                cancel_job = bool(job.task is not None and not job.task.done())
        if cancel_job and job.task is not None:
            # The last interested task is gone: cancel the coroutine currently
            # awaiting httpx so the TCP request is closed immediately.
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)


async def pdf_to_layout(
    pdf_bytes: bytes,
    *,
    filename: str,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    cancel_event: asyncio.Event | None = None,
    on_progress: OcrProgressCallback | None = None,
) -> dict[str, Any]:
    """OCR a PDF with an explicit standalone configuration context.

    The context variable keeps `.env`-backed Pydantic settings private to this
    request while the copied Joincare OCR implementation runs concurrently.
    """
    token = _ACTIVE_SETTINGS.set(settings)
    try:
        return await _pdf_to_layout(
            pdf_bytes,
            filename=filename,
            client=client,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )
    finally:
        _ACTIVE_SETTINGS.reset(token)
