"""PDF Translator compatible layout generation, rendering, and verification.

The OCR service owns recognition and translation.  This module consumes its
auditable translation ledger and applies the PDF Translator rendering model:
each source page is kept as a raster background, source-language regions are
covered narrowly, and translated text is written as searchable vector text in
explicit top-left-origin coordinate boxes.

Tables are represented as deferred ``cover`` elements.  Their source pixels
remain untouched during layout rendering so :mod:`ocr_table_redraw` can replace
the complete table with a searchable vector grid and insert continuation pages
when required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ALLOWED_FONTS = frozenset({"regular", "bold", "mono", "cjk"})
_ALLOWED_ALIGNMENTS = frozenset({"left", "center", "right", "justify"})
_ALLOWED_VALIGNS = frozenset({"top", "middle", "bottom"})
_ALLOWED_ROTATIONS = frozenset({0, 90, 180, 270})


class LayoutPreservingError(RuntimeError):
    """Raised when a complete, layout-faithful artifact cannot be produced."""


@dataclass(frozen=True)
class LayoutRenderResult:
    pdf_path: Path
    report_path: Path
    page_count: int
    element_count: int
    shrunk_elements: int
    background_dpi: int


@dataclass(frozen=True)
class LayoutVerificationResult:
    report_path: Path
    source_page_count: int
    output_page_count: int
    continuation_pages: int
    page_dimensions_match: bool
    minimum_background_similarity: float
    valid: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LayoutPreservingError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LayoutPreservingError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _numeric_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _positive_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return None
    return width, height


def _map_ledger_bbox(
    value: Any,
    page_size: Any,
    page_rect: fitz.Rect,
    *,
    cover_margin: float = 0.0,
) -> tuple[float, float, float, float]:
    bbox = _numeric_bbox(value)
    size = _positive_pair(page_size)
    if bbox is None or size is None:
        raise LayoutPreservingError("Translation ledger region is missing a valid bbox or page_size")
    x0, y0, x1, y1 = bbox
    width, height = size
    mapped = fitz.Rect(
        page_rect.x0 + x0 * page_rect.width / width,
        page_rect.y0 + y0 * page_rect.height / height,
        page_rect.x0 + x1 * page_rect.width / width,
        page_rect.y0 + y1 * page_rect.height / height,
    )
    mapped.x0 -= cover_margin
    mapped.y0 -= cover_margin
    mapped.x1 += cover_margin
    mapped.y1 += cover_margin
    mapped &= page_rect
    if mapped.is_empty or mapped.width <= 1 or mapped.height <= 1:
        raise LayoutPreservingError("Translation ledger region maps to an empty PDF box")
    return tuple(round(float(item), 3) for item in mapped)  # type: ignore[return-value]


def _pixel_rgb(pixmap: fitz.Pixmap, x: int, y: int) -> tuple[int, int, int]:
    x = max(0, min(pixmap.width - 1, x))
    y = max(0, min(pixmap.height - 1, y))
    offset = (y * pixmap.width + x) * pixmap.n
    samples = pixmap.samples
    if pixmap.n == 1:
        value = samples[offset]
        return value, value, value
    return samples[offset], samples[offset + 1], samples[offset + 2]


def _sample_background(page: fitz.Page, bbox: tuple[float, float, float, float]) -> str:
    """Estimate the local paper/cell color from a narrow ring around a box."""
    pixmap = page.get_pixmap(dpi=72, colorspace=fitz.csRGB, alpha=False)
    scale_x = pixmap.width / max(1.0, page.rect.width)
    scale_y = pixmap.height / max(1.0, page.rect.height)
    x0, y0, x1, y1 = bbox
    px0 = int(round(x0 * scale_x))
    py0 = int(round(y0 * scale_y))
    px1 = int(round(x1 * scale_x))
    py1 = int(round(y1 * scale_y))
    samples: list[tuple[int, int, int]] = []
    step_x = max(1, (px1 - px0) // 24)
    step_y = max(1, (py1 - py0) // 16)
    for offset in (2, 3, 4):
        for x in range(px0, px1 + 1, step_x):
            samples.append(_pixel_rgb(pixmap, x, py0 - offset))
            samples.append(_pixel_rgb(pixmap, x, py1 + offset))
        for y in range(py0, py1 + 1, step_y):
            samples.append(_pixel_rgb(pixmap, px0 - offset, y))
            samples.append(_pixel_rgb(pixmap, px1 + offset, y))
    if not samples:
        samples = [(255, 255, 255)]
    rgb = tuple(int(round(statistics.median(channel))) for channel in zip(*samples, strict=True))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _heading_style(bbox: tuple[float, float, float, float], page_rect: fitz.Rect) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    requested = round(min(18.0, max(10.0, height * 0.58)), 2)
    center = (x0 + x1) * 0.5
    centered = abs(center - page_rect.width * 0.5) <= page_rect.width * 0.08
    return {
        "font": "bold",
        "font_size": requested,
        "min_font_size": round(max(8.0, requested * 0.65), 2),
        "line_height": 1.08,
        "align": "center" if centered else "left",
        "valign": "middle",
    }


def generate_layout_from_ledger(
    source_pdf: str | Path,
    ledger_json: str | Path,
    output_json: str | Path,
    *,
    background_dpi: int = 300,
    body_font_size: float = 9.0,
    body_min_font_size: float = 6.0,
) -> Path:
    """Create a PDF Translator schema-v1 layout from the persisted OCR ledger."""
    source_path = Path(source_pdf).resolve()
    ledger_path = Path(ledger_json).resolve()
    output_path = Path(output_json).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if background_dpi < 144:
        raise LayoutPreservingError("Layout background DPI must be at least 144")
    ledger = _read_json(ledger_path)
    if ledger.get("schema_version") != 1:
        raise LayoutPreservingError("Unsupported OCR translation ledger schema")

    with fitz.open(str(source_path)) as source:
        pages: list[dict[str, Any]] = [
            {
                "page": index + 1,
                "width_pt": round(page.rect.width, 3),
                "height_pt": round(page.rect.height, 3),
                "translation_complete": True,
                "elements": [],
            }
            for index, page in enumerate(source)
        ]
        counters = {"body": 0, "heading": 0, "table": 0}

        for role, ledger_key in (("body", "body_regions"), ("heading", "heading_regions")):
            for raw in ledger.get(ledger_key) or []:
                if not isinstance(raw, dict):
                    raise LayoutPreservingError(f"Ledger {ledger_key} entry must be an object")
                try:
                    page_number = int(raw.get("page"))
                except (TypeError, ValueError) as exc:
                    raise LayoutPreservingError(f"Ledger {ledger_key} entry has no page") from exc
                if not 1 <= page_number <= source.page_count:
                    raise LayoutPreservingError(f"Ledger entry references missing page {page_number}")
                page = source[page_number - 1]
                bbox = _map_ledger_bbox(
                    raw.get("bbox"),
                    raw.get("page_size"),
                    page.rect,
                    cover_margin=0.8,
                )
                source_text = str(raw.get("source") or "").strip()
                target_text = str(raw.get("target") or "").strip()
                if not source_text or not target_text:
                    raise LayoutPreservingError(
                        f"Ledger {role} entry on page {page_number} is missing source or target text"
                    )
                counters[role] += 1
                style = (
                    _heading_style(bbox, page.rect)
                    if role == "heading"
                    else {
                        "font": "regular",
                        "font_size": float(body_font_size),
                        "min_font_size": float(body_min_font_size),
                        "line_height": 1.2,
                        "align": "left",
                        "valign": "top",
                    }
                )
                element = {
                    "id": f"p{page_number}-{role}-{counters[role]:03d}",
                    "type": "text",
                    "role": role,
                    "bbox": list(bbox),
                    "source": source_text,
                    "text": target_text,
                    **style,
                    "padding": 1.0,
                    "fill": _sample_background(page, bbox),
                    "text_color": "#000000",
                    "rotation": 0,
                    "protected_literals": list(raw.get("protected_literals") or []),
                }
                pages[page_number - 1]["elements"].append(element)

        for raw in ledger.get("tables") or []:
            if not isinstance(raw, dict):
                raise LayoutPreservingError("Ledger table entry must be an object")
            try:
                page_number = int(raw.get("page"))
            except (TypeError, ValueError) as exc:
                raise LayoutPreservingError("Ledger table entry has no page") from exc
            if not 1 <= page_number <= source.page_count:
                raise LayoutPreservingError(f"Ledger table references missing page {page_number}")
            page = source[page_number - 1]
            bbox = _map_ledger_bbox(raw.get("bbox"), raw.get("page_size"), page.rect)
            preserved = bool(raw.get("preserved_as_image"))
            expected_text = []
            if not preserved:
                for cell in raw.get("cells") or []:
                    if not isinstance(cell, dict):
                        continue
                    target = str(cell.get("target") or "").strip()
                    if target:
                        expected_text.append(target)
            counters["table"] += 1
            pages[page_number - 1]["elements"].append(
                {
                    "id": f"p{page_number}-table-{counters['table']:03d}",
                    "type": "cover",
                    "role": "preserved_table" if preserved else "table",
                    "bbox": list(bbox),
                    "source": "\n".join(
                        str(cell.get("source") or "")
                        for cell in raw.get("cells") or []
                        if isinstance(cell, dict)
                    ),
                    "fill": None,
                    "defer_to": "ocr_vector_table_redraw",
                    "continuation_allowed": not preserved,
                    "expected_text": expected_text,
                    "rows": int(raw.get("rows") or 0),
                    "columns": int(raw.get("columns") or 0),
                }
            )

        for page in pages:
            page["elements"].sort(
                key=lambda item: (
                    float(item["bbox"][1]),
                    float(item["bbox"][0]),
                    item["id"],
                )
            )
        layout = {
            "schema_version": 1,
            "source_pdf": str(source_path),
            "source_sha256": _sha256(source_path),
            "background_dpi": int(background_dpi),
            "generated_from": {
                "type": "ocr_translation_ledger",
                "path": str(ledger_path),
                "sha256": _sha256(ledger_path),
            },
            "pages": pages,
        }
    return _write_json(output_path, layout)


def _layout_pages(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw_pages = spec.get("pages")
    if not isinstance(raw_pages, list):
        raise LayoutPreservingError("Layout pages must be an array")
    result: dict[int, dict[str, Any]] = {}
    for raw in raw_pages:
        if not isinstance(raw, dict):
            raise LayoutPreservingError("Layout page must be an object")
        try:
            page_number = int(raw.get("page"))
        except (TypeError, ValueError) as exc:
            raise LayoutPreservingError("Layout page has no valid page number") from exc
        if page_number in result:
            raise LayoutPreservingError(f"Duplicate layout page {page_number}")
        result[page_number] = raw
    return result


def _validate_layout(source_path: Path, source: fitz.Document, spec: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if spec.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    expected_sha = str(spec.get("source_sha256") or "")
    if not expected_sha or expected_sha != _sha256(source_path):
        issues.append("layout source_sha256 does not match the source PDF")
    try:
        pages = _layout_pages(spec)
    except LayoutPreservingError as exc:
        return [str(exc)]
    expected_pages = set(range(1, source.page_count + 1))
    if set(pages) != expected_pages:
        missing = sorted(expected_pages - set(pages))
        extra = sorted(set(pages) - expected_pages)
        if missing:
            issues.append("layout is missing page(s): " + ", ".join(map(str, missing)))
        if extra:
            issues.append("layout has extra page(s): " + ", ".join(map(str, extra)))
    seen_ids: set[str] = set()
    for page_number, entry in pages.items():
        if not 1 <= page_number <= source.page_count:
            continue
        source_page = source[page_number - 1]
        if entry.get("translation_complete") is not True:
            issues.append(f"page {page_number} is not marked translation_complete")
        for field, actual in (("width_pt", source_page.rect.width), ("height_pt", source_page.rect.height)):
            try:
                declared = float(entry.get(field))
            except (TypeError, ValueError):
                issues.append(f"page {page_number} has no valid {field}")
                continue
            if abs(declared - actual) > 0.2:
                issues.append(f"page {page_number} {field} differs from source")
        elements = entry.get("elements")
        if not isinstance(elements, list):
            issues.append(f"page {page_number} elements must be an array")
            continue
        for index, element in enumerate(elements, start=1):
            label = f"page {page_number} element {index}"
            if not isinstance(element, dict):
                issues.append(f"{label} must be an object")
                continue
            element_id = element.get("id")
            if not isinstance(element_id, str) or not element_id.strip():
                issues.append(f"{label} has no id")
            elif element_id in seen_ids:
                issues.append(f"duplicate element id {element_id}")
            else:
                seen_ids.add(element_id)
            element_type = element.get("type")
            if element_type not in {"text", "cover"}:
                issues.append(f"{label} has unsupported type {element_type!r}")
            bbox = _numeric_bbox(element.get("bbox"))
            if bbox is None:
                issues.append(f"{label} has no valid bbox")
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                issues.append(f"{label} bbox has no area")
            if x0 < 0 or y0 < 0 or x1 > source_page.rect.width or y1 > source_page.rect.height:
                issues.append(f"{label} bbox is outside the source page")
            fill = element.get("fill")
            if fill is not None and (not isinstance(fill, str) or not _HEX_COLOR_RE.fullmatch(fill)):
                issues.append(f"{label} has an invalid fill color")
            if element_type == "text":
                if not str(element.get("text") or "").strip():
                    issues.append(f"{label} has no translated text")
                if element.get("font", "regular") not in _ALLOWED_FONTS:
                    issues.append(f"{label} has an unsupported font")
                if element.get("align", "left") not in _ALLOWED_ALIGNMENTS:
                    issues.append(f"{label} has an unsupported alignment")
                if element.get("valign", "top") not in _ALLOWED_VALIGNS:
                    issues.append(f"{label} has an unsupported vertical alignment")
                if element.get("rotation", 0) not in _ALLOWED_ROTATIONS:
                    issues.append(f"{label} has an unsupported rotation")
                try:
                    requested = float(element.get("font_size", 10))
                    minimum = float(element.get("min_font_size", 6))
                    if requested <= 0 or minimum <= 0 or minimum > requested:
                        raise ValueError
                except (TypeError, ValueError):
                    issues.append(f"{label} has invalid font sizes")
    return issues


def _color(value: str) -> tuple[float, float, float]:
    return tuple(int(value[index : index + 2], 16) / 255 for index in (1, 3, 5))  # type: ignore[return-value]


def _builtin_cjk_font(text: str) -> str | None:
    if re.search(r"[\uac00-\ud7af]", text):
        return "korea-s"
    if re.search(r"[\u3040-\u30ff]", text):
        return "japan-s"
    if re.search(r"[\u3400-\u9fff]", text):
        return "china-s"
    return None


def _fit_textbox(
    text: str,
    rect: fitz.Rect,
    *,
    fontfile: Path | None,
    fontname: str,
    requested: float,
    minimum: float,
    line_height: float,
    align: int,
    rotation: int,
) -> tuple[float, float]:
    current = requested
    last_spare = -1.0
    while current + 1e-6 >= minimum:
        temporary = fitz.open()
        try:
            page = temporary.new_page(width=max(10.0, rect.width + 4), height=max(10.0, rect.height + 4))
            test_rect = fitz.Rect(2, 2, 2 + rect.width, 2 + rect.height)
            font_args = {"fontname": fontname}
            if fontfile is not None:
                font_args["fontfile"] = str(fontfile)
            last_spare = float(
                page.insert_textbox(
                    test_rect,
                    text,
                    **font_args,
                    fontsize=current,
                    lineheight=line_height,
                    align=align,
                    rotate=rotation,
                )
            )
        finally:
            temporary.close()
        if last_spare >= -0.05:
            return round(current, 2), max(0.0, last_spare)
        current = round(current - 0.25, 2)
    raise LayoutPreservingError(
        f"Translated text requires more space than bbox {tuple(round(value, 2) for value in rect)} "
        f"at minimum font size {minimum:.2f} pt"
    )


def _save_document(document: fitz.Document, output: Path, *, label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-{label}-",
        suffix=".pdf",
        dir=output.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        document.save(str(temporary), garbage=4, deflate=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def render_layout_pdf(
    source_pdf: str | Path,
    layout_json: str | Path,
    output_pdf: str | Path,
    *,
    regular_font_path: str | Path,
    bold_font_path: str | Path,
    report_path: str | Path | None = None,
) -> LayoutRenderResult:
    """Render the source background plus translated vector elements."""
    source_path = Path(source_pdf).resolve()
    layout_path = Path(layout_json).resolve()
    output_path = Path(output_pdf).resolve()
    regular_font = Path(regular_font_path).resolve()
    bold_font = Path(bold_font_path).resolve()
    for path, label in (
        (source_path, "source PDF"),
        (layout_path, "layout JSON"),
        (regular_font, "regular font"),
        (bold_font, "bold font"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    spec = _read_json(layout_path)
    background_dpi = int(spec.get("background_dpi") or 300)
    if background_dpi < 144:
        raise LayoutPreservingError("Layout background DPI must be at least 144")

    source = fitz.open(str(source_path))
    output = fitz.open()
    fit_report: list[dict[str, Any]] = []
    try:
        issues = _validate_layout(source_path, source, spec)
        if issues:
            raise LayoutPreservingError("Invalid layout:\n- " + "\n- ".join(issues))
        pages = _layout_pages(spec)
        for page_number, source_page in enumerate(source, start=1):
            page = output.new_page(width=source_page.rect.width, height=source_page.rect.height)
            background = source_page.get_pixmap(dpi=background_dpi, colorspace=fitz.csRGB, alpha=False)
            page.insert_image(page.rect, pixmap=background, keep_proportion=False, overlay=False)
            for element in pages[page_number].get("elements") or []:
                bbox = fitz.Rect(_numeric_bbox(element["bbox"]))
                fill = element.get("fill")
                if fill is not None:
                    page.draw_rect(bbox, color=None, fill=_color(fill), width=0, overlay=True)
                if element.get("type") != "text":
                    fit_report.append(
                        {
                            "page": page_number,
                            "id": element["id"],
                            "type": "cover",
                            "role": element.get("role"),
                            "bbox": list(element["bbox"]),
                            "rendered": fill is not None,
                        }
                    )
                    continue
                padding = float(element.get("padding", 1.5))
                inner = fitz.Rect(
                    bbox.x0 + padding,
                    bbox.y0 + padding,
                    bbox.x1 - padding,
                    bbox.y1 - padding,
                )
                if inner.is_empty:
                    raise LayoutPreservingError(f"Element {element['id']} padding leaves no text area")
                alignment = {
                    "left": fitz.TEXT_ALIGN_LEFT,
                    "center": fitz.TEXT_ALIGN_CENTER,
                    "right": fitz.TEXT_ALIGN_RIGHT,
                    "justify": fitz.TEXT_ALIGN_JUSTIFY,
                }[element.get("align", "left")]
                token = str(element.get("font", "regular"))
                text = str(element["text"])
                builtin_cjk = _builtin_cjk_font(text)
                # PyMuPDF's bundled simplified-Chinese fonts preserve Unicode
                # extraction exactly.  Some CJK TTCs expose compatibility
                # ideographs (for example 行 -> 行), which makes searchable-text
                # verification unreliable even though the glyph is visible.
                fontfile = None if builtin_cjk else (bold_font if token == "bold" else regular_font)
                fontname = builtin_cjk or ("layoutbold" if token == "bold" else "layoutregular")
                requested = float(element.get("font_size", 10))
                minimum = float(element.get("min_font_size", 6))
                line_height = float(element.get("line_height", 1.12))
                rotation = int(element.get("rotation", 0))
                used, spare = _fit_textbox(
                    text,
                    inner,
                    fontfile=fontfile,
                    fontname=fontname,
                    requested=requested,
                    minimum=minimum,
                    line_height=line_height,
                    align=alignment,
                    rotation=rotation,
                )
                valign = element.get("valign", "top")
                if valign == "middle":
                    inner.y0 += spare * 0.5
                elif valign == "bottom":
                    inner.y0 += spare
                font_args = {"fontname": fontname}
                if fontfile is not None:
                    font_args["fontfile"] = str(fontfile)
                inserted = float(
                    page.insert_textbox(
                        inner,
                        text,
                        **font_args,
                        fontsize=used,
                        lineheight=line_height,
                        align=alignment,
                        color=_color(str(element.get("text_color") or "#000000")),
                        rotate=rotation,
                        overlay=True,
                    )
                )
                if inserted < -0.05:
                    raise LayoutPreservingError(f"Element {element['id']} overflowed after preflight")
                fit_report.append(
                    {
                        "page": page_number,
                        "id": element["id"],
                        "type": "text",
                        "role": element.get("role"),
                        "bbox": list(element["bbox"]),
                        "requested_font_size": requested,
                        "used_font_size": used,
                        "shrunk": used < requested,
                        "fits": True,
                    }
                )
        _save_document(output, output_path, label="layout")
    finally:
        output.close()
        source.close()

    resolved_report = (
        Path(report_path).resolve()
        if report_path is not None
        else output_path.with_name(f"{output_path.stem}.layout-render.json")
    )
    report = {
        "schema_version": 1,
        "source_pdf": str(source_path),
        "source_sha256": _sha256(source_path),
        "layout": str(layout_path),
        "layout_sha256": _sha256(layout_path),
        "output_pdf": str(output_path),
        "output_sha256": _sha256(output_path),
        "page_count": len(_layout_pages(spec)),
        "background_dpi": background_dpi,
        "elements": fit_report,
        "overflows": [],
    }
    _write_json(resolved_report, report)
    return LayoutRenderResult(
        pdf_path=output_path,
        report_path=resolved_report,
        page_count=report["page_count"],
        element_count=len(fit_report),
        shrunk_elements=sum(bool(item.get("shrunk")) for item in fit_report),
        background_dpi=background_dpi,
    )


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _merged_intervals(intervals: Iterable[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    clipped = sorted(
        (max(0, start), min(limit, end))
        for start, end in intervals
        if end > 0 and start < limit and end > start
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
    return merged


def _outside_similarity(
    source_page: fitz.Page,
    output_page: fitz.Page,
    boxes: Iterable[tuple[float, float, float, float]],
    *,
    dpi: int,
    margin_pt: float,
) -> dict[str, float]:
    source_pixmap = source_page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    output_pixmap = output_page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    if (source_pixmap.width, source_pixmap.height, source_pixmap.n) != (
        output_pixmap.width,
        output_pixmap.height,
        output_pixmap.n,
    ):
        return {"similarity": 0.0, "mean_absolute_difference": 255.0}
    scale_x = source_pixmap.width / max(1.0, source_page.rect.width)
    scale_y = source_pixmap.height / max(1.0, source_page.rect.height)
    rects = [
        (
            int(math.floor((x0 - margin_pt) * scale_x)),
            int(math.floor((y0 - margin_pt) * scale_y)),
            int(math.ceil((x1 + margin_pt) * scale_x)),
            int(math.ceil((y1 + margin_pt) * scale_y)),
        )
        for x0, y0, x1, y1 in boxes
    ]
    difference_sum = 0
    compared = 0
    channels = source_pixmap.n
    width = source_pixmap.width
    for y in range(source_pixmap.height):
        hidden = _merged_intervals(
            ((x0, x1) for x0, y0, x1, y1 in rects if y0 <= y < y1),
            width,
        )
        cursor = 0
        visible: list[tuple[int, int]] = []
        for start, end in hidden:
            if cursor < start:
                visible.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < width:
            visible.append((cursor, width))
        for start, end in visible:
            byte_start = (y * width + start) * channels
            byte_end = (y * width + end) * channels
            source_slice = source_pixmap.samples[byte_start:byte_end]
            output_slice = output_pixmap.samples[byte_start:byte_end]
            difference_sum += sum(abs(left - right) for left, right in zip(source_slice, output_slice, strict=True))
            compared += byte_end - byte_start
    if compared == 0:
        return {"similarity": 1.0, "mean_absolute_difference": 0.0}
    mean = difference_sum / compared
    return {"similarity": 1 - mean / 255, "mean_absolute_difference": mean}


def verify_layout_output(
    source_pdf: str | Path,
    layout_json: str | Path,
    output_pdf: str | Path,
    report_json: str | Path,
    *,
    source_page_indices: Iterable[int] | None = None,
    continuation_page_groups: Iterable[Iterable[int]] | None = None,
    extra_overlay_regions: Iterable[tuple[int, tuple[float, float, float, float]]] = (),
    repeated_header_texts: Iterable[str] = (),
    similarity_dpi: int = 36,
    similarity_margin: float = 2.0,
    min_similarity: float = 0.985,
) -> LayoutVerificationResult:
    """Strictly verify layout fidelity while allowing explicit table continuations."""
    source_path = Path(source_pdf).resolve()
    layout_path = Path(layout_json).resolve()
    output_path = Path(output_pdf).resolve()
    report_path = Path(report_json).resolve()
    for path, label in ((source_path, "source PDF"), (layout_path, "layout JSON"), (output_path, "output PDF")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    spec = _read_json(layout_path)
    source = fitz.open(str(source_path))
    output = fitz.open(str(output_path))
    issues: list[str] = []
    page_reports: list[dict[str, Any]] = []
    try:
        issues.extend(_validate_layout(source_path, source, spec))
        pages = _layout_pages(spec)
        source_map = tuple(source_page_indices) if source_page_indices is not None else tuple(range(source.page_count))
        groups = (
            tuple(tuple(int(index) for index in group) for group in continuation_page_groups)
            if continuation_page_groups is not None
            else tuple(() for _ in range(source.page_count))
        )
        if len(source_map) != source.page_count:
            issues.append("source page map length differs from source page count")
        if len(groups) != source.page_count:
            issues.append("continuation page group count differs from source page count")
            groups = tuple(() for _ in range(source.page_count))
        physical = [*source_map, *(index for group in groups for index in group)]
        if len(set(physical)) != len(physical) or set(physical) != set(range(output.page_count)):
            issues.append("source/continuation page map does not cover each output page exactly once")
        expected_count = source.page_count + sum(len(group) for group in groups)
        if output.page_count != expected_count:
            issues.append(f"output has {output.page_count} pages; expected {expected_count}")

        overlays: dict[int, list[tuple[float, float, float, float]]] = {}
        for page_idx, bbox in extra_overlay_regions:
            overlays.setdefault(int(page_idx), []).append(tuple(float(item) for item in bbox))

        dimensions_match = True
        for logical_idx in range(source.page_count):
            if logical_idx >= len(source_map) or source_map[logical_idx] not in range(output.page_count):
                dimensions_match = False
                continue
            output_idx = source_map[logical_idx]
            source_page = source[logical_idx]
            output_page = output[output_idx]
            page_issues: list[str] = []
            if (
                abs(source_page.rect.width - output_page.rect.width) > 0.2
                or abs(source_page.rect.height - output_page.rect.height) > 0.2
            ):
                page_issues.append("source page dimensions differ")
                dimensions_match = False
            for continuation_idx in groups[logical_idx] if logical_idx < len(groups) else ():
                if continuation_idx not in range(output.page_count):
                    page_issues.append(f"continuation page index {continuation_idx} is invalid")
                    dimensions_match = False
                    continue
                continuation = output[continuation_idx]
                if (
                    abs(source_page.rect.width - continuation.rect.width) > 0.2
                    or abs(source_page.rect.height - continuation.rect.height) > 0.2
                ):
                    page_issues.append(f"continuation page {continuation_idx + 1} dimensions differ")
                    dimensions_match = False

            output_text = _compact(output_page.get_text("text"))
            grouped_text = output_text + "".join(
                _compact(output[index].get_text("text"))
                for index in groups[logical_idx]
                if 0 <= index < output.page_count
            )
            missing_elements: list[str] = []
            missing_table_text: list[str] = []
            expected_table_text: list[str] = []
            layout_boxes: list[tuple[float, float, float, float]] = []
            for element in pages[logical_idx + 1].get("elements") or []:
                bbox = _numeric_bbox(element.get("bbox"))
                if element.get("type") == "text":
                    if bbox is not None:
                        layout_boxes.append(bbox)
                    expected = _compact(str(element.get("text") or ""))
                    local_text = (
                        _compact(output_page.get_textbox(fitz.Rect(bbox)))
                        if bbox is not None
                        else ""
                    )
                    if expected and expected not in local_text:
                        missing_elements.append(str(element.get("id") or "unnamed"))
                elif element.get("role") == "table":
                    if bbox is not None:
                        layout_boxes.append(bbox)
                    for expected_value in element.get("expected_text") or []:
                        expected = _compact(str(expected_value))
                        if expected:
                            expected_table_text.append(expected)
                elif element.get("fill") is not None and bbox is not None:
                    layout_boxes.append(bbox)
            for expected, count in Counter(expected_table_text).items():
                missing_count = count - grouped_text.count(expected)
                if missing_count > 0:
                    missing_table_text.extend(expected for _ in range(missing_count))
            if missing_elements:
                page_issues.append("searchable translation missing for: " + ", ".join(missing_elements))
            if missing_table_text:
                page_issues.append("searchable table text missing: " + ", ".join(missing_table_text[:12]))

            out_of_bounds = []
            for physical_idx in (output_idx, *groups[logical_idx]):
                if physical_idx not in range(output.page_count):
                    continue
                page = output[physical_idx]
                for block in page.get_text("blocks"):
                    x0, y0, x1, y1 = (float(value) for value in block[:4])
                    if x0 < -0.5 or y0 < -0.5 or x1 > page.rect.width + 0.5 or y1 > page.rect.height + 0.5:
                        out_of_bounds.append([physical_idx + 1, x0, y0, x1, y1])
            if out_of_bounds:
                page_issues.append("text lies outside a media box")

            similarity_boxes = [*layout_boxes, *overlays.get(output_idx, [])]
            similarity = _outside_similarity(
                source_page,
                output_page,
                similarity_boxes,
                dpi=similarity_dpi,
                margin_pt=similarity_margin,
            )
            if similarity["similarity"] < min_similarity:
                page_issues.append(
                    f"background similarity {similarity['similarity']:.5f} is below {min_similarity:.5f}"
                )
            issues.extend(f"Page {logical_idx + 1}: {issue}" for issue in page_issues)
            page_reports.append(
                {
                    "source_page": logical_idx + 1,
                    "output_page": output_idx + 1,
                    "continuation_pages": [index + 1 for index in groups[logical_idx]],
                    "source_size_pt": [round(source_page.rect.width, 3), round(source_page.rect.height, 3)],
                    "output_size_pt": [round(output_page.rect.width, 3), round(output_page.rect.height, 3)],
                    "missing_text_elements": missing_elements,
                    "missing_table_text": missing_table_text,
                    "out_of_bounds_blocks": out_of_bounds,
                    "outside_overlay_similarity": similarity,
                    "issues": page_issues,
                }
            )

        continuation_text = "".join(
            _compact(output[index].get_text("text"))
            for group in groups
            for index in group
            if 0 <= index < output.page_count
        )
        header_counts: dict[str, int] = {}
        for header in repeated_header_texts:
            compact_header = _compact(str(header))
            if compact_header:
                header_counts[compact_header] = header_counts.get(compact_header, 0) + 1
        for header, count in header_counts.items():
            if continuation_text.count(header) < count:
                issues.append("one or more continuation pages are missing a repeated table header")
                break

        similarities = [
            float(page["outside_overlay_similarity"]["similarity"])
            for page in page_reports
        ]
        minimum_similarity = min(similarities, default=1.0)
        report = {
            "schema_version": 1,
            "source_pdf": str(source_path),
            "source_sha256": _sha256(source_path),
            "layout": str(layout_path),
            "layout_sha256": _sha256(layout_path),
            "translated_pdf": str(output_path),
            "translated_sha256": _sha256(output_path),
            "source_page_count": source.page_count,
            "output_page_count": output.page_count,
            "continuation_page_count": sum(len(group) for group in groups),
            "source_page_indices": list(source_map),
            "continuation_page_groups": [list(group) for group in groups],
            "page_dimensions_match": dimensions_match,
            "minimum_background_similarity": minimum_similarity,
            "minimum_required_similarity": min_similarity,
            "pages": page_reports,
            "issues": issues,
            "valid": not issues,
        }
        _write_json(report_path, report)
    finally:
        output.close()
        source.close()

    result = LayoutVerificationResult(
        report_path=report_path,
        source_page_count=int(report["source_page_count"]),
        output_page_count=int(report["output_page_count"]),
        continuation_pages=int(report["continuation_page_count"]),
        page_dimensions_match=bool(report["page_dimensions_match"]),
        minimum_background_similarity=float(report["minimum_background_similarity"]),
        valid=bool(report["valid"]),
    )
    if issues:
        raise LayoutPreservingError("Strict layout verification failed:\n- " + "\n- ".join(issues))
    return result


__all__ = [
    "LayoutPreservingError",
    "LayoutRenderResult",
    "LayoutVerificationResult",
    "generate_layout_from_ledger",
    "render_layout_pdf",
    "verify_layout_output",
]
