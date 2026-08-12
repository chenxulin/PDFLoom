#!/usr/bin/env python3
"""Build and validate layout-preserving translated PDFs.

The source page is rasterized as the visual background. Original-language text
regions are covered and translated text is placed in explicit page-coordinate
boxes. The result keeps the source page geometry while exposing searchable
target-language text.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS = SKILL_DIR / "requirements.txt"


def dependency_error(package: str) -> SystemExit:
    return SystemExit(
        f"Missing dependency: {package}. Install the skill dependencies with:\n"
        f"  python3 -m pip install -r {REQUIREMENTS}"
    )


def require_pymupdf():
    try:
        import pymupdf  # type: ignore
    except ModuleNotFoundError as exc:
        raise dependency_error("PyMuPDF") from exc
    return pymupdf


def require_reportlab():
    try:
        import reportlab  # noqa: F401
    except ModuleNotFoundError as exc:
        raise dependency_error("reportlab") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Output already exists: {path}. Pass --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Refusing to write into non-empty directory: {path}\n"
            "Choose an empty directory or pass --force to replace matching files."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}.")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def numeric_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value):
        return None
    return [float(item) for item in value]


def page_entries(spec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = spec.get("pages")
    if not isinstance(pages, list):
        raise SystemExit("Layout specification must contain a pages array.")
    result: dict[int, dict[str, Any]] = {}
    for entry in pages:
        if not isinstance(entry, dict) or not isinstance(entry.get("page"), int):
            raise SystemExit("Each layout page must be an object with an integer page field.")
        number = entry["page"]
        if number in result:
            raise SystemExit(f"Duplicate layout page: {number}")
        result[number] = entry
    return result


def validate_spec(
    source_path: Path,
    source_document: Any,
    spec: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if spec.get("schema_version") != 1:
        issues.append("schema_version must be 1.")
    expected_sha = spec.get("source_sha256")
    if expected_sha and expected_sha != sha256(source_path):
        issues.append("The layout specification source_sha256 does not match the supplied PDF.")

    try:
        entries = page_entries(spec)
    except SystemExit as exc:
        return [str(exc)]

    missing_pages = [
        page_number
        for page_number in range(1, source_document.page_count + 1)
        if page_number not in entries
    ]
    if missing_pages:
        issues.append(
            "Layout specification is missing source page(s): "
            + ", ".join(str(number) for number in missing_pages)
        )

    seen_ids: set[str] = set()
    allowed_rotations = {0, 90, 180, 270}
    for page_number, entry in entries.items():
        if page_number < 1 or page_number > source_document.page_count:
            issues.append(f"Layout page {page_number} is outside the source page range.")
            continue
        source_page = source_document[page_number - 1]
        if entry.get("translation_complete") is not True:
            issues.append(
                f"Layout page {page_number} must set translation_complete to true after visual review."
            )
        declared_width = entry.get("width_pt")
        declared_height = entry.get("height_pt")
        if declared_width is not None and abs(float(declared_width) - source_page.rect.width) > 0.2:
            issues.append(f"Layout page {page_number} width_pt does not match the source.")
        if declared_height is not None and abs(float(declared_height) - source_page.rect.height) > 0.2:
            issues.append(f"Layout page {page_number} height_pt does not match the source.")

        elements = entry.get("elements", [])
        if not isinstance(elements, list):
            issues.append(f"Layout page {page_number} elements must be an array.")
            continue
        for index, element in enumerate(elements, start=1):
            label = f"page {page_number} element {index}"
            if not isinstance(element, dict):
                issues.append(f"{label} must be an object.")
                continue
            element_id = element.get("id")
            if not isinstance(element_id, str) or not element_id.strip():
                issues.append(f"{label} must have a non-empty id.")
            elif element_id in seen_ids:
                issues.append(f"Duplicate element id: {element_id}")
            else:
                seen_ids.add(element_id)

            element_type = element.get("type", "text")
            if element_type not in {"text", "cover"}:
                issues.append(f"{label} has unsupported type {element_type!r}.")

            bbox = numeric_bbox(element.get("bbox"))
            if bbox is None:
                issues.append(f"{label} bbox must contain four finite numbers.")
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                issues.append(f"{label} bbox must have positive width and height.")
            if x0 < 0 or y0 < 0 or x1 > source_page.rect.width or y1 > source_page.rect.height:
                issues.append(f"{label} bbox is outside the source page.")

            if element_type == "text":
                text = element.get("text")
                if not isinstance(text, str) or not text.strip():
                    issues.append(f"{label} requires non-empty translated text.")
                font_size = element.get("font_size", 10)
                min_font_size = element.get("min_font_size", 6)
                if not isinstance(font_size, (int, float)) or float(font_size) <= 0:
                    issues.append(f"{label} font_size must be positive.")
                if not isinstance(min_font_size, (int, float)) or float(min_font_size) <= 0:
                    issues.append(f"{label} min_font_size must be positive.")
                if (
                    isinstance(font_size, (int, float))
                    and isinstance(min_font_size, (int, float))
                    and float(min_font_size) > float(font_size)
                ):
                    issues.append(f"{label} min_font_size exceeds font_size.")
                if element.get("align", "left") not in {"left", "center", "right", "justify"}:
                    issues.append(f"{label} has an unsupported align value.")
                if element.get("valign", "top") not in {"top", "middle", "bottom"}:
                    issues.append(f"{label} has an unsupported valign value.")
                if element.get("font", "regular") not in {"regular", "bold", "mono", "cjk"}:
                    issues.append(f"{label} has an unsupported font token.")
                rotation = element.get("rotation", 0)
                if rotation not in allowed_rotations:
                    issues.append(f"{label} rotation must be one of 0, 90, 180, or 270.")
    return issues


def register_fonts() -> dict[str, str]:
    require_reportlab()
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = {
        "regular": "Helvetica",
        "bold": "Helvetica-Bold",
        "mono": "Courier",
        "cjk": "Helvetica",
    }
    sans_candidates = [
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
        ),
    ]
    for regular_path, bold_path in sans_candidates:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont("LayoutSans", str(regular_path)))
            pdfmetrics.registerFont(TTFont("LayoutSans-Bold", str(bold_path)))
            fonts["regular"] = "LayoutSans"
            fonts["bold"] = "LayoutSans-Bold"
            break

    mono_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if mono_path.exists():
        pdfmetrics.registerFont(TTFont("LayoutMono", str(mono_path)))
        fonts["mono"] = "LayoutMono"

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        fonts["cjk"] = "STSong-Light"
    except Exception:
        fonts["cjk"] = fonts["regular"]
    return fonts


def parse_color(value: Any, default: str, allow_none: bool = False):
    from reportlab.lib import colors

    if value is None and allow_none:
        return None
    color_value = default if value is None else value
    if not isinstance(color_value, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color_value):
        raise ValueError(f"Expected a color in #RRGGBB format, found {color_value!r}.")
    return colors.HexColor(color_value)


def paragraph_for(
    element: dict[str, Any],
    font_size: float,
    fonts: dict[str, str],
):
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    alignments = {
        "left": TA_LEFT,
        "center": TA_CENTER,
        "right": TA_RIGHT,
        "justify": TA_JUSTIFY,
    }
    font_token = element.get("font", "regular")
    line_height = float(element.get("line_height", 1.12))
    escaped = html.escape(str(element.get("text", "")), quote=False).replace("\n", "<br/>")
    style = ParagraphStyle(
        name=f"layout-{element.get('id', 'element')}",
        fontName=fonts[font_token],
        fontSize=font_size,
        leading=font_size * line_height,
        alignment=alignments[element.get("align", "left")],
        textColor=parse_color(element.get("text_color"), "#000000"),
        leftIndent=0,
        rightIndent=0,
        firstLineIndent=0,
        spaceBefore=0,
        spaceAfter=0,
        splitLongWords=True,
        wordWrap="CJK" if font_token == "cjk" else None,
        allowWidows=1,
        allowOrphans=1,
    )
    return Paragraph(escaped, style)


def effective_box_size(element: dict[str, Any]) -> tuple[float, float, float]:
    bbox = numeric_bbox(element["bbox"])
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    rotation = int(element.get("rotation", 0))
    if rotation in {90, 270}:
        width, height = height, width
    padding = float(element.get("padding", 1.5))
    return width - 2 * padding, height - 2 * padding, padding


def fit_text_element(
    element: dict[str, Any],
    fonts: dict[str, str],
) -> dict[str, Any]:
    available_width, available_height, _ = effective_box_size(element)
    requested = float(element.get("font_size", 10))
    minimum = float(element.get("min_font_size", 6))
    if available_width <= 0 or available_height <= 0:
        return {
            "fits": False,
            "font_size": minimum,
            "content_height": 0,
            "paragraph": None,
            "reason": "padding leaves no usable area",
        }

    current = requested
    while current + 1e-6 >= minimum:
        paragraph = paragraph_for(element, current, fonts)
        _, content_height = paragraph.wrap(available_width, 100000)
        if content_height <= available_height + 0.05:
            return {
                "fits": True,
                "font_size": round(current, 2),
                "content_height": content_height,
                "paragraph": paragraph,
                "reason": None,
            }
        current = round(current - 0.25, 2)
    return {
        "fits": False,
        "font_size": minimum,
        "content_height": content_height,
        "paragraph": paragraph,
        "reason": f"text requires {content_height:.2f} pt but only {available_height:.2f} pt is available",
    }


def draw_cover(canvas: Any, page_height: float, element: dict[str, Any]) -> None:
    bbox = numeric_bbox(element["bbox"])
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    fill = parse_color(element.get("fill"), "#FFFFFF", allow_none=True)
    border_width = float(element.get("border_width", 0))
    canvas.saveState()
    if fill is not None:
        canvas.setFillColor(fill)
    if border_width > 0:
        canvas.setStrokeColor(parse_color(element.get("border_color"), "#000000"))
        canvas.setLineWidth(border_width)
    canvas.rect(
        x0,
        page_height - y1,
        x1 - x0,
        y1 - y0,
        stroke=1 if border_width > 0 else 0,
        fill=1 if fill is not None else 0,
    )
    canvas.restoreState()


def draw_text(
    canvas: Any,
    page_height: float,
    element: dict[str, Any],
    fit: dict[str, Any],
) -> None:
    bbox = numeric_bbox(element["bbox"])
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    rotation = int(element.get("rotation", 0))
    padding = float(element.get("padding", 1.5))
    local_width = box_height if rotation in {90, 270} else box_width
    local_height = box_width if rotation in {90, 270} else box_height
    available_width = local_width - 2 * padding
    available_height = local_height - 2 * padding
    content_height = float(fit["content_height"])
    valign = element.get("valign", "top")
    if valign == "top":
        vertical_offset = available_height - content_height
    elif valign == "middle":
        vertical_offset = (available_height - content_height) / 2
    else:
        vertical_offset = 0

    paragraph = fit["paragraph"]
    if rotation == 0:
        draw_x = x0 + padding
        draw_y = page_height - y1 + padding + vertical_offset
        paragraph.drawOn(canvas, draw_x, draw_y)
        return

    center_x = (x0 + x1) / 2
    center_y = page_height - (y0 + y1) / 2
    canvas.saveState()
    canvas.translate(center_x, center_y)
    canvas.rotate(-rotation)
    draw_x = -local_width / 2 + padding
    draw_y = -local_height / 2 + padding + vertical_offset
    paragraph.drawOn(canvas, draw_x, draw_y)
    canvas.restoreState()


def prepare_source(args: argparse.Namespace) -> None:
    pymupdf = require_pymupdf()
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"Source PDF not found: {source_path}")
    workdir = Path(args.workdir).expanduser().resolve()
    ensure_output_dir(workdir, args.force)
    image_dir = workdir / "source-pages"
    if image_dir.exists() and any(image_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"Rendered pages already exist: {image_dir}. Pass --force to replace matching page images."
        )
    image_dir.mkdir(parents=True, exist_ok=True)

    source_document = pymupdf.open(source_path)
    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(source_document, start=1):
        text = page.get_text("text")
        text_blocks = []
        for block_index, block in enumerate(page.get_text("blocks"), start=1):
            if len(block) > 6 and block[6] != 0:
                continue
            block_text = str(block[4]).strip()
            if not block_text:
                continue
            text_blocks.append(
                {
                    "id": f"p{page_number}-block-{block_index:03d}",
                    "bbox": [round(float(value), 2) for value in block[:4]],
                    "text": block_text,
                }
            )
        pixmap = page.get_pixmap(dpi=args.dpi, alpha=False)
        rendered_name = f"page-{page_number:03d}.png"
        pixmap.save(image_dir / rendered_name)
        pages.append(
            {
                "page": page_number,
                "width_pt": round(page.rect.width, 2),
                "height_pt": round(page.rect.height, 2),
                "rotation": page.rotation,
                "text_characters": len(text.strip()),
                "embedded_images": len(page.get_images(full=True)),
                "rendered_file": f"source-pages/{rendered_name}",
                "rendered_width_px": pixmap.width,
                "rendered_height_px": pixmap.height,
                "pt_per_px_x": round(page.rect.width / pixmap.width, 8),
                "pt_per_px_y": round(page.rect.height / pixmap.height, 8),
                "text_blocks": text_blocks,
            }
        )

    manifest = {
        "source_pdf": str(source_path),
        "source_sha256": sha256(source_path),
        "page_count": source_document.page_count,
        "render_dpi": args.dpi,
        "text_layer_detected": any(page["text_characters"] for page in pages),
        "pages": pages,
    }
    manifest_path = workdir / "source-manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "prepared": str(workdir),
                "manifest": str(manifest_path),
                "pages": source_document.page_count,
            },
            ensure_ascii=False,
        )
    )


def create_template(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_json(manifest_path)
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise SystemExit("Source manifest has no pages array.")
    output = Path(args.output).expanduser().resolve()
    require_new_output(output, args.force)
    spec_pages = []
    for page in pages:
        elements = []
        for block in page.get("text_blocks", []):
            elements.append(
                {
                    "id": block["id"],
                    "type": "text",
                    "bbox": block["bbox"],
                    "source": block["text"],
                    "text": "",
                    "font": "regular",
                    "font_size": 10,
                    "min_font_size": 6,
                    "line_height": 1.12,
                    "align": "left",
                    "valign": "top",
                    "padding": 1.5,
                    "fill": "#FFFFFF",
                    "text_color": "#000000",
                    "rotation": 0,
                }
            )
        spec_pages.append(
            {
                "page": page["page"],
                "width_pt": page["width_pt"],
                "height_pt": page["height_pt"],
                "translation_complete": False,
                "elements": elements,
            }
        )
    spec = {
        "schema_version": 1,
        "source_pdf": manifest.get("source_pdf"),
        "source_sha256": manifest.get("source_sha256"),
        "background_dpi": 300,
        "pages": spec_pages,
    }
    write_json(output, spec)
    print(json.dumps({"template": str(output), "pages": len(spec_pages)}, ensure_ascii=False))


def make_guides(args: argparse.Namespace) -> None:
    pymupdf = require_pymupdf()
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"Source PDF not found: {source_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, args.force)
    source_document = pymupdf.open(source_path)
    spacing = float(args.spacing)
    if spacing <= 0:
        raise SystemExit("Grid spacing must be positive.")
    for page_index, source_page in enumerate(source_document):
        guide_document = pymupdf.open()
        page = guide_document.new_page(
            width=source_page.rect.width,
            height=source_page.rect.height,
        )
        page.show_pdf_page(page.rect, source_document, page_index)
        for x in range(0, int(math.ceil(page.rect.width)) + 1, int(spacing)):
            page.draw_line(
                pymupdf.Point(x, 0),
                pymupdf.Point(x, page.rect.height),
                color=(0.85, 0.15, 0.15),
                width=0.25,
                overlay=True,
            )
            page.insert_text(
                pymupdf.Point(x + 1.5, 9),
                f"x={x}",
                fontsize=4.5,
                color=(0.75, 0.05, 0.05),
                overlay=True,
            )
        for y in range(0, int(math.ceil(page.rect.height)) + 1, int(spacing)):
            page.draw_line(
                pymupdf.Point(0, y),
                pymupdf.Point(page.rect.width, y),
                color=(0.15, 0.35, 0.85),
                width=0.25,
                overlay=True,
            )
            label_y = max(5, y - 1.5)
            page.insert_text(
                pymupdf.Point(1.5, label_y),
                f"y={y}",
                fontsize=4.5,
                color=(0.05, 0.2, 0.75),
                overlay=True,
            )
        pixmap = page.get_pixmap(dpi=args.dpi, alpha=False)
        pixmap.save(output_dir / f"page-{page_index + 1:03d}-grid.png")
        guide_document.close()
    print(
        json.dumps(
            {"guides": str(output_dir), "pages": source_document.page_count, "spacing_pt": spacing},
            ensure_ascii=False,
        )
    )


def build_layout(args: argparse.Namespace) -> None:
    pymupdf = require_pymupdf()
    require_reportlab()
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    source_path = Path(args.source).expanduser().resolve()
    layout_path = Path(args.layout).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"Source PDF not found: {source_path}")
    spec = load_json(layout_path)
    source_document = pymupdf.open(source_path)
    issues = validate_spec(source_path, source_document, spec)
    if issues:
        raise SystemExit("Invalid layout specification:\n- " + "\n- ".join(issues))

    fonts = register_fonts()
    entries = page_entries(spec)
    fits: dict[tuple[int, str], dict[str, Any]] = {}
    fit_report: list[dict[str, Any]] = []
    overflows: list[str] = []
    for page_number in range(1, source_document.page_count + 1):
        elements = entries.get(page_number, {}).get("elements", [])
        for index, element in enumerate(elements, start=1):
            element_id = element.get("id", f"page-{page_number}-element-{index}")
            if element.get("type", "text") == "cover":
                fit_report.append(
                    {
                        "page": page_number,
                        "id": element_id,
                        "type": "cover",
                        "bbox": element["bbox"],
                    }
                )
                continue
            fit = fit_text_element(element, fonts)
            fits[(page_number, element_id)] = fit
            fit_report.append(
                {
                    "page": page_number,
                    "id": element_id,
                    "type": "text",
                    "bbox": element["bbox"],
                    "requested_font_size": float(element.get("font_size", 10)),
                    "used_font_size": fit["font_size"],
                    "shrunk": fit["font_size"] < float(element.get("font_size", 10)),
                    "fits": fit["fits"],
                    "reason": fit["reason"],
                }
            )
            if not fit["fits"]:
                overflows.append(f"page {page_number}, {element_id}: {fit['reason']}")
    if overflows:
        raise SystemExit(
            "Translated text does not fit its layout boxes:\n- "
            + "\n- ".join(overflows)
            + "\nEnlarge the affected bbox, reduce font_size/min_font_size intentionally, or revise the translation without omitting meaning."
        )

    require_new_output(output_path, args.force)
    background_dpi = int(args.background_dpi or spec.get("background_dpi", 300))
    if background_dpi < 144:
        raise SystemExit("background_dpi must be at least 144 for layout-preserving output.")

    pdf_canvas = canvas.Canvas(
        str(output_path),
        pagesize=(
            source_document[0].rect.width,
            source_document[0].rect.height,
        ),
        pageCompression=1,
    )
    pdf_canvas.setTitle(str(spec.get("title", output_path.stem)))
    pdf_canvas.setAuthor("Codex layout-preserving PDF translation workflow")

    for page_number, source_page in enumerate(source_document, start=1):
        width = source_page.rect.width
        height = source_page.rect.height
        pdf_canvas.setPageSize((width, height))
        pixmap = source_page.get_pixmap(dpi=background_dpi, alpha=False)
        background = ImageReader(io.BytesIO(pixmap.tobytes("png")))
        pdf_canvas.drawImage(background, 0, 0, width=width, height=height, mask=None)

        elements = entries.get(page_number, {}).get("elements", [])
        for index, element in enumerate(elements, start=1):
            element_id = element.get("id", f"page-{page_number}-element-{index}")
            draw_cover(pdf_canvas, height, element)
            if element.get("type", "text") == "text":
                draw_text(pdf_canvas, height, element, fits[(page_number, element_id)])
        pdf_canvas.showPage()
    pdf_canvas.save()

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(".layout-report.json")
    )
    report = {
        "output": str(output_path),
        "source_pdf": str(source_path),
        "source_sha256": sha256(source_path),
        "layout": str(layout_path),
        "page_count": source_document.page_count,
        "background_dpi": background_dpi,
        "elements": fit_report,
        "overflows": overflows,
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {"built": str(output_path), "report": str(report_path), "pages": source_document.page_count},
            ensure_ascii=False,
        )
    )


def render_pdf(args: argparse.Namespace) -> None:
    pymupdf = require_pymupdf()
    source_path = Path(args.pdf).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"PDF not found: {source_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, args.force)
    document = pymupdf.open(source_path)
    for page_number, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(dpi=args.dpi, alpha=False)
        pixmap.save(output_dir / f"page-{page_number:03d}.png")
    print(
        json.dumps(
            {"rendered": str(output_dir), "pages": document.page_count},
            ensure_ascii=False,
        )
    )


def compact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized)


def merged_intervals(intervals: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    clipped = [(max(0, start), min(limit, end)) for start, end in intervals if end > 0 and start < limit]
    clipped = [(start, end) for start, end in clipped if end > start]
    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def outside_overlay_similarity(
    source_page: Any,
    output_page: Any,
    elements: list[dict[str, Any]],
    dpi: int,
    margin_pt: float,
) -> dict[str, float]:
    source_pixmap = source_page.get_pixmap(dpi=dpi, alpha=False)
    output_pixmap = output_page.get_pixmap(dpi=dpi, alpha=False)
    if (
        source_pixmap.width != output_pixmap.width
        or source_pixmap.height != output_pixmap.height
        or source_pixmap.n != output_pixmap.n
    ):
        return {"similarity": 0.0, "mean_absolute_difference": 255.0}

    scale_x = source_pixmap.width / source_page.rect.width
    scale_y = source_pixmap.height / source_page.rect.height
    rects = []
    for element in elements:
        bbox = numeric_bbox(element.get("bbox"))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        rects.append(
            (
                int(math.floor((x0 - margin_pt) * scale_x)),
                int(math.floor((y0 - margin_pt) * scale_y)),
                int(math.ceil((x1 + margin_pt) * scale_x)),
                int(math.ceil((y1 + margin_pt) * scale_y)),
            )
        )

    source_samples = source_pixmap.samples
    output_samples = output_pixmap.samples
    channels = source_pixmap.n
    width = source_pixmap.width
    height = source_pixmap.height
    difference_sum = 0
    compared_channels = 0
    for y in range(height):
        intervals = merged_intervals(
            [(x0, x1) for x0, y0, x1, y1 in rects if y0 <= y < y1],
            width,
        )
        cursor = 0
        visible_segments: list[tuple[int, int]] = []
        for start, end in intervals:
            if cursor < start:
                visible_segments.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < width:
            visible_segments.append((cursor, width))
        for start, end in visible_segments:
            byte_start = (y * width + start) * channels
            byte_end = (y * width + end) * channels
            for offset in range(byte_start, byte_end):
                difference_sum += abs(source_samples[offset] - output_samples[offset])
            compared_channels += byte_end - byte_start

    if compared_channels == 0:
        return {"similarity": 1.0, "mean_absolute_difference": 0.0}
    mean_difference = difference_sum / compared_channels
    return {
        "similarity": 1 - mean_difference / 255,
        "mean_absolute_difference": mean_difference,
    }


def verify_layout(args: argparse.Namespace) -> None:
    pymupdf = require_pymupdf()
    source_path = Path(args.source).expanduser().resolve()
    layout_path = Path(args.layout).expanduser().resolve()
    output_path = Path(args.pdf).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"Source PDF not found: {source_path}")
    if not output_path.is_file():
        raise SystemExit(f"Translated PDF not found: {output_path}")

    spec = load_json(layout_path)
    source_document = pymupdf.open(source_path)
    output_document = pymupdf.open(output_path)
    issues = validate_spec(source_path, source_document, spec)
    entries = page_entries(spec)
    page_reports: list[dict[str, Any]] = []

    if output_document.page_count != source_document.page_count:
        issues.append(
            f"Page count differs: source {source_document.page_count}, output {output_document.page_count}."
        )

    common_pages = min(source_document.page_count, output_document.page_count)
    for page_number in range(1, common_pages + 1):
        source_page = source_document[page_number - 1]
        output_page = output_document[page_number - 1]
        page_issues: list[str] = []
        if abs(source_page.rect.width - output_page.rect.width) > 0.2:
            page_issues.append("page width differs from source")
        if abs(source_page.rect.height - output_page.rect.height) > 0.2:
            page_issues.append("page height differs from source")

        output_text = compact_text(output_page.get_text("text"))
        elements = entries.get(page_number, {}).get("elements", [])
        missing_text: list[str] = []
        for element in elements:
            if element.get("type", "text") != "text":
                continue
            expected = compact_text(str(element.get("text", "")))
            if expected and expected not in output_text:
                missing_text.append(str(element.get("id", "unnamed")))
        if missing_text:
            page_issues.append("searchable translation missing for: " + ", ".join(missing_text))

        out_of_bounds = []
        for block in output_page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -0.5 or y0 < -0.5 or x1 > output_page.rect.width + 0.5 or y1 > output_page.rect.height + 0.5:
                out_of_bounds.append([round(value, 2) for value in (x0, y0, x1, y1)])
        if out_of_bounds:
            page_issues.append(f"text outside media box: {out_of_bounds}")

        similarity = None
        if not args.skip_similarity:
            similarity = outside_overlay_similarity(
                source_page,
                output_page,
                elements,
                dpi=args.similarity_dpi,
                margin_pt=args.similarity_margin,
            )
            if similarity["similarity"] < args.min_similarity:
                page_issues.append(
                    "background similarity outside translation boxes "
                    f"{similarity['similarity']:.5f} is below {args.min_similarity:.5f}"
                )

        issues.extend(f"Page {page_number}: {issue}" for issue in page_issues)
        page_reports.append(
            {
                "page": page_number,
                "source_size_pt": [
                    round(source_page.rect.width, 2),
                    round(source_page.rect.height, 2),
                ],
                "output_size_pt": [
                    round(output_page.rect.width, 2),
                    round(output_page.rect.height, 2),
                ],
                "searchable_characters": len(output_page.get_text("text").strip()),
                "missing_text_elements": missing_text,
                "out_of_bounds_blocks": out_of_bounds,
                "outside_overlay_similarity": similarity,
                "issues": page_issues,
            }
        )

    report = {
        "source_pdf": str(source_path),
        "translated_pdf": str(output_path),
        "layout": str(layout_path),
        "source_sha256": sha256(source_path),
        "translated_sha256": sha256(output_path),
        "source_page_count": source_document.page_count,
        "output_page_count": output_document.page_count,
        "pages": page_reports,
        "issues": issues,
        "valid": not issues,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create searchable translated PDFs while preserving source page geometry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Render the source, record page geometry, and create source-manifest.json.",
    )
    prepare.add_argument("source", help="Source PDF")
    prepare.add_argument("--workdir", required=True, help="Empty work directory")
    prepare.add_argument("--dpi", type=int, default=220, help="Source inspection DPI")
    prepare.add_argument("--force", action="store_true", help="Replace matching prepared files")
    prepare.set_defaults(handler=prepare_source)

    template = subparsers.add_parser(
        "template",
        help="Create an empty layout specification from source-manifest.json.",
    )
    template.add_argument("manifest", help="source-manifest.json from the prepare command")
    template.add_argument("--output", required=True, help="Output layout JSON")
    template.add_argument("--force", action="store_true", help="Replace an existing output")
    template.set_defaults(handler=create_template)

    guide = subparsers.add_parser(
        "guide",
        help="Render source pages with a top-left PDF-point coordinate grid.",
    )
    guide.add_argument("source", help="Source PDF")
    guide.add_argument("--output-dir", required=True, help="Directory for guide page images")
    guide.add_argument("--spacing", type=int, default=36, help="Grid spacing in PDF points")
    guide.add_argument("--dpi", type=int, default=170, help="Guide image rendering resolution")
    guide.add_argument("--force", action="store_true", help="Replace matching guide images")
    guide.set_defaults(handler=make_guides)

    build = subparsers.add_parser(
        "build",
        help="Build a layout-preserving translated PDF from a source PDF and layout JSON.",
    )
    build.add_argument("source", help="Source PDF")
    build.add_argument("layout", help="Completed layout JSON")
    build.add_argument("--output", required=True, help="Translated PDF output")
    build.add_argument("--background-dpi", type=int, help="Override background DPI from the layout")
    build.add_argument("--report", help="Optional build report path")
    build.add_argument("--force", action="store_true", help="Replace a known output intentionally")
    build.set_defaults(handler=build_layout)

    verify = subparsers.add_parser(
        "verify",
        help="Validate page geometry, searchable text, bounds, and preserved background.",
    )
    verify.add_argument("source", help="Source PDF")
    verify.add_argument("layout", help="Completed layout JSON")
    verify.add_argument("pdf", help="Translated PDF")
    verify.add_argument("--similarity-dpi", type=int, default=36)
    verify.add_argument("--similarity-margin", type=float, default=2.0)
    verify.add_argument("--min-similarity", type=float, default=0.985)
    verify.add_argument("--skip-similarity", action="store_true")
    verify.set_defaults(handler=verify_layout)

    render = subparsers.add_parser(
        "render",
        help="Render a completed translated PDF for page-by-page visual QA.",
    )
    render.add_argument("pdf", help="Translated PDF")
    render.add_argument("--output-dir", required=True, help="Directory for QA page images")
    render.add_argument("--dpi", type=int, default=170, help="QA rendering DPI")
    render.add_argument("--force", action="store_true", help="Replace matching QA images")
    render.set_defaults(handler=render_pdf)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
