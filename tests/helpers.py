"""Synthetic PDFs and normalized OCR payloads used by tests and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0


def make_native_pdf(path: Path) -> Path:
    document = fitz.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((54, 72), "QUALITY CONTROL SUMMARY", fontsize=18, fontname="hebo")
    page.insert_text(
        (54, 112),
        "This born-digital PDF contains a native searchable text layer.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (54, 138),
        "The assay result meets the approved specification.",
        fontsize=11,
        fontname="helv",
    )
    document.save(str(path))
    document.close()
    return path


def _draw_scan_source() -> fitz.Document:
    document = fitz.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((54, 65), "QUALITY CONTROL REPORT", fontsize=18, fontname="hebo")
    page.insert_text(
        (54, 98),
        "The following results were recorded for batch A-001.",
        fontsize=11,
        fontname="helv",
    )
    left, top, right, bottom = 54.0, 145.0, 541.0, 325.0
    page.draw_rect(fitz.Rect(left, top, right, bottom), color=(0, 0, 0), width=1.0)
    for x in (300.0,):
        page.draw_line((x, top), (x, bottom), color=(0, 0, 0), width=0.8)
    for y in (195.0, 260.0):
        page.draw_line((left, y), (right, y), color=(0, 0, 0), width=0.8)
    values = [
        (70, 176, "Item"),
        (320, 176, "Result"),
        (70, 232, "Appearance"),
        (320, 232, "White powder"),
        (70, 297, "Assay"),
        (320, 297, "99.5 %"),
    ]
    for x, y, text in values:
        page.insert_text((x, y), text, fontsize=12, fontname="helv")
    page.insert_text((54, 370), "Reviewed for release.", fontsize=11, fontname="helv")
    return document


def make_scanned_table_pdf(path: Path) -> Path:
    source = _draw_scan_source()
    pixmap = source[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = pixmap.tobytes("png")
    source.close()
    scanned = fitz.open()
    page = scanned.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_image(page.rect, stream=image)
    scanned.save(str(path), deflate=True)
    scanned.close()
    return path


def fake_ocr_payload(filename: str = "scanned-table.pdf") -> dict[str, Any]:
    scale = 2.0

    def box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
        return [x0 * scale, y0 * scale, x1 * scale, y1 * scale]

    page_size = [PAGE_WIDTH * scale, PAGE_HEIGHT * scale]
    table_html = """<table>
      <tr><th>Item</th><th>Result</th></tr>
      <tr><td>Appearance</td><td>White powder</td></tr>
      <tr><td>Assay</td><td>99.5 %</td></tr>
    </table>"""
    blocks = [
        {
            "page_idx": 0,
            "page_size": page_size,
            "bbox": box(50, 45, 360, 75),
            "text": "QUALITY CONTROL REPORT",
            "type": "title",
            "confidence": 0.99,
        },
        {
            "page_idx": 0,
            "page_size": page_size,
            "bbox": box(50, 82, 500, 108),
            "text": "The following results were recorded for batch A-001.",
            "type": "text",
            "confidence": 0.99,
        },
        {
            "page_idx": 0,
            "page_size": page_size,
            "bbox": box(50, 350, 260, 378),
            "text": "Reviewed for release.",
            "type": "text",
            "confidence": 0.99,
        },
    ]
    for x0, y0, x1, y1, text in (
        (54, 145, 300, 195, "Item"),
        (300, 145, 541, 195, "Result"),
        (54, 195, 300, 260, "Appearance"),
        (300, 195, 541, 260, "White powder"),
        (54, 260, 300, 325, "Assay"),
        (300, 260, 541, 325, "99.5 %"),
    ):
        blocks.append(
            {
                "page_idx": 0,
                "page_size": page_size,
                "bbox": box(x0, y0, x1, y1),
                "text": text,
                "type": "table_text",
                "confidence": 0.99,
            }
        )
    return {
        "provider": "paddleocr-ppstructurev3",
        "markdown": (
            "# QUALITY CONTROL REPORT\n\n"
            "The following results were recorded for batch A-001.\n\n"
            "| Item | Result |\n| --- | --- |\n"
            "| Appearance | White powder |\n| Assay | 99.5 % |"
        ),
        "pages": [{"page_idx": 0, "page_size": page_size, "angle": 0}],
        "blocks": blocks,
        "regions": [
            {
                "page_idx": 0,
                "page_size": page_size,
                "bbox": box(50, 42, 370, 78),
                "type": "title",
                "structured_content": "QUALITY CONTROL REPORT",
            },
            {
                "page_idx": 0,
                "page_size": page_size,
                "bbox": box(54, 145, 541, 325),
                "type": "table",
                "structured_content": table_html,
                "confidence": 0.99,
            },
        ],
        "metadata": {
            "filename": filename,
            "page_count": 1,
            "block_count": len(blocks),
            "region_count": 2,
        },
    }


class FakeOcrClient:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or fake_ocr_payload()
        self.calls: list[Path] = []

    async def recognize(self, path: Path) -> dict[str, Any]:
        self.calls.append(Path(path))
        return self.payload
