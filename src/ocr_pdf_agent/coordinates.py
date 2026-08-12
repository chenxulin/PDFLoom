"""Convert PaddleOCR visual coordinates to PyMuPDF mutation coordinates."""

from __future__ import annotations

from typing import Any

import fitz


def page_rotation(page: fitz.Page) -> int:
    rotation = int(page.rotation or 0) % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported PDF page rotation: {rotation}")
    return rotation


def visual_page_rect(page: fitz.Page) -> fitz.Rect:
    return fitz.Rect(page.rect)


def visual_rect_to_pdf(page: fitz.Page, rect: fitz.Rect, *, clip: bool = True) -> fitz.Rect:
    visual = fitz.Rect(rect)
    if clip:
        visual &= visual_page_rect(page)
    if visual.is_empty:
        return fitz.Rect()
    return visual * page.derotation_matrix


def pdf_rect_to_visual(page: fitz.Page, rect: fitz.Rect, *, clip: bool = True) -> fitz.Rect:
    visual = fitz.Rect(rect) * page.rotation_matrix
    if clip:
        visual &= visual_page_rect(page)
    return visual


def map_ocr_rect(
    segment: fitz.Rect,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    width, height = page_size
    if width <= 0 or height <= 0:
        return fitz.Rect()
    return (
        fitz.Rect(
            segment.x0 + bbox[0] * segment.width / width,
            segment.y0 + bbox[1] * segment.height / height,
            segment.x0 + bbox[2] * segment.width / width,
            segment.y0 + bbox[3] * segment.height / height,
        )
        & segment
    )


def add_visual_redaction(page: fitz.Page, rect: fitz.Rect, **kwargs: Any) -> Any:
    return page.add_redact_annot(visual_rect_to_pdf(page, rect), **kwargs)


def draw_visual_rect(page: fitz.Page, rect: fitz.Rect, **kwargs: Any) -> Any:
    return page.draw_rect(visual_rect_to_pdf(page, rect), **kwargs)


def draw_visual_line(
    page: fitz.Page,
    start: fitz.Point,
    end: fitz.Point,
    **kwargs: Any,
) -> Any:
    start_pdf = fitz.Point(start) * page.derotation_matrix
    end_pdf = fitz.Point(end) * page.derotation_matrix
    return page.draw_line(start_pdf, end_pdf, **kwargs)


def insert_visual_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    **kwargs: Any,
) -> float:
    if "rotate" in kwargs:
        raise TypeError("rotate is controlled by the PDF page rotation")
    return float(
        page.insert_textbox(
            visual_rect_to_pdf(page, rect),
            text,
            rotate=page_rotation(page),
            **kwargs,
        )
    )
