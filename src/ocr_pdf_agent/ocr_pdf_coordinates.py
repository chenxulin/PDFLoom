"""Coordinate adapters for drawing OCR geometry on rotated PDF pages.

PaddleOCR reports coordinates in the visually rendered page coordinate space.
PyMuPDF exposes that same space through ``page.rect``, but page mutation APIs
consume the page's unrotated PDF coordinate space.  On pages whose PDF
``/Rotate`` value is 90, 180, or 270 degrees, passing visual coordinates
directly to those APIs rotates and displaces every new object a second time.

Keep all layout calculations in visual coordinates and cross this module only
at the PDF read / write boundary.
"""

from __future__ import annotations

from typing import Any

import fitz

_RIGHT_ANGLE_ROTATIONS = frozenset({0, 90, 180, 270})


def page_rotation(page: fitz.Page) -> int:
    """Return a normalized PDF page rotation supported by PyMuPDF text APIs."""
    rotation = int(page.rotation or 0) % 360
    if rotation not in _RIGHT_ANGLE_ROTATIONS:
        raise ValueError(f"Unsupported PDF page rotation: {rotation}")
    return rotation


def visual_page_rect(page: fitz.Page) -> fitz.Rect:
    """Return the displayed page rectangle used by OCR and layout planning."""
    return fitz.Rect(page.rect)


def visual_rect_to_pdf(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    clip: bool = True,
) -> fitz.Rect:
    """Convert a displayed-page rectangle to PyMuPDF's unrotated space."""
    visual = fitz.Rect(rect)
    if clip:
        visual &= visual_page_rect(page)
    if visual.is_empty:
        return fitz.Rect()
    return visual * page.derotation_matrix


def pdf_rect_to_visual(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    clip: bool = True,
) -> fitz.Rect:
    """Convert a PyMuPDF text / drawing rectangle to displayed-page space."""
    visual = fitz.Rect(rect) * page.rotation_matrix
    if clip:
        visual &= visual_page_rect(page)
    return visual


def pdf_direction_to_visual(
    page: fitz.Page,
    direction: tuple[float, float],
) -> fitz.Point:
    """Convert a PDF text direction vector to the displayed page direction."""
    matrix = page.rotation_matrix
    return fitz.Point(
        direction[0] * matrix.a + direction[1] * matrix.c,
        direction[0] * matrix.b + direction[1] * matrix.d,
    )


def is_visually_horizontal(
    page: fitz.Page,
    direction: tuple[float, float],
    *,
    tolerance: float = 0.02,
) -> bool:
    """Return whether a PDF text line reads left-to-right after page rotation."""
    visual = pdf_direction_to_visual(page, direction)
    return abs(visual.x - 1.0) <= tolerance and abs(visual.y) <= tolerance


def map_ocr_rect_to_visual(
    segment: fitz.Rect,
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> fitz.Rect:
    """Scale a PaddleOCR box into a visual page or bilingual segment."""
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


def add_visual_redaction(
    page: fitz.Page,
    rect: fitz.Rect,
    **kwargs: Any,
) -> Any:
    """Add a redaction annotation whose input rectangle is visual geometry."""
    return page.add_redact_annot(visual_rect_to_pdf(page, rect), **kwargs)


def draw_visual_rect(
    page: fitz.Page,
    rect: fitz.Rect,
    **kwargs: Any,
) -> Any:
    """Draw a rectangle whose input coordinates are visual page coordinates."""
    return page.draw_rect(visual_rect_to_pdf(page, rect), **kwargs)


def insert_visual_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    **kwargs: Any,
) -> float:
    """Insert visually horizontal text into a visual-coordinate text box."""
    if "rotate" in kwargs:
        raise TypeError("insert_visual_textbox controls rotate from page /Rotate")
    return float(
        page.insert_textbox(
            visual_rect_to_pdf(page, rect),
            text,
            rotate=page_rotation(page),
            **kwargs,
        )
    )


def show_pdf_page_visual(
    target_page: fitz.Page,
    target_rect: fitz.Rect,
    source_document: fitz.Document,
    source_page_index: int,
    *,
    clip: fitz.Rect | None = None,
    **kwargs: Any,
) -> int:
    """Copy a visual source-page region into a visual target-page region.

    ``show_pdf_page`` builds a Form XObject from the source page's unrotated
    content.  Temporarily clearing the source page rotation prevents PyMuPDF
    from applying it again while the target page receives the rotation delta.
    The source document is restored immediately and is never saved here.
    """
    if "rotate" in kwargs:
        raise TypeError("show_pdf_page_visual controls source/target rotation")
    source_page = source_document[source_page_index]
    source_rotation = page_rotation(source_page)
    target_rotation = page_rotation(target_page)
    source_clip = (
        visual_rect_to_pdf(source_page, clip)
        if clip is not None
        else visual_rect_to_pdf(source_page, visual_page_rect(source_page))
    )
    target_pdf_rect = visual_rect_to_pdf(target_page, target_rect)
    try:
        if source_rotation:
            source_page.set_rotation(0)
        return int(
            target_page.show_pdf_page(
                target_pdf_rect,
                source_document,
                source_page_index,
                clip=source_clip,
                rotate=(target_rotation - source_rotation) % 360,
                **kwargs,
            )
        )
    finally:
        if source_rotation:
            source_page.set_rotation(source_rotation)


__all__ = [
    "add_visual_redaction",
    "draw_visual_rect",
    "insert_visual_textbox",
    "is_visually_horizontal",
    "map_ocr_rect_to_visual",
    "page_rotation",
    "pdf_direction_to_visual",
    "pdf_rect_to_visual",
    "show_pdf_page_visual",
    "visual_page_rect",
    "visual_rect_to_pdf",
]
