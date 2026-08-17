"""Classify PDFs with Joincare's current document-level scan policy."""
from __future__ import annotations

from pathlib import Path

import fitz

from .models import PdfKind, PdfProfile, ProcessingRoute


def _page_image_coverage(page: fitz.Page) -> float:
    page_area = max(page.rect.get_area(), 1.0)
    image_area = 0.0
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            image_area += fitz.Rect(rect).get_area()
    return min(1.0, image_area / page_area)


def classify_pdf(
    path: str | Path,
    *,
    min_text_characters: int = 24,
    max_pages: int | None = None,
) -> PdfProfile:
    """Inspect every page and choose the current Joincare document route.

    The stable production rule treats a document as scan-driven when at least
    half of the inspected pages are image-dominant. There is intentionally no
    separate mixed-PDF route: mixed documents follow the same dominant-page
    policy as Joincare's current scanned-PDF path.
    """
    del min_text_characters  # retained for API compatibility with earlier Agent callers
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF input is supported: {pdf_path}")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive or None")

    document = fitz.open(str(pdf_path))
    try:
        if not document.is_pdf:
            raise ValueError(f"Input is not a valid PDF: {pdf_path}")
        page_count = document.page_count
        sample_pages = page_count if max_pages is None else min(page_count, max_pages)
        if sample_pages == 0:
            return PdfProfile(
                PdfKind.EMPTY,
                ProcessingRoute.DIRECT_PDFMATHTRANSLATE,
                0,
                0,
                0,
                0,
                0.0,
                "empty pdf",
                (),
            )

        text_pages = 0
        image_pages = 0
        searchable_image_pages = 0
        coverage_total = 0.0
        for page in document[:sample_pages]:
            text = page.get_text("text").strip()
            if text:
                text_pages += 1
            coverage = _page_image_coverage(page)
            coverage_total += coverage
            if coverage >= 0.45:
                image_pages += 1
                if text:
                    searchable_image_pages += 1

        average_coverage = coverage_total / sample_pages
        image_dominant = image_pages > 0 and image_pages * 2 >= sample_pages
        if image_dominant and searchable_image_pages:
            kind = PdfKind.SEARCHABLE_SCAN
            reason = "image-dominant pages also contain a searchable OCR text layer"
        elif image_dominant:
            kind = PdfKind.IMAGE_SCAN
            reason = "image-dominant pages without usable text layer"
        elif text_pages:
            kind = PdfKind.BORN_DIGITAL
            reason = "native PDF text layer"
        else:
            kind = PdfKind.UNKNOWN
            reason = "no clear text or scan signal"

        route = (
            ProcessingRoute.PADDLEOCR_THEN_PDFMATHTRANSLATE
            if kind in {PdfKind.IMAGE_SCAN, PdfKind.SEARCHABLE_SCAN}
            else ProcessingRoute.DIRECT_PDFMATHTRANSLATE
        )
        return PdfProfile(
            kind=kind,
            route=route,
            pages=page_count,
            native_text_pages=text_pages,
            scan_pages=image_pages,
            searchable_scan_pages=searchable_image_pages,
            average_image_coverage=round(average_coverage, 6),
            reason=reason,
            page_signals=(),
        )
    finally:
        document.close()
