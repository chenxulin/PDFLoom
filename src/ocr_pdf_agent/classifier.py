"""Full-document PDF classification and deterministic route selection."""

from __future__ import annotations

from pathlib import Path

import fitz

from .models import PageSignal, PdfKind, PdfProfile, ProcessingRoute


def _image_coverage(page: fitz.Page) -> tuple[int, float]:
    """Return image count and capped aggregate page coverage.

    Full-page scans normally contain one raster. The capped aggregate is
    intentional: overlapping logos or masks should never produce coverage
    greater than one and do not need expensive polygon-union computation.
    """
    page_area = max(float(page.rect.get_area()), 1.0)
    image_area = 0.0
    images = page.get_images(full=True)
    for image in images:
        for rect in page.get_image_rects(image[0]):
            clipped = fitz.Rect(rect) & page.rect
            if not clipped.is_empty:
                image_area += float(clipped.get_area())
    return len(images), min(1.0, image_area / page_area)


def _page_signal(page: fitz.Page, page_index: int, *, min_text_characters: int) -> PageSignal:
    text = "".join(page.get_text("text").split())
    text_blocks = sum(1 for block in page.get_text("blocks") if len(block) >= 7 and int(block[6]) == 0)
    images, coverage = _image_coverage(page)
    has_text = len(text) >= min_text_characters or (len(text) >= 4 and text_blocks >= 2)
    image_dominant = coverage >= 0.45
    if image_dominant and has_text:
        classification = "searchable_scan"
    elif image_dominant:
        classification = "scan"
    elif has_text:
        classification = "text"
    elif images and coverage >= 0.15:
        classification = "scan"
    else:
        classification = "blank"
    return PageSignal(
        page_index=page_index,
        text_characters=len(text),
        text_blocks=text_blocks,
        images=images,
        image_coverage=round(coverage, 6),
        classification=classification,
    )


def classify_pdf(
    path: str | Path,
    *,
    min_text_characters: int = 24,
    max_pages: int | None = None,
) -> PdfProfile:
    """Inspect the whole document unless an explicit diagnostic limit is set.

    Any image-only page selects the OCR route. This is conservative for mixed
    PDFs and avoids silently dropping scanned appendices merely because the
    cover page has native text.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF input is supported: {pdf_path}")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive or None")

    with fitz.open(str(pdf_path)) as document:
        if not document.is_pdf:
            raise ValueError(f"Input is not a valid PDF: {pdf_path}")
        page_count = document.page_count
        inspected = page_count if max_pages is None else min(page_count, max_pages)
        signals = tuple(
            _page_signal(document[index], index, min_text_characters=min_text_characters)
            for index in range(inspected)
        )

    if page_count == 0:
        return PdfProfile(
            PdfKind.EMPTY,
            ProcessingRoute.DIRECT_PDFMATHTRANSLATE,
            0,
            0,
            0,
            0,
            0.0,
            "empty PDF",
            (),
        )

    native_pages = sum(signal.classification == "text" for signal in signals)
    scan_pages = sum(signal.classification == "scan" for signal in signals)
    searchable_pages = sum(signal.classification == "searchable_scan" for signal in signals)
    average_coverage = sum(signal.image_coverage for signal in signals) / max(1, len(signals))

    if scan_pages and (native_pages or searchable_pages):
        kind = PdfKind.MIXED
        reason = "document contains both native/searchable pages and image-only scan pages"
    elif scan_pages:
        kind = PdfKind.IMAGE_SCAN
        reason = "image-dominant pages do not contain a usable text layer"
    elif searchable_pages and searchable_pages * 2 >= len(signals):
        kind = PdfKind.SEARCHABLE_SCAN
        reason = "image-dominant pages also contain an OCR/searchable text layer"
    elif native_pages or searchable_pages:
        kind = PdfKind.BORN_DIGITAL
        reason = "usable native PDF text layer"
    else:
        kind = PdfKind.UNKNOWN
        reason = "no reliable native text or full-page scan signal"

    needs_ocr = kind in {PdfKind.IMAGE_SCAN, PdfKind.SEARCHABLE_SCAN, PdfKind.MIXED}
    route = (
        ProcessingRoute.PADDLEOCR_THEN_PDFMATHTRANSLATE
        if needs_ocr
        else ProcessingRoute.DIRECT_PDFMATHTRANSLATE
    )
    return PdfProfile(
        kind=kind,
        route=route,
        pages=page_count,
        native_text_pages=native_pages,
        scan_pages=scan_pages,
        searchable_scan_pages=searchable_pages,
        average_image_coverage=round(average_coverage, 6),
        reason=reason,
        page_signals=signals,
    )
