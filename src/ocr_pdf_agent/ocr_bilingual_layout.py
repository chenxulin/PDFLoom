"""Detect target-language page regions in PDFMathTranslate bilingual PDFs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class OcrBilingualLayout:
    mode: str
    logical_page_count: int
    target_page_indices: tuple[int, ...]

    def logical_page_index(self, physical_page_idx: int) -> int | None:
        if self.mode == "side_by_side":
            return (
                physical_page_idx
                if 0 <= physical_page_idx < self.logical_page_count
                else None
            )
        if self.mode == "interleaved":
            logical = physical_page_idx // 2
            return logical if 0 <= logical < self.logical_page_count else None
        return None

    def target_page_index(self, logical_page_idx: int) -> int | None:
        if 0 <= logical_page_idx < len(self.target_page_indices):
            return self.target_page_indices[logical_page_idx]
        return None

    def target_segment(
        self,
        page: fitz.Page,
        physical_page_idx: int,
    ) -> fitz.Rect | None:
        logical = self.logical_page_index(physical_page_idx)
        if logical is None or self.target_page_index(logical) != physical_page_idx:
            return None
        if self.mode == "side_by_side":
            midpoint = page.rect.x0 + page.rect.width / 2
            return fitz.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1)
        return fitz.Rect(page.rect)

    def furniture_segments(
        self,
        page: fitz.Page,
        physical_page_idx: int,
    ) -> list[fitz.Rect]:
        logical = self.logical_page_index(physical_page_idx)
        if logical is None:
            return []
        if self.mode == "side_by_side":
            midpoint = page.rect.x0 + page.rect.width / 2
            return [
                fitz.Rect(page.rect.x0, page.rect.y0, midpoint, page.rect.y1),
                fitz.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1),
            ]
        return [fitz.Rect(page.rect)]


def detect_ocr_bilingual_layout(
    mono_pdf: str | Path,
    bilingual_pdf: str | Path,
) -> OcrBilingualLayout | None:
    """Recognize side-by-side pages or source/target interleaved page pairs."""
    mono_document = fitz.open(str(mono_pdf))
    bilingual_document = fitz.open(str(bilingual_pdf))
    try:
        mono_count = mono_document.page_count
        if mono_count < 1:
            return None
        mono_sizes = [
            (page.rect.width, page.rect.height) for page in mono_document
        ]
        if bilingual_document.page_count == mono_count:
            if all(
                abs(bilingual_document[index].rect.height - mono_height) <= 2
                and bilingual_document[index].rect.width >= mono_width * 1.8
                for index, (mono_width, mono_height) in enumerate(mono_sizes)
            ):
                return OcrBilingualLayout(
                    mode="side_by_side",
                    logical_page_count=mono_count,
                    target_page_indices=tuple(range(mono_count)),
                )
            return None

        if bilingual_document.page_count != mono_count * 2:
            return None
        targets: list[int] = []
        for logical_idx, (mono_width, mono_height) in enumerate(mono_sizes):
            pair = (logical_idx * 2, logical_idx * 2 + 1)
            if not all(
                abs(bilingual_document[index].rect.width - mono_width) <= 2
                and abs(bilingual_document[index].rect.height - mono_height) <= 2
                for index in pair
            ):
                return None
            # PDFMathTranslate's interleaved artifact has a stable source-then-
            # target contract. Similarity is not a safe discriminator here:
            # numeric-heavy scanned source pages can resemble the mono target
            # more closely than the translated page and would then receive
            # destructive typography edits. Always own the second page.
            targets.append(pair[1])
        return OcrBilingualLayout(
            mode="interleaved",
            logical_page_count=mono_count,
            target_page_indices=tuple(targets),
        )
    finally:
        bilingual_document.close()
        mono_document.close()


__all__ = ["OcrBilingualLayout", "detect_ocr_bilingual_layout"]
