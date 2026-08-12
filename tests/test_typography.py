from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from ocr_pdf_agent.ocr_document_typography import (
    _body_plans,
    _body_region_key,
    _BodyRegion,
    _PdfBlock,
)


class TypographyTests(unittest.TestCase):
    def test_one_engine_block_can_cover_two_adjacent_ocr_paragraphs(self) -> None:
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page_size = (600.0, 800.0)
        regions = [
            _BodyRegion(0, (50, 100, 550, 150), page_size),
            _BodyRegion(0, (50, 160, 550, 220), page_size),
        ]
        ocr = {
            "blocks": [
                {
                    "page_idx": 0,
                    "page_size": page_size,
                    "bbox": region.bbox,
                    "text": f"Source paragraph {index}",
                    "type": "text",
                }
                for index, region in enumerate(regions, 1)
            ],
            "regions": [],
        }
        translations = {
            _body_region_key(0, regions[0].bbox): "First translated paragraph.",
            _body_region_key(0, regions[1].bbox): "Second translated paragraph.",
        }
        merged = _PdfBlock(
            fitz.Rect(50, 100, 550, 220),
            "Merged engine output",
            (50.0,),
        )
        font = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")

        with patch(
            "ocr_pdf_agent.ocr_document_typography._pdf_blocks",
            return_value=[merged],
        ):
            plans = _body_plans(
                page,
                page_idx=0,
                segment=fitz.Rect(page.rect),
                body_regions=regions,
                body_translations=translations,
                ocr_result=ocr,
                body_font=font,
            )

        self.assertEqual(2, len(plans))
        self.assertEqual(
            {"First translated paragraph.", "Second translated paragraph."},
            {plan.text for plan in plans},
        )
        self.assertTrue(all(plan.target_rects == (merged.rect,) for plan in plans))
        document.close()


if __name__ == "__main__":
    unittest.main()
