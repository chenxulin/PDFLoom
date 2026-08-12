from __future__ import annotations

import unittest

from ocr_pdf_agent.body import extract_body_blocks
from ocr_pdf_agent.ocr_layer import _blocks

from .helpers import fake_ocr_payload


class OcrFilterTests(unittest.TestCase):
    def test_skips_tiny_low_confidence_speck_and_header_logo(self) -> None:
        payload = fake_ocr_payload()
        page_size = payload["pages"][0]["page_size"]
        payload["blocks"].extend(
            [
                {
                    "page_idx": 0,
                    "page_size": page_size,
                    "bbox": [456, 1189, 463, 1195],
                    "text": "G",
                    "type": "text",
                    "confidence": 0.3247,
                },
                {
                    "page_idx": 0,
                    "page_size": page_size,
                    "bbox": [162, 78, 240, 149],
                    "text": "B3",
                    "type": "header_image",
                    "confidence": 0.765,
                },
            ]
        )

        blocks, _tables, skipped_visuals, skipped_noise = _blocks(payload, 1)

        self.assertNotIn("G", {block.text for block in blocks})
        self.assertNotIn("B3", {block.text for block in blocks})
        self.assertEqual(1, skipped_noise)
        self.assertGreaterEqual(skipped_visuals, 1)
        self.assertNotIn("G", {block.source_text for block in extract_body_blocks(payload)})
        self.assertNotIn("B3", {block.source_text for block in extract_body_blocks(payload)})

    def test_does_not_skip_normal_single_character_text(self) -> None:
        payload = fake_ocr_payload()
        page_size = payload["pages"][0]["page_size"]
        payload["blocks"].append(
            {
                "page_idx": 0,
                "page_size": page_size,
                "bbox": [100, 1100, 145, 1150],
                "text": "结",
                "type": "text",
                "confidence": 0.40,
            }
        )

        blocks, _tables, _visuals, skipped_noise = _blocks(payload, 1)

        self.assertIn("结", {block.text for block in blocks})
        self.assertEqual(0, skipped_noise)


if __name__ == "__main__":
    unittest.main()
