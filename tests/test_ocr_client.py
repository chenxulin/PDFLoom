from __future__ import annotations

import unittest

from ocr_pdf_agent.ocr_client import PaddleOcrClient


class OcrClientTests(unittest.TestCase):
    def test_repairs_common_oos_digit_confusions_everywhere(self) -> None:
        payload = {
            "provider": "paddleocr-ppstructurev3",
            "markdown": "00S / 0OS / O0S; X00S stays unchanged",
            "pages": [],
            "blocks": [{"text": "编号：0OS-25-B-001"}],
            "regions": [{"structured_content": "<table><td>00S</td></table>"}],
            "metadata": {"common_text_corrections": 2},
        }

        corrected = PaddleOcrClient._correct_common_confusions(payload)

        self.assertEqual(corrected["blocks"][0]["text"], "编号：OOS-25-B-001")
        self.assertEqual(
            corrected["regions"][0]["structured_content"],
            "<table><td>OOS</td></table>",
        )
        self.assertEqual(corrected["markdown"], "OOS / OOS / OOS; X00S stays unchanged")
        self.assertEqual(corrected["metadata"]["common_text_corrections"], 7)


if __name__ == "__main__":
    unittest.main()
