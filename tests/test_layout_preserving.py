from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.layout_preserving import (
    LayoutPreservingError,
    generate_layout_from_ledger,
    render_layout_pdf,
    verify_layout_output,
)

from .helpers import PAGE_HEIGHT, PAGE_WIDTH, make_scanned_table_pdf


class LayoutPreservingTests(unittest.TestCase):
    def _ledger(self, path: Path) -> Path:
        payload = {
            "schema_version": 1,
            "translation_order": "page_reading_order_serial",
            "body_regions": [
                {
                    "page": 1,
                    "bbox": [100, 164, 1000, 216],
                    "page_size": [PAGE_WIDTH * 2, PAGE_HEIGHT * 2],
                    "source": "The following results were recorded for batch A-001.",
                    "target": "以下结果记录于批次 A-001。",
                    "protected_literals": ["A-001"],
                },
                {
                    "page": 1,
                    "bbox": [100, 700, 520, 756],
                    "page_size": [PAGE_WIDTH * 2, PAGE_HEIGHT * 2],
                    "source": "Reviewed for release.",
                    "target": "经审核，批准放行。",
                    "protected_literals": [],
                },
            ],
            "heading_regions": [
                {
                    "page": 1,
                    "bbox": [100, 90, 720, 150],
                    "page_size": [PAGE_WIDTH * 2, PAGE_HEIGHT * 2],
                    "source": "QUALITY CONTROL REPORT",
                    "target": "质量控制报告",
                    "protected_literals": [],
                }
            ],
            "tables": [],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_generates_renders_and_strictly_verifies_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_scanned_table_pdf(root / "source.pdf")
            ledger = self._ledger(root / "translation_ledger.json")
            layout = generate_layout_from_ledger(source, ledger, root / "layout.json")
            settings = Settings(
                base_url="https://llm.example.test/v1",
                model_name="test-model",
            )
            rendered = render_layout_pdf(
                source,
                layout,
                root / "rendered.pdf",
                regular_font_path=settings.regular_font_path,
                bold_font_path=settings.bold_font_path,
            )
            verification = verify_layout_output(
                source,
                layout,
                rendered.pdf_path,
                root / "verification.json",
            )

            spec = json.loads(layout.read_text(encoding="utf-8"))
            self.assertEqual(1, spec["schema_version"])
            self.assertTrue(spec["pages"][0]["translation_complete"])
            self.assertEqual(3, len(spec["pages"][0]["elements"]))
            self.assertTrue(verification.valid)
            self.assertGreaterEqual(verification.minimum_background_similarity, 0.985)
            with fitz.open(str(rendered.pdf_path)) as document:
                extracted = "".join(page.get_text("text") for page in document)
            self.assertIn("质量控制报告", extracted)
            self.assertIn("批准放行", extracted)
            self.assertNotIn("QUALITY CONTROL REPORT", extracted)

    def test_verifier_rejects_changes_outside_declared_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_scanned_table_pdf(root / "source.pdf")
            ledger = self._ledger(root / "translation_ledger.json")
            layout = generate_layout_from_ledger(source, ledger, root / "layout.json")
            settings = Settings(
                base_url="https://llm.example.test/v1",
                model_name="test-model",
            )
            rendered = render_layout_pdf(
                source,
                layout,
                root / "rendered.pdf",
                regular_font_path=settings.regular_font_path,
                bold_font_path=settings.bold_font_path,
            )
            tampered = root / "tampered.pdf"
            with fitz.open(str(rendered.pdf_path)) as document:
                document[0].draw_rect(
                    fitz.Rect(340, 500, 560, 760),
                    color=None,
                    fill=(0, 0, 0),
                    overlay=True,
                )
                document.save(str(tampered), garbage=4, deflate=True)

            with self.assertRaises(LayoutPreservingError):
                verify_layout_output(
                    source,
                    layout,
                    tampered,
                    root / "tampered-verification.json",
                )
            report = json.loads((root / "tampered-verification.json").read_text(encoding="utf-8"))
            self.assertFalse(report["valid"])
            self.assertTrue(any("background similarity" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
