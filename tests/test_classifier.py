from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ocr_pdf_agent.classifier import classify_pdf
from ocr_pdf_agent.models import PdfKind, ProcessingRoute

from .helpers import make_native_pdf, make_scanned_table_pdf


class ClassifierTests(unittest.TestCase):
    def test_native_and_scanned_pdfs_choose_different_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = make_native_pdf(root / "native.pdf")
            scan = make_scanned_table_pdf(root / "scan.pdf")

            native_profile = classify_pdf(native)
            scan_profile = classify_pdf(scan)

            self.assertEqual(PdfKind.BORN_DIGITAL, native_profile.kind)
            self.assertEqual(
                ProcessingRoute.DIRECT_PDFMATHTRANSLATE,
                native_profile.route,
            )
            self.assertEqual(PdfKind.IMAGE_SCAN, scan_profile.kind)
            self.assertEqual(
                ProcessingRoute.PADDLEOCR_THEN_PDFMATHTRANSLATE,
                scan_profile.route,
            )


if __name__ == "__main__":
    unittest.main()
