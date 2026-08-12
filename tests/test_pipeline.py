from __future__ import annotations

import json
import tempfile
import unicodedata
import unittest
from pathlib import Path

import fitz

from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.llm import StaticTranslator
from ocr_pdf_agent.models import ProcessingRoute
from ocr_pdf_agent.pdfmath import CopyPdfEngine
from ocr_pdf_agent.pipeline import PdfTranslationPipeline

from .helpers import FakeOcrClient, make_native_pdf, make_scanned_table_pdf


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, root: Path) -> Settings:
        return Settings(
            api_key="offline-test-key",
            base_url="https://llm.example.test/v1",
            model_name="test-model",
            target_language="zh-CN",
            paddleocr_service_token="offline-ocr-token",
            storage_dir=root / "storage",
            max_workers=2,
            strict_output_qa=False,
        )

    async def test_native_pdf_bypasses_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_native_pdf(root / "native.pdf")
            ocr = FakeOcrClient()
            engine = CopyPdfEngine()
            pipeline = PdfTranslationPipeline(
                self.settings(root),
                ocr_client=ocr,
                pdf_engine=engine,
                translator=StaticTranslator(),
            )

            result = await pipeline.translate(source, output_dir=root / "native-output")

            self.assertEqual(ProcessingRoute.DIRECT_PDFMATHTRANSLATE, result.profile.route)
            self.assertEqual([], ocr.calls)
            self.assertEqual("source.pdf", engine.inputs[0].name)
            self.assertTrue(result.artifacts.translated_pdf.is_file())
            manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertTrue(manifest["validation"]["page_dimensions_match"])

    async def test_scanned_pdf_runs_ocr_and_redraws_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_scanned_table_pdf(root / "scan.pdf")
            ocr = FakeOcrClient()
            engine = CopyPdfEngine()
            translator = StaticTranslator(
                {
                    "QUALITY CONTROL REPORT": "质量控制报告",
                    "The following results were recorded for batch [[JTBL000|A-001.]]": (
                        "以下结果记录于批次 [[JTBL000|A-001.]]"
                    ),
                    "Reviewed for release.": "经审核，批准放行。",
                    "Item": "项目",
                    "Result": "结果",
                    "Appearance": "外观",
                    "White powder": "白色粉末",
                    "Assay": "含量测定",
                }
            )
            pipeline = PdfTranslationPipeline(
                self.settings(root),
                ocr_client=ocr,
                pdf_engine=engine,
                translator=translator,
            )

            result = await pipeline.translate(source, output_dir=root / "scan-output")

            self.assertEqual(
                ProcessingRoute.PADDLEOCR_THEN_PDFMATHTRANSLATE,
                result.profile.route,
            )
            self.assertEqual(1, len(ocr.calls))
            self.assertEqual("ocr_pdfmathtranslate_input.pdf", engine.inputs[0].name)
            self.assertEqual(1, result.table_stats.tables_redrawn)
            self.assertEqual(3, result.body_stats.blocks_translated)
            self.assertEqual(3, result.body_stats.blocks_redrawn)
            self.assertEqual(6, result.table_stats.cells_redrawn)
            self.assertEqual(5, result.table_stats.cells_translated)
            self.assertTrue(result.artifacts.ocr_json.is_file())
            self.assertTrue(result.artifacts.ocr_input_pdf.is_file())
            self.assertTrue(result.artifacts.translation_ledger.is_file())
            self.assertTrue(result.artifacts.layout_json.is_file())
            self.assertTrue(result.artifacts.layout_render_report.is_file())
            self.assertTrue(result.artifacts.layout_verification.is_file())
            ledger = json.loads(result.artifacts.translation_ledger.read_text(encoding="utf-8"))
            translations = {
                item["source"]: item["target"]
                for item in [*ledger["body_regions"], *ledger["heading_regions"]]
            }
            translations.update(
                {cell["source"]: cell["target"] for table in ledger["tables"] for cell in table["cells"]}
            )
            self.assertEqual("含量测定", translations["Assay"])
            self.assertEqual("经审核，批准放行。", translations["Reviewed for release."])
            with fitz.open(str(result.artifacts.translated_pdf)) as document:
                extracted = "".join(page.get_text("text") for page in document)
            compact = "".join(unicodedata.normalize("NFKC", extracted).split())
            self.assertIn("99.5%", compact)
            self.assertIn("以下结果记录于批次A-001.", compact)
            self.assertIn("项目", compact)
            self.assertIn("含量测定", compact)
            self.assertIn("放行", compact)
            self.assertNotIn("检测", compact)
            self.assertNotIn("发布", compact)
            self.assertNotIn("Assay", compact)
            self.assertNotIn("Reviewed for release", extracted)
            manifest = json.loads(result.artifacts.manifest.read_text(encoding="utf-8"))
            self.assertTrue(manifest["validation"]["page_dimensions_match"])
            self.assertTrue(manifest["validation"]["layout"]["valid"])
            layout = json.loads(result.artifacts.layout_json.read_text(encoding="utf-8"))
            roles = {
                element.get("role")
                for page in layout["pages"]
                for element in page["elements"]
            }
            self.assertEqual({"body", "heading", "table"}, roles)
            release_index = translator.calls.index("Reviewed for release.")
            self.assertIn("Previous context", translator.contexts[release_index])
            self.assertGreater(release_index, translator.calls.index("Assay"))

    async def test_long_table_adds_verified_continuation_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_scanned_table_pdf(root / "scan.pdf")
            payload = FakeOcrClient().payload
            rows = "".join(
                f"<tr><td>B-{index:03d}</td><td>{90 + index / 10:.1f} %</td></tr>"
                for index in range(1, 61)
            )
            payload["regions"][1]["structured_content"] = (
                "<table><tr><th>Item</th><th>Result</th></tr>" + rows + "</table>"
            )
            translator = StaticTranslator(
                {
                    "QUALITY CONTROL REPORT": "质量控制报告",
                    "The following results were recorded for batch [[JTBL000|A-001.]]": (
                        "以下结果记录于批次 [[JTBL000|A-001.]]"
                    ),
                    "Reviewed for release.": "经审核，批准放行。",
                    "Item": "项目",
                    "Result": "结果",
                }
            )
            pipeline = PdfTranslationPipeline(
                self.settings(root),
                ocr_client=FakeOcrClient(payload),
                pdf_engine=CopyPdfEngine(),
                translator=translator,
            )

            result = await pipeline.translate(source, output_dir=root / "scan-output")

            self.assertGreaterEqual(result.table_stats.continuation_pages, 1)
            with fitz.open(str(result.artifacts.translated_pdf)) as document:
                self.assertEqual(1 + result.table_stats.continuation_pages, document.page_count)
            verification = json.loads(result.artifacts.layout_verification.read_text(encoding="utf-8"))
            self.assertTrue(verification["valid"])
            self.assertEqual(result.table_stats.continuation_pages, verification["continuation_page_count"])


if __name__ == "__main__":
    unittest.main()
