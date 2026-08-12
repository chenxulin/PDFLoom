from __future__ import annotations

import unittest

from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.llm import StaticTranslator
from ocr_pdf_agent.tables import extract_tables, translate_tables

from .helpers import fake_ocr_payload


class TableTests(unittest.TestCase):
    def test_extracts_complete_grid(self) -> None:
        tables = extract_tables(fake_ocr_payload())
        self.assertEqual(1, len(tables))
        table = tables[0]
        self.assertEqual((3, 2), (table.row_count, table.column_count))
        self.assertEqual(6, len(table.cells))
        self.assertEqual("99.5 %", table.cells[-1].source_text)

    def test_html_rowspan_and_colspan_are_preserved(self) -> None:
        payload = fake_ocr_payload()
        payload["regions"][1]["structured_content"] = (
            "<table><tr><th rowspan='2'>Item</th><th colspan='2'>Results</th></tr>"
            "<tr><th>Initial</th><th>Final</th></tr>"
            "<tr><td>Assay</td><td>98.0 %</td><td>99.5 %</td></tr></table>"
        )
        table = extract_tables(payload)[0]
        self.assertEqual((3, 3), (table.row_count, table.column_count))
        self.assertEqual(2, table.cells[0].row_span)
        self.assertEqual(2, table.cells[1].column_span)


class TableTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_row_context_and_enforces_assay_term(self) -> None:
        translator = StaticTranslator(
            {
                "Item": "项目",
                "Result": "结果",
                "Appearance": "外观",
                "White powder": "白色粉末",
                # A bad model answer must not leak into the table.
                "Assay": "检测",
            }
        )
        plan = await translate_tables(
            fake_ocr_payload(),
            translator,
            Settings(
                base_url="https://llm.example.test/v1",
                model_name="test-model",
                target_language="zh-CN",
                max_workers=2,
            ),
        )
        assay = next(cell for cell in plan.tables[0].cells if cell.source_text == "Assay")
        self.assertEqual("含量测定", assay.target_text)
        assay_index = translator.calls.index("Assay")
        context = translator.contexts[assay_index]
        self.assertIn("Table headers: Item | Result", context)
        self.assertIn("Current source row: Assay | 99.5 %", context)
        self.assertIn("assay -> 含量测定", context)


if __name__ == "__main__":
    unittest.main()
