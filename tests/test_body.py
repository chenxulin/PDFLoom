from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from ocr_pdf_agent.body import (
    BodyRedrawError,
    _expand_into_following_whitespace,
    translate_body,
)
from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.llm import StaticTranslator

from .helpers import fake_ocr_payload


class BodyTranslationTests(unittest.IsolatedAsyncioTestCase):
    def test_expands_only_into_bounded_following_whitespace(self) -> None:
        source = fitz.Rect(10, 10, 100, 20)
        following = fitz.Rect(10, 40, 100, 50)

        expanded = _expand_into_following_whitespace(
            source,
            [following],
            fitz.Rect(0, 0, 200, 200),
        )

        self.assertGreater(expanded.y1, source.y1)
        self.assertLess(expanded.y1, following.y0)

    async def test_skips_blocks_already_rendered_in_target_language(self) -> None:
        payload = fake_ocr_payload()
        payload["blocks"] = [payload["blocks"][0]]
        payload["blocks"][0]["text"] = "质量控制报告"
        payload["regions"] = [payload["regions"][0]]
        translator = StaticTranslator({"质量控制报告": "Quality Control Report"})
        with TemporaryDirectory() as temporary:
            rendered = Path(temporary) / "translated.pdf"
            document = fitz.open()
            page = document.new_page(width=595, height=842)
            page.insert_text((54, 68), "QUALITY CONTROL REPORT", fontsize=14)
            document.save(rendered)
            document.close()

            plan = await translate_body(
                payload,
                translator,
                Settings(
                    base_url="https://llm.example.test/v1",
                    model_name="test-model",
                    target_language="en",
                ),
                existing_pdf=rendered,
            )

        self.assertEqual([], translator.calls)
        self.assertEqual(0, plan.translated_blocks)
        self.assertTrue(plan.blocks[0].engine_rendered)
        self.assertFalse(plan.blocks[0].redraw)

    async def test_release_mistranslation_retries_then_fails(self) -> None:
        translator = StaticTranslator({"Reviewed for release.": "已审核发布。"})
        with self.assertRaisesRegex(BodyRedrawError, "release->放行"):
            await translate_body(
                fake_ocr_payload(),
                translator,
                Settings(
                    base_url="https://llm.example.test/v1",
                    model_name="test-model",
                    target_language="zh-CN",
                ),
            )
        self.assertEqual(2, translator.calls.count("Reviewed for release."))
        retry_context = translator.contexts[-1]
        self.assertIn("previous answer violated", retry_context.casefold())
        self.assertIn("release -> 放行", retry_context)
