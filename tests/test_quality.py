from __future__ import annotations

import unittest

from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.pipeline import _output_quality


class QualityTests(unittest.TestCase):
    def test_llm_endpoint_and_model_are_required_without_fallbacks(self) -> None:
        self.assertTrue(Settings.model_fields["base_url"].is_required())
        self.assertTrue(Settings.model_fields["model_name"].is_required())

    def test_rejects_unchanged_cross_language_output(self) -> None:
        source = "This document contains enough English text to require translation."
        result = _output_quality(source, source, "zh-CN")
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["failures"]), 1)

    def test_accepts_target_language_output(self) -> None:
        source = "This document contains enough English text to require translation."
        result = _output_quality(source, "本文档包含需要翻译的英文内容。", "zh-CN")
        self.assertTrue(result["passed"])

    def test_rejects_wrong_cmc_terms_even_when_output_is_chinese(self) -> None:
        source = "Assay reviewed for release"
        result = _output_quality(source, "检测结果已审核发布", "zh-CN")
        self.assertFalse(result["passed"])
        self.assertEqual(2, len(result["terminology_requirements"]))
        self.assertTrue(any("assay -> 含量测定" in item for item in result["failures"]))
        self.assertTrue(any("release -> 放行" in item for item in result["failures"]))

    def test_accepts_required_cmc_terms(self) -> None:
        result = _output_quality(
            "Assay reviewed for release",
            "含量测定结果已审核并批准放行",
            "zh-CN",
        )
        self.assertTrue(result["passed"])

    def test_empty_primary_ocr_token_uses_migration_fallback(self) -> None:
        settings = Settings(
            base_url="https://llm.example.test/v1",
            model_name="test-model",
            paddleocr_service_token="",
            attachment_ocr_service_token="fallback-token",
        )
        self.assertEqual("fallback-token", settings.ocr_token)


if __name__ == "__main__":
    unittest.main()
