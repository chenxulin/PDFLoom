from __future__ import annotations

import unittest

from ocr_pdf_agent.terminology import (
    exact_preferred_target,
    missing_requirements,
    normalize_target_output,
    requirements_for,
)


class TerminologyTests(unittest.TestCase):
    def test_assay_has_exact_pharmaceutical_target(self) -> None:
        requirements = requirements_for("Assay", "zh-CN")
        self.assertEqual("含量测定", exact_preferred_target("Assay", "zh-CN"))
        self.assertEqual(("含量测定",), tuple(item.required_target for item in requirements))

    def test_release_is_required_inside_a_sentence(self) -> None:
        requirements = requirements_for("Reviewed for release.", "zh-CN")
        self.assertEqual(("放行",), tuple(item.required_target for item in requirements))
        self.assertEqual((), missing_requirements("经审核，批准放行。", requirements))
        self.assertEqual(requirements, missing_requirements("已审核发布。", requirements))

    def test_constraints_do_not_apply_to_other_target_languages(self) -> None:
        self.assertEqual((), requirements_for("Assay released", "en-US"))

    def test_exact_english_cmc_table_headers(self) -> None:
        self.assertEqual(
            "TOC parallel samples/ppb",
            exact_preferred_target("TOC平行样/ppb", "en"),
        )
        self.assertEqual("Batch No.", exact_preferred_target("产品批号", "en"))
        self.assertEqual(
            "3.2.4 Test Utensils",
            exact_preferred_target("3.2.4检验用具", "en-US"),
        )
        self.assertEqual(
            "Test utensils: All sample vials used for testing are clean and dry.",
            exact_preferred_target(
                "检验用具：检验用进样瓶均为已清洁干燥的检验用具。",
                "en",
            ),
        )

    def test_repeated_english_acronym_is_collapsed(self) -> None:
        self.assertEqual(
            "This OOS resulted from laboratory causes.",
            normalize_target_output(
                "This OOS OOS resulted from laboratory causes.",
                "en",
            ),
        )
        self.assertEqual(
            "This OOS resulted from laboratory causes.",
            normalize_target_output(
                "This OOS OOS OOS resulted from laboratory causes.",
                "en-US",
            ),
        )

    def test_target_normalization_preserves_non_acronym_repetition(self) -> None:
        self.assertEqual(
            "The result was very very low.",
            normalize_target_output("The result was very very low.", "en"),
        )
        self.assertEqual("OOS OOS", normalize_target_output("OOS OOS", "zh-CN"))


if __name__ == "__main__":
    unittest.main()
