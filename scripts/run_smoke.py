#!/usr/bin/env python3
"""Run both routes with either deterministic doubles or live dependencies."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz

from ocr_pdf_agent.config import Settings
from ocr_pdf_agent.llm import StaticTranslator
from ocr_pdf_agent.pdfmath import CopyPdfEngine
from ocr_pdf_agent.pipeline import PdfTranslationPipeline
from tests.helpers import FakeOcrClient, make_native_pdf, make_scanned_table_pdf


def _scan_semantic_qa(path: Path) -> dict[str, Any]:
    with fitz.open(str(path)) as document:
        extracted = "\n".join(page.get_text("text") for page in document)
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", extracted))
    folded = compact.casefold()
    required = ("含量测定", "放行", "99.5")
    forbidden = ("检测", "发布", "assay", "reviewedforrelease")
    required_checks = {term: term in compact for term in required}
    forbidden_checks = {term: term not in folded for term in forbidden}
    failures = [f"missing required term: {term}" for term, passed in required_checks.items() if not passed]
    failures.extend(
        f"forbidden mistranslation/source residue: {term}"
        for term, passed in forbidden_checks.items()
        if not passed
    )
    return {
        "passed": not failures,
        "required": required_checks,
        "forbidden": forbidden_checks,
        "failures": failures,
        "extracted_text": extracted.strip(),
    }


async def run(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    scan = make_scanned_table_pdf(samples / "scanned-table.pdf")
    native = make_native_pdf(samples / "born-digital.pdf")
    if args.offline:
        settings = Settings(
            api_key="offline-test-key",
            base_url="https://llm.invalid/v1",
            model_name="offline-static-translator",
            paddleocr_service_token="offline-ocr-token",
            target_language="zh-CN",
            strict_output_qa=False,
            storage_dir=root / "storage",
        )
        static = StaticTranslator(
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
        scan_pipeline = PdfTranslationPipeline(
            settings,
            ocr_client=FakeOcrClient(),
            pdf_engine=CopyPdfEngine(),
            translator=static,
        )
        native_pipeline = PdfTranslationPipeline(
            settings,
            ocr_client=FakeOcrClient(),
            pdf_engine=CopyPdfEngine(),
            translator=static,
        )
    else:
        settings = Settings(storage_dir=root / "storage")
        scan_pipeline = PdfTranslationPipeline(settings)
        native_pipeline = PdfTranslationPipeline(settings)

    def progress(label: str):
        return lambda stage, pct, message: print(f"[{label} {pct:6.2f}%] {stage}: {message}", flush=True)

    results = []
    semantic_failures: list[str] = []
    for label, source, pipeline in (
        ("scan", scan, scan_pipeline),
        ("native", native, native_pipeline),
    ):
        result = await pipeline.translate(
            source,
            output_dir=root / label,
            progress=progress(label),
        )
        semantic_qa = _scan_semantic_qa(result.artifacts.translated_pdf) if label == "scan" else None
        if semantic_qa and not semantic_qa["passed"]:
            semantic_failures.extend(semantic_qa["failures"])
        results.append(
            {
                "sample": label,
                "input": str(source),
                "kind": result.profile.kind.value,
                "route": result.profile.route.value,
                "translated_pdf": str(result.artifacts.translated_pdf),
                "manifest": str(result.artifacts.manifest),
                "translation_ledger": (
                    str(result.artifacts.translation_ledger) if result.artifacts.translation_ledger else None
                ),
                "body_stats": result.body_stats.__dict__,
                "table_stats": result.table_stats.__dict__,
                "semantic_qa": semantic_qa,
                "timings_ms": result.timings_ms,
            }
        )
    summary: dict[str, Any] = {
        "mode": "offline" if args.offline else "live",
        "passed": not semantic_failures,
        "failures": semantic_failures,
        "results": results,
    }
    summary_path = root / "smoke-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    if semantic_failures:
        raise RuntimeError("Scanned-PDF semantic QA failed: " + "; ".join(semantic_failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("test-results/smoke"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
