"""Command-line entry point for classification, translation, and serving."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .classifier import classify_pdf
from .config import Settings
from .pipeline import PdfTranslationPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocr-pdf-agent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    classify = subcommands.add_parser("classify", help="Classify a PDF and show its route")
    classify.add_argument("pdf", type=Path)
    translate = subcommands.add_parser("translate", help="Translate one PDF")
    translate.add_argument("pdf", type=Path)
    translate.add_argument("--output-dir", type=Path)
    translate.add_argument("--source-language", default=None)
    translate.add_argument("--target-language", default=None)
    serve = subcommands.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8010)
    return parser


async def _translate(args: argparse.Namespace) -> None:
    settings = Settings()
    overrides = {
        key: value
        for key, value in {
            "source_language": args.source_language,
            "target_language": args.target_language,
        }.items()
        if value
    }
    if overrides:
        settings = settings.model_copy(update=overrides)
    pipeline = PdfTranslationPipeline(settings)

    def progress(stage: str, pct: float, message: str) -> None:
        print(f"[{pct:6.2f}%] {stage}: {message}", flush=True)

    result = await pipeline.translate(args.pdf, output_dir=args.output_dir, progress=progress)
    print(
        json.dumps(
            {
                "job_id": result.job_id,
                "kind": result.profile.kind.value,
                "route": result.profile.route.value,
                "translated_pdf": str(result.artifacts.translated_pdf),
                "bilingual_pdf": (
                    str(result.artifacts.bilingual_pdf) if result.artifacts.bilingual_pdf else None
                ),
                "manifest": str(result.artifacts.manifest),
                "layout_json": (
                    str(result.artifacts.layout_json) if result.artifacts.layout_json else None
                ),
                "layout_verification": (
                    str(result.artifacts.layout_verification)
                    if result.artifacts.layout_verification
                    else None
                ),
                "body_stats": result.body_stats.__dict__,
                "table_stats": result.table_stats.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "classify":
        print(json.dumps(classify_pdf(args.pdf).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "translate":
        asyncio.run(_translate(args))
    else:
        import uvicorn

        uvicorn.run("ocr_pdf_agent.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
