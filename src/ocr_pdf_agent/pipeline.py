"""End-to-end route orchestration for native and scanned PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

import fitz

from .classifier import classify_pdf
from .config import Settings
from .llm import Translator
from .models import (
    BodyRedrawStats,
    PdfEngineArtifacts,
    PipelineArtifacts,
    PipelineResult,
    ProcessingRoute,
    TableRedrawStats,
)
from .ocr_client import PaddleOcrClient
from .ocr_document_typography import OcrBodyTranslationPlan, restore_ocr_document_typography
from .ocr_heading_typography import OcrHeadingTranslationPlan
from .ocr_searchable_pdf import build_ocr_searchable_pdf
from .ocr_serial_translation import translate_ocr_content_serially
from .ocr_table_redraw import OcrTableTranslationPlan, redraw_ocr_tables
from .pdfmath import PdfEngine, PdfMathTranslateEngine
from .terminology import missing_requirements, requirements_for


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_artifact(source: Path, destination: Path) -> Path:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _pdf_text(path: Path) -> str:
    with fitz.open(str(path)) as document:
        return "\n".join(page.get_text("text") for page in document)


def _output_quality(
    source_text: str,
    target_text: str,
    target_language: str,
    *,
    enforce_cmc_terminology: bool = True,
) -> dict[str, Any]:
    normalized_source = unicodedata.normalize("NFKC", source_text)
    normalized_target = unicodedata.normalize("NFKC", target_text)
    source_compact = re.sub(r"\s+", "", normalized_source).casefold()
    target_compact = re.sub(r"\s+", "", normalized_target).casefold()
    similarity = (
        SequenceMatcher(None, source_compact[:20000], target_compact[:20000], autojunk=False).ratio()
        if source_compact and target_compact
        else 0.0
    )
    source_latin = len(re.findall(r"[A-Za-z]", normalized_source))
    source_cjk = len(re.findall(r"[\u3400-\u9fff]", normalized_source))
    target_latin = len(re.findall(r"[A-Za-z]", normalized_target))
    target_cjk = len(re.findall(r"[\u3400-\u9fff]", normalized_target))
    target = target_language.lower().replace("_", "-")
    failures: list[str] = []
    if not target_compact:
        failures.append("translated PDF has no extractable text")
    if source_latin >= 20 and target.startswith(("zh", "ja", "ko")) and target_cjk < 3:
        failures.append("target PDF does not contain the requested CJK target language")
    if source_cjk >= 5 and target.startswith("en") and target_latin < 10:
        failures.append("target PDF does not contain the requested English target language")
    if similarity >= 0.97 and (
        (source_latin >= 20 and target.startswith(("zh", "ja", "ko")))
        or (source_cjk >= 5 and target.startswith("en"))
    ):
        failures.append("target text is effectively identical to source text")
    requirements = requirements_for(normalized_source, target_language) if enforce_cmc_terminology else ()
    missing_terms = missing_requirements(target_compact, requirements)
    failures.extend(
        f"mandatory terminology missing: {requirement.source_term} -> {requirement.required_target}"
        for requirement in missing_terms
    )
    return {
        "source_characters": len(source_compact),
        "target_characters": len(target_compact),
        "source_target_similarity": round(similarity, 6),
        "source_latin_characters": source_latin,
        "source_cjk_characters": source_cjk,
        "target_latin_characters": target_latin,
        "target_cjk_characters": target_cjk,
        "terminology_requirements": [
            {
                "source": requirement.source_term,
                "target": requirement.required_target,
                "passed": requirement not in missing_terms,
            }
            for requirement in requirements
        ],
        "passed": not failures,
        "failures": failures,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_ocr_translation_ledger(
    path: Path,
    body_plan: OcrBodyTranslationPlan,
    heading_plan: OcrHeadingTranslationPlan,
    table_plan: OcrTableTranslationPlan,
) -> Path:
    payload = {
        "schema_version": 1,
        "translation_order": "page_reading_order_serial",
        "body_regions": [
            {
                "page": region.page_idx + 1,
                "bbox": list(region.bbox),
                "page_size": list(region.page_size),
                "source": region.source_text,
                "target": region.target_text,
                "protected_literals": list(region.protected_values),
            }
            for region in body_plan.regions
        ],
        "heading_regions": [
            {
                "page": region.page_idx + 1,
                "bbox": list(region.bbox),
                "page_size": list(region.page_size),
                "source": region.source_text,
                "target": region.target_text,
                "protected_literals": list(region.protected_values),
            }
            for region in heading_plan.regions
        ],
        "tables": [
            {
                "table": table.index + 1,
                "page": table.page_idx + 1,
                "bbox": list(table.bbox),
                "page_size": list(table.page_size),
                "rows": table.row_count,
                "columns": table.column_count,
                "preserved_as_image": table.preserve_as_image,
                "cells": [
                    {
                        "row": cell.row + 1,
                        "column": cell.column + 1,
                        "row_span": cell.row_span,
                        "column_span": cell.column_span,
                        "source": cell.source_text,
                        "target": cell.target_text,
                        "translated": cell.translated,
                        "protected_literals": list(cell.protected_values),
                    }
                    for cell in table.cells
                ],
            }
            for table in table_plan.tables
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class _Manifest:
    def __init__(self, path: Path, initial: dict[str, Any]) -> None:
        self.path = path
        self.data = initial
        self.write()

    def update(self, **changes: Any) -> None:
        self.data.update(changes)
        self.write()

    def stage(self, name: str, pct: float, message: str) -> None:
        self.data["stage"] = name
        self.data["progress"] = round(max(0.0, min(100.0, pct)), 2)
        self.data["message"] = message
        history = self.data.setdefault("progress_history", [])
        history.append(
            {
                "stage": name,
                "progress": self.data["progress"],
                "message": message,
                "timestamp": int(time.time()),
            }
        )
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(self.data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class PdfTranslationPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ocr_client: Any | None = None,
        pdf_engine: PdfEngine | None = None,
        translator: Translator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.ocr_client = ocr_client
        self.pdf_engine = pdf_engine or PdfMathTranslateEngine(self.settings)
        self.translator = translator

    async def translate(
        self,
        input_pdf: str | Path,
        *,
        output_dir: str | Path | None = None,
        job_id: str | None = None,
        progress: Any = None,
    ) -> PipelineResult:
        source = Path(input_pdf).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as stream:
            magic = stream.read(5)
        if source.suffix.lower() != ".pdf" or magic != b"%PDF-":
            raise ValueError("Only valid PDF uploads are accepted")
        if source.stat().st_size > self.settings.max_upload_mib * 1024 * 1024:
            raise ValueError(f"PDF exceeds {self.settings.max_upload_mib} MiB limit")

        identifier = job_id or uuid4().hex
        destination = (
            Path(output_dir).resolve()
            if output_dir is not None
            else (self.settings.storage_dir / "jobs" / identifier).resolve()
        )
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "manifest.json"
        display_filename = source.name
        if job_id and manifest_path.is_file():
            try:
                queued_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                display_filename = str(
                    (queued_manifest.get("input") or {}).get("filename") or display_filename
                )
            except (OSError, ValueError, TypeError):
                pass
        manifest = _Manifest(
            manifest_path,
            {
                "schema_version": 1,
                "job_id": identifier,
                "status": "running",
                "stage": "starting",
                "progress": 0.0,
                "message": "Preparing PDF",
                "created_at": int(time.time()),
                "input": {
                    "filename": display_filename,
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                },
                "options": {
                    "source_language": self.settings.source_language,
                    "target_language": self.settings.target_language,
                },
                "artifacts": {},
                "progress_history": [],
            },
        )
        timings: dict[str, int] = {}
        started = time.monotonic()

        def emit(stage: str, pct: float, message: str) -> None:
            manifest.stage(stage, pct, message)
            if progress:
                progress(stage, pct, message)

        try:
            archived_source = destination / "source.pdf"
            _copy_artifact(source, archived_source)

            mark = time.monotonic()
            emit("classify", 1.0, "Classifying every PDF page")
            profile = await asyncio.to_thread(classify_pdf, archived_source)
            timings["classify"] = int((time.monotonic() - mark) * 1000)
            needs_ocr = profile.route == ProcessingRoute.PADDLEOCR_THEN_PDFMATHTRANSLATE
            self.settings.validate_for_translation(needs_ocr=needs_ocr)
            manifest.update(profile=profile.to_dict())

            ocr_json_path: Path | None = None
            ocr_input_path: Path | None = None
            translation_ledger_path: Path | None = None
            ocr_result: dict[str, Any] | None = None
            engine_input = archived_source
            if needs_ocr:
                mark = time.monotonic()
                emit("paddleocr", 3.0, "Running PaddleOCR PP-StructureV3")
                client = self.ocr_client or PaddleOcrClient(self.settings)
                owns_client = self.ocr_client is None
                try:
                    ocr_result = await client.recognize(archived_source)
                finally:
                    if owns_client:
                        await client.aclose()
                timings["paddleocr"] = int((time.monotonic() - mark) * 1000)
                ocr_json_path = destination / "ocr_ppstructurev3.json"
                ocr_json_path.write_text(
                    json.dumps(ocr_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                emit("ocr-layer", 12.0, "Building a clean OCR source layer")
                mark = time.monotonic()
                ocr_input_path = destination / "ocr_pdfmathtranslate_input.pdf"
                layer = await asyncio.to_thread(
                    build_ocr_searchable_pdf,
                    archived_source,
                    ocr_input_path,
                    ocr_result,
                    input_profile=profile.kind.value,
                    # The standalone Agent intentionally exposes only the
                    # stable v1 engine. Its input must therefore mask scanned
                    # source glyphs and supply one visible OCR text layer.
                    source_layer_mode="visible_masked",
                )
                engine_input = layer.pdf_path
                timings["ocr_layer"] = int((time.monotonic() - mark) * 1000)
                manifest.update(ocr_layer=_json_safe(asdict(layer)))

            mark = time.monotonic()

            def engine_progress(stage: str, pct: float, message: str) -> None:
                base = 18.0 if needs_ocr else 4.0
                span = 66.0 if needs_ocr else 82.0
                emit(stage, base + span * max(0.0, min(100.0, pct)) / 100.0, message)

            engine_artifacts: PdfEngineArtifacts = await self.pdf_engine.translate(
                engine_input,
                destination / "pdfmathtranslate",
                on_progress=engine_progress,
            )
            timings["pdfmathtranslate"] = int((time.monotonic() - mark) * 1000)

            table_stats = TableRedrawStats()
            body_stats = BodyRedrawStats()
            working_mono_pdf = engine_artifacts.mono_pdf
            table_result = None
            if ocr_result is not None:
                mark = time.monotonic()
                emit(
                    "content-translate-serial",
                    84.5,
                    "Translating OCR headings, prose, and tables in page order",
                )
                serial = await translate_ocr_content_serially(
                    ocr_result=ocr_result,
                    settings=self.settings,
                    translator=self.translator,
                    on_progress=lambda stage, pct, msg: emit(
                        stage,
                        84.5 + pct * 0.075,
                        msg,
                    ),
                )
                translation_ledger_path = _write_ocr_translation_ledger(
                    destination / "translation_ledger.json",
                    serial.body_plan,
                    serial.heading_plan,
                    serial.table_plan,
                )
                emit(
                    "typography",
                    92.0,
                    "Redrawing complete OCR paragraphs and headings with unified typography",
                )
                typography = await asyncio.to_thread(
                    restore_ocr_document_typography,
                    ocr_result=ocr_result,
                    translated_pdf=engine_artifacts.mono_pdf,
                    bilingual_pdf=engine_artifacts.bilingual_pdf,
                    # ``build_ocr_searchable_pdf`` applies any OCR-detected
                    # right-angle correction.  Use that same coordinate
                    # system while restoring protected visuals and rebuilding
                    # body typography, exactly as Joincare's scan route does.
                    source_pdf=ocr_input_path,
                    body_translation_plan=serial.body_plan,
                    heading_translation_plan=serial.heading_plan,
                )
                body_stats = BodyRedrawStats(
                    blocks_detected=(serial.body_plan.region_count + serial.heading_plan.region_count),
                    blocks_translated=(
                        serial.body_plan.translated_regions + serial.heading_plan.translated_regions
                    ),
                    blocks_redrawn=(typography.repaired_body_blocks + typography.headings.repaired_headings),
                    protected_literals=(
                        serial.body_plan.protected_values + serial.heading_plan.protected_values
                    ),
                )
                if serial.table_plan.table_count:
                    emit(
                        "table-redraw",
                        95.0,
                        "Replacing source tables with fitted 9 pt vector tables",
                    )
                    table_result = await asyncio.to_thread(
                        redraw_ocr_tables,
                        ocr_result=ocr_result,
                        plan=serial.table_plan,
                        translated_pdf=working_mono_pdf,
                        bilingual_pdf=engine_artifacts.bilingual_pdf,
                        body_font_path=typography.body_font_path,
                        bold_font_path=typography.headings.heading_font_path,
                        source_page_indices=tuple(
                            getattr(typography, "source_page_indices", ()) or ()
                        ),
                        continuation_page_indices=tuple(
                            getattr(typography, "continuation_page_indices", ())
                            or ()
                        ),
                    )
                    table_stats = TableRedrawStats(
                        tables_detected=serial.table_plan.table_count,
                        tables_redrawn=table_result.redrawn_tables,
                        cells_redrawn=table_result.redrawn_cells,
                        cells_translated=serial.table_plan.translated_cells,
                        protected_literals=serial.table_plan.protected_values,
                        preserved_image_tables=(serial.table_plan.image_preserved_tables),
                        continuation_pages=table_result.continuation_pages,
                    )
                manifest.update(
                    serial_translation={
                        "translated_items": serial.translated_items,
                        "model_requests": serial.model_requests,
                        "order": "page_reading_order_serial",
                    },
                    typography=_json_safe(asdict(typography)),
                    automatic_qa="disabled",
                )
                timings["ocr_translate_redraw"] = int((time.monotonic() - mark) * 1000)

            emit("validate", 97.0, "Validating output artifacts")
            translated = _copy_artifact(
                working_mono_pdf,
                destination / "translated.pdf",
            )
            bilingual = (
                _copy_artifact(engine_artifacts.bilingual_pdf, destination / "bilingual.pdf")
                if engine_artifacts.bilingual_pdf is not None
                else None
            )
            with fitz.open(str(translated)) as document:
                if document.page_count < 1:
                    raise RuntimeError("Translated PDF has no pages")
                final_page_count = document.page_count
                final_text_characters = sum(
                    len(page.get_text("text").strip()) for page in document
                )

            # Match Joincare's current scanned-PDF policy: the OCR-specific
            # automatic quality gate is deliberately disabled. The renderer may
            # append continuation pages for expanded prose or tables, so a
            # source-page-count/dimension gate would reject valid deliveries.
            page_dimensions_match: bool | None = None
            output_quality: dict[str, Any] | None = None
            if ocr_result is None:
                with fitz.open(str(archived_source)) as source_document, fitz.open(
                    str(translated)
                ) as document:
                    if document.page_count != source_document.page_count:
                        raise RuntimeError(
                            "Translated PDF page count differs from the uploaded PDF"
                        )
                    page_dimensions_match = all(
                        abs(
                            document[index].rect.width
                            - source_document[index].rect.width
                        )
                        <= 0.1
                        and abs(
                            document[index].rect.height
                            - source_document[index].rect.height
                        )
                        <= 0.1
                        for index in range(document.page_count)
                    )
                    if not page_dimensions_match:
                        raise RuntimeError(
                            "Translated PDF page dimensions differ from the uploaded PDF"
                        )
                output_quality = _output_quality(
                    _pdf_text(archived_source),
                    _pdf_text(translated),
                    self.settings.target_language,
                    enforce_cmc_terminology=self.settings.enforce_cmc_terminology,
                )
                if self.settings.strict_output_qa and not output_quality["passed"]:
                    raise RuntimeError(
                        "Output translation QA failed: "
                        + "; ".join(output_quality["failures"])
                    )
            timings["total"] = int((time.monotonic() - started) * 1000)
            artifact_data = {
                "source_pdf": str(archived_source),
                "translated_pdf": str(translated),
                "translated_sha256": _sha256(translated),
                "bilingual_pdf": str(bilingual) if bilingual else None,
                "ocr_json": str(ocr_json_path) if ocr_json_path else None,
                "ocr_input_pdf": str(ocr_input_path) if ocr_input_path else None,
                "translation_ledger": (
                    str(translation_ledger_path) if translation_ledger_path else None
                ),
            }
            manifest.update(
                status="completed",
                stage="completed",
                progress=100.0,
                message="Translation completed",
                completed_at=int(time.time()),
                engine={
                    "name": "pdfmathtranslate-v1",
                    "source_language": engine_artifacts.source_language,
                    "target_language": engine_artifacts.target_language,
                    "word_count": engine_artifacts.word_count,
                    "duration_ms": engine_artifacts.duration_ms,
                },
                body_redraw=_json_safe(asdict(body_stats)),
                table_redraw=_json_safe(asdict(table_stats)),
                validation={
                    "final_page_count": final_page_count,
                    "page_dimensions_match": page_dimensions_match,
                    "final_text_characters": final_text_characters,
                    "translation_quality": output_quality,
                    "automatic_qa": "disabled" if ocr_result is not None else "enabled",
                    "layout": None,
                },
                timings_ms=timings,
                artifacts=artifact_data,
            )
            artifacts = PipelineArtifacts(
                output_dir=destination,
                translated_pdf=translated,
                bilingual_pdf=bilingual,
                manifest=manifest_path,
                ocr_json=ocr_json_path,
                ocr_input_pdf=ocr_input_path,
                translation_ledger=translation_ledger_path,
            )
            return PipelineResult(
                job_id=identifier,
                input_pdf=archived_source,
                profile=profile,
                artifacts=artifacts,
                body_stats=body_stats,
                table_stats=table_stats,
                timings_ms=timings,
            )
        except Exception as exc:
            timings["total"] = int((time.monotonic() - started) * 1000)
            manifest.update(
                status="failed",
                stage="failed",
                message=str(exc),
                failed_at=int(time.time()),
                error={"type": type(exc).__name__, "message": str(exc)},
                timings_ms=timings,
            )
            raise
