"""Isolated adapter around the public PDFMathTranslate v1 runtime."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from string import Template
from typing import Any, Protocol
from uuid import uuid4

import fitz
from tenacity import stop_after_attempt, stop_any

from .config import Settings
from .language import detect_pdf_language, pdfmath_language
from .llm import normalize_openai_base_url
from .models import PdfEngineArtifacts

logger = logging.getLogger(__name__)
Progress = Callable[[str, float, str], None]
_ENGINE_LOCK = threading.Lock()
_LAYOUT_MODEL: Any | None = None


class PdfMathTranslateError(RuntimeError):
    pass


class PdfEngine(Protocol):
    async def translate(
        self,
        input_pdf: str | Path,
        output_dir: str | Path,
        *,
        on_progress: Progress | None = None,
    ) -> PdfEngineArtifacts: ...


def _patch_optional_tencent_types() -> None:
    try:
        module = importlib.import_module("tencentcloud.tmt.v20180321.models")
    except Exception:
        return
    missing = [
        name for name in ("TextTranslateRequest", "TextTranslateResponse") if not hasattr(module, name)
    ]
    if not missing:
        return

    class UnsupportedTencentTextTranslate:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise PdfMathTranslateError(
                "The installed Tencent SDK is incompatible with pdf2zh; use OpenAI-compatible mode"
            )

    for name in missing:
        setattr(module, name, UnsupportedTencentTextTranslate)


@contextmanager
def _bounded_retries(max_attempts: int) -> Iterator[None]:
    _patch_optional_tencent_types()
    converter = importlib.import_module("pdf2zh.converter")
    original_retry = getattr(converter, "retry", None)
    if not callable(original_retry):
        yield
        return

    def bounded_retry(*args: Any, **kwargs: Any):
        attempt_stop = stop_after_attempt(max_attempts)
        existing_stop = kwargs.get("stop")
        kwargs["stop"] = stop_any(existing_stop, attempt_stop) if existing_stop else attempt_stop
        kwargs["reraise"] = True
        return original_retry(*args, **kwargs)

    converter.retry = bounded_retry
    try:
        yield
    finally:
        converter.retry = original_retry


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_pdf2zh() -> tuple[Callable[..., Any], type[Any]]:
    _patch_optional_tencent_types()
    try:
        # This import order avoids an initialization interaction observed in
        # pdf2zh 1.9.11 when its package __init__ eagerly loads everything.
        importlib.import_module("pdf2zh.converter")
        high_level = importlib.import_module("pdf2zh.high_level")
        doclayout = importlib.import_module("pdf2zh.doclayout")
    except Exception as exc:
        raise PdfMathTranslateError(f"Unable to import pdf2zh==1.9.11: {exc}") from exc
    translate = getattr(high_level, "translate", None)
    model_type = getattr(doclayout, "OnnxModel", None)
    if not callable(translate) or not callable(getattr(model_type, "load_available", None)):
        raise PdfMathTranslateError("pdf2zh does not expose the expected public API")
    return translate, model_type


def _prompt(settings: Settings) -> Template:
    domain = f"\n{settings.domain_prompt.strip()}" if settings.domain_prompt.strip() else ""
    return Template(
        f"{settings.system_prompt}{domain}\n\n"
        "You are translating a structured scientific PDF. Keep formula placeholders "
        "such as {{v0}}, citations, units, chemical formulas and identifiers unchanged. "
        "Return only the translated text.\n\n"
        "Source language: $lang_in\nTarget language: $lang_out\n\nText:\n$text"
    )


def _count_words(path: Path) -> int:
    with fitz.open(str(path)) as document:
        return sum(len(page.get_text("words")) for page in document)


class PdfMathTranslateEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def translate(
        self,
        input_pdf: str | Path,
        output_dir: str | Path,
        *,
        on_progress: Progress | None = None,
    ) -> PdfEngineArtifacts:
        path = Path(input_pdf).resolve()
        destination = Path(output_dir).resolve()
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"PDFMathTranslate requires an existing PDF: {path}")
        destination.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(self._translate_serialized, path, destination, on_progress)

    def _translate_serialized(
        self,
        path: Path,
        destination: Path,
        on_progress: Progress | None,
    ) -> PdfEngineArtifacts:
        # pdf2zh 1.9.11 patches retry decorators and maintains module-level
        # model state. Serialize engine calls inside one worker process; API
        # replicas remain the horizontal-scaling boundary.
        with _ENGINE_LOCK:
            return self._translate_sync(path, destination, on_progress)

    def _translate_sync(
        self,
        path: Path,
        destination: Path,
        on_progress: Progress | None,
    ) -> PdfEngineArtifacts:
        started = time.monotonic()
        source_language = detect_pdf_language(path, self.settings.source_language)
        target_language = pdfmath_language(self.settings.target_language)

        def emit(stage: str, pct: float, message: str) -> None:
            if on_progress:
                on_progress(stage, max(0.0, min(100.0, pct)), message)

        emit("pdfmathtranslate", 1.0, "Loading PDFMathTranslate layout model")
        translate, model_type = _load_pdf2zh()
        global _LAYOUT_MODEL
        if _LAYOUT_MODEL is None:
            _LAYOUT_MODEL = model_type.load_available()
        model = _LAYOUT_MODEL
        cancellation = threading.Event()

        def callback(progress: object) -> None:
            try:
                current = int(progress.n)
                total = int(progress.total)
            except (TypeError, ValueError):
                return
            if total > 0:
                emit(
                    "pdfmathtranslate",
                    5.0 + 88.0 * current / total,
                    f"Translating PDF page {min(current, total)}/{total}",
                )

        stage_dir = destination / f".pdfmath-input-{uuid4().hex}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged = stage_dir / path.name
        shutil.copy2(path, staged)
        environment = {
            "HF_ENDPOINT": self.settings.pdfmathtranslate_hf_endpoint,
        }
        envs = {
            "OPENAI_API_KEY": self.settings.llm_key,
            "OPENAI_BASE_URL": normalize_openai_base_url(self.settings.base_url),
            "OPENAI_MODEL": self.settings.model_name,
            "OPENAI_STREAM": "false",
        }
        try:
            with (
                _temporary_environment(environment),
                _bounded_retries(self.settings.pdfmathtranslate_max_attempts),
            ):
                results = translate(
                    files=[str(staged)],
                    output=str(destination),
                    lang_in=source_language,
                    lang_out=target_language,
                    service="openai",
                    thread=min(16, self.settings.max_workers),
                    callback=callback,
                    cancellation_event=cancellation,
                    model=model,
                    envs=envs,
                    prompt=_prompt(self.settings),
                    ignore_cache=self.settings.ignore_translation_cache,
                )
        finally:
            staged.unlink(missing_ok=True)
            with suppress(OSError):
                stage_dir.rmdir()
        if not results:
            raise PdfMathTranslateError("PDFMathTranslate returned no output artifacts")
        try:
            mono_value, bilingual_value = results[0]
        except (TypeError, ValueError) as exc:
            raise PdfMathTranslateError("PDFMathTranslate returned an invalid artifact tuple") from exc
        mono = Path(str(mono_value)).resolve()
        bilingual = Path(str(bilingual_value)).resolve() if bilingual_value else None
        if not mono.is_file() or mono.stat().st_size == 0:
            raise PdfMathTranslateError("PDFMathTranslate mono output is missing or empty")
        if bilingual is not None and (not bilingual.is_file() or bilingual.stat().st_size == 0):
            bilingual = None
        emit("pdfmathtranslate", 100.0, "PDFMathTranslate completed")
        return PdfEngineArtifacts(
            mono_pdf=mono,
            bilingual_pdf=bilingual,
            source_language=source_language,
            target_language=target_language,
            word_count=_count_words(path),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


class CopyPdfEngine:
    """Offline engine used to verify routing and post-processing."""

    def __init__(self) -> None:
        self.inputs: list[Path] = []

    async def translate(
        self,
        input_pdf: str | Path,
        output_dir: str | Path,
        *,
        on_progress: Progress | None = None,
    ) -> PdfEngineArtifacts:
        source = Path(input_pdf).resolve()
        destination = Path(output_dir).resolve() / "engine-mono.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.inputs.append(source)
        if on_progress:
            on_progress("pdfmathtranslate", 100.0, "Offline copy engine completed")
        return PdfEngineArtifacts(destination, None, "en", "zh", _count_words(source), 1)
