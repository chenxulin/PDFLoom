"""Shared value objects for routing and translation artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class PdfKind(StrEnum):
    EMPTY = "empty"
    BORN_DIGITAL = "born_digital"
    IMAGE_SCAN = "image_scan"
    SEARCHABLE_SCAN = "searchable_scan"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProcessingRoute(StrEnum):
    DIRECT_PDFMATHTRANSLATE = "direct_pdfmathtranslate"
    PADDLEOCR_THEN_PDFMATHTRANSLATE = "paddleocr_then_pdfmathtranslate"


@dataclass(frozen=True)
class PageSignal:
    page_index: int
    text_characters: int
    text_blocks: int
    images: int
    image_coverage: float
    classification: str


@dataclass(frozen=True)
class PdfProfile:
    kind: PdfKind
    route: ProcessingRoute
    pages: int
    native_text_pages: int
    scan_pages: int
    searchable_scan_pages: int
    average_image_coverage: float
    reason: str
    page_signals: tuple[PageSignal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["route"] = self.route.value
        return result


@dataclass(frozen=True)
class OcrLayerArtifact:
    pdf_path: Path
    inserted_blocks: int
    skipped_table_blocks: int
    skipped_visual_blocks: int
    skipped_noise_blocks: int
    removed_text_pages: int
    masked_regions: int


@dataclass(frozen=True)
class PdfEngineArtifacts:
    mono_pdf: Path
    bilingual_pdf: Path | None
    source_language: str
    target_language: str
    word_count: int
    duration_ms: int


@dataclass(frozen=True)
class TableRedrawStats:
    tables_detected: int = 0
    tables_redrawn: int = 0
    cells_redrawn: int = 0
    cells_translated: int = 0
    protected_literals: int = 0
    preserved_image_tables: int = 0
    continuation_pages: int = 0


@dataclass(frozen=True)
class BodyRedrawStats:
    blocks_detected: int = 0
    blocks_translated: int = 0
    blocks_redrawn: int = 0
    protected_literals: int = 0


@dataclass(frozen=True)
class PipelineArtifacts:
    output_dir: Path
    translated_pdf: Path
    bilingual_pdf: Path | None
    manifest: Path
    ocr_json: Path | None = None
    ocr_input_pdf: Path | None = None
    translation_ledger: Path | None = None


@dataclass(frozen=True)
class PipelineResult:
    job_id: str
    input_pdf: Path
    profile: PdfProfile
    artifacts: PipelineArtifacts
    body_stats: BodyRedrawStats = field(default_factory=BodyRedrawStats)
    table_stats: TableRedrawStats = field(default_factory=TableRedrawStats)
    timings_ms: dict[str, int] = field(default_factory=dict)


ProgressCallback = Any
