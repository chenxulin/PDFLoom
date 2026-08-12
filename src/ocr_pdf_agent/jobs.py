"""Small disk-backed job registry for the HTTP service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .pipeline import PdfTranslationPipeline


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs_dir = settings.storage_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._tasks: set[asyncio.Task[Any]] = set()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(char not in "0123456789abcdef" for char in job_id.lower()):
            raise ValueError("Invalid job id")
        return self.jobs_dir / job_id

    def read(self, job_id: str) -> dict[str, Any]:
        manifest = self.job_dir(job_id) / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(job_id)
        return json.loads(manifest.read_text(encoding="utf-8"))

    def dispatch(
        self,
        job_id: str,
        source: Path,
        *,
        settings_overrides: dict[str, Any] | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._run(job_id, source, settings_overrides or {}),
            name=f"ocr-pdf-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job_id: str, source: Path, settings_overrides: dict[str, Any]) -> None:
        async with self._semaphore:
            effective = self.settings.model_copy(update=settings_overrides)
            pipeline = PdfTranslationPipeline(effective)
            try:
                await pipeline.translate(
                    source,
                    output_dir=self.job_dir(job_id),
                    job_id=job_id,
                )
            except Exception:
                # The pipeline persists a sanitized failure in manifest.json.
                return

    async def shutdown(self) -> None:
        if not self._tasks:
            return
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
