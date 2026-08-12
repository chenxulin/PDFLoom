"""Async client for the dedicated PaddleOCR PP-StructureV3 service."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx

from .config import Settings


class OcrServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_OOS_OCR_CONFUSION = re.compile(
    r"(?<![A-Za-z0-9])(?:0OS|O0S|00S)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


class PaddleOcrClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.paddleocr_timeout_seconds,
                write=120.0,
                pool=10.0,
            ),
            trust_env=False,
        )

    async def __aenter__(self) -> PaddleOcrClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        base = self.settings.paddleocr_api_url
        suffix = self.settings.paddleocr_api_path
        if suffix and base.endswith(suffix):
            base = base[: -len(suffix)]
        response = await self._client.get(f"{base.rstrip('/')}/health")
        return self._json_response(response)

    async def recognize(self, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.settings.max_upload_mib * 1024 * 1024:
            raise ValueError(f"PDF exceeds {self.settings.max_upload_mib} MiB upload limit")
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise ValueError("Only valid PDF files are accepted")

        last_error: Exception | None = None
        for attempt in range(1, self.settings.paddleocr_attempts + 1):
            try:
                data = path.read_bytes()
                response = await self._client.post(
                    self.settings.ocr_endpoint,
                    headers={"X-OCR-Service-Token": self.settings.ocr_token},
                    files={"file": (path.name, data, "application/pdf")},
                )
                payload = self._json_response(response)
                return self._correct_common_confusions(self._validate_payload(payload))
            except OcrServiceError as exc:
                last_error = exc
                if exc.status_code is not None and exc.status_code < 500:
                    raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            if attempt < self.settings.paddleocr_attempts:
                await asyncio.sleep(2 ** (attempt - 1))
        raise OcrServiceError(f"OCR request failed after bounded retries: {last_error}") from last_error

    @staticmethod
    def _json_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            excerpt = re.sub(r"\s+", " ", response.text or "").strip()[:500]
            if response.status_code in {401, 403}:
                message = "OCR authentication failed"
            elif response.status_code == 413:
                message = "PDF exceeds the OCR service upload limit"
            else:
                message = "OCR service request failed"
            suffix = f": {excerpt}" if excerpt else ""
            raise OcrServiceError(
                f"{message} (HTTP {response.status_code}){suffix}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OcrServiceError("OCR service did not return JSON") from exc
        if not isinstance(payload, dict):
            raise OcrServiceError("OCR service returned a non-object JSON value")
        return payload

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("provider") != "paddleocr-ppstructurev3":
            raise OcrServiceError("OCR response has an unexpected provider")
        for key in ("pages", "blocks", "regions"):
            if not isinstance(payload.get(key), list):
                raise OcrServiceError(f"OCR response field {key!r} must be a list")
        if not isinstance(payload.get("markdown", ""), str):
            raise OcrServiceError("OCR response field 'markdown' must be text")
        return payload

    @staticmethod
    def _correct_common_confusions(payload: dict[str, Any]) -> dict[str, Any]:
        """Apply the parent's high-confidence pharmaceutical acronym repair."""
        corrections = 0

        def corrected(value: Any) -> str:
            nonlocal corrections
            result, count = _OOS_OCR_CONFUSION.subn("OOS", str(value or ""))
            corrections += count
            return result

        for block in payload.get("blocks") or []:
            if isinstance(block, dict):
                block["text"] = corrected(block.get("text"))
        for region in payload.get("regions") or []:
            if not isinstance(region, dict):
                continue
            structured = region.get("structured_content")
            if isinstance(structured, str):
                region["structured_content"] = corrected(structured)
        payload["markdown"] = corrected(payload.get("markdown"))
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        metadata["common_text_corrections"] = int(metadata.get("common_text_corrections") or 0) + corrections
        return payload
