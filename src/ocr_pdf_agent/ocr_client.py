"""Standalone adapter for Joincare's PP-StructureV3 OCR gateway."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .ocr_gateway import _correct_common_ocr_confusions, pdf_to_layout


class OcrServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PaddleOcrClient:
    """Use the production-equivalent page OCR flow with standalone settings."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        # A caller-supplied client is retained for deterministic integration
        # tests. Normal service calls let the gateway own the short-lived client
        # so its content-addressed disk cache remains enabled.
        self._client = client

    async def __aenter__(self) -> PaddleOcrClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        # The Agent never owns a caller-supplied client. Gateway-created
        # clients are closed by its request coroutine.
        return None

    async def health(self) -> dict[str, Any]:
        endpoint = self.settings.ocr_endpoint
        suffix = self.settings.paddleocr_api_path
        if suffix and endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.settings.paddleocr_timeout_seconds,
                write=120.0,
                pool=10.0,
            ),
            trust_env=False,
        )
        try:
            response = await client.get(f"{endpoint.rstrip('/')}/health")
            return self._json_response(response)
        except httpx.HTTPError as exc:
            raise OcrServiceError(f"OCR health request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def recognize(
        self,
        pdf_path: str | Path,
        *,
        cancel_event: Any | None = None,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.settings.max_upload_mib * 1024 * 1024:
            raise ValueError(
                f"PDF exceeds {self.settings.max_upload_mib} MiB upload limit"
            )
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise ValueError("Only valid PDF files are accepted")

        try:
            return await pdf_to_layout(
                path.read_bytes(),
                filename=path.name,
                settings=self.settings,
                client=self._client,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
        except OcrServiceError:
            raise
        except ValueError as exc:
            raise OcrServiceError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise OcrServiceError(f"OCR request failed: {exc}") from exc

    @staticmethod
    def _json_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            detail = (response.text or "").strip().replace("\n", " ")[:500]
            message = (
                "OCR authentication failed"
                if response.status_code in {401, 403}
                else "OCR service request failed"
            )
            suffix = f": {detail}" if detail else ""
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
    def _correct_common_confusions(payload: dict[str, Any]) -> dict[str, Any]:
        return _correct_common_ocr_confusions(payload)
