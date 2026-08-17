"""Environment-only settings for the standalone service.

The service deliberately has no database or dependency on the original
application's settings tables. Secrets are represented as ``SecretStr`` so a
settings object can be logged without revealing credentials.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SYSTEM_PROMPT = """You are a rigorous professional document translator.
Translate faithfully and return only the translated text. Preserve every number,
date, percentage, unit, chemical formula, identifier, URL, citation and placeholder
character-for-character. Do not invent, omit, summarize, or annotate content.
For table cells, keep the translation concise and on one line."""

DEFAULT_CMC_DOMAIN_PROMPT = """This document belongs to the pharmaceutical/CMC domain.
Use formal terminology suitable for quality standards, batch records, analytical
methods, stability studies and regulatory submissions. Interpret short table labels
from their headers and neighbouring cells, never as isolated general-language words.
When the target is Chinese, use these mandatory conventions where applicable:
- Assay as a quality-control test/item -> 含量测定, never 检测.
- release/released in batch disposition -> 放行/批准放行, never 发布.
- batch -> 批次; specification -> 质量标准 or 规定, according to grammar.
Facts, acceptance criteria and protected literals must remain exactly traceable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: SecretStr = SecretStr("")
    base_url: str
    model_name: str
    target_language: str = "zh-CN"
    source_language: str = Field(
        default="auto",
        validation_alias=AliasChoices("SOURCE_LANGUAGE", "PDFMATH_TRANSLATE_SOURCE_LANGUAGE"),
    )
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    domain_prompt: str = DEFAULT_CMC_DOMAIN_PROMPT
    enforce_cmc_terminology: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_workers: int = Field(default=4, ge=1, le=100)
    llm_timeout_seconds: float = Field(default=180.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)

    paddleocr_enabled: bool = True
    paddleocr_api_url: str = Field(
        default="http://192.168.1.88:18093",
        validation_alias=AliasChoices("PADDLEOCR_API_URL", "JOINCARE_OCR_BASE_URL", "OCR_API_URL"),
    )
    paddleocr_api_path: str = "/api/v1/structure"
    paddleocr_service_token: SecretStr = SecretStr("")
    attachment_ocr_service_token: SecretStr = SecretStr("")
    joincare_ocr_token: SecretStr = SecretStr("")
    ocr_service_token: SecretStr = SecretStr("")
    paddleocr_timeout_seconds: float = Field(default=3600.0, gt=0)
    paddleocr_attempts: int = Field(default=2, ge=1, le=3)
    # Keep the standalone service aligned with Joincare's remote OCR contract:
    # only the OCR proxy is downsampled, requests are globally bounded, and
    # normalized results can be safely reused from the disk cache.
    paddleocr_max_concurrent_requests: int = Field(default=2, ge=1, le=20)
    paddleocr_proxy_target_dpi: int = Field(default=600, ge=72, le=1200)
    paddleocr_downsample_threshold_dpi: int = Field(default=600, ge=72, le=2400)
    paddleocr_cache_enabled: bool = True
    paddleocr_cache_dir: Path | None = None
    paddleocr_orientation_retry: bool = True

    pdfmathtranslate_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias=AliasChoices("PDFMATH_TRANSLATE_MAX_ATTEMPTS", "PDFMATHTRANSLATE_MAX_ATTEMPTS"),
    )
    pdfmathtranslate_hf_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("PDFMATH_TRANSLATE_HF_ENDPOINT", "PDFMATHTRANSLATE_HF_ENDPOINT"),
    )
    ignore_translation_cache: bool = False
    strict_output_qa: bool = True

    storage_dir: Path = PROJECT_ROOT / "storage"
    max_upload_mib: int = Field(default=200, ge=1, le=2048)
    max_concurrent_jobs: int = Field(default=2, ge=1, le=20)
    service_api_token: SecretStr = SecretStr("")

    table_font_size: float = Field(default=9.0, ge=6.0, le=16.0)
    table_min_font_size: float = Field(default=6.0, ge=4.0, le=12.0)
    table_line_height: float = Field(default=1.25, ge=1.0, le=2.0)
    regular_font_path: Path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    bold_font_path: Path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

    @field_validator("base_url", "paddleocr_api_url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return normalized

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MODEL_NAME must not be empty")
        return normalized

    @field_validator("paddleocr_api_path")
    @classmethod
    def normalize_api_path(cls, value: str) -> str:
        normalized = value.strip()
        return "/" + normalized.lstrip("/") if normalized else ""

    @property
    def llm_key(self) -> str:
        return self.api_key.get_secret_value().strip()

    @property
    def ocr_token(self) -> str:
        # Empty migration variables are common in Compose environments. Check
        # each value explicitly instead of using AliasChoices, which correctly
        # treats an empty first alias as present and would hide later fallbacks.
        return next(
            (
                value.get_secret_value().strip()
                for value in (
                    self.paddleocr_service_token,
                    self.attachment_ocr_service_token,
                    self.joincare_ocr_token,
                    self.ocr_service_token,
                )
                if value.get_secret_value().strip()
            ),
            "",
        )

    @property
    def access_token(self) -> str:
        return self.service_api_token.get_secret_value().strip()

    @property
    def ocr_endpoint(self) -> str:
        if self.paddleocr_api_url.endswith(self.paddleocr_api_path):
            return self.paddleocr_api_url
        return f"{self.paddleocr_api_url}{self.paddleocr_api_path}"

    def validate_for_translation(self, *, needs_ocr: bool) -> None:
        if not self.llm_key:
            raise ValueError("API_KEY is required for PDF translation")
        if needs_ocr and (not self.paddleocr_enabled or not self.ocr_token):
            raise ValueError(
                "Scanned PDFs require PADDLEOCR_SERVICE_TOKEN "
                "(ATTACHMENT_OCR_SERVICE_TOKEN is also accepted for migration)"
            )
