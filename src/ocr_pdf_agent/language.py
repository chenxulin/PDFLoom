"""Resolve the source-language code required by PDFMathTranslate."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-z]")


def normalize_language(value: str, *, default: str = "auto") -> str:
    raw = (value or default).strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "detect": "auto",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "chinese": "zh",
        "en-us": "en",
        "en-gb": "en",
        "english": "en",
        "ja-jp": "ja",
        "japanese": "ja",
        "ko-kr": "ko",
        "korean": "ko",
    }
    return aliases.get(raw, raw or default)


def detect_language(text: str, *, fallback: str = "en") -> str:
    han = len(_HAN.findall(text))
    kana = len(_KANA.findall(text))
    hangul = len(_HANGUL.findall(text))
    latin = len(_LATIN.findall(text))
    if kana >= 2 and kana >= max(1, han // 8):
        return "ja"
    if hangul >= 2:
        return "ko"
    if han >= 6 and (han / max(1, han + latin) >= 0.12 or latin < 8):
        return "zh"
    if latin >= 5:
        return "en"
    if han:
        return "zh"
    return normalize_language(fallback, default="en")


def detect_pdf_language(path: str | Path, configured: str) -> str:
    normalized = normalize_language(configured)
    if normalized != "auto":
        return normalized
    parts: list[str] = []
    with fitz.open(str(path)) as document:
        for page_index in range(min(5, document.page_count)):
            parts.append(document[page_index].get_text("text"))
            if sum(map(len, parts)) >= 12000:
                break
    return detect_language("\n".join(parts)[:12000])


def pdfmath_language(value: str) -> str:
    normalized = normalize_language(value, default="en")
    return "zh" if normalized.startswith("zh") else normalized
