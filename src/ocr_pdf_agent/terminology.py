"""Built-in CMC terminology constraints for high-risk short translations."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminologyRequirement:
    source_term: str
    required_target: str
    reason: str
    exact_source: bool = False


_ZH_RULES = (
    TerminologyRequirement(
        source_term="assay",
        required_target="含量测定",
        reason="pharmaceutical quality-control test name",
        exact_source=True,
    ),
    TerminologyRequirement(
        source_term="release",
        required_target="放行",
        reason="pharmaceutical batch disposition",
    ),
)

_EXACT_EN_TARGETS = {
    "2、oos项目描述": "2. OOS Item Description",
    "3、实验室调查": "3. Laboratory Investigation",
    "3.1调查计划": "3.1 Investigation Plan",
    "3.2调查内容": "3.2 Investigation Content",
    "3.2.2检验设备：": "3.2.2 Testing Equipment:",
    "3.2.4检验用具": "3.2.4 Test Utensils",
    "3.2.5检测环境": "3.2.5 Testing Environment",
    "4、实验室调查分析": "4. Laboratory Investigation Analysis",
    "6、影响性评估": "6. Impact Assessment",
    "检验用具：检验用进样瓶均为已清洁干燥的检验用具。": (
        "Test utensils: All sample vials used for testing are clean and dry."
    ),
    "产品批号": "Batch No.",
    "取样点编号": "Sampling Point No.",
    "toc平行样/ppb": "TOC parallel samples/ppb",
    "平均值/ppb": "Average/ppb",
    "符合": "Conforms",
    "符合，超行动限": "Conforms; exceeds action limit",
}

_REPEATED_EN_ACRONYM = re.compile(r"\b(?P<acronym>[A-Z][A-Z0-9-]{1,})\b(?:\s+(?P=acronym)\b)+")


def requirements_for(source_text: str, target_language: str) -> tuple[TerminologyRequirement, ...]:
    """Return only constraints that apply to this source fragment."""
    target = target_language.lower().replace("_", "-")
    if not target.startswith("zh"):
        return ()
    normalized = " ".join(source_text.casefold().split())
    result: list[TerminologyRequirement] = []
    for rule in _ZH_RULES:
        matches = bool(re.search(rf"\b{re.escape(rule.source_term)}\w*\b", normalized))
        if matches:
            result.append(rule)
    return tuple(result)


def exact_preferred_target(
    source_text: str,
    target_language: str,
) -> str | None:
    normalized = " ".join(source_text.casefold().split())
    target = target_language.lower().replace("_", "-")
    if target.startswith("en") and normalized in _EXACT_EN_TARGETS:
        return _EXACT_EN_TARGETS[normalized]
    for rule in requirements_for(source_text, target_language):
        if rule.exact_source and normalized == rule.source_term:
            return rule.required_target
    return None


def normalize_target_output(text: str, target_language: str) -> str:
    """Remove model-only duplicate acronym runs without altering source literals."""
    target = target_language.lower().replace("_", "-")
    if target.startswith("en"):
        return _REPEATED_EN_ACRONYM.sub(r"\g<acronym>", text)
    return text


def requirement_instruction(requirements: tuple[TerminologyRequirement, ...]) -> str:
    if not requirements:
        return ""
    mappings = "; ".join(
        f"{rule.source_term} -> {rule.required_target} ({rule.reason})" for rule in requirements
    )
    return f"Mandatory terminology for this fragment: {mappings}."


def missing_requirements(
    target_text: str,
    requirements: tuple[TerminologyRequirement, ...],
) -> tuple[TerminologyRequirement, ...]:
    return tuple(rule for rule in requirements if rule.required_target not in target_text)


__all__ = [
    "TerminologyRequirement",
    "exact_preferred_target",
    "missing_requirements",
    "normalize_target_output",
    "requirement_instruction",
    "requirements_for",
]
