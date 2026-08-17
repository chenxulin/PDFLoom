"""Font discovery shared by the scanned-PDF rendering stages."""

from __future__ import annotations

from pathlib import Path


def resolve_cjk_font(preferred: str | Path | None = None) -> Path:
    """Find a usable CJK font in the standalone runtime."""
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend(
        (
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
        )
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No CJK font found; install a Noto CJK font or configure one")
