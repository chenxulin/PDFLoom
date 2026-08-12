#!/usr/bin/env python3
"""Generate one image-only scan and one born-digital PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.helpers import make_native_pdf, make_scanned_table_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("test-results/samples"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(make_scanned_table_pdf(args.output_dir / "scanned-table.pdf"))
    print(make_native_pdf(args.output_dir / "born-digital.pdf"))


if __name__ == "__main__":
    main()
