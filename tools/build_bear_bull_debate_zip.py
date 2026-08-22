#!/usr/bin/env python3
"""Package the ``src/bear_bull_debate`` package into a ZIP for Google Colab upload.

This script is the portable equivalent of running, from the project root::

    zip -r bear_bull_debate_src.zip src/bear_bull_debate

It uses only the Python standard library (``zipfile``), so it works on any OS
without a ``zip`` binary. The resulting archive preserves the
``src/bear_bull_debate/...`` paths, which the Colab notebook
(``notebooks/bear_bull_debate_colab.ipynb``) auto-locates after extraction.

Usage::

    python tools/build_bear_bull_debate_zip.py                # -> bear_bull_debate_src.zip
    python tools/build_bear_bull_debate_zip.py --output /tmp/out.zip
"""

import argparse
import zipfile
from pathlib import Path

# Project root = parent of the `tools/` directory holding this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_ROOT / "src" / "bear_bull_debate"
DEFAULT_OUTPUT = PROJECT_ROOT / "bear_bull_debate_src.zip"


def collect_files() -> list[Path]:
    """Return the package's files, skipping bytecode caches."""
    if not PACKAGE_DIR.is_dir():
        raise SystemExit(f"Package directory not found: {PACKAGE_DIR}")

    files = [
        p
        for p in sorted(PACKAGE_DIR.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.endswith(".pyc")
    ]
    if not files:
        raise SystemExit(f"No files found under {PACKAGE_DIR}")
    return files


def build_zip(output: Path) -> int:
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            # Preserve `src/bear_bull_debate/...` arcnames, matching the shell cmd.
            arcname = path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(path, arcname)

    print(f"Wrote {output} ({len(files)} files)")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output ZIP path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    build_zip(args.output)


if __name__ == "__main__":
    main()
