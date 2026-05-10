#!/usr/bin/env python3
"""
Scan a dataset for files whose extension is .jpg/.jpeg/.png but whose actual
content is WebP, then re-encode them to the format implied by the extension.

Usage:
  python3 fix_mislabeled_webp.py dataset --dry-run
  python3 fix_mislabeled_webp.py dataset --backup

Requires Pillow:
  python3 -m pip install Pillow
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from PIL import Image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_webp_by_magic(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"


def target_format_for_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def convert_file(path: Path, *, backup: bool, quality: int) -> None:
    target_format = target_format_for_suffix(path)
    backup_path = path.with_suffix(path.suffix + ".webp_backup")
    tmp = path.with_suffix(path.suffix + ".tmp")

    with Image.open(path) as im:
        if im.format != "WEBP":
            return
        if target_format == "JPEG":
            out = im.convert("RGB")
            out.save(tmp, target_format, quality=quality, optimize=True)
        else:
            out = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
            out.save(tmp, target_format, optimize=True)

    if backup:
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        shutil.copy2(path, backup_path)

    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix .jpg/.jpeg/.png files that are actually WebP.")
    parser.add_argument("root", nargs="?", default="dataset", help="Dataset root directory to scan. Default: dataset")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be converted.")
    parser.add_argument("--backup", action="store_true", help="Keep a .webp_backup copy before overwriting.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, default: 95")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    webp_mislabeled = [p for p in candidates if is_webp_by_magic(p)]

    if not webp_mislabeled:
        print("No mislabeled WebP files found among .jpg/.jpeg/.png files.")
        return 0

    for p in webp_mislabeled:
        action = "would convert" if args.dry_run else "converting"
        print(f"{action}: {p}")
        if not args.dry_run:
            convert_file(p, backup=args.backup, quality=args.quality)

    print(f"Done. Files matched: {len(webp_mislabeled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
