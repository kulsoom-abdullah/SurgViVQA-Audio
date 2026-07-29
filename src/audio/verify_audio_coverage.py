"""
Independent audio-coverage check.

This verifier RECOMPUTES the K2 keys from the split jsonl. It deliberately does
NOT read manifest.jsonl to decide what to check -- a manifest-driven check would
only be comparing the manifest against itself and would pass even if generation
had silently skipped rows.

Usage:
    python -m src.audio.verify_audio_coverage --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.audio.generate_audio import (
    AUDIO_ROOT, MIN_VALID_BYTES, SPLIT_DATA, k2_key,
)


def verify(split: str) -> int:
    split_path = SPLIT_DATA[split]
    if not split_path.exists():
        print(f"MISSING SPLIT FILE: {split_path}")
        return 1

    rows = [json.loads(line) for line in split_path.open()]
    audio_dir = AUDIO_ROOT / split

    missing: list[tuple[str, str]] = []
    for row in rows:
        key = k2_key(row)
        path = audio_dir / f"{key}.mp3"
        if not path.exists():
            missing.append((key, "absent"))
        elif path.stat().st_size <= MIN_VALID_BYTES:
            missing.append((key, f"only {path.stat().st_size} bytes"))

    found = len(rows) - len(missing)
    print(f"COVERAGE: {found}/{len(rows)}")

    if missing:
        print(f"MISSING OR TOO SMALL: {len(missing)} (first 10)")
        for key, why in missing[:10]:
            print(f"  {audio_dir.name}/{key}.mp3 -- {why}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify per-row audio coverage")
    parser.add_argument("--split", choices=sorted(SPLIT_DATA), required=True)
    args = parser.parse_args()
    sys.exit(verify(args.split))


if __name__ == "__main__":
    main()
