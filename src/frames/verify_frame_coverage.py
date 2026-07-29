"""
Independent frame-coverage gate.

Why this exists: load_frames() in evaluate_checkpoint.py does NOT fail when a
frame is missing -- it appends a solid black 224x224 image and carries on
(evaluate_checkpoint.py:115). An eval run against a partial frame transfer
therefore completes normally and prints a plausible accuracy computed on blank
images. Nothing downstream flags it. This gate is the only thing standing
between a truncated rsync and a silently invalid number.

The resolution logic below is copied from load_frames() lines 102-107 rather
than simplified. A tidier rule here would be checking something the harness will
not actually do, which defeats the purpose of the gate.

Standalone by design: stdlib only, no torch/PIL/transformers, so it can run on a
fresh pod before any heavy dependency is installed.

Usage:
    python src/frames/verify_frame_coverage.py --frames_dir data/frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve_frame(frame_name: str, frames_dir: str) -> Path:
    """Mirror of evaluate_checkpoint.py load_frames() lines 102-107.

    Returns the path the harness would settle on -- which may not exist; the
    caller decides what that means.
    """
    vid_id = frame_name.rsplit('_', 1)[0] if '_' in frame_name else "unknown"
    path = Path(frames_dir) / vid_id / f"{frame_name}.jpg"

    if not path.exists():
        frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
        path = Path(frames_dir) / vid_id / f"{frame_num}.jpg"

    return path


def verify(eval_data_path: str, frames_dir: str) -> int:
    data_path = Path(eval_data_path)
    if not data_path.exists():
        print(f"MISSING EVAL DATA: {data_path}")
        return 1

    rows = [json.loads(line) for line in data_path.open()]

    row_slots = 0
    distinct: dict[str, Path] = {}
    for row in rows:
        for frame_name in row["frames"]:
            row_slots += 1
            if frame_name not in distinct:
                distinct[frame_name] = resolve_frame(frame_name, frames_dir)

    bad: list[tuple[str, str]] = []
    for frame_name, path in sorted(distinct.items()):
        if not path.exists():
            bad.append((frame_name, f"absent (would become a BLACK image): {path}"))
        elif path.stat().st_size == 0:
            bad.append((frame_name, f"zero bytes: {path}"))

    ok_distinct = len(distinct) - len(bad)
    bad_names = {name for name, _ in bad}
    ok_slots = sum(
        1 for row in rows for frame_name in row["frames"] if frame_name not in bad_names
    )

    print(f"FRAME COVERAGE: {ok_distinct}/{len(distinct)} distinct frames, "
          f"{ok_slots}/{row_slots} row-slots")

    if bad:
        print(f"UNRESOLVED: {len(bad)} distinct frame(s) (first 10)")
        for frame_name, why in bad[:10]:
            print(f"  {frame_name} -- {why}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frame coverage for the eval harness")
    parser.add_argument("--eval_data_path", default="data/test_multivideo.jsonl",
                        help="Eval JSONL to read frame references from")
    parser.add_argument("--frames_dir", default="data/frames",
                        help="Frames root, as passed to evaluate_checkpoint.py --frames_dir")
    args = parser.parse_args()
    sys.exit(verify(args.eval_data_path, args.frames_dir))


if __name__ == "__main__":
    main()
