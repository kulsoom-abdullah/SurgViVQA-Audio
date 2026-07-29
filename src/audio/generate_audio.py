"""
K2-keyed TTS generation for the SurgViVQA audio channel.

Filenames are keyed by K2 -- {video_id}_{id}_{question_type} -- because the bare
`id` is NOT unique: it repeats across videos and across splits, with different
questions attached. A flat {id}.mp3 namespace would let a train-set question
overwrite a test-set file. K2 is collision-free across all 3,700 corpus rows.

Synthesis is deduplicated. The corpus has only 20 distinct question strings, one
per question_type, so the audio content of a row is fully determined by
(question_type, voice, rate). That triple is the CANONICAL unit; every corpus row
is a hardlink onto its canonical file. Train would otherwise be 2,302 syntheses
instead of 460.

Usage:
    python -m src.audio.generate_audio --split test --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.audio.voice_split import (
    SEED, REPO_ROOT, OUTPUT_PATH as VOICE_SPLIT_PATH, parse_roster,
    COLLAPSED_PERSONAS, EXCLUDED_VOICES,
)

# edge-tts is a network RPC to Microsoft, not local CPU. Concurrency is capped
# accordingly, and every write is atomic so an interrupted run never leaves a
# partial file that a later resume would mistake for finished work.
MAX_CONCURRENCY = 4
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2, 8, 32)
MIN_VALID_BYTES = 1024

SPLIT_DATA = {
    "train": REPO_ROOT / "data" / "train_multivideo.jsonl",
    "eval": REPO_ROOT / "data" / "eval_multivideo.jsonl",
    "test": REPO_ROOT / "data" / "test_multivideo.jsonl",
}

AUDIO_ROOT = REPO_ROOT / "data" / "audio"
CANONICAL_ROOT = AUDIO_ROOT / "_canonical"

REQUIRED_FIELDS = ("video_id", "id", "question_type", "question")

RATE_MIN_PCT = -12
RATE_MAX_PCT = 12


def k2_key(row: dict) -> str:
    """The one key scheme. Collision-free across train+eval+test (3700/3700)."""
    return f"{row['video_id']}_{row['id']}_{row['question_type']}"


@dataclass(frozen=True)
class PlanRow:
    row_id: str
    video_id: str
    question_type: str
    question: str
    k2: str
    voice: str
    rate_pct: int
    canonical_path: Path
    output_path: Path


def load_split_rows(split: str) -> list[dict]:
    """Load a split and abort on a missing field or a K2 collision."""
    path = SPLIT_DATA[split]
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    rows = [json.loads(line) for line in path.open()]

    for i, row in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if f not in row or row[f] in (None, "")]
        if missing:
            raise AssertionError(f"{path}:{i + 1}: row missing required field(s) {missing}")

    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[k2_key(row)].append(i + 1)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    if collisions:
        detail = "\n".join(
            f"    {k}  <- lines {v}" for k, v in sorted(collisions.items())[:20]
        )
        raise AssertionError(
            f"{path}: K2 key is not unique -- {len(collisions)} colliding key(s):\n{detail}"
        )

    return rows


def load_voices(split: str) -> list[str]:
    if not VOICE_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{VOICE_SPLIT_PATH} not found -- run `python -m src.audio.voice_split --write` first"
        )
    payload = json.loads(VOICE_SPLIT_PATH.read_text())
    voices = sorted(payload["splits"][split])
    if not voices:
        raise AssertionError(f"No voices assigned to split {split!r}")
    return voices


def rate_for(split: str, question_type: str, voice: str) -> int:
    """One rate per (question_type, voice) pair -- NOT per row.

    Per-row rates would defeat the canonical dedup: every row would become its
    own synthesis.
    """
    rng = random.Random(f"{SEED}:{split}:{question_type}:{voice}")
    return rng.randint(RATE_MIN_PCT, RATE_MAX_PCT)


def canonical_path(split: str, question_type: str, voice: str, rate_pct: int) -> Path:
    return CANONICAL_ROOT / split / f"{question_type}__{voice}__r{rate_pct:+d}.mp3"


def build_plan(split: str) -> tuple[list[PlanRow], dict[tuple[str, str], PlanRow]]:
    """Assign voices round-robin within each question_type and resolve paths.

    Rows are ordered by (id, k2) rather than id alone: `id` is not unique inside a
    split (train repeats 256 of them), so id-only sorting would leave the tie order
    dependent on file order. k2 is unique, making the order a total order.
    """
    rows = load_split_rows(split)
    voices = load_voices(split)

    by_qtype: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_qtype[row["question_type"]].append(row)

    plan: list[PlanRow] = []
    canonical: dict[tuple[str, str], PlanRow] = {}

    for question_type in sorted(by_qtype):
        ordered = sorted(by_qtype[question_type], key=lambda r: (r["id"], k2_key(r)))
        for i, row in enumerate(ordered):
            voice = voices[i % len(voices)]
            rate_pct = rate_for(split, question_type, voice)
            cpath = canonical_path(split, question_type, voice, rate_pct)
            entry = PlanRow(
                row_id=row["id"],
                video_id=row["video_id"],
                question_type=question_type,
                question=row["question"],
                k2=k2_key(row),
                voice=voice,
                rate_pct=rate_pct,
                canonical_path=cpath,
                output_path=AUDIO_ROOT / split / f"{k2_key(row)}.mp3",
            )
            plan.append(entry)
            canonical.setdefault((question_type, voice), entry)

    # A canonical file speaks exactly one question, so a question_type carrying two
    # different question strings would silently mis-speak some rows.
    per_qtype_questions = collections.defaultdict(set)
    for entry in plan:
        per_qtype_questions[entry.question_type].add(entry.question)
    ambiguous = {k: v for k, v in per_qtype_questions.items() if len(v) > 1}
    if ambiguous:
        raise AssertionError(
            f"{split}: question_type(s) map to >1 question text, so (question_type, voice) "
            f"is not a valid canonical unit: {sorted(ambiguous)}"
        )

    return plan, canonical


def print_dry_run(split: str) -> tuple[int, int]:
    plan, canonical = build_plan(split)
    qtypes = {e.question_type for e in plan}
    voices = {e.voice for e in plan}

    print(f"--- {split} ---")
    print(f"  rows                      : {len(plan)}")
    print(f"  distinct question_types   : {len(qtypes)}")
    print(f"  distinct voices           : {len(voices)}")
    print(f"  canonical to synthesize   : {len(canonical)}")
    print(f"  hardlinks to create       : {len(plan)}")
    print(f"  distinct rates in use     : {len({e.rate_pct for e in plan})}")
    print("  examples:")
    for e in plan[:3]:
        print(f"    {e.row_id} -> {e.k2}")
        print(f"        voice={e.voice}  rate={e.rate_pct:+d}%")
        print(f"        canonical={e.canonical_path.relative_to(REPO_ROOT)}")
    print()
    return len(canonical), len(plan)


def _is_complete(path: Path) -> bool:
    """A canonical file counts as done only if it exists AND is plausibly audio."""
    return path.exists() and path.stat().st_size > MIN_VALID_BYTES


async def _synthesize_one(entry: PlanRow, sem: asyncio.Semaphore) -> str | None:
    """Synthesize one canonical file. Returns None on success, else an error string.

    Writes to {path}.tmp and os.replace()s into position, so a killed run leaves
    either nothing or a complete file -- never a zero-byte stub that the resume
    check would skip.
    """
    import edge_tts

    path = entry.canonical_path
    if _is_complete(path):
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    async with sem:
        last_error = "unknown"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                comm = edge_tts.Communicate(
                    entry.question, entry.voice, rate=f"{entry.rate_pct:+d}%"
                )
                await comm.save(str(tmp))
                size = tmp.stat().st_size
                if size <= MIN_VALID_BYTES:
                    raise ValueError(f"synthesized file is only {size} bytes")
                os.replace(tmp, path)
                return None
            except Exception as e:  # noqa: BLE001 - failures are collected, not swallowed
                last_error = f"attempt {attempt}/{MAX_ATTEMPTS}: {type(e).__name__}: {e}"
                tmp.unlink(missing_ok=True)
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(BACKOFF_SECONDS[attempt - 1])
        return last_error


async def _synthesize_all(entries: list[PlanRow]) -> dict[Path, str]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    done = 0
    total = len(entries)
    failures: dict[Path, str] = {}

    async def run(entry: PlanRow):
        nonlocal done
        err = await _synthesize_one(entry, sem)
        done += 1
        if err:
            failures[entry.canonical_path] = err
        if done % 20 == 0 or done == total:
            print(f"    {done}/{total} canonical  ({len(failures)} failed)", flush=True)

    await asyncio.gather(*(run(e) for e in entries))
    return failures


def _sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size


def _duration_seconds(path: Path) -> float | None:
    """mp3 duration via ffprobe; None (never a fabricated 0.0) if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return round(float(out.stdout.strip()), 3)
    except Exception:
        return None


def _link_or_copy(canonical: Path, output: Path) -> str:
    """Hardlink the per-row path onto its canonical file. Returns 'link' or 'copy'."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.samefile(canonical):
            return "link"
        output.unlink()
    try:
        os.link(canonical, output)
        return "link"
    except OSError:
        shutil.copy2(canonical, output)
        return "copy"


def generate(split: str) -> int:
    """Synthesize, hardlink, and write the manifest for one split."""
    import edge_tts

    plan, canonical = build_plan(split)
    entries = sorted(canonical.values(), key=lambda e: e.canonical_path.name)

    already = sum(1 for e in entries if _is_complete(e.canonical_path))
    print(f"=== {split}: {len(entries)} canonical files "
          f"({already} already present, resuming), {len(plan)} rows ===")

    failures = asyncio.run(_synthesize_all(entries))

    if failures:
        print(f"\n!! {len(failures)} canonical synthesis FAILURES -- hardlinks NOT created:")
        for path, err in sorted(failures.items()):
            print(f"   {path.name}: {err}")
        return 1

    synthesized = len(entries) - already
    print(f"  canonical complete: {len(entries)} ({synthesized} newly synthesized)")

    # Per-canonical stats, computed once and shared by every row pointing at it.
    stats: dict[Path, tuple[str, int, float | None]] = {}
    for e in entries:
        sha, size = _sha256_and_size(e.canonical_path)
        stats[e.canonical_path] = (sha, size, _duration_seconds(e.canonical_path))

    roster = {v.short_name: v for v in parse_roster()}
    counts = collections.Counter()
    manifest_path = AUDIO_ROOT / split / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as mf:
        for e in plan:
            counts[_link_or_copy(e.canonical_path, e.output_path)] += 1
            sha, size, duration = stats[e.canonical_path]
            voice = roster[e.voice]
            mf.write(json.dumps({
                "row_id": e.row_id,
                "video_id": e.video_id,
                "question_type": e.question_type,
                "k2_key": e.k2,
                "voice": e.voice,
                "locale": voice.locale,
                "gender": voice.gender,
                "rate_pct": e.rate_pct,
                "question": e.question,
                "canonical_path": str(e.canonical_path.relative_to(REPO_ROOT)),
                "sha256": sha,
                "bytes": size,
                "duration_s": duration,
                "split": split,
            }) + "\n")

    print(f"  hardlinks: {counts['link']}   copies (os.link unavailable): {counts['copy']}")
    print(f"  manifest : {manifest_path.relative_to(REPO_ROOT)} ({len(plan)} lines)")

    header = {
        "seed": SEED,
        "split": split,
        "voice_split_path": str(VOICE_SPLIT_PATH.relative_to(REPO_ROOT)),
        "voice_split_sha256": hashlib.sha256(VOICE_SPLIT_PATH.read_bytes()).hexdigest(),
        "edge_tts_version": edge_tts.__version__,
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "key_scheme": "K2",
        "rows": len(plan),
        "canonical_files": len(entries),
        "canonical_newly_synthesized": synthesized,
        "hardlinks": counts["link"],
        "copies": counts["copy"],
        "voices": sorted({e.voice for e in plan}),
        "collapsed_personas": sorted(COLLAPSED_PERSONAS),
        "excluded_voices": sorted(EXCLUDED_VOICES),
        "max_concurrency": MAX_CONCURRENCY,
        "duration_probe_failures": sum(1 for v in stats.values() if v[2] is None),
    }
    header_path = AUDIO_ROOT / split / "run_header.json"
    header_path.write_text(json.dumps(header, indent=2) + "\n")
    print(f"  header   : {header_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="K2-keyed TTS generation")
    parser.add_argument("--split", choices=sorted(SPLIT_DATA), action="append",
                        help="Split to process; repeatable. Default: all three.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the plan; synthesize nothing.")
    args = parser.parse_args()

    splits = args.split or ["train", "eval", "test"]

    if args.dry_run:
        print(f"DRY RUN (no network, no writes)   SEED={SEED}")
        print(f"key scheme: K2 = {{video_id}}_{{id}}_{{question_type}}")
        print()
        totals = {}
        for split in splits:
            totals[split] = print_dry_run(split)
        print("SUMMARY  split: canonical / hardlinks")
        for split, (c, h) in totals.items():
            print(f"  {split:<6} {c:>5} / {h:>5}")
        return

    rc = 0
    for split in splits:
        rc |= generate(split)
    sys.exit(rc)


if __name__ == "__main__":
    main()
