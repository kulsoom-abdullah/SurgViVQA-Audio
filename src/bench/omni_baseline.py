#!/usr/bin/env python3
"""
omni_baseline.py — Stage 5. The off-the-shelf omni baseline run (arm A1).

ONE condition: Parity-B, no vocabulary hint. The V1 vocabulary arm was cut before any
data was collected (PRE_REGISTRATION.md §10).

Generation goes through omni_adapters.py — the SAME module the Gate 3 probe verified.
That is the entire point of the Stage 4.5 extraction: `adapters_sha256` is written into
both artifacts/probe/{model}_probe.json and this run's manifest, so "the probe verified
the code that produced this number" is a string comparison rather than an assumption.
Unequal hashes mean the probe result does not transfer to this run.

WHAT THIS FILE DOES NOT DO:

  - It does not time anything. `latency_s` is the third value returned by
    Adapter.run(), measured inside run() around the generate call. A timer here would
    wrap frame decoding and audio loading too, producing a second timing path that
    disagrees with the manifest's declared TIMING_BRACKET. One timing path or none.
  - It does not define the strict rule. See _load_scoring() below.
  - It does not swallow row errors. There is no `except: continue`. A run that quietly
    drops rows reports an accuracy over a denominator nobody chose.

Usage:
    python src/bench/omni_baseline.py --model minicpmo45 \
        --frames-root data/frames --audio-dir data/audio/test

    # dry run, no GPU, no model download
    python src/bench/omni_baseline.py --model minicpmo45 --stub --limit 5 \
        --frames-root dataset/frames --audio-dir data/audio/test \
        --out /tmp/stubrun
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

from omni_adapters import (        # shared with verify_combined_path.py — see Stage 4.5
    ADAPTERS_SHA256, MAX_NEW_TOKENS, NEUTRAL_INSTRUCTION, PARITY_B_SHA256, SYNC_SOURCE,
    Adapter, assert_no_leak, k2_key, load_frames_for_row, make_adapter,
)

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_DIR = _BENCH_DIR.parent.parent

INPUT_MODE = "audio_only"          # the regime M_ft = 0.571 was measured in


# --------------------------------------------------------------------------------------
# Scoring — lifted from freeze_configs.py, not reimplemented
# --------------------------------------------------------------------------------------

def _load_scoring():
    """Return (lenient, strict) as defined in src/bench/freeze_configs.py.

    freeze_configs cannot be imported: its assertion body runs at module level, reads
    data/, and rewrites configs/parity.yaml. So the scoring definitions are lifted out of
    its AST and exec'd in isolation, leaving exactly ONE source of truth for the strict
    rule — freeze_configs.py:30-46, the same text Stage 2 verified against cell 3 (strict
    and the stored `correct` field both 0.5710, 0 rows disagreeing).

    A copy-with-a-drift-check was the alternative. This is the same length and has no
    copy to drift. If a name goes missing the run aborts here, before the model loads,
    rather than scoring 1000 rows under a rule nobody verified.
    """
    want = {"CARVE", "norm", "lenient", "strict"}
    tree = ast.parse((_BENCH_DIR / "freeze_configs.py").read_text())

    def bound(node):
        if isinstance(node, ast.FunctionDef):
            return node.name
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    keep = [n for n in tree.body if bound(n) in want]
    ns: dict = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]),   # noqa: S102
                 "<freeze_configs:scoring>", "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit(
            f"freeze_configs.py no longer defines {sorted(missing)} at module level. "
            "The strict rule is pinned to that file; fix the lift or the run scores "
            "under an unverified rule.")
    return ns["lenient"], ns["strict"]


lenient_match, strict_match = _load_scoring()


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

def _git(*cmd) -> str:
    """Pinned to the repo dir so a run launched from anywhere reports THIS repo."""
    try:
        return subprocess.check_output(["git", *cmd], cwd=_REPO_DIR,
                                       stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"GIT_ERROR({e.returncode}): {e.output.decode().strip()}"
    except Exception as e:                                       # noqa: BLE001
        return f"GIT_ERROR: {type(e).__name__}: {e}"


def _env() -> dict:
    """Environment facts. Tolerant of missing torch/transformers so the --stub dry run
    works on a laptop; a real run would die at adapter.load() anyway, loudly."""
    out = {"venv": sys.prefix, "python": sys.version.split()[0]}
    try:
        import torch
        out["torch"] = torch.__version__                      # carries the +cuXXX build
        out["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_arch"] = f"sm_{cap[0]}{cap[1]}"
        else:
            out["gpu"] = out["gpu_arch"] = "NO CUDA DEVICE"
    except Exception as e:                                       # noqa: BLE001
        out["torch"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception as e:                                       # noqa: BLE001
        out["transformers"] = f"UNAVAILABLE: {type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------------------
# Stub adapter — dry-run only
# --------------------------------------------------------------------------------------

class StubAdapter(Adapter):
    """Exercises the harness with no GPU and no download: row loading, K2 audio
    resolution, frame resolution, the leak gate, both scorers, jsonl and manifest
    writing. Emits a fixed answer, so its accuracy is meaningless by construction and the
    manifest is flagged `stub_adapter: true`."""

    TIMING_BRACKET = "STUB — no generate call; latency is not a measurement"

    def load(self) -> None:
        self.notes.append("STUB ADAPTER — no model loaded, outputs are constant")

    def run(self, images, audio_path):
        rendered = f"<|user|><|image_1|>x{len(images)}<|audio_1|>{NEUTRAL_INSTRUCTION}<|end|>"
        return rendered, "no", 0.0


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 — off-the-shelf omni baseline (A1)")
    ap.add_argument("--model", required=True, choices=["phi4mm", "minicpmo45"])
    ap.add_argument("--frames-root", type=Path, required=True,
                    help="REQUIRED, no default: dataset/frames locally, data/frames on "
                         "the pod. A default is correct on exactly one machine.")
    ap.add_argument("--audio-dir", type=Path, required=True,
                    help="REQUIRED. Clips are {video_id}_{id}_{question_type}.mp3 (K2).")
    ap.add_argument("--test-manifest", type=Path, default=Path("data/test_multivideo.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline"))
    ap.add_argument("--omni-mode", action="store_true",
                    help="minicpmo45 only: interleaved omni path")
    ap.add_argument("--limit", type=int, default=None, help="dry runs only")
    ap.add_argument("--stub", action="store_true",
                    help="dry run: substitute a constant-output adapter, load no model")
    args = ap.parse_args()

    if args.stub and args.limit is None:
        raise SystemExit("--stub requires --limit; it must not look like a real run")

    rows = [json.loads(l) for l in open(args.test_manifest)]
    if args.limit:
        rows = rows[:args.limit]

    # Data-derived, from the FULL test set regardless of --limit: types whose gold answer
    # is constant across all 1000 rows cannot separate a model from a constant emitter.
    # 7 such types (350 rows); the remaining 13 types (650 rows) are the primary read.
    all_rows = [json.loads(l) for l in open(args.test_manifest)]
    golds_by_type: dict[str, set] = {}
    for r in all_rows:
        golds_by_type.setdefault(r["question_type"], set()).add(r["short_answer"])
    discriminative = {k for k, v in golds_by_type.items() if len(v) > 1}

    # Resolve every audio clip BEFORE loading the model. A missing clip is a hard failure,
    # never a silence fallback: an audio_only arm running on silence still prints a
    # plausible accuracy, and that is this project's original failure mode.
    # Keyed by the K2 key, not by `id`: bare ids are not unique corpus-wide, and keying a
    # dict on one is how the wrong clip reaches the wrong row silently.
    audio_paths: dict[str, Path] = {}
    missing = []
    for r in rows:
        key = k2_key(r["video_id"], r["id"], r["question_type"])
        p = args.audio_dir / f"{key}.mp3"
        if p.exists():
            audio_paths[key] = p
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"{len(missing)}/{len(rows)} audio clips absent from "
                         f"{args.audio_dir}; first 5: {missing[:5]}. This arm is "
                         f"audio-only — there is no fallback.")
    print(f"audio coverage: {len(rows)}/{len(rows)} clips resolved")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}.jsonl"
    man_path = args.out / f"{args.model}.manifest.json"

    adapter = StubAdapter(name=args.model) if args.stub \
        else make_adapter(args.model, args.omni_mode)

    # Manifest is written BEFORE the model loads and backfilled after the run. A run that
    # dies at row 900 still leaves a record of what it was, which is the only reason the
    # aborted Phase B runs were auditable at all.
    manifest = dict(
        stage="5 — off-the-shelf omni baseline (A1)",
        status="started",
        model=args.model,
        model_id=getattr(adapter, "MODEL_ID", None),
        revision=getattr(adapter, "REVISION", None),
        attn_implementation=getattr(adapter, "ATTN_IMPL", None),
        omni_mode=args.omni_mode,
        stub_adapter=args.stub,
        condition="Parity-B, single condition (V1 vocabulary arm cut pre-data, §10)",
        instruction=NEUTRAL_INSTRUCTION,
        instruction_sha256=PARITY_B_SHA256,
        input_mode=INPUT_MODE,
        decoding="greedy (do_sample=False, num_beams=1)",
        max_new_tokens=MAX_NEW_TOKENS,
        frames_per_row=8,
        batch_size=1,
        # Compare against artifacts/probe/{model}_probe.json. Equal => Gate 3 verified
        # this generate path. Unequal => it verified a different one.
        adapters_sha256=ADAPTERS_SHA256,
        sync_source=SYNC_SOURCE,
        # NOT comparable across adapters: Phi-4 brackets generate() alone, MiniCPM-o's
        # tightest available bracket is the whole chat() call, which preprocesses inside.
        timing_bracket=adapter.TIMING_BRACKET,
        scoring=("lenient = gold.lower() in pred.lower(); strict = word-boundary regex "
                 "with carve-outs. Both lifted from src/bench/freeze_configs.py:30-46."),
        frames_root=str(args.frames_root),
        audio_dir=str(args.audio_dir),
        test_manifest=str(args.test_manifest),
        n_rows_requested=len(rows),
        discriminative_types=sorted(discriminative),
        degenerate_types=sorted(set(golds_by_type) - discriminative),
        git_sha=_git("rev-parse", "HEAD"),
        git_ref=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **_env(),
    )
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote pre-run manifest -> {man_path}")

    print(f"[{args.model}] loading...")
    adapter.load()
    for n in adapter.notes:
        print(f"  ! {n}")

    n_lenient = n_strict = 0
    latencies: list[float] = []
    results: list[dict] = []

    # Rows are written as they complete. A crash at row 900 keeps 899 rows of rented-GPU
    # output instead of discarding them.
    with out_path.open("w") as fh:
        for i, row in enumerate(rows, 1):
            key = k2_key(row["video_id"], row["id"], row["question_type"])
            images = load_frames_for_row(args.frames_root, row)
            rendered, pred, latency_s = adapter.run(images, str(audio_paths[key]))

            # Gate 1, per row and blocking. No try/except: a leaked prompt invalidates the
            # arm, so continuing past one would only produce an unusable number slowly.
            assert_no_leak(rendered, row["question"], f"{args.model}/{row['id']}",
                           NEUTRAL_INSTRUCTION)

            lo = bool(lenient_match(row["short_answer"], pred))
            st = bool(strict_match(row["short_answer"], pred))
            n_lenient += lo
            n_strict += st
            latencies.append(latency_s)

            rec = dict(
                question_id=row["id"], question_type=row["question_type"],
                question=row["question"], ground_truth=row["answer"],
                short_answer=row["short_answer"], predicted_answer=pred,
                correct=int(lo), strict_match=int(st),
                discriminative=int(row["question_type"] in discriminative),
                latency_s=latency_s, input_mode=INPUT_MODE,
                audio_source_id=key, audio_found=True, model=args.model)
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            if i == 1:
                (args.out / f"{args.model}_rendered_prompt.txt").write_text(rendered)
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  lenient={n_lenient/i*100:.1f}%  "
                      f"strict={n_strict/i*100:.1f}%", flush=True)

    import statistics
    disc = [r for r in results if r["discriminative"]]
    pct = lambda rs, k: (sum(r[k] for r in rs) / len(rs) * 100) if rs else None

    manifest.update(
        status="complete",
        n_rows=len(results),
        accuracy_lenient=pct(results, "correct"),
        accuracy_strict=pct(results, "strict_match"),
        n_discriminative_rows=len(disc),
        accuracy_lenient_discriminative=pct(disc, "correct"),
        accuracy_strict_discriminative=pct(disc, "strict_match"),
        latency_median_s=statistics.median(latencies),
        latency_p90_s=sorted(latencies)[max(0, int(0.9 * len(latencies)) - 1)],
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    m = manifest
    print(f"\n  ALL-{m['n_rows']}   lenient={m['accuracy_lenient']:.2f}%  "
          f"strict={m['accuracy_strict']:.2f}%")
    print(f"  DISC-{m['n_discriminative_rows']}    "
          f"lenient={m['accuracy_lenient_discriminative']:.2f}%  "
          f"strict={m['accuracy_strict_discriminative']:.2f}%")
    print(f"  latency median={m['latency_median_s']:.3f}s p90={m['latency_p90_s']:.3f}s "
          f"[{m['timing_bracket']}]")
    print(f"  wrote {out_path}\n  wrote {man_path}")
    print("\n  Next, before any number is reported: Gate 2 by eye on "
          f"{args.out}/{args.model}_rendered_prompt.txt, the §5.2 format-failure audit, "
          "and the §5.4 per-type table (question_type + correct + strict_match are on "
          "every row).")
    if args.stub:
        print("\n  STUB RUN — accuracies above are constant-output artifacts, not results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
