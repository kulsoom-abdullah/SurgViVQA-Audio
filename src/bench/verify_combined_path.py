#!/usr/bin/env python3
"""
verify_combined_path.py — screening probe for omni-model baselines.

Decides whether a candidate is worth a full evaluation run, without renting a pod for
one. Answers four questions per candidate:

  0. Is decoding DETERMINISTIC?      (control — without it, Q3 is uninterpretable)
  1. Does 8 images + 1 audio clip survive a single prompt without erroring?
  2. Does the rendered prompt leak question text?   (Gate 2, PRE_REGISTRATION §4.2)
  3. IS THE AUDIO PATHWAY ACTUALLY CONNECTED?       (Gate 3, PRE_REGISTRATION §4.4)

Q3 is the reason this file exists: a model can accept audio, run clean, and produce a
plausible accuracy number while the audio pathway is silently dead, in which case the
output is a function of the images and the answer prior alone. That failure already
invalidated one result on this project.

Q0 is why Q3 is trustworthy. If sampling is on, two runs of the SAME audio differ by
chance, "outputs differ" proves nothing, and the probe passes while measuring noise.
Both candidates' model cards use do_sample=True in their examples. We force greedy and
then verify greed took effect.

SCOPE: phi4mm and minicpmo45. Gemma 4 / Nemotron 3 are scoped-not-run.

Usage:
    # 1. build pairs from real artifacts (no GPU needed)
    python verify_combined_path.py --build-pairs \
        --audio-dir data/audio/test --test-manifest data/test_multivideo.jsonl \
        --out-pairs data/probe_pairs.json

    # 2. probe (one venv per model — see PRE_REGISTRATION §11)
    python verify_combined_path.py --model phi4mm     --pairs data/probe_pairs.json \
        --frames data/frames/002-004
    python verify_combined_path.py --model minicpmo45 --pairs data/probe_pairs.json \
        --frames data/frames/002-004
"""

import argparse
import json
import random
import sys
from pathlib import Path

from omni_adapters import (          # shared with omni_baseline.py — see Stage 4.5
    ADAPTERS_SHA256, MAX_NEW_TOKENS, NEUTRAL_INSTRUCTION, PARITY_B_SHA256, SYNC_SOURCE,
    Adapter, assert_no_leak, content_tokens, k2_key, load_frames, load_frames_for_row,
    make_adapter, question_trigrams,
)

N_PAIRS = 10
SENSITIVITY_THRESHOLD = 0.8          # >= 8/10 pairs must differ


# --------------------------------------------------------------------------------------
# Probe-pair construction from real artifacts
# --------------------------------------------------------------------------------------

def build_pairs(audio_dir: Path, test_manifest: Path, out_pairs: Path,
                seed: int = 0) -> None:
    """Pair clips from DIFFERENT question_types so the two spoken questions genuinely differ.

    Keys are constructed FORWARD from data/test_multivideo.jsonl. Filenames are never
    parsed.

    An earlier version split the stem on "_" and read parts[1] as the id. That is wrong:
    the id is itself `qa_NNNNNN`, so `002-004_qa_002496_scope_outside` yielded
    id="qa" and question_type="002496_scope_outside". Two silent failures followed —
    every clip got a unique pseudo-type (destroying the cross-type guarantee, which then
    held only by luck), and `qtext.get("qa")` never hit, so the leak gate would have been
    fed "003353 lesion site" instead of the real question text. A gate checking the wrong
    tokens passes prompts that genuinely leak.

    Reverse-engineering a filename whose fields can each contain the delimiter is not
    fixable by counting fields. Forward construction has no ambiguity, and it verifies
    audio coverage for free.
    """
    rows = [json.loads(l) for l in open(test_manifest)]
    print(f"manifest: {len(rows)} rows from {test_manifest}")

    recs, missing = [], []
    for r in rows:
        key = k2_key(r["video_id"], r["id"], r["question_type"])
        path = audio_dir / f"{key}.mp3"
        if not path.exists():
            missing.append(key)
        recs.append({"path": path, "qid": r["id"], "qtype": r["question_type"],
                     "question": r["question"]})
    if missing:
        raise SystemExit(
            f"{len(missing)} expected clips absent from {audio_dir}; first 5: "
            f"{missing[:5]}. Pair building requires full audio coverage.")
    print(f"audio coverage: {len(recs)}/{len(rows)} clips resolved")

    by_type: dict[str, list[dict]] = {}
    for rec in recs:
        by_type.setdefault(rec["qtype"], []).append(rec)
    types = sorted(by_type)
    print(f"question_types: {len(types)}")
    if len(types) != 20:
        raise SystemExit(f"expected 20 question_types, got {len(types)}: {types}")

    rng = random.Random(seed)
    pairs = []
    for i in range(N_PAIRS):
        ta, tb = rng.sample(types, 2)
        ra, rb = rng.choice(by_type[ta]), rng.choice(by_type[tb])

        # Invariants asserted, not assumed. The old code claimed cross-type pairing while
        # sampling rows; here each is checked per pair and aborts on violation.
        if ra["qtype"] == rb["qtype"]:
            raise SystemExit(f"pair {i}: same question_type {ra['qtype']}")
        if ra["question"] == rb["question"]:
            raise SystemExit(f"pair {i}: identical question text")
        # 1000 paths over ~180 inodes (20 types x 9 voices, hardlinked onto canonical
        # clips), so different paths can be the SAME FILE -> identical output for a
        # trivially correct reason, failing Gate 3b spuriously.
        if ra["path"].stat().st_ino == rb["path"].stat().st_ino:
            raise SystemExit(
                f"pair {i}: {ra['path'].name} and {rb['path'].name} share inode "
                f"{ra['path'].stat().st_ino} despite differing question_type")

        pairs.append({
            "audio_a": str(ra["path"]), "audio_b": str(rb["path"]),
            "type_a": ra["qtype"], "type_b": rb["qtype"],
            "id_a": ra["qid"], "id_b": rb["qid"],
            "text_a": ra["question"], "text_b": rb["question"],
        })

    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    out_pairs.write_text(json.dumps(pairs, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out_pairs}")
    for p_ in pairs:
        print(f"  {p_['type_a']:<26} | {p_['text_a'][:44]}")
        print(f"  {p_['type_b']:<26} | {p_['text_b'][:44]}\n")


def image_sensitivity(adapter, test_manifest: Path, audio_dir: Path,
                      frames_root: Path, out_dir: Path) -> bool:
    """Gate 3c — is the IMAGE pathway live?

    Gate 3b varied the audio with frames held FIXED, so it proves nothing about vision:
    a model receiving zero images, or only the first, passes 3b identically. The task is
    VQA over frames, so a vision-less baseline produces an unfairly low number — a
    confound pointing toward flattering the fine-tune (PRE_REGISTRATION §8.1).

    Design: hold the AUDIO constant (one file, so literally one inode) and swap in two
    disjoint frame sets drawn from the same question_type with DIFFERENT gold answers.
    Same question, same voice, different pictures, opposite truths. If vision is live the
    outputs must differ; if they also match their golds, vision is not merely attached but
    being used.
    """
    rows = [json.loads(l) for l in open(test_manifest)]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["question_type"], []).append(r)

    best = None
    for t in sorted(by_type):
        first: dict[str, dict] = {}
        for r in by_type[t]:
            first.setdefault(r["short_answer"], r)
        if len(first) < 2:
            continue
        (a1, ra), (a2, rb) = sorted(first.items())[:2]
        ov = len(set(ra["frames"]) & set(rb["frames"]))
        if best is None or ov < best[0]:
            best = (ov, t, ra, rb, a1, a2)
    if best is None:
        raise SystemExit("no question_type with two distinct gold answers")
    ov, qtype, ra, rb, gold_a, gold_b = best

    # ra['question_type'] == qtype by construction (ra came from by_type[qtype]); passed
    # explicitly anyway so the key is read off the row, not off loop state.
    audio = audio_dir / f"{k2_key(ra['video_id'], ra['id'], ra['question_type'])}.mp3"
    if not audio.exists():
        raise SystemExit(f"audio missing: {audio}")

    print(f"\n  [Q3c] image sensitivity")
    print(f"    question_type : {qtype}")
    print(f"    question      : {ra['question']}")
    print(f"    audio (both)  : {audio.name}")
    print(f"    frame overlap : {ov}/8")
    print(f"    row A {ra['id']} gold={gold_a!r} | row B {rb['id']} gold={gold_b!r}")

    _, out_a, *_ = adapter.run(load_frames_for_row(frames_root, ra), str(audio))
    _, out_b, *_ = adapter.run(load_frames_for_row(frames_root, rb), str(audio))
    differs = out_a.strip().lower() != out_b.strip().lower()
    hit_a = gold_a.lower() in out_a.lower()
    hit_b = gold_b.lower() in out_b.lower()

    print(f"    frames A -> {out_a[:60]!r}  gold-match={hit_a}")
    print(f"    frames B -> {out_b[:60]!r}  gold-match={hit_b}")
    print(f"    {'DIFFER — image pathway live' if differs else 'IDENTICAL — VISION DEAD'}")
    if differs and hit_a and hit_b:
        print("    both match gold: images are not merely attached, they are being used")
    elif differs:
        print("    outputs differ but do not both match gold: images reach the model;")
        print("    whether they are used correctly is what Stage 5 measures")

    (out_dir / f"{adapter.name}_image_sensitivity.json").write_text(json.dumps({
        "model": adapter.name, "question_type": qtype, "question": ra["question"],
        "audio": str(audio), "frame_overlap": ov,
        "row_a": ra["id"], "gold_a": gold_a, "out_a": out_a, "gold_match_a": hit_a,
        "row_b": rb["id"], "gold_b": gold_b, "out_b": out_b, "gold_match_b": hit_b,
        "differs": differs, "passed": differs}, indent=2))
    return differs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pairs", action="store_true")
    ap.add_argument("--audio-dir", type=Path, default=Path("data/audio/test"))
    ap.add_argument("--test-manifest", type=Path,
                    default=Path("data/test_multivideo.jsonl"),
                    help="REQUIRED for pair building. There is no manifest-less fallback: "
                         "the old degraded mode fed question_type strings to the leak gate "
                         "and could activate silently even with a manifest present.")
    ap.add_argument("--out-pairs", type=Path, default=Path("data/probe_pairs.json"))
    ap.add_argument("--model", choices=["phi4mm", "minicpmo45"])
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--frames", type=Path, required=True,
                    help="REQUIRED, no default. The frames root is machine-specific: "
                         "dataset/frames locally, data/frames on the pod after rsync. A "
                         "default was correct on exactly one machine and cost a failed "
                         "Stage 4a invocation after the model had already loaded.")
    ap.add_argument("--image-sensitivity", action="store_true",
                    help="Gate 3c: same audio, disjoint frame sets, different golds. "
                         "Gate 3b holds frames fixed and therefore proves nothing about "
                         "vision.")
    ap.add_argument("--omni-mode", action="store_true",
                    help="minicpmo45 only: probe the interleaved omni path instead")
    ap.add_argument("--out", type=Path, default=Path("artifacts/probe"))
    args = ap.parse_args()

    if args.build_pairs:
        build_pairs(args.audio_dir, args.test_manifest, args.out_pairs)
        return 0
    if not args.model or not args.pairs:
        raise SystemExit("need --model and --pairs (or --build-pairs)")

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(args.pairs.read_text())
    images = load_frames(args.frames)

    adapter = make_adapter(args.model, args.omni_mode)
    print(f"[{args.model}] loading...")
    adapter.load()
    for n in adapter.notes:
        print(f"  ! {n}")

    # ---- Q0: determinism control -----------------------------------------------------
    # Without this, "outputs differ" cannot be attributed to the audio. Both model cards
    # use do_sample=True in their examples; we forced greedy and now verify it landed.
    _, det_a, _ = adapter.run(images, pairs[0]["audio_a"])
    _, det_b, _ = adapter.run(images, pairs[0]["audio_a"])
    deterministic = det_a.strip().lower() == det_b.strip().lower()
    print(f"\n  [Q0] same audio twice: "
          f"{'DETERMINISTIC' if deterministic else 'NONDETERMINISTIC'}")
    if not deterministic:
        print("  FAIL — decoding is not greedy. The audio-sensitivity result below")
        print("  would measure sampling noise, not the audio pathway. Fix do_sample /")
        print("  temperature / seed before interpreting anything.")
        print(f"    run1={det_a[:60]!r}\n    run2={det_b[:60]!r}")
        (args.out / f"{args.model}_probe.json").write_text(json.dumps(
            {"model": args.model, "deterministic": False, "passed": False,
             "adapters_sha256": ADAPTERS_SHA256, "sync_source": SYNC_SOURCE,
             "timing_bracket": adapter.TIMING_BRACKET}, indent=2))
        return 1

    # ---- Q1-Q3 -----------------------------------------------------------------------
    n_differ, rows = 0, []
    for i, pair in enumerate(pairs):
        rendered_a, out_a, _ = adapter.run(images, pair["audio_a"])
        rendered_b, out_b, _ = adapter.run(images, pair["audio_b"])

        assert_no_leak(rendered_a, pair["text_a"], f"{args.model}/pair{i}/a",
                       NEUTRAL_INSTRUCTION)
        assert_no_leak(rendered_b, pair["text_b"], f"{args.model}/pair{i}/b",
                       NEUTRAL_INSTRUCTION)

        differs = out_a.strip().lower() != out_b.strip().lower()
        n_differ += differs
        rows.append({"pair": i, "type_a": pair.get("type_a"), "type_b": pair.get("type_b"),
                     "out_a": out_a, "out_b": out_b, "differs": differs})
        print(f"  pair {i}: {'DIFFER' if differs else 'IDENTICAL':9} "
              f"| a={out_a[:40]!r} | b={out_b[:40]!r}")
        if i == 0:
            (args.out / f"{args.model}_rendered_prompt.txt").write_text(rendered_a)

    rate = n_differ / len(pairs)
    passed = rate >= SENSITIVITY_THRESHOLD

    img_ok = None
    if args.image_sensitivity:
        img_ok = image_sensitivity(adapter, args.test_manifest, args.audio_dir,
                                   args.frames.parent, args.out)
        passed = passed and img_ok

    (args.out / f"{args.model}_probe.json").write_text(json.dumps({
        "model": args.model, "revision": getattr(adapter, "REVISION", None),
        "omni_mode": args.omni_mode, "deterministic": True,
        "sensitivity": rate, "image_sensitivity": img_ok,
        # Compare against the Stage 5 manifest's adapters_sha256. Equal => the gates below
        # were passed by the same code the run measured. Unequal => they were not.
        "adapters_sha256": ADAPTERS_SHA256, "sync_source": SYNC_SOURCE,
        "timing_bracket": adapter.TIMING_BRACKET,
        "passed": passed, "rows": rows}, indent=2))

    print(f"\n[{args.model}] deterministic: yes | audio sensitivity: "
          f"{n_differ}/{len(pairs)} = {rate:.0%}")
    if passed:
        print("  PASS — audio pathway is live. Cleared for full evaluation.")
        print("  Next: read artifacts/probe/*_rendered_prompt.txt by eye (Gate 2).")
        return 0

    print("  FAIL — identical outputs across different questions, with decoding")
    print("  confirmed deterministic. The audio is NOT reaching the model.")
    print("  Any accuracy number from this configuration measures the image prior.")
    print("  Check: code path / placeholder scheme, transformers version, processor")
    print("  dropping the audio kwarg, sample rate (16 kHz mono?), init_audio, adapter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
