#!/usr/bin/env python3
"""
question_identification.py — does the model recover WHICH question it heard?

Motivation
----------
Under audio-only input the question exists nowhere except the audio. Accuracy conflates
two abilities: hearing the question, and answering it in the scored format. This script
isolates the first.

The un-fine-tuned base model is a transcription model, so when handed audio it frequently
reproduces the question instead of answering it. That behaviour is a measurement
opportunity: the corpus contains exactly 20 distinct question strings, so "which question
did the model echo" is a clean 20-way classification with a 5% chance rate.

Method
------
For each prediction, take content tokens (stopwords removed) and compute Jaccard overlap
against the content tokens of each of the 20 corpus questions. The argmax is the question
the output most resembles. A prediction is counted as "echo-like" when that best overlap
clears a threshold; among echo-like rows we report how often the argmax is the question
that was actually spoken on that row.

Two controls, because a metric that cannot fail proves nothing:
  1. Below-threshold rows should identify at roughly chance.
  2. A permutation null (shuffling which question is treated as gold) should collapse
     to 1/20.

Usage
-----
    python src/analysis/question_identification.py \
        --results results/baseline/cell2_base_audio.jsonl \
        --eval_data_path data/test_multivideo.jsonl \
        --out results/baseline/question_identification.json

Reproduces RESULTS.md §4.1.
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

# Function words only. Content-bearing words that happen to be common in this corpus
# ("visible", "many", "motion") are deliberately NOT stopped -- removing them would let a
# near-match on generic phrasing pass as identification.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "this", "that", "it", "and", "or", "be", "been", "with", "from", "there", "here",
    "does", "do", "did", "you", "can", "these", "those", "by", "as",
}

THRESHOLDS = (0.20, 0.35, 0.50)
PRIMARY_THRESHOLD = 0.20
N_PERMUTATIONS = 200
SEED = 0


def content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def best_match(pred: str, question_tokens: list) -> tuple:
    """Return (index of best-matching question, jaccard score)."""
    p = content_tokens(pred)
    if not p:
        return None, 0.0
    scores = [
        (len(p & q) / len(p | q) if (p | q) else 0.0, i)
        for i, q in enumerate(question_tokens)
    ]
    score, idx = max(scores)
    return idx, score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--eval_data_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    eval_rows = [json.loads(l) for l in args.eval_data_path.open()]
    question_by_type = {r["question_type"]: r["question"] for r in eval_rows}
    questions = sorted(set(question_by_type.values()))
    if len(questions) != len(question_by_type):
        raise SystemExit(
            f"expected one distinct question per type; got {len(questions)} questions "
            f"for {len(question_by_type)} types"
        )
    q_tokens = [content_tokens(q) for q in questions]
    chance = 1.0 / len(questions)

    results = [json.loads(l) for l in args.results.open()]

    per_threshold = {}
    for thr in THRESHOLDS:
        n_above = n_correct = 0
        for r in results:
            gold_idx = questions.index(question_by_type[r["question_type"]])
            idx, score = best_match(r["predicted_answer"], q_tokens)
            if idx is not None and score >= thr:
                n_above += 1
                n_correct += idx == gold_idx
        per_threshold[f"{thr:.2f}"] = {
            "n_echo_like": n_above,
            "echo_like_rate": n_above / len(results),
            "n_correct_question": n_correct,
            "precision": (n_correct / n_above) if n_above else None,
        }

    # Control 1: rows BELOW the primary threshold should sit near chance.
    below_n = below_correct = 0
    for r in results:
        gold_idx = questions.index(question_by_type[r["question_type"]])
        idx, score = best_match(r["predicted_answer"], q_tokens)
        if idx is None or score < PRIMARY_THRESHOLD:
            below_n += 1
            below_correct += idx == gold_idx
    below = {
        "n": below_n,
        "precision": (below_correct / below_n) if below_n else None,
    }

    # Control 2: permutation null. Shuffle which question counts as gold for each type;
    # identification should collapse to chance.
    rng = random.Random(SEED)
    null_rates = []
    for _ in range(N_PERMUTATIONS):
        shuffled = questions[:]
        rng.shuffle(shuffled)
        remap = dict(zip(questions, shuffled))
        n = c = 0
        for r in results:
            idx, score = best_match(r["predicted_answer"], q_tokens)
            if idx is not None and score >= PRIMARY_THRESHOLD:
                n += 1
                c += questions[idx] == remap[question_by_type[r["question_type"]]]
        if n:
            null_rates.append(c / n)
    null = {
        "n_permutations": len(null_rates),
        "mean": sum(null_rates) / len(null_rates) if null_rates else None,
        "max": max(null_rates) if null_rates else None,
    }

    summary = {
        "results_file": str(args.results),
        "eval_data_path": str(args.eval_data_path),
        "n_rows": len(results),
        "n_distinct_questions": len(questions),
        "chance_rate": chance,
        "primary_threshold": PRIMARY_THRESHOLD,
        "by_threshold": per_threshold,
        "control_below_threshold": below,
        "control_permutation_null": null,
        "seed": SEED,
        "stopwords": sorted(STOPWORDS),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    p = per_threshold[f"{PRIMARY_THRESHOLD:.2f}"]
    print(f"rows                      : {len(results)}")
    print(f"distinct corpus questions : {len(questions)}  (chance {chance:.3f})")
    for thr in THRESHOLDS:
        d = per_threshold[f"{thr:.2f}"]
        prec = f"{d['precision']:.3f}" if d["precision"] is not None else "n/a"
        print(
            f"  jaccard >= {thr:.2f}      : {d['n_echo_like']:4d} echo-like "
            f"({d['echo_like_rate']:.1%})  correct question {d['n_correct_question']}"
            f"/{d['n_echo_like']} = {prec}"
        )
    print(f"control, below threshold  : n={below['n']}  precision={below['precision']:.3f}")
    print(
        f"control, permutation null : mean={null['mean']:.3f}  max={null['max']:.3f} "
        f"over {null['n_permutations']} shuffles"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
