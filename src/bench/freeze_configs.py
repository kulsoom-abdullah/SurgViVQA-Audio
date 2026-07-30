#!/usr/bin/env python3
"""
Stage 2 — freeze configs and run the pre-rental sanity assertions.

Writes configs/parity.yaml and configs/vocab_v1.txt, then asserts every value the
screening depends on. Aborts on any mismatch; nothing gets rented until this exits 0.

The Parity-B instruction is extracted BY BYTES from src/evaluate_checkpoint.py and
verified against a pinned sha256. A line-matching extract would have even odds of
capturing the include_question variant, of which the bare instruction is a suffix.

    python src/bench/freeze_configs.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PARITY_B_SHA256 = "19cdcc49804887546caa625547573347cd09b3303bf7143f365e1f58a991f410"
PARITY_B_LEN = 69

EXPECTED_VOCAB = [
    "<5", "NBI", "absent", "adenoma", "advancing", "ascending", "catheter", "complete",
    "descending", "down", "forceps", "hyperplastic", "left", "lower-left", "lower-right",
    "no", "rectum", "right", "sigma", "snare", "stable", "up", "upper-left",
    "upper-right", "withdrawing", "yes",
]
EXPECTED_DEGENERATE = [
    "blue_dye_presence", "endoscope_visibility", "lesion_histology_extended",
    "lesion_size_range", "lighting_mode", "tool_catheter_check", "tool_identification",
]

CARVE = {"complete": ["completely"], "down": ["downward"], "up": ["upward"]}


def norm(s: str) -> str:
    return s.lower().strip().strip('.,!?;:"\'')


def lenient(gold: str, pred: str) -> bool:
    return gold.lower() in pred.lower()


def strict(gold: str, pred: str) -> bool:
    g, p = norm(gold), norm(pred)
    if re.search(r'(?<![\w-])' + re.escape(g) + r'(?![\w-])', p):
        return True
    return any(re.search(r'(?<![\w-])' + re.escape(v) + r'(?![\w-])', p)
               for v in CARVE.get(g, []))


ok = True


def check(label: str, got, want=None, cond=None) -> None:
    global ok
    passed = cond if cond is not None else (got == want)
    ok &= bool(passed)
    tail = "" if want is None else f"  (expect {want})"
    print(f"  [{'PASS' if passed else 'FAIL'}] {label:<52} {got}{tail}")


# ---- 2a: extract and freeze Parity-B by bytes ------------------------------------------
print("=== 2a. Parity-B instruction ===")
src_lines = Path("src/evaluate_checkpoint.py").read_text().split("\n")
line309 = src_lines[308]
m = re.search(r'"([^"]*)"', line309)
if not m:
    raise SystemExit("could not extract a quoted string from line 309")
PARITY_B = m.group(1)
h = hashlib.sha256(PARITY_B.encode()).hexdigest()
check("extracted length", len(PARITY_B), PARITY_B_LEN)
check("sha256", h[:16] + "...", cond=(h == PARITY_B_SHA256))
check("no 'User Question' in extract", "User Question" not in PARITY_B, True)
check("no f-string brace in extract", "{" not in PARITY_B, True)

Path("configs").mkdir(exist_ok=True)
Path("configs/parity.yaml").write_text(
    f'parity_b_instruction: "{PARITY_B}"\n'
    f'parity_b_sha256: "{PARITY_B_SHA256}"\n'
    f'parity_b_length: {PARITY_B_LEN}\n'
    f'source: "src/evaluate_checkpoint.py:309"\n'
)
print("  wrote configs/parity.yaml")

# ---- 2b: vocab V1 from TRAIN only ------------------------------------------------------
print("\n=== 2b. configs/vocab_v1.txt (TRAIN only) ===")
train = [json.loads(l) for l in open("data/train_multivideo.jsonl")]
test = [json.loads(l) for l in open("data/test_multivideo.jsonl")]
vocab = sorted({r["short_answer"] for r in train})
check("entry count", len(vocab), 26)
check("matches expected list exactly", vocab == EXPECTED_VOCAB, True)
test_answers = {r["short_answer"] for r in test}
check("test answers", len(test_answers), 21)
check("test answers unreachable under V1", len(test_answers - set(vocab)), 0)
Path("configs/vocab_v1.txt").write_text("\n".join(vocab) + "\n")
print("  wrote configs/vocab_v1.txt")

# ---- 2d: sanity assertions -------------------------------------------------------------
print("\n=== 2d. test set shape ===")
check("row count", len(test), 1000)
check("distinct question_type", len({r["question_type"] for r in test}), 20)
check("distinct question", len({r["question"] for r in test}), 20)
pairs = {(r["question_type"], r["question"]) for r in test}
check("(type, question) is 1:1", len(pairs), 20)

print("\n=== 2d. cell3 recompute ===")
cell3 = [json.loads(l) for l in open("results/cell3/cell3_finetuned_audio.jsonl")]
check("cell3 rows", len(cell3), 1000)
field_acc = sum(r["correct"] for r in cell3) / len(cell3)
strict_acc = sum(strict(r["short_answer"], r["predicted_answer"]) for r in cell3) / len(cell3)
disagree = sum(bool(r["correct"]) != strict(r["short_answer"], r["predicted_answer"])
               for r in cell3)
check("correct field accuracy", f"{field_acc:.4f}", "0.5710")
check("strict rule accuracy", f"{strict_acc:.4f}", "0.5710")
check("rows where they disagree", disagree, 0)

print("\n=== 2d. discriminative predicate (capacity_control.py:147-151) ===")
golds_by_type: dict[str, set] = {}
for r in test:
    golds_by_type.setdefault(r["question_type"], set()).add(r["short_answer"])
discriminative = {k for k, v in golds_by_type.items() if len(v) > 1}
degenerate = sorted(set(golds_by_type) - discriminative)
disc_rows = [r for r in test if r["question_type"] in discriminative]
check("discriminative types", len(discriminative), 13)
check("discriminative rows", len(disc_rows), 650)
check("degenerate types match exactly", degenerate == EXPECTED_DEGENERATE, True)

print("\n=== 2d. transferred prior (train majority per type) ===")
from collections import Counter
maj = {}
for t in golds_by_type:
    c = Counter(r["short_answer"] for r in train if r["question_type"] == t)
    maj[t] = c.most_common(1)[0][0] if c else None
prior_all = sum(maj[r["question_type"]] == r["short_answer"] for r in test) / len(test)
prior_650 = sum(maj[r["question_type"]] == r["short_answer"] for r in disc_rows) / len(disc_rows)
check("prior on full-1000", f"{prior_all:.4f}", "0.5730")
check("prior on 650", f"{prior_650:.4f}", "0.4200")
per_type_chance = sum(1 / len(golds_by_type[t]) for t in discriminative) / len(discriminative)
print(f"  (per-type chance on the 650, for contrast: {per_type_chance:.4f} -- not the prior)")

print("\n" + ("ALL STAGE 2 ASSERTIONS PASSED" if ok else "*** STAGE 2 FAILED ***"))
raise SystemExit(0 if ok else 1)
