#!/usr/bin/env python3
"""
stage6_baseline_audit.py — turn results/baseline/minicpmo45.jsonl into a reportable
result, or establish that it is not one.

Local only, no GPU. Steps mirror PRE_REGISTRATION.md:

    step1   is the lenient/strict equality real, or a scoring-path bug?
    step2   §5.2 format-failure audit (blocking on all reporting)
    step3   §5.4 stratification, all 20 question types
    step4   §6 lookup and §7 abandonment criteria

The scorers are the SAME ones the run used — lifted from src/bench/freeze_configs.py by
omni_baseline._load_scoring(). Re-deriving them here would let the audit disagree with the
thing it is auditing.

    python src/analysis/stage6_baseline_audit.py step1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

# omni_baseline imports omni_adapters, which imports PIL. Only the scoring lift is needed
# here, so pull it directly rather than dragging the adapter stack into a laptop session.
import ast


def _load_scoring():
    """Byte-for-byte the lift omni_baseline.py performs, so the audit scores exactly as
    the run did. Duplicated as a function body, not as a rule: the RULE still lives only
    in freeze_configs.py and is read from disk here."""
    want = {"CARVE", "norm", "lenient", "strict"}
    fc = Path(__file__).resolve().parent.parent / "bench" / "freeze_configs.py"
    tree = ast.parse(fc.read_text())

    def bound(node):
        if isinstance(node, ast.FunctionDef):
            return node.name
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    keep = [n for n in tree.body if bound(n) in want]
    ns: dict = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<freeze_configs:scoring>", "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit(f"freeze_configs.py no longer defines {sorted(missing)}")
    return ns["lenient"], ns["strict"], ns["CARVE"]


lenient, strict, CARVE = _load_scoring()

RESULTS = Path("results/baseline/minicpmo45.jsonl")


def load():
    return [json.loads(l) for l in open(RESULTS)]


# ----------------------------------------------------------------------------------------
# STEP 1 — is the 0-disagreement equality real?
# ----------------------------------------------------------------------------------------

def step1() -> None:
    rows = load()
    print(f"rows: {len(rows)}\n")

    # (a) gold appears INSIDE a longer word -> lenient hits, strict must miss.
    #     "up" in "upper left", "no" in "not visible". These are the cases that SHOULD
    #     split the two rules; if any exist while the stored disagreement count is 0,
    #     the scoring path is broken.
    #     A row here only splits the rules if NO carve-out rescues strict — the carve-outs
    #     exist precisely to credit these morphological variants, and every carve-out
    #     variant contains its gold as a prefix ("complete" in "completely"), so whenever
    #     a carve-out fires lenient has already hit. Carve-outs therefore cannot produce
    #     a strict-only hit, and cannot produce a disagreement in either direction.
    inside_word, inside_word_rescued = [], []
    for r in rows:
        g, p = r["short_answer"].lower(), r["predicted_answer"].lower()
        if g not in p:
            continue
        flat = " ".join(p.split())
        if re.search(r"(?<![\w-])" + re.escape(g) + r"(?![\w-])", flat):
            continue                                   # standalone token, rules agree
        if any(re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", flat)
               for v in CARVE.get(g, [])):
            inside_word_rescued.append(r)              # carve-out rescues strict
        else:
            inside_word.append(r)                      # genuine split; must disagree

    # (b) carve-out fired: strict credits a morphological variant.
    carve_fired = []
    for r in rows:
        g = r["short_answer"].lower().strip()
        for variant in CARVE.get(g, []):
            if re.search(r"(?<![\w-])" + re.escape(variant) + r"(?![\w-])",
                         " ".join(r["predicted_answer"].lower().split())):
                carve_fired.append((r, variant))
                break

    # (c) recompute both rules from scratch against the stored fields
    recomputed = [(bool(lenient(r["short_answer"], r["predicted_answer"])),
                   bool(strict(r["short_answer"], r["predicted_answer"])),
                   bool(r["correct"]), bool(r["strict_match"])) for r in rows]
    stored_disagree = sum(a != b for _, _, a, b in recomputed)
    recomp_disagree = sum(a != b for a, b, _, _ in recomputed)
    lenient_drift = sum(a != c for a, _, c, _ in recomputed)
    strict_drift = sum(b != d for _, b, _, d in recomputed)

    words = [len(r["predicted_answer"].split()) for r in rows]

    print("=== (a) gold inside a longer word (lenient hit, strict must miss) ===")
    print(f"  genuine splits (no carve-out rescue): {len(inside_word)}")
    for r in inside_word[:15]:
        print(f"    {r['question_type']:<26} gold={r['short_answer']!r:<16} "
              f"pred={r['predicted_answer'][:56]!r}  correct={r['correct']} "
              f"strict={r['strict_match']}")
    print(f"  rescued by a carve-out (rules agree by design): {len(inside_word_rescued)}")
    for r in inside_word_rescued:
        print(f"    {r['question_type']} | gold={r['short_answer']!r} | "
              f"correct={r['correct']} strict={r['strict_match']}")
        print(f"      FULL PREDICTION: {r['predicted_answer']!r}")

    print(f"\n=== (b) carve-out fired ({CARVE}) ===")
    print(f"  count: {len(carve_fired)}")
    for r, v in carve_fired[:15]:
        print(f"    {r['question_type']:<26} gold={r['short_answer']!r:<14} via {v!r:<12} "
              f"pred={r['predicted_answer'][:46]!r}  correct={r['correct']} "
              f"strict={r['strict_match']}")
    also_lenient = sum(1 for r, _ in carve_fired
                       if r["short_answer"].lower() in r["predicted_answer"].lower())
    print(f"  of those, lenient ALSO hit: {also_lenient}/{len(carve_fired)}")

    print("\n=== (c) predicted_answer length, in words ===")
    print(f"  median {statistics.median(words):.0f} | p90 {sorted(words)[int(.9*len(words))-1]} "
          f"| max {max(words)} | min {min(words)}")
    print(f"  rows with 1 word: {sum(w == 1 for w in words)}  "
          f"| <= 3 words: {sum(w <= 3 for w in words)}  "
          f"| > 15 words: {sum(w > 15 for w in words)}")

    print("\n=== (d) scoring-path integrity ===")
    print(f"  stored   correct != strict_match : {stored_disagree}")
    print(f"  recomputed lenient != strict     : {recomp_disagree}")
    print(f"  stored correct differs from recomputed lenient : {lenient_drift}")
    print(f"  stored strict_match differs from recomputed strict : {strict_drift}")

    bug = (len(inside_word) > 0 and stored_disagree == 0) or lenient_drift or strict_drift
    print(f"\n  VERDICT: {'*** SCORING PATH SUSPECT — STOP ***' if bug else 'equality is real'}")


# ----------------------------------------------------------------------------------------
# STEP 2 — §5.2 format-failure audit, extended to both directions
# ----------------------------------------------------------------------------------------

CELL3 = Path("results/cell3/cell3_finetuned_audio.jsonl")
SEED = 20260731


def carve_census(rows: list[dict], label: str) -> list[dict]:
    """Every row a carve-out credited. Run on BOTH arms: if the baseline is scored under a
    rule that can over-credit, the fine-tune must be checked under the same rule or the
    comparison silently favours one side."""
    hits = []
    for r in rows:
        g = r["short_answer"].lower().strip()
        flat = " ".join(r["predicted_answer"].lower().split())
        if re.search(r"(?<![\w-])" + re.escape(g) + r"(?![\w-])", flat):
            continue                                   # credited on the gold itself
        for v in CARVE.get(g, []):
            if re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", flat):
                hits.append(r)
                break
    print(f"=== carve-out census: {label} ({len(rows)} rows) ===")
    print(f"  rows credited ONLY via a carve-out: {len(hits)}")
    for r in hits:
        print(f"    {r['question_type']} | gold={r['short_answer']!r} | "
              f"correct={r['correct']}")
        print(f"      {r['predicted_answer']!r}")
    return hits


def step2() -> None:
    rows = load()
    cell3 = [json.loads(l) for l in open(CELL3)]

    # --- addition 1: symmetric scrutiny -------------------------------------------------
    carve_census(rows, "BASELINE MiniCPM-o 4.5")
    print()
    carve_census(cell3, "M_ft cell 3 (fine-tuned, audio-only)")

    rng = random.Random(SEED)
    wrong = [r for r in rows if not r["strict_match"]]
    right = [r for r in rows if r["strict_match"]]

    print(f"\n\n=== SAMPLE A: 30 random INCORRECT rows (seed {SEED}; pool {len(wrong)}) ===")
    print("classify: content failure | format failure (right content, lexically missed)\n")
    for i, r in enumerate(rng.sample(wrong, 30), 1):
        print(f"A{i:02d}  {r['question_id']}  {r['question_type']}")
        print(f"     Q    {r['question']}")
        print(f"     gold {r['short_answer']!r}")
        print(f"     pred {r['predicted_answer']!r}\n")

    print(f"\n=== SAMPLE B: 30 random CORRECT rows (seed {SEED}; pool {len(right)}) ===")
    print("classify: true credit | false credit (wrong content, lexically matched)")
    print("A false credit cannot appear among incorrect rows, so this direction needs")
    print("its own sample. §5.2 as written specified only the first direction.\n")
    for i, r in enumerate(rng.sample(right, 30), 1):
        print(f"B{i:02d}  {r['question_id']}  {r['question_type']}")
        print(f"     gold {r['short_answer']!r}")
        print(f"     pred {r['predicted_answer']!r}\n")

    # --- addition 3: census, not sample -------------------------------------------------
    longtail = [r for r in rows if len(r["predicted_answer"].split()) > 15]
    print(f"\n=== CENSUS C: ALL {len(longtail)} predictions over 15 words ===")
    print("Every one, not a sample — a census of a small well-defined subset cannot be")
    print("accused of favourable sampling, and this tail is where format failures concentrate.\n")
    for i, r in enumerate(sorted(longtail, key=lambda x: -len(x["predicted_answer"].split())), 1):
        print(f"C{i:02d}  {r['question_id']}  {r['question_type']}  "
              f"({len(r['predicted_answer'].split())} words, scored correct={r['correct']})")
        print(f"     gold {r['short_answer']!r}")
        print(f"     pred {r['predicted_answer']!r}\n")


# ----------------------------------------------------------------------------------------
# STEP 3 — §5.4 stratification, all 20 types
# ----------------------------------------------------------------------------------------

TEST = Path("data/test_multivideo.jsonl")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used rather than normal-approximation because several types
    sit near 0 or 1, where the normal interval runs outside [0, 1] and is meaningless."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def step3() -> None:
    rows = load()
    test = [json.loads(l) for l in open(TEST)]

    golds: dict[str, set] = {}
    for r in test:
        golds.setdefault(r["question_type"], set()).add(r["short_answer"])

    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["question_type"], []).append(r)

    print("§5.4 stratification — BASELINE MiniCPM-o 4.5, strict scoring, frozen rule")
    print("Per-type transferred prior is deliberately ABSENT: §2.2 forbids it in a")
    print("baseline row — the prior is unavailable to a zero-shot arm.\n")
    hdr = f"{'question_type':<28}{'n':>4}{'acc':>8}{'Wilson 95% CI':>20}{'chance':>8}  note"
    print(hdr)
    print("-" * len(hdr))

    collapse = []
    for t in sorted(by_type):
        rs = by_type[t]
        n = len(rs)
        k = sum(r["strict_match"] for r in rs)
        acc = k / n
        lo, hi = wilson(k, n)
        chance = 1 / len(golds[t])
        notes = []
        notes.append("suggestive only" if n >= 30 else "UNDERPOWERED, excluded from claims")
        if len(golds[t]) == 1:
            notes.append("degenerate")
        if acc < 0.10 and chance > 0.30:
            notes.append("*** CATEGORY COLLAPSE ***")
            collapse.append((t, acc, chance, n))
        print(f"{t:<28}{n:>4}{acc:>8.3f}   [{lo:.3f}, {hi:.3f}]{chance:>8.3f}  {'; '.join(notes)}")

    print("\nNo per-type comparison is called significant: with 20 types the")
    print("multiple-comparison problem makes single-type significance meaningless.")
    print("The honest use of this table is detecting category collapse, not finding winners.")

    print(f"\n=== category-collapse flags (acc < 0.10 where chance > 0.30) ===")
    if not collapse:
        print("  none")
    for t, acc, ch, n in collapse:
        print(f"  {t}: acc={acc:.3f} vs chance={ch:.3f} (n={n})")


# ----------------------------------------------------------------------------------------
# STEP 4 — §6 lookup and §7 abandonment criteria
# ----------------------------------------------------------------------------------------

def step4() -> None:
    rows = load()
    n = len(rows)
    P = sum(r["strict_match"] for r in rows) / n
    disc = [r for r in rows if r["discriminative"]]
    P650 = sum(r["strict_match"] for r in disc) / len(disc)
    lat = statistics.median(r["latency_s"] for r in rows)

    print("*** GOVERNING OUTCOME: §5.2 THRESHOLD BREACHED (33.3% format failures > 20%) ***")
    print("The strict number is NOT reportable as a capability measurement. The §5.2 remedy")
    print("is unavailable: it routes the capability claim to lenient, but lenient and strict")
    print("agree on all 1000 rows, so there is no second metric to fall back to. No")
    print("comparison to M_ft's 0.571 / 0.495 is licensed on this evidence.")
    print("Everything below is A MATTER OF RECORD, explicitly NON-CAPABILITY.\n")

    print("=== §6 band lookup (non-capability, for the record) ===")
    print(f"  P_650 = {P650:.4f}   -> §6.1 band: 0.40 <= P_650 < 0.495")
    print(f"  P     = {P:.4f}   -> §6.2 band: 0.40 <= P     < 0.571")

    print("\n=== §7 abandonment criteria ===")
    print(f"  1. P_650 >= 0.495 ?   {P650:.4f} >= 0.495 -> {P650 >= 0.495}")
    print(f"  2. P     >= 0.571 ?   {P:.4f} >= 0.571 -> {P >= 0.571}")
    print(f"  3. severely weakened if 0.40 <= P_650 < 0.495 AND baseline median latency")
    print(f"     <= the fine-tune's.")
    print(f"     first conjunct : 0.40 <= {P650:.4f} < 0.495 -> {0.40 <= P650 < 0.495}  (LIVE)")
    print(f"     second conjunct: baseline median {lat:.3f}s brackets model.chat() WHOLE")
    print(f"       call including internal preprocessing; the fine-tune's ~0.93s figure was")
    print(f"       measured under a different bracket and possibly different hardware.")
    print(f"       UNRESOLVED pending reconciliation. Not claimed to fail merely because")
    print(f"       {lat:.3f} > 0.93 — the two numbers do not measure the same interval.")
    print(f"     criterion 3 verdict: UNRESOLVED")

    print("\n=== labelled comparison: distance above the transferred prior (0.4200) ===")
    print("  §2.2 forbids the prior in a baseline results row, so it lives here, separately.")
    print(f"    baseline disc-650  {P650:.4f}  ->  {(P650 - 0.4200)*100:+.2f} points")
    print(f"    M_ft     disc-650  0.4950  ->  {(0.4950 - 0.4200)*100:+.2f} points")
    print("  The baseline figure is NOT a capability measurement (see governing outcome).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step1", "step2", "step3", "step4"])
    a = ap.parse_args()
    {"step1": step1, "step2": step2, "step3": step3, "step4": step4}[a.step]()
