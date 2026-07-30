# Pre-Registration — Off-the-Shelf Omni Baseline for SurgViVQA-Audio

**Written:** 2026-07-28 · **Revised:** 2026-07-29 (v3, before any baseline model was run)
**Status:** written before any baseline model was downloaded or run.
**Amendable:** §10 only, with reason and timestamp. Nothing above §10 changes after the
first row of baseline data exists.

**v2 changes (all pre-data):** dropped the zero-shot-vs-zero-shot framing as not
like-for-like; scoped the transferred-prior floor to my fine-tune only; pinned the model
revision; added §11 environment verification.

**v3 changes (all pre-data):** probe two candidates, run one, per the §3.1 decision table;
MiniCPM-o 4.5 preferred on the architectural-parallel argument (§3.2); added Gate 3a
determinism control without which Gate 3b is uninterpretable (§4.4); added vocabulary
conditions V0/V1, both run and reported, V0 primary (§3.4); two-venv plan with the
confirmed transformers conflict and per-model attention settings (§11).

---

## 1. The question

An interviewer asked why this system was built rather than using an off-the-shelf omni
model. I did not have an answer. This study produces one, and accepts the possibility that
the answer is "you shouldn't have."

**Formal question:** on spoken-question surgical VQA (audio-only question, 8 video frames,
short text answer), **does an off-the-shelf omni model already do this, zero-shot, without
the work I did?**

The comparison that answers it is **Phi-4-multimodal zero-shot vs my fine-tuned model.**
That is the only comparison this study is built to support.

---

## 2. Existing numbers, fixed as of this document

Test set: 1,000 rows, 20 question types, one held-out colonoscopy patient
(REAL-Colon 002-004). No baseline model has been run against it.

### 2.1 The comparison target

| Quantity | Symbol | Full 1000 | Discriminative-650 |
|---|---|---|---|
| **My fine-tuned model, audio-only** | `M_ft` | **0.571** | **0.495** |

Verified 2026-07-29 by recomputation from `results/cell3/cell3_finetuned_audio.jsonl`:
the `correct` field yields 0.5710, and the strict word-boundary regex with the
complete/down/up carve-outs yields **0.5710 on the same rows — 0 disagreements, 0.00 pp
delta**. The claim that strict and lenient coincide for this arm is confirmed, not assumed.

### 2.2 Context rows — NOT comparators

| Quantity | Value | Why it is not a comparator |
|---|---|---|
| Qwen2-Audio-7B-Transcription, audio-only | 0.107 | A model **fine-tuned specifically to transcribe**, not a general-purpose model performing badly. Measuring a general-purpose omni model against a transcription-specialised one is not like-for-like and would flatter the baseline for reasons unrelated to architecture. **Provenance to record before publication** — which harness and which instruction string produced it. `src/evaluate_zeroshot.py` is ruled out (vision+text only, no audio path), but the instruction regime behind 0.107 is not yet established, and the row must be labelled with it or dropped. |
| Transferred-prior floor `F` | 0.573 / **0.420** | Derived from **my training labels**. A zero-shot model has never seen them and cannot be held to a bar built from them. Bar for `M_ft` only. |
| Per-type chance `1/\|A_t\|` | 0.648 / **0.458** | Reference line for zero-shot arms — but see §2.2a: unusable on full-1000. |

**Neither the floor nor 0.107 appears in a row, column, or chart containing a baseline
number.** Both live in their own clearly-labelled context section.

### 2.2a Floor and chance, disambiguated — CORRECTED 2026-07-29

Both were recomputed from `data/train_multivideo.jsonl` and `data/test_multivideo.jsonl`.
Earlier drafts of this document used **0.459 as the transferred-prior floor on the
discriminative-650. That was wrong.**

| Definition | Full 1000 | Disc-650 |
|---|---|---|
| Train-majority per type (**transferred prior**) | **0.5730** ✔ reproduces | **0.4200** |
| Per-type chance `1/\|A_t\|` | 0.6475 | **0.4577** |
| Test-majority per type (oracle, not a prior) | 0.6480 | 0.4585 |

0.459 is **per-type chance** (0.4577), not the transferred prior. The transferred prior on
the 650 is **0.420**. Consequences:

- `M_ft` on the 650 is **+7.5 pp over the transferred prior** (0.495 vs 0.420), not +3.6.
  The fine-tune's margin is roughly double what earlier drafts recorded. The correction
  runs in my favour, which is why it needed checking rather than accepting.
- `M_ft` is **+3.7 pp over per-type chance** (0.495 vs 0.4577). Both margins get reported;
  they answer different questions and neither substitutes for the other.
- **Per-type chance is unusable on full-1000.** The 7 degenerate types have `|A_t| = 1`, so
  chance is 1.0 across 350 rows and the aggregate inflates to 0.6475 — above `M_ft` itself.
  Independent confirmation that discriminative-650 must be the primary set.

**One per-type finding that aggregate reporting would bury:** on `lesion_site` the train
majority is `descending`, but the held-out patient's test answers are only
`{sigma, rectum}` — so the transferred prior scores **0/50** there. That is a real
train/test distribution shift on the held-out patient, not noise, and it is a substantial
part of why the 650 floor (0.420) sits below chance (0.458). It goes in the writeup by name.

### 2.3 The fact about my own model that constrains everything below

`M_ft` (0.571) is **below** the transferred-prior floor (0.573) on the full set — by 0.2 pp,
confirmed by recomputation. The only place my fine-tune shows a margin over its own prior
is the discriminative-650
subset: 0.495 against a transferred prior of 0.420, **+7.5 points** (§2.2a).

Consequences, binding:

1. **Aggregate accuracy on 1000 rows is not the headline for any arm.** For my model it is
   dominated by answer priors and sits below its own floor.
2. **Discriminative-650 is the primary evaluation set. Full-1000 is secondary.** Stated
   now, before baseline data exists, so it cannot later look like a subset chosen after
   seeing which one was favourable.
3. The `F`-vs-`M_ft` comparison stays in the writeup even though it is unflattering. It is
   the honest characterisation of what the fine-tune achieved.

---

## 3. Arms

### 3.1 Candidates — probe two, run one

| Arm | Model | Condition | Role |
|---|---|---|---|
| **A1** | **MiniCPM-o 4.5** (9B, rev `44151b3`) | zero-shot | **Preferred run** — see §3.2 |
| **A1′** | **Phi-4-multimodal-instruct** (5.58B, rev `93f923e`) | zero-shot | Fallback run / safer probe |
| **A2** | Qwen2-VL + Whisper-large-v3-turbo, QLoRA FT | fine-tuned | **Comparison target** (`M_ft`) |
| C1 | Qwen2-Audio-7B-Transcription | as measured | Context only (§2.2) |
| C2 | Transferred-prior floor | n/a | Context only, **A2 section exclusively** |

**Decision rule, fixed before probing:**

| Probe outcome | Full run | Reporting |
|---|---|---|
| Both pass | **MiniCPM-o 4.5** | Phi-4 reported as *screened and probed, not run*, with its probe result |
| Only MiniCPM-o passes | MiniCPM-o 4.5 | Phi-4 reported as *probe failed*, with the failure mode |
| Only Phi-4 passes | Phi-4 | MiniCPM-o reported as *probe failed*, with the failure mode |
| Neither passes | **None** | Probe failure is the result. No accuracy number is produced or reported. |

The last row is a real outcome, not a contingency to improvise around. A model whose audio
pathway cannot be shown to be live produces no reportable number.

### 3.2 Why MiniCPM-o is preferred despite Phi-4 being the safer run

MiniCPM-o 4.5 is SigLip2 + **Whisper-medium** + Qwen3-8B — a Whisper encoder projected
into an LLM. That is SurgViVQA-Audio's design, built by a better-resourced team. It answers
**"is my design sound, and what does the well-resourced version of it reach?"**

Phi-4 cannot answer that. It is a different design (mixture-of-LoRAs, its own conformer
speech encoder) solving the same task, so it answers only "does *something* off-the-shelf
do this." Useful, narrower.

Phi-4 is the safer run — documented multi-image+audio template, MIT, smallest, fewest
unknowns. MiniCPM-o is the more informative one. The probe decides on evidence rather than
on which desk research read more comfortably.

### 3.3 Scope

Gemma 4 12B, Qwen2.5-Omni and Nemotron 3 Nano Omni are screened and scoped but **not run
and not probed**. Scope reopens only on an explicit decision after A1's number exists.

### 3.4 Vocabulary conditions — both run, both reported, never merged

| Condition | Instruction | Role |
|---|---|---|
| **V0 — no hint** | Parity-B string alone | **PRIMARY.** Matches the regime `M_ft` was measured under, so the comparison is parity-clean. |
| **V1 — global vocabulary** | Parity-B string + union of gold answers across all 20 types | **SECONDARY.** Bounds the vocabulary confound. |

V1 is parity-safe: a 20-type union does not identify which type a row belongs to, so no
question content reaches the model as text. It converts *"the baseline scored low, possibly
because it could not guess my answer space"* from a caveat into a measured bound.

**V1 never becomes the headline.** It is reported as its own row, labelled
"global answer vocabulary supplied", beneath V0. If V1 ≫ V0, that is a finding about
answer-space mismatch and is reported as such — not as the baseline's capability.

### 3.5 The asymmetry this comparison contains, stated up front

A1 is zero-shot. A2 is fine-tuned on 20 in-domain question types and **has seen the answer
vocabulary**. This is not a flaw — it is the question being asked. But it means **A2 holds
a structural advantage that is not architectural**, and every table, slide and README
sentence labels A2 "fine-tuned, in-domain" wherever it appears, including if the result is
favourable. V1 exists to size that advantage.

---

## 4. Prompt parity protocol

This project's original result was invalidated because the question arrived as text
alongside the audio. The rules below are gates, not guidelines. A violated gate aborts the
run.

### 4.1 The invariant

For every arm and every row: **the question content reaches the model through audio and
through nothing else.** Not in the user turn, not in an instruction string, not in a system
prompt, not injected by a chat template, not in a filename passed as metadata.

### 4.2 Gate 1 — programmatic leak assertion (per row, blocking)

Runs against the **fully rendered prompt string**, post-template. Reading the template is
not checking the template; templates inject content at render time.

Two layers, both in `verify_combined_path.py`:

- **Content tokens** — every non-function word of the question must be absent.
  The stopword list is deliberately minimal. An earlier draft excluded "what", "how",
  "many", "visible", "see" — which are content-bearing in surgical VQA — reducing
  *"how many polyps are visible"* to the single token `{polyps}`. A template injecting
  "how many are visible" would have passed. Fixed and unit-tested against that case.
- **Contiguous trigrams** — the question's 3-grams, function words included, must be
  absent from the normalised prompt. Catches injected phrasing whose individual words are
  each innocuous.

Fails loud, aborts, does not skip the row. A skipped row is a silently biased test set.

### 4.3 Gate 2 — rendered-prompt artifacts (per arm, manual, blocking)

Dump three rendered prompts to `artifacts/prompts/` and read them. The leak assertion only
catches question content; it does not catch a template that helpfully appends
"Answer the question above."

### 4.4 Gate 3 — determinism control, then audio-sensitivity (blocking)

**Gate 3a — determinism control.** Same images, the **same** audio clip, twice. Outputs
must be **identical**.

This runs first because without it Gate 3b is uninterpretable. Both candidates' model
cards use `do_sample=True` in their examples; under sampling, two runs of the same audio
differ by chance, "outputs differ" proves nothing, and the probe would pass while
measuring noise. We force greedy (`do_sample=False`, `num_beams=1`) and then verify greed
actually took effect. A nondeterministic result aborts before Gate 3b is even scored.

**Gate 3b — audio-sensitivity.** Same images, two **different** audio clips, 10 pairs.
Outputs must differ on **≥ 8 of 10**.

If they do not, the audio pathway is not connected, and any accuracy number from that
configuration measures the image prior and the answer distribution. This is the specific
failure that already cost this project one result, and it is invisible in aggregate
metrics.

**Pair construction.** Pairs are built from the real eval artifacts —
`data/audio/test/{video_id}_{id}_{question_type}.mp3`, 1000 clips — never from freshly
synthesised audio, because the point is to test the pathway the full run will use. Each
pair draws its two clips from **different `question_type`s**, so the two spoken questions
genuinely differ; same-type pairs can share a gold answer, which would make identical
outputs look like a dead pathway when they are in fact correct. One fixed set of 8 frames
throughout — the test is whether output tracks the audio, not whether it is right.

### 4.5 What parity does *not* mean

**Same information, each model's native format** — not byte-identical prompt strings.
Layouts are recorded in `artifacts/prompts/`.

### 4.6 The instruction string — RESOLVED 2026-07-29, no re-measurement needed

The blocking prerequisite is discharged. `src/evaluate_checkpoint.py:309` shows the exact
string the `audio_only` arm ran under, and therefore the regime `M_ft = 0.571` was measured
in:

> `Answer the question concisely based on the visual and audio evidence.`

**This string is adopted verbatim as Parity-B.** It carries no question content, and the
`include_question=False` branch provably cannot interpolate `sample['question']` — the
question can only enter through the f-string in the sibling branch.

Adopting it rather than my earlier invented string has two consequences, both good:

1. **A2 does not need re-measuring.** The ~2 GPU-hour prerequisite is gone.
2. **It is the cleaner string anyway.** Tested against all 20 distinct question strings,
   the existing string collides with the leak gate **once**; my invented string
   ("Answer the spoken question about these video frames in a few words") collides
   **four times** — `frames` and `these` appear in four of the twenty questions. Had I
   frozen my own wording, the gate would have aborted on 200 of 1000 rows.

**The one collision, and the gate fix it forces.** `based` is a content token of
*"How severe is the fluid-based occlusion?"* and also appears in the instruction's
"based on the visual and audio evidence". That is a false positive: the instruction is a
known constant, not leakage. Fix — **excise the frozen instruction from the rendered prompt
before running the token and trigram checks**, then assert on the remainder. The instruction
itself is audited once by eye under Gate 2 (§4.3), which is the right place for a constant.
Implemented in `verify_combined_path.py`.

`configs/parity.yaml` freezes the string above, byte-identical, for V0 and V1 and for any
re-run of A2.

## 5. Metrics

### 5.1 Two scoring variants, both reported

- **Strict:** exact match after the project's existing answer normalizer.
- **Lenient:** gold answer appears as a substring in the first 15 generated tokens.

**Strict is the headline** (it is the regime `M_ft` was measured under). Lenient is the
robustness check and carries extra weight here — see §5.4.

**Direction-agreement rule:** if the two variants disagree on the *direction* of the
A1-vs-A2 comparison, that comparison is reported as inconclusive. It is not resolved by
picking the favourable one.

### 5.2 Format-failure audit (blocking, before any number is reported)

Sample 30 random incorrect outputs **per condition (V0 and V1)** and classify each as:

- **content failure** — answered the question, answer wrong
- **format failure** — answered correctly inside a paragraph, refused, described the images
  instead of answering, or emitted a reasoning trace

**If > 20% of sampled failures are format failures, the strict number is not reportable as
a capability measurement.** It is reported as a prompting/format result, with lenient
carrying the capability claim, and the writeup says which.

A broken baseline is not a beaten baseline. If Phi-4 scores 0.05 because it writes
paragraphs, "my model beat Phi-4 by 45 points" is a false statement about a true number.

### 5.3 Reference line for the zero-shot arm: per-type chance

The transferred-prior floor is unavailable to A1 (§2.2). The reference line for a zero-shot
arm is **per-question-type chance**:

> `chance_t = 1 / |A_t|`, where `A_t` = distinct gold answers observed for question type
> `t` in the test set. Aggregate over the 650 = `Σ_t (n_t / N) · chance_t`.

**Critical caveat, pre-registered:** chance assumes knowledge of the answer set `A_t`. A1
does not have it — the question arrives only as audio and no options appear in the prompt.
Therefore:

- **A1 scoring below per-type chance is NOT evidence it is worse than random.** It is
  evidence of **answer-vocabulary mismatch** — saying "several" where gold is "3".
- A below-chance result **triggers the §5.2 audit and the lenient read**, and does not
  license a capability claim in either direction.

### 5.4 Stratification — mandatory

**Every arm reports accuracy per question type. All 20.** An aggregate hid a total category
failure on this project once; it does not get a second chance.

Per type: `n`, accuracy, Wilson 95% CI, per-type chance. (Per-type transferred-prior floor
in the A2 section only.)

~50 rows/type average → 95% CI ≈ ±14 points. Therefore:

- **n ≥ 30**: reported with CI, described as suggestive, never a standalone finding.
- **n < 30**: in the table for completeness, marked underpowered, excluded from claims.
- **No per-type comparison is called significant.** With 20 types the multiple-comparison
  problem makes single-type significance meaningless. The honest use of stratification here
  is detecting **category collapse**, not finding winners.

**Category-collapse trigger:** any arm scoring < 0.10 on a type whose per-type chance
exceeds 0.30 is named in the writeup, regardless of aggregate — including if it is A2.

### 5.5 Latency

Per-example wall-clock, prompt-in to answer-out, same card, same batch size,
`torch.cuda.synchronize()` either side. Median and p90 per arm.

Included because the architectural case for audio input here was never primarily accuracy —
it is latency reduction against an ASR→text pipeline and elimination of ASR error
propagation, in a setting where the surgeon's hands are occupied and speech is the only
free input channel. If that is the real argument it should be measured, and a 5.58B
baseline is exactly the model that could undercut it.

---

## 6. Pre-registered interpretation

`P` = the **run model's** zero-shot accuracy, condition **V0**, strict. Which model that
is (MiniCPM-o 4.5 or Phi-4) is decided by the probe per §3.1, not by this table. Bands are
identical either way — set before any data exists, so the conclusion is a lookup, not a
negotiation. V1 is reported alongside and is never substituted into these bands.

### 6.1 Primary read — discriminative-650

| Outcome | Interpretation |
|---|---|
| `P₆₅₀ ≥ 0.495` | An off-the-shelf model matches my fine-tune **where my only real margin lives**. Argument abandoned — §7. |
| `0.40 ≤ P₆₅₀ < 0.495` | Off-the-shelf gets close zero-shot. Fine-tune bought a modest margin; value case shifts to latency. |
| `P₆₅₀ < 0.40` | Substantial gap. The fine-tune did work that off-the-shelf does not do. |

### 6.2 Secondary read — full 1000

| Outcome | Interpretation |
|---|---|
| `P ≥ 0.571` | Off-the-shelf zero-shot matches my fine-tune. Abandoned regardless of §6.1. |
| `0.40 ≤ P < 0.571` | Task substantially solvable zero-shot; the fine-tune's contribution is smaller than the headline suggests. |
| `P < 0.40` | Fine-tune motivated on the full set too. |

### 6.3 Outcomes that are not results

- **`P` low *and* format-failure rate > 20%** — measured prompting, not capability. Not
  reportable as a win. Re-report under lenient.
- **`P` low and Gate 3 marginal, or Gate 3a nondeterministic** — failed run. Do not report.
- **`P` below per-type chance** — §5.3. Vocabulary mismatch, not incapability.

---

## 7. Abandonment criteria

The argument: *a task-specific VLM + speech-encoder fine-tune is worth building for
spoken-question surgical VQA rather than using an off-the-shelf omni model.*

**Abandoned if either:**

1. `P₆₅₀ ≥ 0.495` (V0) — off-the-shelf zero-shot matches my fine-tune on the primary set.
2. `P ≥ 0.571` (V0) — same on the full set.

**Severely weakened if:** `0.40 ≤ P₆₅₀ < 0.495` **and** the baseline's median latency ≤ my system's.
No meaningful accuracy margin and no speed margin leaves no argument worth making.

**What abandonment means concretely,** written now so it is not renegotiated later:

- The resume bullet stays in the pending column and does not ship.
- The README leads with the negative result.
- The interview answer becomes: *"I ran it. A 5.6B off-the-shelf omni model matches my
  fine-tune zero-shot. The fine-tune wasn't worth the compute, and here's the ablation that
  told me."* That is a better answer than the one I currently don't have, and a stronger
  signal than a favourable number.

**NOT abandoned merely because:**

- The baseline beats the 0.107 context row. That number is a transcription-specialised
  model and is not a comparator (§2.2).
- The baseline clears the 0.573 transferred prior. Not a bar it is held to (§2.2).
- The baseline beats A2 on a single question type. §5.4 — no single-type claims, either
  direction.
- The baseline fails to produce parseable output. §5.2.
- **V1 beats A2 while V0 does not.** V1 is the secondary condition (§3.4); it bounds the
  vocabulary confound, it does not replace the primary comparison.

---

## 8. Known confounds

| Confound | Direction | Mitigation |
|---|---|---|
| `M_ft` measured under a different instruction regime | either | §4.6 blocking prerequisite — re-measure under Parity-B |
| **A2 has seen the answer vocabulary; A1 has not** | **favours A2** | Cannot be removed — it is the question being asked. Declared prominently; lenient scoring partly absorbs it; see §8.1 |
| **TTS voice/prosody matched to A2's training distribution** | **favours A2** | Cannot be removed. Declared in full sentences, not a footnote. Optional: 100-row slice with a second TTS voice as sensitivity check |
| **8-frame sampling scheme tuned for A2** | **favours A2** | Same 8 frames to both arms; declared |
| Output verbosity differs by model family | favours A2 | Dual scoring §5.1 + audit §5.2 |
| Phi-4 wrong code path / placeholder scheme | either, silent | Gate 3 §4.4 + pinned revision |
| One held-out patient | n/a | Restated as an existing limitation |

### 8.1 The thing to say out loud in the writeup

**Every known confound in this study points the same way — toward flattering my model.**
A2 knows the answer vocabulary, was trained on this TTS voice, and gets a frame-sampling
scheme chosen for it. Therefore:

- If **Phi-4 wins anyway**, the result is strong and should be trusted.
- If **my model wins narrowly**, the margin is not trustworthy and must not be reported as
  though it were.

**Optional secondary condition (requires sign-off, not pre-registered as running):** re-run
A1 with the *global* answer vocabulary — the union of gold answers across all 20 types —
appended to the instruction. This is parity-safe (a 20-type union does not identify which
type a row belongs to) and closes most of the vocabulary gap. One extra run of the same
model on the same pod, ~$1.50. Decide before the run or not at all.

---

## 9. What this study does not claim

- Not a claim about omni models as a class. One model, one task, one patient.
- Not a claim that Phi-4 is bad at audio VQA. A claim about Phi-4 zero-shot on colonoscopy
  VQA with TTS questions.
- Not a benchmark. `n=1` patient, TTS audio, one frame-sampling scheme.
- No result becomes a resume bullet until §4.4 and §5.2 have both passed.

---

## 10. Deviation log

Amendments after first data. Every entry: date, what changed, why, what it does to the read.
Entries marked **pre-data** happened before any baseline row existed and therefore cannot
have been influenced by a result; they are recorded because instrument stability is part of
what this document attests to.

| Date | Stage | Deviation | Reason | Effect on interpretation |
|---|---|---|---|---|
| 2026-07-29 | 2c, **pre-data** | `build_pairs()` rewritten: filename parsing replaced with forward key construction from the test manifest; manifest-less fallback deleted; per-pair invariants (question_type, question text, inode) changed from documented to asserted | The old parser split the clip stem on `_` and read `parts[1]` as the id, but the id is itself `qa_NNNNNN`. Result: every clip got a unique pseudo-type, so cross-type pairing held only by luck, and the question-text lookup missed on every row — **the leak gate would have been checked against `"003353 lesion site"` instead of the real question**, passing prompts that genuinely leak. Caught by the Stage 2d eyeball check before any pod was rented. | None on the read. No baseline data existed. The invalid `data/probe_pairs.json` was deleted and regenerated. Recorded because the failure was in the gate itself, and a gate that silently checks the wrong thing is the exact hazard §4.2 exists to prevent. |

---

## 11. Environment — two venvs, one pod session

Both pins are **transcribed from model cards and have not been validated on a GPU.**
Precedent: a pin previously validated against a CPU-only bitsandbytes build did not work
on GPU. Verify before anything else.

### 11.1 The venv conflict is confirmed, not assumed

| | Phi-4-multimodal | MiniCPM-o 4.5 |
|---|---|---|
| transformers | **4.48.2** (remote code breaks ≥4.50) | **4.51.0** exactly — card: other versions "may have compatibility issues (under investigation)" |
| torch | 2.6.0 (card) | `>=2.3.0,<=2.8.0` |
| attention | **eager** | **sdpa** — MiniCPM does **not** support eager |
| extra | — | `minicpmo-utils>=1.0.5`, `torchaudio<=2.8.0`, **ffmpeg** (mp3 decode) |
| revision | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | `44151b3` |

**Two venvs, mandatory.** 4.48.2 and 4.51.0 cannot coexist. torch 2.6.0 satisfies both
ranges, so torch can be common; transformers cannot.

**On "skip flash-attn on both":** honoured, but not by using eager on both. MiniCPM
supports `sdpa` or `flash_attention_2` and **not** eager, so MiniCPM-o gets **sdpa** —
torch-native, no compiled wheel, no build step. Same intent, model-legal.

```bash
python -m venv ~/venv-phi4    && ~/venv-phi4/bin/pip install \
  "transformers==4.48.2" "torch==2.6.0" accelerate soundfile pillow scipy peft

python -m venv ~/venv-minicpm && ~/venv-minicpm/bin/pip install \
  "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0" \
  "minicpmo-utils>=1.0.5" librosa pillow
sudo apt update && sudo apt install -y ffmpeg   # mp3 decode for librosa
```

### 11.2 Verify before renting time on anything else

```bash
python - <<'EOF'
import torch, transformers
cc = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), "| sm_%d%d" % cc,
      "| %.0f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
assert torch.cuda.is_available(), "no GPU visible"
assert cc < (12, 0), "Blackwell (sm_120) — bitsandbytes kernels absent, wrong card"
assert torch.cuda.get_device_properties(0).total_memory > 44e9, "need a 48 GB card"
EOF
```

**Card requirement: 48 GB, Ada or Ampere — A6000, L40S, A40. Not Blackwell** (sm_120 lacks
bitsandbytes kernels). The assertion above fails fast rather than discovering it mid-run.

### 11.3 Headroom

Phi-4: 11.15 GB weights. MiniCPM-o: 9B BF16 ≈ 18 GB weights; the card's own efficiency
table reports 19.0 GB in bf16. Both fit 48 GB with room for 8 frames of vision tokens.
`init_tts=False` on MiniCPM-o — text output only, and it saves memory.

### 11.4 MiniCPM-o settings that are arm-defining

- `enable_thinking=False` — **mandatory.** Instruct and thinking modes ship in one model;
  a reasoning trace ahead of a one-word answer destroys exact-match scoring.
- `use_tts_template=False`, `init_tts=False` — text out only.
- `use_image_id=False`, `max_slice_nums=1` — the card's own multi-frame setting.
- Audio: 16 kHz mono float ndarray via `librosa.load(path, sr=16000, mono=True)`.
- `omni_mode`: the card's omni examples interleave frame/audio-segment pairs synced on a
  timeline. Ours is 8 static frames + one complete spoken question — the plain chat path.
  `--omni-mode` probes the alternative if the plain path fails Gate 3b.

---

## Run order

1. ~~Confirm the `M_ft` instruction regime.~~ **DONE (§4.6)** — string recovered from
   `src/evaluate_checkpoint.py:309`, adopted as Parity-B. No re-measurement. Note there is
   no existing leak *checker* to stress-test: the old script relies on branch structure
   (`include_question`), which is sound for a first-party script but not transferable to
   third-party chat templates — hence the assertions in §4.2.
2. Freeze `configs/parity.yaml` (Parity-B string) and `configs/vocab_v1.txt` (the global
   answer-vocabulary union, V0/V1 both frozen before any run).
3. Build probe pairs from real artifacts, no GPU needed:
   `verify_combined_path.py --build-pairs --audio-dir data/audio/test --manifest ...`
4. Rent one 48 GB Ada/Ampere pod. §11.2 verification. **Blocking.**
5. Two venvs (§11.1). Probe **both** candidates:
   `--model minicpmo45` and `--model phi4mm`. **Blocking on Gate 3a then 3b.**
6. Apply the §3.1 decision table. If neither passes, stop — that is the result.
7. Full 1000-row run of the selected model, **conditions V0 and V1**, both scoring
   variants, per-type stratification, latency.
8. Format-failure audit, 30 samples per condition. **Blocking on reporting.**
9. Fill the §6 lookup from V0. Do not reinterpret it.
