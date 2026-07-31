# Detailed Results

Companion to the [README results section](../README.md#results). Every number here traces
to a committed prediction file with a run manifest recording model revision, quantization,
prompt string, grading rule, and environment.

---

## 1. Evaluation design

**Task.** A spoken question about 8 sampled video frames; the model returns a short text
answer. The question reaches the model as **audio only** — the text prompt is a fixed
instruction containing no question content:

```
Answer the question concisely based on the visual and audio evidence.
```

This string is byte-identical across training, evaluation, and every comparison arm. An
earlier version of this project delivered the question as text *alongside* the audio, which
made the audio channel unattributable; see [§8](#8-superseded-results).

**Splits.** Source data is [REAL-Colon](https://doi.org/10.25452/figshare.plus.22202866)
(CC BY), study 002.

| split | rows | videos |
|---|---|---|
| train | 2,302 | 002-001, 002-002, 002-003 |
| eval | 398 | 002-001, 002-002, 002-003 |
| test | 1,000 | 002-004 |

Test is a **held-out patient**, frame-disjoint from train and eval (0 shared frames,
verified). Test is stratified at 50 rows per question type across 20 types.

**Audio.** Generated with edge-tts across 41 English voices, split **23 / 9 / 9** across
train / eval / test with **no voice appearing in two splits**, stratified by accent region
and gender, seeded. Audio files are keyed `{video_id}_{id}_{question_type}` — a composite
key adopted after a flat `{id}` scheme was found to collide (78 collisions across the
combined corpus, 18 with contradictory answers). Coverage is gated at 100% before any run;
a missing audio file raises rather than substituting silence.

---

## 2. Grading rule

Predictions are free text; gold answers are short tokens. The rule is:

1. Lowercase, strip whitespace and surrounding punctuation.
2. Match gold as a whole token: `(?<![\w-])gold(?![\w-])`.
3. Documented morphological carve-out: `complete` ← *completely*, `down` ← *downward*,
   `up` ← *upward*.

This replaced a plain substring check. The substring rule credited `"no"` inside *cannot*,
*none*, and *adenoma* — 33.7% of test rows have gold `"no"`. Regrading the three baseline
cells under the strict rule:

| | substring | strict | delta |
|---|---|---|---|
| base, text question | 36.60% | 36.60% | 0.00 |
| base, audio question | 10.70% | 9.00% | −1.70 |
| fine-tuned, audio question | 57.10% | 57.10% | 0.00 |

Only the un-fine-tuned base moved, because it is the only arm that hedges. The fine-tuned
results were never inflated by the lenient rule.

---

## 3. Floors

**Test-prior floor.** For each question type, the most common answer *in the test set*,
emitted every time. Uses no video.

| | full 1,000 | discriminative 650 |
|---|---|---|
| test-prior floor | **64.8%** | **45.8%** |
| global majority (one string for all rows) | 35.0% | 30.8% |

**Transferred-prior floor.** The most common answer *in the training set* per question
type, scored on test — the score reachable by identifying the question by ear and answering
from the training distribution: **57.3%** on the full 1,000.

This floor is noisy. Two question types are near coin flips in training
(`lesion_histology_extended` 43 vs 42, `tool_identification` 37 vs 35 vs 13) while being
single-class in test, so the floor moves 5 points depending on which side the training
majority happens to fall. Read it as **~52–57%**, not as a precise threshold.

The fine-tuned model scores **57.1%**, inside that range. On the full test set it is
statistically indistinguishable from a system that hears which question was asked and
answers from the training prior with no reference to the video.

**Why the 650 is the headline.** Seven of 20 question types have a single gold answer
across all 50 test rows. On those 350 rows a constant string scores 100% and no model can
be distinguished from a lookup table. The remaining 13 types (650 rows) are the only place
visual discrimination can be observed.

---

## 4. Baseline cells

All three use the same weights — `kulsoom-abdullah/Qwen2-Audio-7B-Transcription`, an
audio-adapted Qwen2-VL — with only the input and the adapter varying.

| cell | model | question delivered as | accuracy (1,000) |
|---|---|---|---|
| 1 | base, no adapter | text | 36.6% |
| 2 | base, no adapter | audio | 10.7% |
| 3 | + task LoRA | audio | **57.1%** |

**Cell 1 establishes that the base cannot do this task in any modality.** At 36.6% it is
1.6 points above answering `"no"` to all 1,000 rows, and 28 points below the test-prior
floor. Sample failures are not near misses — for a question about anatomical site it
answered *"The lesion is located in the eye."*

Consequently **cell 3 − cell 2 measures task acquisition, not audio comprehension.** The
fine-tuning delta cannot be attributed to the audio channel, because the base could not
perform the task through the text channel either.

### 4.1 Audio comprehension, measured directly

Cell 2's low accuracy is a *format* failure, not a perception failure. Given audio and no
question text, the base model frequently reproduced the question it had heard instead of
answering it — behaviour consistent with its transcription fine-tuning.

Scoring each cell-2 output against all 20 corpus questions and taking the best lexical
match:

| overlap threshold | echo-like rows | correct question identified |
|---|---|---|
| ≥ 0.20 | 488 / 1,000 (48.8%) | 472 / 488 = **96.7%** |
| ≥ 0.35 | 414 / 1,000 (41.4%) | 414 / 414 = **100.0%** |
| ≥ 0.50 | 402 / 1,000 (40.2%) | 402 / 402 = **100.0%** |
| chance (1 of 20) | — | 5.0% |
| control: rows below 0.20 | 512 | 11.3% |
| control: permutation null, 200 shuffles | — | mean 4.9%, max 25.6% |

Identification is stable across thresholds. Reproduce with:

```
python src/analysis/question_identification.py \
  --results results/baseline/cell2_base_audio.jsonl \
  --eval_data_path data/test_multivideo.jsonl \
  --out results/baseline/question_identification.json
```

The audio channel transmits the question essentially intact, before any task fine-tuning.

---

## 5. Cross-model capacity control

**Question.** Is weak visual grounding a model-capacity limit, or do the sampled frames not
contain the answers?

**Design.** Four stock vision-language models, zero-shot, question delivered as **text**, on
the identical 1,000 rows with the identical 8 frames at 384px, the identical prompt string,
and the identical grading rule. Audio is not involved, so capacity is the only variable.
All loaded 4-bit NF4 with bf16 compute, greedy, `max_new_tokens=128`.

### 5.1 Per question type

Vocabulary-normalized (see [§6](#6-vocabulary-normalization-audit)). Floor here is the
**test-majority** constant-emitter rate for that type, which for every type in this table
also equals per-type chance `1/|A_t|` except `lesion_screen_position` (26 vs 25, because
its four classes are not exactly even). §5.3 below uses the **train-majority** floor, which
differs on `lesion_site` and `tool_identification` only — see the note under that table.

| question type | classes | floor | this model | Qwen2-VL-7B | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-32B |
|---|---|---|---|---|---|---|---|
| `fluid_occlusion_level` | 2 | 50 | **80** | 0 | 0 | 0 | 14 |
| `flush_action` | 2 | 50 | 50 | 52 | 68 | 58 | **74** |
| `lesion_motion_direction` | 5 | 20 | 20 | 22 | 20 | 24 | 12 |
| `lesion_screen_position` | 4 | 26 | 26 | 34 | 22 | 22 | **48** |
| `lesion_site` | 2 | 50 | 22 | 8 | 0 | 0 | 10 |
| `mucosa_visibility` | 2 | 50 | 48 | 72 | 64 | 72 | **80** |
| `nbi_status` | 2 | 50 | 50 | 50 | 64 | **94** | 52 |
| `occlusion_check` | 2 | 50 | 50 | 86 | 52 | 54 | **88** |
| `scope_backward_motion` | 2 | 50 | 50 | 58 | 50 | 50 | 52 |
| `scope_forward_motion` | 2 | 50 | 54 | 50 | 56 | 60 | **76** |
| `scope_motion` | 2 | 50 | 48 | 52 | 56 | 58 | 56 |
| `scope_motion_type` | 2 | 50 | 50 | 50 | 50 | 52 | 48 |
| `scope_outside` | 2 | 50 | **96** | 74 | 54 | 52 | **96** |
| `blue_dye_presence` | 1 | 100 | **100** | 56 | 84 | 54 | 98 |
| `endoscope_visibility` | 1 | 100 | **100** | 90 | **100** | 98 | **100** |
| `lesion_histology_extended` | 1 | 100 | 0 | 0 | 0 | 62 | 36 |
| `lesion_size_range` | 1 | 100 | **100** | 0 | 0 | 0 | 0 |
| `lighting_mode` | 1 | 100 | **100** | 2 | 4 | 84 | 2 |
| `tool_catheter_check` | 1 | 100 | 98 | 62 | 98 | 96 | **100** |
| `tool_identification` | 1 | 100 | 0 | 26 | 0 | 48 | 70 |

### 5.2 Totals

| | full 1,000 | discriminative 650 |
|---|---|---|
| **floor (constant emitter)** | **64.8** | **45.8** |
| Qwen2-VL-7B | 42.2 | 46.8 |
| Qwen2.5-VL-7B | 42.1 | 42.8 |
| Qwen3-VL-8B | 51.9 | 45.8 |
| **this model (audio question)** | **57.1** | **49.5** |
| Qwen3-VL-32B | 55.6 | **54.3** |

### 5.3 By question family

Margin over floor. Family assignment is a judgment call and is documented in
[§9](#9-known-limitations).

| family | n | floor | this model | Qwen2-VL-7B | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-32B |
|---|---|---|---|---|---|---|---|
| A. Static scene state | 300 | 50.0 | +12.3 | +5.7 | +0.3 | +5.0 | **+17.3** |
| B. Screen-space position | 50 | 26.0 | +0.0 | +8.0 | −4.0 | −4.0 | **+22.0** |
| C. Temporal / motion | 250 | 44.0 | +0.4 | +2.4 | +2.4 | +4.8 | +4.8 |
| D. Anatomical / diagnostic | 200 | 87.5 | −57.0 | −79.0 | −87.5 | −60.0 | −58.5 |

Floors are **per-type chance**, which equals the test-majority floor for families A, C and
D. Family B is the sole divergence — its four classes are not exactly even, giving a test
floor of 26.0 against chance 25.0; the table uses 26.0.

Family D's floor is 87.5 because that *is* per-type chance here: three of its four types
are degenerate (one test answer, chance 100%) and `lesion_site` is binary, so
`(0.5×50 + 50 + 50 + 50) / 200 = 87.5`.

**Train-majority reading, recorded for audit.** Under the transferred prior the D row is
floor 50.0 → −19.5 / −41.5 / −50.0 / −22.5 / −21.0. It is not used, because on `lesion_site`
and `tool_identification` the training majority answer never occurs in this patient's test
answers, so the constant emitter scores 0/50 on both and the floor collapses for reasons
unrelated to difficulty. Families A, B and C are identical under either definition.

- **A** — `nbi_status`, `mucosa_visibility`, `occlusion_check`, `fluid_occlusion_level`,
  `scope_outside`, `flush_action`
- **B** — `lesion_screen_position`
- **C** — `scope_motion`, `scope_motion_type`, `scope_forward_motion`,
  `scope_backward_motion`, `lesion_motion_direction`
- **D** — `lesion_site`, `lesion_histology_extended`, `tool_identification`,
  `lesion_size_range`

### 5.4 Output diversity

Mean distinct prediction strings per 50-row question type:

| family | this model | Qwen2-VL-7B | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-32B |
|---|---|---|---|---|---|
| A | 1.7 | 3.5 | 2.0 | 3.5 | 33.7 |
| C | 1.4 | 2.4 | 1.4 | 3.6 | 34.4 |

The fine-tuned model emits roughly **one string per question type**. Where it wins it is
usually because that constant is the correct answer for a single-class type; where it
genuinely discriminates — `scope_outside` at 96% and `fluid_occlusion_level` at 80%, both
on balanced 25/25 splits — it uses two strings correctly. Those two types are the real
visual results.

### 5.5 Conclusions

**Capacity is not the limit for temporal questions.** Across 7B→33B and two generations,
every model lands within +0.4 to +4.8 of the floor on family C, and
`lesion_motion_direction` stays at 12–24% against a 20% floor for all five. The dataset
samples 8 frames at a fixed offset regardless of where the labeled motion occurs; a visual
audit of failure cases found the labeled motion frequently absent from the sampled frames
entirely. No encoder recovers signal that was never captured.

**Capacity *is* the limit for screen-space position.** Qwen3-VL-32B reaches +22 over floor
where this model sits at exactly the floor with one constant string. Both at 384px, both on
the same frames. This corrects an earlier claim in this repository that attributed
`lesion_screen_position` failure to input resolution.

**`lesion_site` is likely unanswerable from these inputs.** All five systems score 0–22%
against 50% chance — `lesion_site` is binary in the held-out video (`sigma` / `rectum`).
Below chance on a binary type is what supports the claim; "floor" is ambiguous in this
document and does not carry it. Distinguishing sigmoid colon from rectum is done by insertion depth
and navigational landmarks, not from a close-up view of mucosa. This is a
benchmark-construction issue rather than a model failure.

**Two honest losses.** Stock models beat this one on `mucosa_visibility` (48% vs 64–80%)
and `occlusion_check` (50% vs 86–88%), with standard answer vocabulary and no grading
artifact available as an excuse.

---

## 6. Vocabulary normalization audit

The fine-tuned model was trained on the gold answer vocabulary; stock models were not. Strict
token matching therefore penalizes stock models for phrasing rather than for being wrong —
all four scored exactly 0/50 on `lesion_screen_position` while producing outputs like
*"top left quadrant of the image"* against gold `upper-left`.

**Applied map:** quadrant synonyms (`top`/`bottom` ← `upper`/`lower`, and word-order
variants such as *"right lower quadrant"*, applied symmetrically across all four quadrants
so no class is favoured within a type), plus `sigma` ← *sigmoid*.

**Effect on the discriminative 650:**

| | strict | vocabulary-aware | delta |
|---|---|---|---|
| Qwen2-VL-7B | 44.2 | 46.8 | +2.6 |
| Qwen2.5-VL-7B | 41.1 | 42.8 | +1.7 |
| Qwen3-VL-8B | 44.2 | 45.8 | +1.6 |
| Qwen3-VL-32B | 49.8 | 54.3 | +4.5 |
| **this model** | **49.5** | **49.5** | **+0.00** |

The fine-tuned model moves by exactly zero on every question type — it never phrases an
answer any other way. That asymmetry is the point of the audit, and it means the map credits
phrasing rather than leaking correctness.

**Candidates inspected and rejected**, recorded so the map cannot be tuned to raise scores:

| candidate | rows | rejected because |
|---|---|---|
| `forceps` ← *grasper* | 9 | domain judgment, not phrasing; a grasper is not unambiguously forceps |
| `absent` ← *none / no fluid / clear / minimal* | 19 | every match was spurious — *"small amounts of clear fluid"* and *"mild to moderate"* assert fluid is **present**; `clear` modifies fluid transparency, not absence. Would have added +20 points to one model on one type on false credit. |
| `<5` ← *0-5 / 1-5 / 0.1-0.5 mm* | 57 | largest score-raiser and least defensible — most predictions were *"10-20 mm"*, plainly wrong |

The rejected `absent` candidate was tested specifically because it would have undercut this
project's strongest single result (`fluid_occlusion_level` at 80%). It did not; the 80%
stands.

---

## 7. Latency

**Question.** Does feeding audio directly to the model beat transcribing first?

**Design.** Same fine-tuned weights, same frames, same greedy decoding, same token budget,
interleaved A/B/A/B per sample to distribute thermal drift, 5 warmup iterations discarded,
`torch.cuda.synchronize()` around every timed region, n=100 stratified 5 per question type.
Medians with IQR; RTX A6000.

| stage | ASR pipeline | direct audio |
|---|---|---|
| transcription (Whisper decode) | 73.3 ms | — |
| prefill | 870.1 ms | 1270.1 ms |
| decode per generated token | 62.0 ms | 62.0 ms |
| **end to end** | **1677.5 ms** | **1895.6 ms** |
| input tokens | 1,294 | 2,795 |
| generated tokens (median) | 11 | 10 |

`t_total` IQRs do not overlap (1556.9–1806.5 vs 1796.8–2025.1), so the direction is not
noise. Per-token decode is identical to within 0.0 ms, so the raw decode difference was an
output-length artifact, not architecture.

**Result: 0.88×. Direct audio is 13% slower, and the entire difference is prefill.** The
audio path carries 1,502 additional input tokens because the speech encoder emits a fixed
1,500 tokens for a 30-second window regardless of utterance length; the questions here run
2.8–4.1 seconds, so roughly 88% of that context is encoded silence. Skipping transcription
saves 73 ms against a 400 ms prefill penalty.

**ASR fidelity control:** the transcription arm matched the corpus question on 99/100 rows,
so it is a genuine pipeline and not handicapped by mistranscription.

Transcription cost scales with speech length while the audio prefix does not, so longer
utterances would shift the balance. Extrapolating from this single operating point puts the
crossover near 50–60 spoken words. That is an estimate, not a measurement.

---

## 8. Superseded results

An earlier version of this project reported **63.4%** accuracy. That figure does not
reproduce from any committed prediction file and no run manifest exists for it. It was
measured under a prompt that delivered the question as text alongside the audio tokens,
making the audio channel unattributable. It has been withdrawn.

Similarly withdrawn:

- A **2.5× latency speedup**, which compared a separate Whisper pass plus a full text-VQA
  pass against a single direct pass. Holding weights, frames, decoding, and token budget
  constant gives 0.88× (see [§7](#7-latency)).
- Zero-shot generation numbers of 36.0 / 39.4 / 54.1% for Qwen2 / 2.5 / 3.0, which used an
  unrecorded quantization and token budget and a lenient substring scorer. The same models
  re-run under the current harness with manifests give 42.2 / 42.1 / 51.9%.
- A **74.0%** figure for a fine-tuned Qwen3-VL on text questions, which was scored from an
  `exact_match` field rather than the harness's grading rule and has no manifest. Not
  reproducible under the current setup; not carried forward.

---

## 9. Known limitations

**Question-family assignment is a judgment call.** `flush_action` is grouped as static
scene state on the reasoning that a flush jet is visible in a single frame, though it is
defensible as a temporal event. `lesion_site` is grouped as anatomical/diagnostic rather
than spatial, since colonic segment is not determinable from mucosal appearance.

**Audio is synthetic.** Clean TTS across 41 voices with mild seeded rate jitter, no noise,
no reverberation, no real operating-room conditions. The transcription arm's 99% match rate
reflects clean speech; a real acoustic environment would favour direct audio input more than
these results show.

**Accent coverage does not match the deployment population.** `en-IN` voices are included as
the nearest available South Asian English and are explicitly *not* labeled as
Pakistani-accented. One `en-IN` voice appears in the held-out test split, which supports
reporting per-voice accuracy but does not license any claim about accent robustness.

**Only 20 distinct question strings exist in the corpus**, one per question type. The audio
channel therefore carries at most ~4.3 bits of question identity, and question recognition
from audio is a 20-way classification rather than open-vocabulary speech understanding.

**Single test patient.** All 1,000 test rows come from video 002-004. Held-out-patient
generalization is measured on one patient.

**The stock-model comparison is not like-for-like** and every known confound favours this
model: it knows the answer vocabulary, was trained on these TTS voices, and receives a frame
sampling scheme chosen for it. A narrow win should not be trusted; the loss to Qwen3-VL-32B
on the discriminative 650 should be.

---

## 10. Future work

Ordered by expected value, none of it run:

1. **Targeted frame resampling** — re-extract frames aligned to where the labeled motion
   occurs, rather than at a fixed offset. This is the precondition for any temporal result;
   §5.5 shows no model recovers motion from the current sampling.
2. **Native video input** — feed frames through the `videos=` pathway with temporal patch
   merging rather than as independent images. Requires retraining. Would not fix motion
   questions on its own, since temporal encoding cannot recover motion absent from the
   frames.
3. **Variable-length audio encoding** — truncating or pooling the audio token budget to
   actual utterance duration would remove roughly 88% of the prefill cost measured in §7 and
   bring direct audio to approximate latency parity.
4. **Off-the-shelf omni-model comparison** — Phi-4-multimodal and MiniCPM-o are screened and
   pre-registered but not run. The question is whether a general-purpose omni model already
   does this task without task-specific fine-tuning.
5. **Acoustic robustness** — all evaluation audio is clean TTS across 41 voices
   with mild rate jitter. Adding background noise, reverberation, and varied SNR
   would test whether direct audio input degrades more gracefully than an ASR
   pipeline, which is the untested half of the error-propagation argument.
6. **Denser frame sampling and higher resolution** — 16 or 32 frames, 512px or above.
