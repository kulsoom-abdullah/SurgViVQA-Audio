# Audio-Channel Ablation — Pre-Registration

Registered **before** running any arm, so the verdicts can't be reverse-fit to the
numbers. The ablation probes one question: **did the audio channel actually contribute
to the fine-tuned model's 63.4%, or did the model ride on the question text that was
inadvertently present in the prompt during both training and eval?**

Provenance of the confound (already established, read-only):
- Training kept the question as text: [src/train_vqa.py:168](../../src/train_vqa.py) + audio inject [:184–193](../../src/train_vqa.py)
- Eval kept the question as text: [src/evaluate_checkpoint.py:131](../../src/evaluate_checkpoint.py) + audio inject
- The zero-shot audio baseline did NOT (question via audio only): [baselines/utils.py:260](../../baselines/utils.py)

All arms run through the **same** harness ([src/evaluate_checkpoint.py](../../src/evaluate_checkpoint.py),
`--input_mode`), greedy decode, substring scorer — only the prompt content changes.

## The four arms

| `--input_mode` | audio | question text | what it tests |
| :-- | :-: | :-: | :-- |
| `audio_text` | self | ✅ | **sanity gate** — reproduces training/deployment |
| `text_only` | ✗ | ✅ | how well the model does with **no audio** (the confound test) |
| `mismatched_audio` | **other** | ✅ | is the audio **content** read? (in-distribution control) |
| `audio_only` | self | ✗ | can it run **hands-free** on audio alone? |

## (i) Sanity gate — nothing is interpretable until this passes

`audio_text` on the committed hero checkpoint (`checkpoint-1000`) over the full
`data/test_multivideo.jsonl` (1000 samples) **must reproduce 63.4% within noise**
before any other arm is read.

- Pre-registered band: **≥ 61.4%** (63.4 − 2 pp). Binomial SE at p=0.634, n=1000 is
  ≈ 1.5 pp, so ±2 pp is ~1.3 SE.
- If it lands **materially off** (e.g. < 60% or > 67%), **STOP** — the harness or
  environment differs from the original run (wrong transformers build → audio pathway
  inert; different frames; different edge-tts voice/version). Fix that first. A failed
  gate is itself a finding worth recording, not a number to explain away.

## (ii) Run order if time/budget is short

1. `audio_text`  ← gate (mandatory)
2. `text_only`   ← the confound test; highest information per dollar
3. `mismatched_audio` ← cleanest "is audio read" signal (in-distribution)
4. `audio_only`  ← **deferrable**; interesting for the hands-free story but its negative is ambiguous

Stopping after arm 3 still answers the core question.

## Decision rules (pre-registered)

Define, vs. the `audio_text` gate result G:
- **"≈ G"** = within ±3 pp of G (~2 SE; statistically indistinguishable)
- **"materially below"** = more than **5 pp** below G (well outside noise)
- 3–5 pp = **inconclusive** band; report as such, do not round to a verdict

**text_only** (the confound test):
- `text_only ≈ G` → **audio was decorative.** The model achieves the same score with the
  audio removed; the text shortcut carried it. The "answers spoken questions" claim is
  **not defensible as worded.** (Robust conclusion — removing the supposedly load-bearing
  channel costs nothing.)
- `text_only materially below G` → audio contributed something; interpret alongside the
  other arms (a mechanical drop from breaking the trained layout is a partial confound, so
  this direction is weaker evidence than the equality case).

**mismatched_audio** (iii — the in-distribution control):
- `mismatched ≈ G` → **audio content is ignored.** Correct text + wrong spoken question
  yields the same score, so the model is not reading what the audio says.
- `mismatched materially below G` → **audio content is read and interferes when wrong** —
  evidence the audio channel is genuinely used.

**audio_only** (hands-free; interpret with care):
- `audio_only ≈ G` → **strong, publishable positive** — audio alone is sufficient.
- `audio_only` collapses → **ambiguous**, NOT proof audio is worthless: the model never saw
  an audio-only prompt in training, so this arm carries a train/test distribution shift.

## Interpretation weights (why the arms are not equal)

The checkpoint was trained with audio+text both present. Removing a channel post-hoc is a
distribution shift, and it cuts differently per arm:
- `text_only ≈ G` and `mismatched ≈ G` are **robust** (they show audio is *unused* despite
  being present in-format).
- `audio_only` collapse is **soft** (could be "audio empty" OR "format never seen").

The only way to *cleanly isolate* audio is to **retrain** (audio-only, or with text-dropout).
This ablation is a cheap, honest post-hoc probe — it can convincingly show audio was
*ignored*, but a collapse cannot by itself prove audio is *useless*. The commit message and
script docstring say "probes… post-hoc," not "isolates," for exactly this reason.

## How results feed the claims audit

- The **"question asked out loud" verdict** ([CLAIMS_AUDIT.md](CLAIMS_AUDIT.md)) is held
  `PENDING` on `text_only` + `mismatched_audio`.
- The **"+17 points from fine-tuning"** attribution also resolves here: `text_only` on the
  fine-tuned model vs. the zero-shot **text** baseline (44%, [baseline1](../../baselines/baseline1_text_image.py))
  is the clean fine-tuning delta on a fixed modality. If `text_only ≈ G ≈ 63%`, the "+17–19"
  is largely a real *fine-tuning-on-text* gain (with audio adding ~0), which is honest and
  still a contribution — just not an *audio* result.
