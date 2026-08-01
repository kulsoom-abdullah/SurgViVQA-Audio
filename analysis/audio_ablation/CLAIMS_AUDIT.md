# Claims Audit — Audio, Speed, Hardware

Read-only audit of the public-facing claims, with `file:line` evidence. Two rows that are
pure code-reading — (d) latency and (e) the "+17" baseline — are **closed now**. The
audio-accuracy verdict is held `PENDING` the ablation (see [PRE_REGISTRATION.md](PRE_REGISTRATION.md)).

---

## (d) Latency benchmark — what each arm's prompt contained

The zero-shot latency baselines were **methodologically clean** — each arm delivered the
question through exactly one channel, no double-delivery:

| Baseline | Question delivered as | Latency | Evidence |
| :-- | :-- | :-- | :-- |
| 1 · Text+Image | **text** (`sample['question']`) | ~1,736 ms | [baseline1_text_image.py:70–71](../../baselines/baseline1_text_image.py) → `build_text_vqa_messages` |
| 2 · Audio+Image | **audio only** (fixed instruction, no question text) | ~930 ms | [baseline2_audio_image.py:102](../../baselines/baseline2_audio_image.py) → [utils.py:260](../../baselines/utils.py) `"Answer the question based on the audio and images."` |
| 3 · ASR→Text+Image | **ASR-transcribed text** | ~2,300 ms | [baseline3_asr_pipeline.py:88–98](../../baselines/baseline3_asr_pipeline.py) (Whisper transcribe → `build_text_vqa_messages`) |

**The 2.5× claim** ([README.md:86](../../README.md)): "2.5× faster… by skipping intermediate
transcription (0.9s vs 2.3s)" = **Baseline 2 (930 ms) vs Baseline 3 (2,300 ms)**;
2300/930 = 2.47× ✓ (also 1.07 vs 0.43 samples/s). This is a **fair architectural comparison**:
both answer the same audio question, one by direct audio graft (no ASR), one via a Whisper
ASR pass. The audio arm here is genuinely audio-only, so the latency number is **not** inflated
by the text-shortcut confound that affects the *accuracy* story.

Two honest scope notes to keep with the claim:
1. The benchmarked audio arm is **zero-shot audio-only** (930 ms). The *deployed* fine-tuned
   model runs audio **+ text** (more tokens → marginally slower), so 930 ms is the
   architecture's floor, not the deployed model's measured latency. The ASR pass it avoids
   still dominates, so ~2.5× holds directionally.
2. Speed came with a zero-shot **accuracy** cost (Baseline 2 = 46% vs Baseline 3 = 62%). That
   trade-off is separate from — and should not be quietly folded into — the speed claim.

**Verdict (d): SUPPORTED** as an architectural latency claim, with the two scope notes.

---

## (e) The "+17" zero-shot baseline — what was its input

"+17–19 percentage points via domain-specific training" ([baseline_results.txt:28](../../docs/notes/baseline_results.txt);
[README.md:76](../../README.md) "+17.4 points") = fine-tuned **63.4%** − zero-shot **46%**.

- Zero-shot **audio** baseline (46%): question via **audio only**, fixed generic instruction,
  **no question text** — [utils.py:260](../../baselines/utils.py) + [baseline2_audio_image.py:102](../../baselines/baseline2_audio_image.py).
- Fine-tuned model (63.4%): question via **audio + text** — [evaluate_checkpoint.py:131](../../src/evaluate_checkpoint.py).

So the +17 spans **two simultaneous changes**: (a) fine-tuning on 2,300 samples **and**
(b) adding the question as text to a prompt that previously had none. Part of the jump could be
the text shortcut, not learned audio understanding. There is no zero-shot *audio+text* baseline
in the repo, so the fine-tuning delta cannot be cleanly isolated from the committed numbers.

**Verdict (e): CONFOUNDED as an "audio understanding" gain.** The `text_only` arm resolves it:
`text_only` (fine-tuned, no audio) vs. the zero-shot **text** baseline 44%
([baseline1](../../baselines/baseline1_text_image.py)) is the clean fine-tuning delta on a fixed
modality. If `text_only ≈ 63%`, the "+17–19" is a genuine *fine-tuning-on-text* gain with audio
adding ≈0 — honest, still a contribution, just not an audio result.

---

## Claim → Verdict

| # | Claim (as worded) | Verdict | Basis |
| :-: | :-- | :-- | :-- |
| 1 | **Résumé:** multi-GPU DDP QLoRA fine-tune on 2× RTX 4090 | ✅ **SUPPORTED** | W&B run `4owcddle`: `gpu_count=2`, two 4090 UUIDs, 89% mean util both, 348.7 min → [docs/run_evidence/](../../docs/run_evidence/). Independent: `checkpoint-1000/` has `rng_state_0` **and** `_1`. |
| 2 | **"Model answers questions asked out loud"** — [README.md:2](../../README.md) "hear… audio (no ASR)", [:83](../../README.md) "matching text-based approaches **while using raw audio**", [baseline_results.txt:30](../../docs/notes/baseline_results.txt) "correctly interprets audio questions" | ⏳ **PENDING ablation** | Confound: training + eval + demo fed the question as text **and** audio ([train_vqa.py:168](../../src/train_vqa.py), [evaluate_checkpoint.py:131](../../src/evaluate_checkpoint.py), [app.py:137](../../src/app.py)). **Pre-registered rule:** `text_only ≥ 61.4%` ⇒ audio decorative ⇒ claim **not defensible as worded**; `mismatched_audio` drop ⇒ audio **is** read. |
| 3 | **2.5× faster** by skipping ASR ([README.md:86](../../README.md)) | ✅ **SUPPORTED** (scoped) | Row (d). Fair Baseline 2 vs 3 architectural comparison; note zero-shot audio-only latency and the separate accuracy trade-off. |

**Not yet auditable from the repo:** the verbatim résumé bullet and deck one-liners — I audited
the in-repo equivalents (column "Claim"). If the résumé states an *audio-accuracy* result, that
portion inherits row 2's `PENDING` status; the hardware/DDP portion is row 1 (clean).

## Status

- (d), (e): **closed** (code-reading complete).
- Row 2 / "+17" attribution: **open**, resolves when `text_only` + `mismatched_audio` run
  (gated on `audio_text` reproducing 63.4%). No verdict will be written for row 2 until then.
