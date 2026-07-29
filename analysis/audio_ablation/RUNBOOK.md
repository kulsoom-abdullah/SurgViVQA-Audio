# Audio-Channel Ablation — RunPod Runbook

Runs the four `--input_mode` arms of [src/evaluate_checkpoint.py](../../src/evaluate_checkpoint.py)
on the committed hero checkpoint. Read [PRE_REGISTRATION.md](PRE_REGISTRATION.md) first — the
`audio_text` sanity gate must pass before any other arm is interpretable.

## ⚠️ The one thing that will silently ruin the run

The audio pathway (Whisper encoder → projector → scatter into the 1500 audio-token
positions) lives **only in the bundled forked transformers**
([transformers_fork/…/modeling_qwen2_vl.py:1727–1753](../../transformers_fork/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)).
`requirements.txt` pins **stock** `transformers>=4.47.0`, which has no audio pathway — with
it, `input_features` is silently ignored and every "audio" arm is mechanically inert while
still producing plausible numbers.

**You must editable-install the fork after requirements**, exactly as
[scripts/setup_runpod_venv.sh:105](../../scripts/setup_runpod_venv.sh) does:

```bash
pip install -r requirements.txt
cd transformers_fork && pip install -e . && cd ..   # supersedes the stock wheel — MANDATORY
```

Liveness check before spending money — with the fork, injecting 1500 audio tokens but no
features raises a token/feature-count `ValueError`
([modeling_qwen2_vl.py:1744–1749](../../transformers_fork/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)).
The ablation also detects an inert pathway behaviorally: if `mismatched_audio ≈ audio_text`
**and** `audio_only` collapses, the audio is being ignored — by the model *or* by a missing
fork. Confirm the fork import resolves:

```bash
python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)"
# must point at .../transformers_fork/src/transformers, NOT a site-packages wheel
```

## Files that must be on the pod

| Item | Source | Notes |
| :-- | :-- | :-- |
| Repo (code **+ `transformers_fork/`**) | git clone | fork ships in-repo |
| Base model `kulsoom-abdullah/Qwen2-Audio-7B-Transcription` | HF, auto-downloads | ~16 GB; grafted Whisper base |
| Adapter `checkpoint-1000` | HF `kulsoom-abdullah/surgvivqa-qwen7b-audio` | `--checkpoint_path` points here |
| Test set `data/test_multivideo.jsonl` | in repo | 1000 samples, video 002-004 |
| Frames `data/frames/002-004/…` | Figshare / tarball (gitignored) | see `scripts/download_frames_figshare.sh` |
| Test audio `*.mp3` | **regenerate, see below** | needed by every arm except `text_only` |

### edge-tts audio regeneration

The TTS `.mp3`s are gitignored and not redistributed, so regenerate them from the question
text on the pod (no dataset re-download needed):

```bash
python scripts/generate_all_audio.py   # (or scripts/generate_audio_multivideo.sh)
```

- **Match the original voice/settings.** The `audio_text` gate reproducing 63.4% depends on
  the spoken audio being close to the original TTS; a different edge-tts voice/version shifts
  the Whisper features and can move borderline predictions. If the gate misses, suspect the
  voice first.
- `mismatched_audio` needs **no extra audio** — it reuses the same test-set `.mp3` pool,
  just paired differently (below).

### mismatched-audio pairing (automatic)

Handled in-script by `build_mismatch_map()` — deterministic, no manual step. Each sample is
paired with the next sample (with wraparound) whose `short_answer` **differs**, so the spoken
question always asks something with a different answer than the on-screen text question. The
chosen source is written to every result row as `audio_source_id` for audit. For a stronger
cross-question-type mismatch, edit `build_mismatch_map()` to also require a different
`question_type`.

## Commands (run order per pre-registration)

```bash
CKPT=./checkpoint-1000            # local path to the downloaded adapter
COMMON="--checkpoint_path $CKPT --eval_data_path data/test_multivideo.jsonl \
        --frames_dir data/frames --audio_dir data/audio --max_image_size 384"

# 1. GATE — must reproduce ~63.4% (>=61.4%) or STOP
python src/evaluate_checkpoint.py $COMMON --input_mode audio_text \
       --output_file analysis/audio_ablation/results/arm_audio_text.jsonl

# 2. confound test
python src/evaluate_checkpoint.py $COMMON --input_mode text_only \
       --output_file analysis/audio_ablation/results/arm_text_only.jsonl

# 3. is the audio content read? (in-distribution control)
python src/evaluate_checkpoint.py $COMMON --input_mode mismatched_audio \
       --output_file analysis/audio_ablation/results/arm_mismatched_audio.jsonl

# 4. deferrable — hands-free
python src/evaluate_checkpoint.py $COMMON --input_mode audio_only \
       --output_file analysis/audio_ablation/results/arm_audio_only.jsonl
```

Each run prints overall accuracy + a per-question-type breakdown and writes one JSONL row
per sample (`input_mode`, `audio_source_id`, `predicted_answer`, `exact_match`).

## Runtime & cost (estimates, not measured)

Basis: ~1.2 s/query on 1× RTX 4090 ([README.md:287](../../README.md)), 1000 samples, batch=1,
4-bit base.

| GPU | Model load | Per arm (1000) | 3 arms (gate+2) | 4 arms |
| :-- | :-- | :-- | :-- | :-- |
| RTX 4090 (24 GB) | ~3–5 min | ~25–30 min | ~1.5 h | ~2 h |
| A6000 (48 GB) | ~3–5 min | ~30–40 min | ~2 h | ~2.5 h |

Cost is rough — **check current RunPod rates**; community 4090 has run ≈ $0.34–0.44/hr and
A6000 ≈ $0.49–0.79/hr, so a full 4-arm sweep is roughly **$1–2**. 24 GB is sufficient (4-bit
7B + Whisper encoder); the A6000's headroom mainly buys safety, not speed.

> Fine print in the script header: this is a **post-hoc probe**, not a clean isolation — the
> checkpoint was trained with audio+text both present. See PRE_REGISTRATION.md for how to
> read each arm.
