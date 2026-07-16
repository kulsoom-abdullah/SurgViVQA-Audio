# Run Evidence — Full QLoRA Fine-Tune Hardware Record

Machine-readable evidence for the training hardware behind the hero checkpoint
(`checkpoint-1000`, 63.4% overall). Exported from Weights & Biases so the hardware
claim in the top-level README does not rest on prose alone.

**Run:** [`surgical-vqa-multivideo-overnight-20260116-055704`](https://wandb.ai/AI_Healthcare/surgical-vqa/runs/4owcddle) (`4owcddle`)

| Field | Value |
| :--- | :--- |
| State | finished |
| Started | 2026-01-16 05:58:02 UTC |
| Runtime | 348.7 min (~5.8 h) |
| Program | `/workspace/SurgViVQA-Audio/src/train_vqa.py` |
| Code commit | [`a999243`](https://github.com/kulsoom-abdullah/SurgViVQA-Audio/commit/a99924330edc1beb691ea59d3cb4ee45f3c60029) |
| GPU | NVIDIA GeForce RTX 4090 × **2** (24 GB each, Ada) |
| CUDA | 12.4 |
| Host / OS | `11045dcf6e2d` (RunPod container) / Linux 6.8.0 |

## Files

| File | Contents |
| :--- | :--- |
| `run_4owcddle_summary.json` | One-glance provenance (run, hardware, commit, runtime) |
| `run_4owcddle_metadata.json` | W&B `wandb-metadata.json`: GPU model, `gpu_count`, per-device UUIDs, CUDA, git commit |
| `run_4owcddle_config.json` | Full 218-key training config (LoRA, QLoRA, batch/accum, DDP flags) |
| `run_4owcddle_system_metrics.json` | Per-GPU utilization / memory / power / temp summary |

## What this establishes

1. **Two physical RTX 4090s.** `gpu_count: 2`, with two distinct device UUIDs
   (`GPU-1cfcb1ca…`, `GPU-99d5dbb5…`) recorded by the W&B agent at run start.
2. **Both were actually working, not merely present.** Mean utilization
   gpu.0 = 89.3%, gpu.1 = 89.7% (symmetry ratio 1.005); each device exceeded 5%
   utilization in 239/255 samples, drawing ~312 W and holding ~17–18 GiB
   (peak ~22–23 GiB) of its 24 GB.
3. **The load pattern is data-parallel.** Near-identical *concurrent* utilization
   across both devices under one training program is the signature of DDP
   replication. Pipeline/model-parallel sharding (`device_map="auto"`, which is
   deliberately disabled at `src/train_vqa.py:270`) instead yields asymmetric or
   alternating per-device load.
4. **It is the full fine-tune.** 348.7 min matches the README's "350 minutes";
   the program is `src/train_vqa.py`; the run began 49 minutes after its code
   commit `a999243` (2026-01-16 00:09 EST → run start 00:58 EST).

## Known limitation

`config.world_size` is **absent** (`null`) from the W&B config. This is a
Hugging Face artifact, not a gap in the run: `TrainingArguments.world_size` is a
`@property`, not a dataclass field, so `to_dict()` never serializes it and no HF
run of this vintage logs it. The device count is therefore evidenced by
`metadata.gpu_count` and the per-GPU system-metrics streams
(`system.gpu.0.*`, `system.gpu.1.*`) rather than by a config key.

Related: W&B logs only rank 0 under `torchrun`, but its system sampler reads the
whole node — which is why both GPUs appear from a single logging process.

## Regenerating

Requires `wandb login` with access to the `AI_Healthcare` entity.

```bash
pip install wandb
wandb login
python3 - <<'PY'
import wandb, json
run = wandb.Api().run("AI_Healthcare/surgical-vqa/4owcddle")
print(json.dumps(run.metadata, indent=2))   # gpu, gpu_count, git commit
PY
```

`metadata.email` is stripped from `run_4owcddle_metadata.json` (PII). Token fields
in the config arrive pre-masked by W&B as `<HUB_TOKEN>` / `<PUSH_TO_HUB_TOKEN>`.
