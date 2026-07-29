"""
Latency benchmark: ASR pipeline (arm A) vs direct audio input (arm B).

Both arms answer the SAME question from the SAME spoken audio using the SAME
fine-tuned VLM weights and the same 8 frames. The only variable is how the
spoken question reaches the model:

  arm A (asr_pipeline) : whisper encode -> full autoregressive decode to text
                         -> "User Question: {transcript}" + frames -> VLM
  arm B (direct_audio) : whisper encoder + projector -> 1500 audio tokens
                         + frames -> VLM   (the deployed path)

Arm A transcribes rather than reading the corpus question. Feeding it the
ground-truth string would hand it an accuracy it has not earned and hide the
cost of the ASR pass, which is the whole thing being measured.

TIMING DECOMPOSITION -- read this before citing the numbers.

Arm A's stages are genuinely serial: ASR finishes, then the VLM runs. So
    t_total ~= t_asr + t_prefill + t_decode.

Arm B has no separate ASR stage. Its audio encode + projection happen INSIDE
the first VLM forward (modeling_qwen2_vl.py:1728-1739), so they are already
part of t_prefill. Timing them as an extra serial stage would mean running the
encoder twice and charging arm B for work it only does once. Instead
t_audio_enc is measured in a separate, untimed probe and reported as a
COMPONENT OF t_prefill, never added to it. Hence for arm B:
    t_total ~= t_prefill + t_decode,  with t_audio_enc <= t_prefill.

Usage:
    python src/bench/latency_bench.py --n_per_type 5 --out results/bench
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList,
    WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor,
)
import librosa

AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"
WHISPER_ID = "openai/whisper-large-v3-turbo"
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500
MAX_NEW_TOKENS = 32
WARMUP_ITERS = 5
SEED = 20260728

# Byte-identical to evaluate_checkpoint.py:306 / :309
PROMPT_WITH_QUESTION = "User Question: {q}\nAnswer the question concisely based on the visual and audio evidence."
PROMPT_AUDIO_ONLY = "Answer the question concisely based on the visual and audio evidence."


def k2_key(row):
    return f"{row['video_id']}_{row['id']}_{row['question_type']}"


def load_frames(frame_names, frames_dir, max_size=384):
    """Identical resolution to evaluate_checkpoint.py load_frames()."""
    images = []
    for frame_name in frame_names:
        vid_id = frame_name.rsplit('_', 1)[0] if '_' in frame_name else "unknown"
        path = Path(frames_dir) / vid_id / f"{frame_name}.jpg"
        if not path.exists():
            frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
            path = Path(frames_dir) / vid_id / f"{frame_num}.jpg"
        if path.exists():
            img = Image.open(path).convert("RGB")
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
            images.append(img)
        else:
            raise FileNotFoundError(f"frame not resolvable: {frame_name}")
    return images


class FirstTokenTimer(StoppingCriteria):
    """Records the wall time at which the first token is available.

    Synchronizes once, on the first call only -- enough to make the prefill/decode
    split real rather than an artifact of CUDA's async queue, without serializing
    every decode step.
    """

    def __init__(self):
        self.first_token_time = None

    def reset(self):
        self.first_token_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()
        return False


def sync():
    torch.cuda.synchronize()


def build_sample(eval_data, n_per_type, seed):
    """Stratified: n_per_type rows from each question_type, fixed seed."""
    by_type = {}
    for i, row in enumerate(eval_data):
        by_type.setdefault(row["question_type"], []).append(i)
    rng = random.Random(seed)
    chosen = []
    for qtype in sorted(by_type):
        idxs = sorted(by_type[qtype])
        if len(idxs) < n_per_type:
            raise RuntimeError(f"{qtype} has only {len(idxs)} rows, need {n_per_type}")
        chosen.extend(rng.sample(idxs, n_per_type))
    chosen.sort()
    return chosen


class Bench:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"

        print("Loading VLM (4-bit) + cell3 adapter ...", flush=True)
        tok = AutoTokenizer.from_pretrained(AUDIO_ADAPTED_MODEL_ID, trust_remote_code=True, use_fast=False)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            AUDIO_ADAPTED_MODEL_ID, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa", trust_remote_code=True)
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base, args.checkpoint_path)
        self.model.eval()

        proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)
        if tok.pad_token_id == tok.eos_token_id or tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        proc.tokenizer = tok
        self.tokenizer, self.processor = tok, proc
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)

        print("Loading Whisper for arm A ...", flush=True)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_ID, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.whisper_proc = WhisperProcessor.from_pretrained(WHISPER_ID)

        self.stopper = FirstTokenTimer()
        self.criteria = StoppingCriteriaList([self.stopper])

    # ---------- shared pieces ----------

    def audio_features(self, path):
        y, _ = librosa.load(path, sr=16000, mono=True)
        f = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        return f.input_features.to(self.device).to(torch.bfloat16)

    def vlm_batch(self, images, prompt_text):
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=images, return_tensors="pt")

    def _generate(self, input_ids, batch, input_features):
        """Run generation, returning (text, n_in, n_out, t_prefill, t_decode)."""
        attn = torch.ones_like(input_ids)
        self.stopper.reset()
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                input_features=input_features,
                pixel_values=batch.pixel_values.to(self.device).to(torch.bfloat16),
                image_grid_thw=batch.image_grid_thw.to(self.device),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.criteria)
        sync()
        t1 = time.perf_counter()
        first = self.stopper.first_token_time or t1
        n_in = input_ids.shape[1]
        n_out = out.shape[1] - n_in
        text = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return text, n_in, n_out, first - t0, t1 - first

    # ---------- arm A ----------

    def arm_a(self, row, images):
        feats = self.audio_features(Path(self.args.audio_dir) / f"{k2_key(row)}.mp3")
        sync()
        t_start = time.perf_counter()

        # --- ASR: encode + full autoregressive decode ---
        sync()
        a0 = time.perf_counter()
        with torch.no_grad():
            ids = self.whisper.generate(feats, max_new_tokens=64, do_sample=False,
                                        language="en", task="transcribe")
        sync()
        t_asr = time.perf_counter() - a0
        transcript = self.whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        # --- VLM on the TRANSCRIBED text (not the corpus question) ---
        batch = self.vlm_batch(images, PROMPT_WITH_QUESTION.format(q=transcript))
        input_ids = batch.input_ids.to(self.device)
        text, n_in, n_out, t_pre, t_dec = self._generate(input_ids, batch, None)

        sync()
        t_total = time.perf_counter() - t_start
        return dict(arm="A_asr_pipeline", answer=text, transcript=transcript,
                    t_asr=t_asr, t_audio_enc=None, t_prefill=t_pre, t_decode=t_dec,
                    t_total=t_total, n_input_tokens=n_in, n_generated_tokens=n_out)

    # ---------- arm B ----------

    def arm_b(self, row, images):
        feats = self.audio_features(Path(self.args.audio_dir) / f"{k2_key(row)}.mp3")

        batch = self.vlm_batch(images, PROMPT_AUDIO_ONLY)
        audio_tokens = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=self.device)
        hdr = self.tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False,
                                    return_tensors="pt").to(self.device)
        ftr = self.tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False,
                                    return_tensors="pt").to(self.device)
        pre_len = len(self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
        vision = batch.input_ids[:, pre_len:].to(self.device)
        input_ids = torch.cat([hdr, audio_tokens, ftr, vision], dim=1)

        sync()
        t_start = time.perf_counter()
        text, n_in, n_out, t_pre, t_dec = self._generate(input_ids, batch, feats)
        sync()
        t_total = time.perf_counter() - t_start

        # Probe only: encoder+projector cost, ALREADY inside t_prefill above.
        # Measured after the timed region so it never inflates t_total.
        inner = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        t_audio_enc = None
        if hasattr(inner, "audio_encoder"):
            sync()
            e0 = time.perf_counter()
            with torch.no_grad():
                h = inner.audio_encoder(feats.to(inner.audio_projector.weight.dtype)).last_hidden_state
                inner.audio_projector(h)
            sync()
            t_audio_enc = time.perf_counter() - e0

        return dict(arm="B_direct_audio", answer=text, transcript=None,
                    t_asr=None, t_audio_enc=t_audio_enc, t_prefill=t_pre, t_decode=t_dec,
                    t_total=t_total, n_input_tokens=n_in, n_generated_tokens=n_out)

    # ---------- driver ----------

    def run(self):
        eval_data = [json.loads(l) for l in open(self.args.eval_data_path)]
        idxs = build_sample(eval_data, self.args.n_per_type, SEED)
        if self.args.limit:
            idxs = idxs[:self.args.limit]
        print(f"sampled {len(idxs)} rows (seed={SEED}, {self.args.n_per_type}/type)", flush=True)

        warm_row = eval_data[idxs[0]]
        warm_imgs = load_frames(warm_row["frames"], self.args.frames_dir)
        print(f"warmup: {WARMUP_ITERS} iters per arm ...", flush=True)
        for _ in range(WARMUP_ITERS):
            self.arm_a(warm_row, warm_imgs)
            self.arm_b(warm_row, warm_imgs)

        records = []
        for n, i in enumerate(idxs, 1):
            row = eval_data[i]
            images = load_frames(row["frames"], self.args.frames_dir)
            for fn in (self.arm_a, self.arm_b):      # INTERLEAVED A/B per sample
                rec = fn(row, images)
                rec.update(row_index=i, row_id=row["id"], k2_key=k2_key(row),
                           question_type=row["question_type"], question=row["question"],
                           short_answer=row["short_answer"])
                records.append(rec)
            if n % 10 == 0 or n == len(idxs):
                print(f"  {n}/{len(idxs)}", flush=True)
        return idxs, records


def summarize(records, idxs, args):
    def med_iqr(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [min(vals)] * 3
        return dict(median=statistics.median(vals), q1=q[0], q3=q[2],
                    iqr=q[2] - q[0], n=len(vals))

    out = {}
    for arm in ("A_asr_pipeline", "B_direct_audio"):
        rs = [r for r in records if r["arm"] == arm]
        out[arm] = {k: med_iqr([r[k] for r in rs])
                    for k in ("t_asr", "t_audio_enc", "t_prefill", "t_decode", "t_total")}
        # Raw t_decode mixes output length with per-step cost: an arm that emits
        # more tokens looks slower even at identical per-token speed. Normalizing
        # isolates the architectural difference.
        out[arm]["t_decode_per_token"] = med_iqr(
            [r["t_decode"] / r["n_generated_tokens"] for r in rs if r["n_generated_tokens"] > 0])
        out[arm]["n_input_tokens"] = med_iqr([r["n_input_tokens"] for r in rs])
        out[arm]["n_generated_tokens"] = med_iqr([r["n_generated_tokens"] for r in rs])
        out[arm]["n_samples"] = len(rs)
        out[arm]["empty_answers"] = sum(1 for r in rs if not r["answer"].strip())
        out[arm]["zero_token_generations"] = sum(1 for r in rs if r["n_generated_tokens"] == 0)

    a = out["A_asr_pipeline"]["t_total"]["median"]
    b = out["B_direct_audio"]["t_total"]["median"]
    out["speedup_A_over_B"] = a / b if b else None

    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()
    arm_a = [r for r in records if r["arm"] == "A_asr_pipeline"]
    matches = sum(1 for r in arm_a if norm(r["transcript"] or "") == norm(r["question"]))
    out["asr_exact_match_rate"] = matches / len(arm_a) if arm_a else None
    out["asr_exact_matches"] = matches

    out["environment"] = dict(
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        transformers=__import__("transformers").__version__,
        transformers_path=__import__("transformers").__file__,
        dtype="bfloat16", quantization="4-bit nf4 double-quant (VLM), bf16 (Whisper)",
        max_new_tokens=MAX_NEW_TOKENS, warmup_iters=WARMUP_ITERS, seed=SEED,
        n_per_type=args.n_per_type, checkpoint_path=args.checkpoint_path,
        whisper=WHISPER_ID, sampled_row_indices=idxs,
    )
    out["timing_note"] = (
        "Arm A stages are serial: t_total ~= t_asr + t_prefill + t_decode. "
        "Arm B has no separate ASR stage; its audio encode+projection occur inside "
        "the first VLM forward, so t_audio_enc is a COMPONENT of t_prefill (probe "
        "measured outside the timed region) and must not be added to it."
    )
    return out


def main():
    p = argparse.ArgumentParser(description="ASR-pipeline vs direct-audio latency benchmark")
    p.add_argument("--checkpoint_path", default="./checkpoints/cell3_audio_only")
    p.add_argument("--eval_data_path", default="data/test_multivideo.jsonl")
    p.add_argument("--frames_dir", default="data/frames")
    p.add_argument("--audio_dir", default="data/audio/test")
    p.add_argument("--n_per_type", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="cap total samples (smoke)")
    p.add_argument("--out", default="results/bench")
    args = p.parse_args()

    bench = Bench(args)
    idxs, records = bench.run()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "latency.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, idxs, args)
    (outdir / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    A, B = summary["A_asr_pipeline"], summary["B_direct_audio"]
    print("\n" + "=" * 78)
    print(f"{'stage':<16}{'arm A (ASR)':>18}{'arm B (direct)':>18}{'delta':>16}")
    print("-" * 78)
    for k in ("t_asr", "t_audio_enc", "t_prefill", "t_decode", "t_decode_per_token", "t_total"):
        av, bv = A[k], B[k]
        a_s = f"{av['median']*1000:.1f} ms" if av else "—"
        b_s = f"{bv['median']*1000:.1f} ms" if bv else "—"
        d_s = f"{(av['median']-bv['median'])*1000:+.1f} ms" if av and bv else "—"
        print(f"{k:<20}{a_s:>16}{b_s:>16}{d_s:>16}")
    print("-" * 78)

    def iqr_ms(d):
        return f"{d['q1']*1000:.1f}-{d['q3']*1000:.1f} ms"

    def iqr_n(d):
        return f"{d['q1']:.0f}-{d['q3']:.0f}"

    for k in ("t_decode_per_token", "t_total"):
        print(f"{k + ' IQR':<20}{iqr_ms(A[k]):>16}{iqr_ms(B[k]):>16}")
    print("-" * 78)
    print(f"{'input tokens':<20}{A['n_input_tokens']['median']:>16.0f}{B['n_input_tokens']['median']:>16.0f}")
    print(f"{'gen tokens (median)':<20}{A['n_generated_tokens']['median']:>16.0f}{B['n_generated_tokens']['median']:>16.0f}")
    print(f"{'gen tokens IQR':<20}{iqr_n(A['n_generated_tokens']):>16}{iqr_n(B['n_generated_tokens']):>16}")
    print("=" * 78)
    print(f"speedup (A/B on t_total): {summary['speedup_A_over_B']:.2f}x   n={A['n_samples']}")
    print(f"ASR exact-match rate: {summary['asr_exact_match_rate']*100:.1f}% "
          f"({summary['asr_exact_matches']}/{A['n_samples']})")
    print(f"empty answers: A={A['empty_answers']}  B={B['empty_answers']}")


if __name__ == "__main__":
    main()
