"""
Capacity control: does a larger / newer stock VLM close the visual-grounding gap?

Audio is NOT involved. Every arm receives the question as TEXT, so model capacity
is the only variable. If all four stock models sit at the prior on the same
question types the fine-tuned model does, the limitation is the sampled frames
not containing the answer -- not capacity.

Arms, smallest to largest:
    Qwen/Qwen2-VL-7B-Instruct      Qwen2VLForConditionalGeneration
    Qwen/Qwen2.5-VL-7B-Instruct    Qwen2_5_VLForConditionalGeneration
    Qwen/Qwen3-VL-8B-Instruct      Qwen3VLForConditionalGeneration
    Qwen/Qwen3-VL-32B-Instruct     Qwen3VLForConditionalGeneration

All zero-shot, no fine-tuning, 4-bit NF4 with bf16 compute so the four are
comparable to each other and to the cell-1 baseline.

Usage:
    python src/bench/capacity_control.py --model qwen2-vl-7b --limit 20
    python src/bench/capacity_control.py --model qwen2-vl-7b
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

MAX_NEW_TOKENS = 128
# Greedy decoding is prefix-deterministic, so the first 32 generated token IDs of a
# 128-token generation are bit-identical to what max_new_tokens=32 would have
# produced. Slicing the IDs (rather than decoding then re-encoding, which can shift
# token boundaries) reproduces the cell-1 budget exactly.
TRUNC_TOKENS = 32
MAX_IMAGE_SIZE = 384

# Byte-identical to evaluate_checkpoint.py:306. "and audio evidence" is retained
# even though no audio is present in this experiment: holding the wording constant
# is what makes these numbers comparable to the cell-1 text_only baseline. This is
# deliberate, not a copy-paste bug.
PROMPT = "User Question: {q}\nAnswer the question concisely based on the visual and audio evidence."

MODELS = {
    "qwen2-vl-7b":   dict(repo="Qwen/Qwen2-VL-7B-Instruct",   cls="Qwen2VLForConditionalGeneration"),
    "qwen2.5-vl-7b": dict(repo="Qwen/Qwen2.5-VL-7B-Instruct", cls="Qwen2_5_VLForConditionalGeneration"),
    "qwen3-vl-8b":   dict(repo="Qwen/Qwen3-VL-8B-Instruct",   cls="Qwen3VLForConditionalGeneration"),
    "qwen3-vl-32b":  dict(repo="Qwen/Qwen3-VL-32B-Instruct",  cls="Qwen3VLForConditionalGeneration"),
}

# Carve-out: the gold token appears only as a morphological variant in fluent
# answers ("completely obscured" for gold "complete"). Documented so the rule is
# auditable rather than an unexplained special case.
CARVE_OUT = {"complete": ["completely"], "down": ["downward"]}
GRADING_RULE = (
    r"normalized whitespace + lowercase, then regex (?<![\w-])GOLD(?![\w-]); "
    r"plus carve-out crediting complete->completely and down->downward. "
    r"substring_correct (gold in pred) also recorded for parity with cell 1/3."
)


def grade_strict(gold: str, pred: str) -> bool:
    g = gold.strip().lower()
    p = " ".join(pred.lower().split())
    if re.search(r"(?<![\w-])" + re.escape(g) + r"(?![\w-])", p):
        return True
    return any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", p)
               for a in CARVE_OUT.get(g, []))


def grade_substring(gold: str, pred: str) -> bool:
    return gold.lower() in pred.lower()


def load_frames(frame_names, frames_dir, max_size=MAX_IMAGE_SIZE):
    """Copied verbatim from evaluate_checkpoint.py load_frames() so frame sampling
    and resize are identical -- a different rule here would change what the models
    see and break comparability with cell 1."""
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
            # Fail loudly. A black placeholder would silently turn "model cannot
            # see the answer" into "model was shown nothing".
            raise FileNotFoundError(f"frame not resolvable: {frame_name}")
    return images


def resolve_class(name):
    import transformers
    cls = getattr(transformers, name, None)
    if cls is not None:
        return cls, name
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText, f"AutoModelForImageTextToText (fallback, {name} absent)"


def repo_revision(repo):
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha
    except Exception as e:
        return f"UNRESOLVED: {type(e).__name__}: {e}"


def run(key, args):
    import transformers
    spec = MODELS[key]
    repo, want_cls = spec["repo"], spec["cls"]

    model_cls, cls_used = resolve_class(want_cls)
    print(f"=== {repo} ===", flush=True)
    print(f"  class: {cls_used}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = model_cls.from_pretrained(repo, quantization_config=bnb, device_map={"": 0})
    model.eval()
    processor = AutoProcessor.from_pretrained(repo)

    eval_data = [json.loads(l) for l in open(args.eval_data_path)]
    if args.limit:
        eval_data = eval_data[:args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{key}.jsonl"

    # Data-derived: question types whose gold answer is constant across the whole
    # test set cannot discriminate a model from a constant emitter. 7 such types
    # (350 rows) exist; the remaining 13 types (650 rows) are the real signal.
    all_rows = [json.loads(l) for l in open(args.eval_data_path)]
    golds_by_type = {}
    for r in all_rows:
        golds_by_type.setdefault(r["question_type"], set()).add(r["short_answer"])
    discriminative = {k for k, v in golds_by_type.items() if len(v) > 1}

    results, n_correct, n_sub = [], 0, 0
    for i, sample in enumerate(eval_data, 1):
        images = load_frames(sample["frames"], args.frames_dir)
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": PROMPT.format(q=sample["question"])})
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        n_in = batch["input_ids"].shape[1]
        gen_ids = out[0][n_in:]
        n_gen = gen_ids.shape[0]
        tok = processor.tokenizer
        pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
        pred32 = tok.decode(gen_ids[:TRUNC_TOKENS], skip_special_tokens=True).strip()

        strict = grade_strict(sample["short_answer"], pred)
        strict32 = grade_strict(sample["short_answer"], pred32)
        sub = grade_substring(sample["short_answer"], pred)
        n_correct += strict
        n_sub += sub
        results.append(dict(
            question_id=sample["id"], question_type=sample["question_type"],
            question=sample["question"], short_answer=sample["short_answer"],
            predicted_answer=pred, predicted_answer_at32=pred32,
            correct=int(strict), correct_at32=int(strict32),
            substring_correct=int(sub),
            substring_correct_at32=int(grade_substring(sample["short_answer"], pred32)),
            n_input_tokens=n_in, n_generated_tokens=n_gen,
            hit_cap=int(n_gen >= MAX_NEW_TOKENS),
            discriminative=int(sample["question_type"] in discriminative),
            model=repo))
        if i % 50 == 0 or i == len(eval_data):
            print(f"  {i}/{len(eval_data)}  acc@128={n_correct/i*100:.1f}%", flush=True)

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    import statistics
    disc = [r for r in results if r["discriminative"]]
    pct = lambda rs, k: (sum(r[k] for r in rs) / len(rs) * 100) if rs else None
    gen_counts = [r["n_generated_tokens"] for r in results]

    manifest = dict(
        model_id=repo, revision_sha=repo_revision(repo), model_class=cls_used,
        quantization="4-bit nf4, double-quant, bf16 compute",
        prompt=PROMPT,
        prompt_note=("byte-identical to evaluate_checkpoint.py:306, including "
                     "'and audio evidence' though no audio is present -- constant "
                     "wording is required for comparability with cell 1"),
        grading_rule=GRADING_RULE,
        max_new_tokens=MAX_NEW_TOKENS, trunc_tokens=TRUNC_TOKENS,
        trunc_note=("acc@32 slices the first 32 generated token IDs of the same "
                    "greedy generation; prefix-determinism makes this identical to "
                    "running with max_new_tokens=32, and comparable to cell 1/3"),
        decoding="greedy (do_sample=False)", batch_size=1,
        max_image_size=MAX_IMAGE_SIZE, frames_per_sample=8,
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__, transformers=transformers.__version__,
        n_rows=len(results),
        accuracy_at128=pct(results, "correct"),
        accuracy_at32=pct(results, "correct_at32"),
        accuracy_substring_at128=pct(results, "substring_correct"),
        n_discriminative_rows=len(disc),
        accuracy_at128_discriminative=pct(disc, "correct"),
        accuracy_at32_discriminative=pct(disc, "correct_at32"),
        discriminative_types=sorted(discriminative),
        degenerate_types=sorted(set(golds_by_type) - discriminative),
        median_generated_tokens=statistics.median(gen_counts),
        max_generated_tokens=max(gen_counts),
        n_hit_cap=sum(r["hit_cap"] for r in results),
        utc_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    (outdir / f"{key}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    m = manifest
    print(f"  ALL-1000   acc@32={m['accuracy_at32']:.2f}%  acc@128={m['accuracy_at128']:.2f}%")
    print(f"  DISC-{m['n_discriminative_rows']}    acc@32={m['accuracy_at32_discriminative']:.2f}%  "
          f"acc@128={m['accuracy_at128_discriminative']:.2f}%")
    print(f"  gen tokens median={m['median_generated_tokens']:.0f} max={m['max_generated_tokens']}  "
          f"hit 128-cap: {m['n_hit_cap']}/{m['n_rows']}")
    print(f"  wrote {out_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Capacity control across stock VLMs")
    p.add_argument("--model", required=True, choices=sorted(MODELS) + ["all"])
    p.add_argument("--eval_data_path", default="data/test_multivideo.jsonl")
    p.add_argument("--frames_dir", default="data/frames")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/capacity")
    args = p.parse_args()

    keys = ["qwen2-vl-7b", "qwen2.5-vl-7b", "qwen3-vl-8b", "qwen3-vl-32b"] \
        if args.model == "all" else [args.model]
    for k in keys:
        run(k, args)


if __name__ == "__main__":
    main()
