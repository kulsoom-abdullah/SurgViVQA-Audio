"""
Standalone evaluation script for surgical VQA model
Runs after training completes, with memory-efficient settings

Usage:
python evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/surgical_vqa_50samples \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir dataset/frames \
    --audio_dir audio/out_002-001 \
    --output_file results/eval_results.jsonl \
    --batch_size 1 \
    --input_mode audio_text

--input_mode selects the ablation arm. All arms share this exact harness
(loader, greedy decode, substring scorer); only the prompt content changes,
so the arms are provably identical except the intended variable.

  audio_text       (default) reproduces training/deployment: 1500 spoken-question
                   audio tokens + the question as text. Use this as the sanity gate.
  text_only        question as text, NO audio tokens / features (standard Qwen2-VL layout).
  audio_only       question via audio only; question text removed from the prompt.
  mismatched_audio correct question text, but the audio is a DIFFERENT sample's
                   spoken question (in-distribution control: if the model reads audio,
                   the wrong audio should pull the answer off).

IMPORTANT (post-hoc probe, not a clean isolation): the checkpoint was TRAINED with
audio+text both present. text_only / audio_only therefore carry a train/test prompt
mismatch and their DROPS are ambiguous; mismatched_audio stays in-distribution and is
the cleanest test of whether the audio content is read at all. Audio is only live when
the FORKED transformers is installed (see the ablation runbook); with stock transformers
the audio pathway is silently inert.
"""

import warnings
# Suppress "copying from non-meta parameter" warnings during checkpoint loading
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*meta parameter.*", category=UserWarning)

import torch
import json
import argparse
import sys
import subprocess
import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, WhisperFeatureExtractor
from PIL import Image
import librosa

# Using your merged Stage 1 + Stage 2 checkpoint from HuggingFace
AUDIO_ADAPTED_MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

_REPO_DIR = Path(__file__).resolve().parent

def _git(*cmd):
    """Run a git command against THIS script's repo, not the process CWD.

    On failure returns the literal error string, never None: an absent .git, an
    unborn HEAD, or a missing git binary must be visible in the manifest rather
    than silently blank.
    """
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=_REPO_DIR, stderr=subprocess.STDOUT
        ).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"GIT_ERROR({e.returncode}): {e.output.decode().strip()}"
    except Exception as e:
        return f"GIT_ERROR: {type(e).__name__}: {e}"

def _git_sha():
    """Current commit for run provenance."""
    return _git("rev-parse", "HEAD")

def _git_ref():
    """Branch name, or the literal 'HEAD' when detached — that IS the detachment
    signal, since `rev-parse HEAD` returns an ordinary sha either way."""
    return _git("rev-parse", "--abbrev-ref", "HEAD")

def _git_dirty():
    """True when the working tree differs from HEAD (untracked files count).

    A run of uncommitted code recording a clean-looking sha is the provenance
    failure the manifest exists to prevent.
    """
    out = _git("status", "--porcelain")
    return out if out.startswith("GIT_ERROR") else bool(out)

def k2_key(row):
    """Audio filename key. The bare `id` is NOT unique — it repeats across videos
    and splits with different questions attached, so a flat {id}.mp3 lookup can
    silently return a different question's audio. K2 is collision-free across all
    3700 corpus rows."""
    return f"{row['video_id']}_{row['id']}_{row['question_type']}"

def load_frames(frame_names, frames_dir, max_size=384):
    """Load and resize frames"""
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
            images.append(Image.new('RGB', (224, 224), color='black'))

    return images

def build_mismatch_map(eval_data):
    """Deterministically pair each sample with a DIFFERENT sample whose short_answer
    differs, so the spoken (audio) question conflicts with the correct text question.

    Returns a list `m` where eval_data[m[i]] supplies the audio for sample i. Pairing
    scans forward with wraparound to the first differing short_answer, so the map is
    reproducible run-to-run and every sample's audio asks a question with a different
    answer than its own. Written to each result row as audio_source_id for auditability.
    """
    n = len(eval_data)
    mapping = []
    for i in range(n):
        j = (i + 1) % n
        steps = 0
        while (eval_data[j].get('short_answer', '').lower()
               == eval_data[i].get('short_answer', '').lower()) and steps < n:
            j = (j + 1) % n
            steps += 1
        mapping.append(j)
    return mapping

def evaluate(args):
    print("="*80)
    print("SURGICAL VQA EVALUATION")
    print("="*80)

    # ------------------------------------------------------------------
    # Pre-flight (runs BEFORE model load): load eval data, resolve the
    # audio files this arm requires, and write a run manifest so an
    # aborted run still records what was attempted. A silence run (missing
    # audio) must never be mistakable for a real audio measurement, hence
    # the coverage gate below.
    # ------------------------------------------------------------------
    print(f"📁 Loading eval data: {args.eval_data_path}")
    eval_data = [json.loads(line) for line in open(args.eval_data_path)]
    print(f"✓ Loaded {len(eval_data)} samples")

    # Ablation arm: precompute the audio-source pairing only when needed
    mismatch_map = build_mismatch_map(eval_data) if args.input_mode == "mismatched_audio" else None

    use_audio = args.input_mode in ("audio_text", "audio_only", "mismatched_audio")
    if use_audio and args.input_mode == "mismatched_audio":
        # mismatched_audio speaks a DIFFERENT sample's question, so the required
        # files are the paired sources, not the sample ids themselves.
        required_keys = [k2_key(eval_data[mismatch_map[i]]) for i in range(len(eval_data))]
    elif use_audio:
        required_keys = [k2_key(s) for s in eval_data]
    else:
        required_keys = []

    required = len(required_keys)
    missing_paths = [str(Path(args.audio_dir) / f"{key}.mp3") for key in required_keys
                     if not (Path(args.audio_dir) / f"{key}.mp3").exists()]
    found = required - len(missing_paths)
    coverage_fraction = (found / required) if required else None

    if use_audio:
        print(f"AUDIO COVERAGE: {found}/{required} found in {args.audio_dir}")

    # Run manifest — written before model load; total/correct/accuracy filled after.
    manifest = {
        "argv": sys.argv,
        "audio_dir": str(Path(args.audio_dir).resolve()),
        "eval_data_path": str(Path(args.eval_data_path).resolve()),
        "audio_required": required,
        "audio_found": found if use_audio else 0,
        "coverage_fraction": coverage_fraction,
        "input_mode": args.input_mode,
        "allow_missing_audio": args.allow_missing_audio,
        "checkpoint_path": args.checkpoint_path,
        "key_scheme": "K2",
        "adapter_loaded": not args.no_adapter,
        "git_sha": _git_sha(),
        "git_ref": _git_ref(),
        "git_dirty": _git_dirty(),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total": None,
        "correct": None,
        "accuracy": None,
    }
    manifest_path = Path(str(args.output_file) + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)

    print("RUN MANIFEST:")
    for k, v in manifest.items():
        print(f"  {k}: {v}")

    if use_audio and missing_paths:
        print(f"  {min(5, len(missing_paths))} of {len(missing_paths)} required audio files missing (first 5):")
        for m in missing_paths[:5]:
            print(f"    MISSING: {m}")
        if not args.allow_missing_audio:
            print("❌ Refusing to run: required audio files are missing and "
                  "--allow_missing_audio was not set. A silent run is not a valid "
                  "audio measurement.")
            sys.exit(1)

    # Load base model + LoRA adapters
    print(f"\n⏳ Loading base model and adapters from: {args.checkpoint_path}")

    # Load tokenizer/processor from base model (not from PEFT checkpoint)
    print(f"Loading tokenizer from base: {AUDIO_ADAPTED_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(AUDIO_ADAPTED_MODEL_ID, trust_remote_code=True, use_fast=False)

    # Load base model (4-bit quantized)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        AUDIO_ADAPTED_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    # Load LoRA adapters — skipped for the un-fine-tuned base (baseline cells 1 and 2)
    if args.no_adapter:
        print("⚠️  --no_adapter: using the audio-adapted BASE model, no LoRA adapters")
        model = base_model
    else:
        from peft import PeftModel
        print(f"Loading LoRA adapters from: {args.checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    model.eval()

    # Load processor from Qwen2-VL base
    print("Loading processor from Qwen2-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)
    processor.tokenizer = tokenizer

    # Fix PAD/EOS if needed
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.pad_token_id is None:
        print("🔧 Setting dedicated pad token for generation")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        # Note: We don't resize embeddings here since checkpoint already has it
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # (eval data, mismatch map, audio coverage, and run manifest were resolved
    #  in the pre-flight above, before model load.)

    # Run evaluation
    results = []
    correct = 0
    total = 0

    print(f"\n🚀 Running evaluation (input_mode={args.input_mode}, batch_size={args.batch_size})...")

    for idx, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        try:
            # --- Resolve ablation arm (see --input_mode) ---
            # use_audio comes from the pre-flight; one definition only, since the
            # summary block below reports coverage off the same predicate.
            # only audio_only drops the question text; mismatched_audio keeps the CORRECT
            # text and swaps the audio, so it stays in the training (audio+text) distribution
            include_question = args.input_mode in ("audio_text", "text_only", "mismatched_audio")
            if args.input_mode == "mismatched_audio":
                audio_row = eval_data[mismatch_map[idx]]  # spoken question from a DIFFERENT sample
            else:
                audio_row = sample

            # Load audio (only when the arm uses it)
            input_features = None
            audio_found_row = None
            if use_audio:
                audio_path = Path(args.audio_dir) / f"{k2_key(audio_row)}.mp3"
                audio_found_row = audio_path.exists()
                if audio_found_row:
                    y, _ = librosa.load(audio_path, sr=16000, mono=True)
                else:
                    y = torch.zeros(16000 * 2)
                audio_inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
                input_features = audio_inputs.input_features.to(model.device).to(torch.bfloat16)

            # Load frames
            images = load_frames(sample['frames'], args.frames_dir, max_size=args.max_image_size)

            # Build prompt. Instruction wording is held CONSTANT across arms; the only
            # variables are (a) audio presence and (b) question-text presence.
            if include_question:
                prompt_text = f"User Question: {sample['question']}\nAnswer the question concisely based on the visual and audio evidence."
            else:
                # bare instruction, no dangling "User Question:" label
                prompt_text = "Answer the question concisely based on the visual and audio evidence."
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = processor(text=[text], images=images, return_tensors="pt")

            if use_audio:
                # Inject 1500 audio tokens ahead of the vision/text content (training layout)
                AUDIO_TOKEN_ID = 151657
                NUM_AUDIO_TOKENS = 1500

                audio_tokens = torch.tensor([[AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS], device=model.device)
                audio_header = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
                audio_footer = tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").to(model.device)

                user_prefix_len = len(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
                vision_content = batch.input_ids[:, user_prefix_len:].to(model.device)

                input_ids = torch.cat([audio_header, audio_tokens, audio_footer, vision_content], dim=1)
            else:
                # text_only: no audio tokens and no audio features -> standard Qwen2-VL layout
                input_ids = batch.input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    pixel_values=batch.pixel_values.to(model.device).to(torch.bfloat16),
                    image_grid_thw=batch.image_grid_thw.to(model.device),
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            predicted = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check answer
            is_correct = sample['short_answer'].lower() in predicted.lower()

            result = {
                'question_id': sample['id'],
                'question_type': sample.get('question_type', 'unknown'),
                'question': sample['question'],
                'ground_truth': sample['answer'],
                'short_answer': sample['short_answer'],
                'predicted_answer': predicted,
                'correct': int(is_correct),
                'exact_match': int(is_correct),
                'input_mode': args.input_mode,
                'audio_source_id': k2_key(audio_row) if use_audio else None,
                'audio_found': audio_found_row if use_audio else None
            }
            results.append(result)

            total += 1
            if is_correct:
                correct += 1

            # Free memory
            del input_features, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error on sample {sample['id']}: {e}")
            continue

    # Save results
    print(f"\n💾 Saving results to: {args.output_file}")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print summary
    accuracy = (correct / total * 100) if total > 0 else 0

    # Backfill the run manifest with final results (it was written pre-model).
    manifest["total"] = total
    manifest["correct"] = correct
    manifest["accuracy"] = accuracy
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf, indent=2)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Input mode: {args.input_mode}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    if use_audio:
        # Printed for EVERY audio arm, not just degraded ones, so the line's
        # presence is never itself the warning — only the bracket is.
        pct = coverage_fraction * 100 if coverage_fraction is not None else 0.0
        flag = "  [SILENCE FALLBACK ACTIVE]" if found < required else ""
        print(f"AUDIO COVERAGE: {found}/{required} ({pct:.1f}%){flag}")
    print(f"Goal: Beat 46% baseline (audio+image zero-shot)")
    print("="*80)

    # Question type breakdown
    if 'question_type' in eval_data[0]:
        print("\nAccuracy by Question Type:")
        type_stats = {}
        for result in results:
            qtype = result['question_type']
            if qtype not in type_stats:
                type_stats[qtype] = {'correct': 0, 'total': 0}
            type_stats[qtype]['total'] += 1
            type_stats[qtype]['correct'] += result['correct']

        for qtype, stats in sorted(type_stats.items()):
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {qtype}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    return results, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate surgical VQA checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to trained checkpoint (required unless --no_adapter)")
    parser.add_argument("--eval_data_path", type=str, required=True,
                       help="Path to eval JSONL file")
    parser.add_argument("--frames_dir", type=str, required=True,
                       help="Path to frames directory")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Path to audio directory")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep at 1 for memory)")
    parser.add_argument("--max_image_size", type=int, default=384,
                       help="Max image dimension")
    parser.add_argument("--input_mode", type=str, default="audio_text",
                       choices=["audio_text", "text_only", "audio_only", "mismatched_audio"],
                       help="Ablation arm (default audio_text = training/deployment layout). "
                            "See module docstring; audio_text is the sanity gate.")
    parser.add_argument("--allow_missing_audio", action="store_true", default=False,
                       help="Proceed even if audio files are missing (runs on silence). "
                            "Off by default: a silent run is not a valid audio measurement.")
    parser.add_argument("--no_adapter", action="store_true", default=False,
                       help="Skip the LoRA adapters and evaluate the un-fine-tuned "
                            "audio-adapted BASE model (baseline cells 1 and 2). "
                            "Recorded as adapter_loaded=false in the run manifest.")

    args = parser.parse_args()
    if not args.no_adapter and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required unless --no_adapter is set")
    evaluate(args)

if __name__ == "__main__":
    main()
