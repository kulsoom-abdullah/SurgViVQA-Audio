"""
Train Audio-Grafted Qwen2-VL for Surgical VQA (QLoRA)
- Quantized base model (4-bit) + bf16 LoRA adapters
- All 8 frames at 384px for both train and eval
- SDPA attention (compatible with quantized weights)
- W&B tracking enabled

Launch command for full training (50-sample overfit test):
python src/train_vqa.py \
    --output_dir ./checkpoints/surgical_vqa_50 \
    --run_name "surgical-vqa-50-samples" \
    --train_data_path test_set/in_002-001.jsonl \
    --eval_data_path test_set/out_002-001.jsonl \
    --frames_dir data/frames \
    --audio_dir data/audio/in_002-001 \
    --eval_audio_dir data/audio/out_002-001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --logging_steps 5 \
    --report_to wandb

Note: Gradient checkpointing is enabled manually in the code with non-reentrant mode for DDP+QLoRA compatibility.

Memory usage on RTX 4090 24GB:
- Training: ~21GB (batch_size=1, 8 frames @ 384px)
- Eval: ~19GB (batch_size=1, 8 frames @ 384px)
"""

import os
import sys
import json
import torch
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
from PIL import Image
import librosa

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    WhisperFeatureExtractor,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model
)
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress "copying from non-meta parameter" warnings during checkpoint loading
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*meta parameter.*",
    category=UserWarning,
)

# Using your merged Stage 1 + Stage 2 checkpoint from HuggingFace
MODEL_ID = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"
MAX_LENGTH = 1024

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=MODEL_ID)

@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    frames_dir: str = field(default=None)
    audio_dir: str = field(default=None)
    eval_audio_dir: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_bnb_8bit")  # 8-bit AdamW for memory efficiency
    max_seq_length: int = field(default=MAX_LENGTH)
    early_stopping_patience: int = field(default=0)  # 0 = disabled, >0 = enable early stopping

class SurgicalVQADataset(Dataset):
    def __init__(self, data_path, frames_dir, audio_dir, processor, tokenizer, feature_extractor,
                 max_eval_frames=None, is_eval=False):
        self.data = [json.loads(line) for line in open(data_path, 'r')]
        self.frames_dir = frames_dir
        self.audio_dir = audio_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_token_id = 151657
        self.num_audio_tokens = 1500
        self.max_eval_frames = max_eval_frames
        self.is_eval = is_eval

    def _select_frames(self, frames):
        """Select subset of frames for eval to reduce memory"""
        if not self.is_eval or self.max_eval_frames is None or len(frames) <= self.max_eval_frames:
            return frames
        # Evenly sample frames to preserve temporal context
        stride = len(frames) / self.max_eval_frames
        return [frames[int(i * stride)] for i in range(self.max_eval_frames)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]

        # 1. Load Audio
        audio_path = Path(self.audio_dir) / f"{sample['id']}.mp3"
        if not audio_path.exists():
            y = torch.zeros(16000 * 2)
        else:
            y, _ = librosa.load(audio_path, sr=16000, mono=True)

        audio_inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        input_features = audio_inputs.input_features.squeeze(0)

        # 2. Load Images (Resized to 384px for SDPA memory efficiency)
        images = []
        selected_frames = self._select_frames(sample['frames'])
        for frame_name in selected_frames:
            if '_' in frame_name:
                vid_id = frame_name.rsplit('_', 1)[0]
            else:
                vid_id = "unknown"

            path = Path(self.frames_dir) / vid_id / f"{frame_name}.jpg"
            if not path.exists():
                frame_num = frame_name.rsplit('_', 1)[1] if '_' in frame_name else frame_name
                path = Path(self.frames_dir) / vid_id / f"{frame_num}.jpg"

            if path.exists():
                img = Image.open(path).convert("RGB")
                if max(img.size) > 384:
                    img.thumbnail((384, 384))
                images.append(img)
            else:
                images.append(Image.new('RGB', (224, 224), color='black'))

        # 3. Create Prompt
        content = [{"type": "image"} for _ in range(len(images))]
        question_text = f"User Question: {sample['question']}\nAnswer the question concisely based on the visual and audio evidence."
        content.append({"type": "text", "text": question_text})

        conversation = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample['answer']}]}
        ]

        # 4. Process Inputs
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        batch = self.processor(text=[text], images=images, padding=True, return_tensors="pt")

        input_ids = batch.input_ids.squeeze(0)
        labels = input_ids.clone()

        # 5. Inject Audio Tokens
        audio_tokens = torch.tensor([self.audio_token_id] * self.num_audio_tokens, dtype=torch.long)

        audio_header = self.tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").squeeze(0)
        audio_footer = self.tokenizer.encode("<|audio_eos|>\n", add_special_tokens=False, return_tensors="pt").squeeze(0)

        user_prefix_len = len(self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
        vision_content = input_ids[user_prefix_len:]
        vision_labels = labels[user_prefix_len:]

        final_input_ids = torch.cat([audio_header, audio_tokens, audio_footer, vision_content])

        # CRITICAL FIX: Only train on assistant's answer, mask everything else
        ignore_idx = -100

        # Find where assistant response starts
        assistant_start_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_start_len = len(assistant_start_ids)

        # Mask everything before assistant response
        final_labels = torch.cat([
            torch.full_like(audio_header, ignore_idx),
            torch.full_like(audio_tokens, ignore_idx),
            torch.full_like(audio_footer, ignore_idx),
            vision_labels
        ])

        # Find assistant start position in final sequence
        for i in range(len(final_labels) - assistant_start_len):
            if torch.equal(final_input_ids[i:i+assistant_start_len], torch.tensor(assistant_start_ids)):
                # Mask everything up to and including "<|im_start|>assistant\n"
                final_labels[:i+assistant_start_len] = ignore_idx
                break

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "input_features": input_features,
            "pixel_values": batch.pixel_values.squeeze(0),
            "image_grid_thw": batch.image_grid_thw.squeeze(0)
        }

@dataclass
class DataCollatorForSurgicalVQA:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        input_features = [f["input_features"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Explicitly cast to bf16 for FlashAttention compatibility
        pixel_values_tensor = torch.cat(pixel_values, dim=0).to(torch.bfloat16)
        input_features_tensor = torch.stack(input_features).to(torch.bfloat16)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "input_features": input_features_tensor,
            "pixel_values": pixel_values_tensor,
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
            "attention_mask": input_ids_padded.ne(self.tokenizer.pad_token_id)
        }

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load Quantized Model with SDPA (QLoRA: 4-bit base + bf16 LoRA)
    print("⏳ Loading 4-bit quantized model with SDPA attention...")
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        # device_map="auto", #have 2 GPUs now
        attn_implementation="sdpa",  # Works with quantized weights
        trust_remote_code=True
    )

    print("✓ Model loaded with 4-bit quantization (QLoRA ready)")

    # Hybrid Processor
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)

    # CRITICAL FIX: Separate PAD and EOS tokens to prevent generation loops
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.pad_token_id is None:
        print("🔧 Setting dedicated pad token (was same as EOS)")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, use_fast=False)
    processor.tokenizer = tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # LoRA Adapters for surgical VQA fine-tuning
    print("🧠 Adding LoRA Adapters...")
    target_regex = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=target_regex,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["audio_projector"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # CRITICAL FIX: Enable input grads for QLoRA + DDP + gradient checkpointing
    print("🔧 Configuring for multi-GPU training...")
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Use non-reentrant gradient checkpointing (required for DDP + QLoRA)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("✓ DDP-compatible gradient checkpointing enabled")

    # Prepare Datasets
    print(f"📁 Loading Train Data: {data_args.train_data_path}")
    train_dataset = SurgicalVQADataset(
        data_args.train_data_path,
        data_args.frames_dir,
        data_args.audio_dir,
        processor, tokenizer, feature_extractor,
        max_eval_frames=None,  # Train uses all 8 frames
        is_eval=False
    )

    eval_dataset = None
    if data_args.eval_data_path:
        print(f"📁 Loading Eval Data: {data_args.eval_data_path}")
        print(f"   Using all 8 frames at 384px (same as training)")
        eval_audio_path = data_args.eval_audio_dir if data_args.eval_audio_dir else data_args.audio_dir

        eval_dataset = SurgicalVQADataset(
            data_args.eval_data_path,
            data_args.frames_dir,
            eval_audio_path,
            processor, tokenizer, feature_extractor,
            max_eval_frames=None,  # Eval uses all 8 frames (same as training)
            is_eval=True
        )

    # Setup callbacks
    callbacks = []
    if training_args.early_stopping_patience > 0 and eval_dataset is not None:
        print(f"⏹️  Early stopping enabled (patience={training_args.early_stopping_patience})")
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSurgicalVQA(tokenizer),
        callbacks=callbacks if callbacks else None,
    )

    print("🚀 Starting Training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
