"""
Train Qwen 3.0-VL for Surgical VQA (Text+Image Only, NO Audio)
- For fair comparison with Qwen 2.0 + audio
- Same train/eval/test split, same hyperparameters, same evaluation
- QLoRA: Quantized base model (4-bit) + bf16 LoRA adapters
- All 8 frames at 384px for both train and eval
- W&B tracking enabled

Experimental hypothesis:
Can Qwen 3.0's superior vision encoder LEARN temporal reasoning from training data?
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

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoProcessor,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model
)
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*meta parameter.*", category=UserWarning)

# Import Qwen3 VL class (dynamic import for compatibility)
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError("Qwen3VLForConditionalGeneration not available. Update transformers: pip install --upgrade transformers")

MAX_LENGTH = 1024

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")

@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    frames_dir: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_bnb_8bit")
    max_seq_length: int = field(default=MAX_LENGTH)
    early_stopping_patience: int = field(default=0)
    ddp_find_unused_parameters: bool = field(default=False)

class SurgicalVQADataset(Dataset):
    """Dataset for Qwen 3.0 (text+image only, NO audio)"""
    def __init__(self, data_path, frames_dir, processor, tokenizer,
                 max_eval_frames=None, is_eval=False):
        self.data = [json.loads(line) for line in open(data_path, 'r')]
        self.frames_dir = frames_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_eval_frames = max_eval_frames
        self.is_eval = is_eval

    def _select_frames(self, frames):
        """Select subset of frames for eval to reduce memory"""
        if not self.is_eval or self.max_eval_frames is None or len(frames) <= self.max_eval_frames:
            return frames
        stride = len(frames) / self.max_eval_frames
        return [frames[int(i * stride)] for i in range(self.max_eval_frames)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]

        # Load Images (Resized to 384px for memory efficiency)
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

        # Create Prompt (same format as Qwen 2.0, but without audio reference)
        content = [{"type": "image"} for _ in range(len(images))]
        question_text = f"User Question: {sample['question']}\nAnswer the question concisely based on the visual evidence."
        content.append({"type": "text", "text": question_text})

        conversation = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": sample['answer']}]}
        ]

        # Process Inputs
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        batch = self.processor(text=[text], images=images, padding=True, return_tensors="pt")

        input_ids = batch.input_ids.squeeze(0)
        labels = input_ids.clone()

        # Mask everything except assistant's answer
        ignore_idx = -100
        assistant_start_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_start_len = len(assistant_start_ids)

        # Find assistant start position
        for i in range(len(labels) - assistant_start_len):
            if torch.equal(input_ids[i:i+assistant_start_len], torch.tensor(assistant_start_ids)):
                labels[:i+assistant_start_len] = ignore_idx
                break

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": batch.pixel_values.squeeze(0),
            "image_grid_thw": batch.image_grid_thw.squeeze(0)
        }

@dataclass
class DataCollatorForSurgicalVQA:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Cast to bf16 for FlashAttention compatibility
        pixel_values_tensor = torch.cat(pixel_values, dim=0).to(torch.bfloat16)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "pixel_values": pixel_values_tensor,
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
            "attention_mask": input_ids_padded.ne(self.tokenizer.pad_token_id)
        }

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load Quantized Model (QLoRA: 4-bit base + bf16 LoRA)
    print("="*80)
    print(f"QWEN 3.0-VL FINE-TUNING (Text+Image Only)")
    print("="*80)
    print(f"⏳ Loading 4-bit quantized model: {model_args.model_name_or_path}")

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 like baselines
        trust_remote_code=True
    )

    print("✓ Model loaded with 4-bit quantization (QLoRA ready)")

    # Load Processor & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # Set dedicated pad token (prevent generation loops)
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.pad_token_id is None:
        print("🔧 Setting dedicated pad token (was same as EOS)")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        processor.tokenizer = tokenizer
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # LoRA Adapters (auto-detect layer structure for Qwen 3.0)
    print("🧠 Adding LoRA Adapters...")

    # Qwen 3.0 has different architecture: model.language_model.layers.X instead of model.layers.X
    # Auto-detect the correct layer prefix
    layer_names = [name for name, _ in model.named_modules()]

    if any("language_model.layers" in name for name in layer_names):
        # Qwen 3.0 architecture
        target_regex = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        print("   Detected Qwen 3.0 architecture (language_model.layers)")
    else:
        # Qwen 2.0 architecture
        target_regex = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        print("   Detected Qwen 2.0 architecture (layers)")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=target_regex,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable for multi-GPU training
    print("🔧 Configuring for multi-GPU training...")
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("✓ DDP-compatible gradient checkpointing enabled")

    # Prepare Datasets
    print(f"📁 Loading Train Data: {data_args.train_data_path}")
    train_dataset = SurgicalVQADataset(
        data_args.train_data_path,
        data_args.frames_dir,
        processor, tokenizer,
        max_eval_frames=None,
        is_eval=False
    )

    eval_dataset = None
    if data_args.eval_data_path:
        print(f"📁 Loading Eval Data: {data_args.eval_data_path}")
        print(f"   Using all 8 frames at 384px (same as training)")
        eval_dataset = SurgicalVQADataset(
            data_args.eval_data_path,
            data_args.frames_dir,
            processor, tokenizer,
            max_eval_frames=None,
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
    print(f"   Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Max epochs: {training_args.num_train_epochs}")
    print(f"   Early stopping: {training_args.early_stopping_patience > 0}")
    print("")

    trainer.train()
    trainer.save_model(training_args.output_dir)

    print("")
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Checkpoint saved to: {training_args.output_dir}")
    print("")
    print("Next: Evaluate on test set (1,000 held-out samples)")

if __name__ == "__main__":
    train()
