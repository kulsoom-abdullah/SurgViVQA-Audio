# SurgViVQA-Audio: 3-Baseline Comparison Experiment

## Objective
Demonstrate the value of direct audio processing in multimodal VQA by comparing inference speed and accuracy across three approaches.

## Experimental Setup

### Dataset
- **Source**: SurgViVQA (out_template.jsonl - natural questions)
- **Test Set**: Start with 50-100 samples, then scale to full 5,200
- **Metrics**: Inference time (seconds/sample) + Answer accuracy (exact match)

### Three Baselines

#### Baseline 1: Text + Image → Model (Control)
**Description**: Traditional VQA - Your current Qwen2-VL-Audio-Graft model using text questions
- **Input**: Text question + Image frames
- **Model**: Qwen2-VL-7B (text encoder + vision encoder)
- **Purpose**: Baseline performance without audio

#### Baseline 2: Audio + Image → Model (Your Innovation!)
**Description**: Direct audio processing - Your grafted model
- **Input**: Audio question (synthesized via edge-tts) + Image frames
- **Model**: Qwen2-VL-Audio-Graft (Whisper encoder + Qwen2-VL)
- **Purpose**: Show direct audio processing capability

#### Baseline 3: Audio → ASR → Text + Image → Model (Traditional Pipeline)
**Description**: Two-stage pipeline - Realistic comparison
- **Input**: Audio question → Whisper ASR → Text + Image frames
- **Model**: Whisper (ASR) → Qwen2-VL-7B (text encoder + vision encoder)
- **Purpose**: Show overhead of traditional ASR preprocessing

## Expected Results

### Inference Speed Comparison
```
Baseline 1 (Text+Image):        ~X seconds/sample  (fastest - no audio)
Baseline 2 (Audio+Image):       ~Y seconds/sample  (YOUR MODEL)
Baseline 3 (ASR→Text+Image):    ~Z seconds/sample  (slowest - 2 stages)
```

**Expected**: Z > Y (Baseline 3 slower due to ASR overhead)

**Value Proposition**: If Y ≈ X and Y < Z, you demonstrate:
1. Direct audio processing saves time vs ASR pipeline
2. Audio modality doesn't slow down inference significantly
3. Single-stage > two-stage for real-time applications

### Accuracy Comparison
```
Baseline 1 (Text+Image):        ~A% accuracy  (control)
Baseline 2 (Audio+Image):       ~B% accuracy  (YOUR MODEL)
Baseline 3 (ASR→Text+Image):    ~C% accuracy  (may have ASR errors)
```

**Expected**: B ≥ C (Audio direct processing at least as good as ASR pipeline)

**Potential findings**:
- If B > C: Audio features help! (e.g., prosody, emphasis)
- If B = C: Audio is as good without ASR overhead
- If B < A < C: Text is still best (but audio eliminates ASR step)

## Metrics to Track

### Per-Sample Metrics
- `question_id`: Sample identifier
- `question_type`: Category (e.g., nbi_status, mucosa_visibility)
- `ground_truth`: Correct answer
- `predicted_answer`: Model output
- `exact_match`: 1 if correct, 0 otherwise
- `inference_time_ms`: Milliseconds to generate answer

### Aggregate Metrics
- **Accuracy**: Exact match rate (%)
- **Average Inference Time**: Mean time per sample (ms)
- **Median Inference Time**: Median time (ms)
- **Throughput**: Samples per second
- **By Question Type**: Accuracy breakdown by category

## Implementation Plan

### Phase 1: Setup (Start Small)
1. ✓ Generate audio for all questions (edge-tts)
2. Create test set (50-100 samples with diverse question types)
3. Implement inference scripts for each baseline
4. Implement evaluation harness

### Phase 2: Baseline Implementations

#### Baseline 1: Text + Image
```python
# Pseudocode
for sample in test_set:
    start = time()
    output = model.generate(
        text=sample['question'],
        images=sample['frames']
    )
    inference_time = time() - start
    accuracy = check_answer(output, sample['ground_truth'])
```

#### Baseline 2: Audio + Image (Your Model)
```python
# Pseudocode
for sample in test_set:
    audio = load_audio(sample['audio_path'])  # Pre-generated
    start = time()
    output = model.generate(
        audio=audio,
        images=sample['frames']
    )
    inference_time = time() - start
    accuracy = check_answer(output, sample['ground_truth'])
```

#### Baseline 3: ASR → Text + Image
```python
# Pseudocode
for sample in test_set:
    audio = load_audio(sample['audio_path'])

    # Stage 1: ASR
    start = time()
    transcribed_text = whisper.transcribe(audio)
    asr_time = time() - start

    # Stage 2: VQA
    vqa_start = time()
    output = model.generate(
        text=transcribed_text,
        images=sample['frames']
    )
    vqa_time = time() - vqa_start

    total_time = asr_time + vqa_time
    accuracy = check_answer(output, sample['ground_truth'])
```

### Phase 3: Evaluation
1. Run all 3 baselines on test set
2. Compute metrics
3. Generate comparison tables and plots
4. Statistical significance testing

### Phase 4: Scale Up
1. Run on full dataset (5,200 samples)
2. Generate final results for paper/presentation

## Directory Structure
```
SurgViVQA-Audio/
├── audio/                      # Generated TTS audio files
│   ├── in_template/           # Robotic questions
│   └── out_template/          # Natural questions
├── test_set/                  # Small test set for debugging
│   ├── test_samples.jsonl    # 50-100 samples
│   └── test_audio/           # Audio for test set
├── baselines/                 # Inference scripts
│   ├── baseline1_text_image.py
│   ├── baseline2_audio_image.py
│   └── baseline3_asr_pipeline.py
├── evaluation/
│   ├── evaluate.py           # Main evaluation script
│   └── metrics.py            # Metric computation
└── results/
    ├── test_results.json     # Debug results
    ├── full_results.json     # Full dataset results
    └── plots/                # Visualizations
```

## Next Steps
1. Read your model code to understand inference API
2. Generate audio for all 10,400 questions
3. Create test set with 50-100 diverse samples
4. Implement baseline inference scripts
5. Run small-scale test
6. Scale to full dataset

## Questions to Clarify
- What's your model loading/inference code structure?
- Do you have image frames already, or need to extract from videos?
- What's the expected format for image input (single frame, multiple frames, video)?
- Are there any specific question types you want to prioritize in the test set?
