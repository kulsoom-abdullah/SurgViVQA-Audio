# 🔍 Data Viewer

Moved out of `README.md`; linked from its Setup section.

Lightweight local browser for inspecting dataset samples and overlaying SFT/zero-shot model predictions. No GPU or extra dependencies required.

**Generate the viewer:**
```bash
# Browse-only (no predictions)
python scripts/generate_viewer_html.py --frames_dir dataset/frames

# With predictions overlay (e.g. qwen3 zero-shot)
python scripts/generate_viewer_html.py \
    --frames_dir dataset/frames \
    --predictions results/qwen3_zeroshot_test.jsonl \
    --out viewer/data_viewer_qwen3.html
```

**Launch:**
```bash
python -m http.server 8080
# open http://localhost:8080/viewer/data_viewer.html
```

**Usage:**
- Set frame directory to wherever REAL-Colon frames are extracted (`dataset/frames/`)
- Optionally load a predictions JSONL to overlay model outputs
- Toggle between `in_template` / `out_template` question phrasing
- Filter by question type to examine specific failure modes
- Filter by correct / wrong (when predictions loaded)
- Keyboard: `←` / `→` to navigate samples

**Download frames from Figshare:**
```bash
# Step 1: see what's available and get per-video download commands
python scripts/download_sample_frames.py

# Step 2: after extracting, verify coverage
python scripts/download_sample_frames.py --verify --output_dir dataset/frames
```

**Purpose:** Confirm whether 8-frame sequences contain sufficient visual signal for each question type — foundation for documenting SFT failure modes before GRPO.

> **⚠️ Under revision (July 2026):** the viewer can display a sample alongside another sample's question and answer. Treat its output as unverified.
