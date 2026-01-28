# Professional Setup for Audio-Grafted Qwen2-VL

This document shows how to properly package your custom transformers fork for reproducible research.

## 🎯 Current Status (Works, but not professional)
- Modified transformers code lives in HuggingFace model repo subfolder
- Users must download and manually configure paths
- Not pip-installable

## 🏆 Professional Solution: GitHub + Pip Install

### Step 1: Create GitHub Repo for Modified Transformers

```bash
# On your Mac
cd /Users/kulsoom/workspace/learning/Qwen2-VL-Audio-Graft

# Initialize the transformers fork as a separate repo
cd transformers_fork
git init
git add .
git commit -m "Initial commit: Audio-grafted Qwen2-VL transformers fork"

# Create repo on GitHub: https://github.com/kulsoom-abdullah/transformers-qwen-audio
# Then push:
git remote add origin https://github.com/kulsoom-abdullah/transformers-qwen-audio.git
git branch -M main
git push -u origin main
```

### Step 2: Add Setup File (for pip install)

Create `transformers_fork/setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="transformers-qwen-audio",
    version="0.1.0",
    description="Transformers fork with audio-grafted Qwen2-VL support",
    author="Kulsoom Abdullah",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",  # Base version you forked from
        "accelerate",
        "pillow",
        "librosa",
    ],
)
```

### Step 3: Update Your Project's README

```markdown
## Installation

This project uses a custom fork of Transformers with audio-grafted Qwen2-VL:

\`\`\`bash
# Install modified transformers
pip install git+https://github.com/kulsoom-abdullah/transformers-qwen-audio.git

# Install project dependencies
pip install -r requirements.txt
\`\`\`

## Running Baseline 2 (Audio-Grafted Model)

\`\`\`bash
python baselines/baseline2_audio_image.py \\
    --test_file test_set/in_002-001.jsonl \\
    --frames_dir dataset/frames \\
    --audio_dir audio/in_002-001 \\
    --output results/baseline2_in.jsonl
\`\`\`
```

### Step 4: Update baseline2.py for pip-installed fork

Once your fork is pip-installable, you don't need the `sys.path.insert()` hack:

```python
# Just import normally - pip installed the modified version
from transformers import Qwen2VLForConditionalGeneration
```

## 🎓 Why This Matters for Your Resume

**Before (Current):**
- "Modified transformers locally"
- Requires manual setup
- Hard to reproduce

**After (Professional):**
- "Developed custom transformers fork for audio-visual models"
- One-line installation: `pip install git+https://github.com/...`
- Works on any machine (Lambda, Colab, local)
- Shows software engineering best practices

## 📊 Comparison of Approaches

| Approach | Setup Time | Reproducibility | Professional? |
|----------|-----------|-----------------|---------------|
| scp from Mac | Manual | Low | ❌ No |
| Download from HF | 1 minute | Medium | ⚠️ Okay |
| GitHub + pip | 5 minutes (once) | High | ✅ Yes |

## ⏭️ Next Steps

1. **Today:** Use `setup_fork_from_hf.sh` to run Baseline 2
2. **This week:** Push fork to GitHub and make it pip-installable
3. **Portfolio:** Update README with professional installation instructions
