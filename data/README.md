# Data Directory

This directory should contain your surgical VQA data:

```
data/
├── frames/
│   ├── 002-001/
│   │   ├── 002-001_18743.jpg
│   │   └── ...
│   └── ...
├── audio/
│   ├── in_002-001/
│   │   ├── qa_000357.mp3
│   │   └── ...
│   └── out_002-001/
│       └── ...
```

## Data is NOT in Git

Large data files are excluded via `.gitignore`.

### On RunPod Network Volume:
Data lives at `/workspace/SurgViVQA-Audio/data/` and persists between pod sessions.

### First-time setup:
Upload data once:
```bash
# From your Mac
scp -r audio frames ubuntu@RUNPOD_IP:/workspace/SurgViVQA-Audio/data/
```

### Alternative: Cloud storage
For stateless infrastructure, sync from R2/S3:
```bash
rclone sync r2:surgvqa-data/audio ./data/audio
rclone sync r2:surgvqa-data/frames ./data/frames
```
