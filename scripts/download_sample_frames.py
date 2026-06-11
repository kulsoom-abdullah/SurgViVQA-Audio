#!/usr/bin/env python3
"""
Query Figshare API for the REAL-Colon dataset and verify downloaded frames.

Step 1 — see what's on Figshare before downloading anything:
    python scripts/download_sample_frames.py

Step 2 — after extracting frames, verify they're all present:
    python scripts/download_sample_frames.py --verify --output_dir dataset/frames

    # Restrict to specific video_ids:
    python scripts/download_sample_frames.py --verify \
        --output_dir dataset/frames --video_ids 002-001 002-003
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict

FIGSHARE_ARTICLE_ID = 22202866
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

_frame_cache: dict[str, str | None] = {}


def find_frame(base_dir: str, stem: str, video_id: str) -> str | None:
    """Return the first existing path for a frame stem, or None. Results are cached."""
    if stem in _frame_cache:
        return _frame_cache[stem]
    candidates = [
        os.path.join(base_dir, f"{stem}.jpg"),
        os.path.join(base_dir, f"{stem}.png"),
        os.path.join(base_dir, video_id, f"{stem}.jpg"),
        os.path.join(base_dir, video_id, f"{stem}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            _frame_cache[stem] = path
            return path
    _frame_cache[stem] = None
    return None


def query_figshare() -> None:
    print(f"Querying Figshare API: {FIGSHARE_API_URL}\n")
    try:
        req = urllib.request.Request(FIGSHARE_API_URL, headers={"User-Agent": "SurgViVQA-viewer/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            article = json.loads(resp.read())
    except Exception as exc:
        print(f"ERROR: Could not reach Figshare API: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Title:     {article.get('title', 'N/A')}")
    print(f"DOI:       {article.get('doi', 'N/A')}")
    print(f"Published: {article.get('published_date', 'N/A')}")

    files = article.get("files", [])
    if not files:
        print("\nNo files listed in API response.")
        return

    # Highlight which files correspond to our SurgViVQA video_ids
    OUR_VIDEOS = {"002-001", "002-002", "002-003", "002-004", "002-005", "002-006"}

    print(f"\n{'File name':<55} {'Size':>10}  Download URL")
    print("-" * 120)
    total_bytes = 0
    our_frames: list[dict] = []
    for f in files:
        size = f.get("size", 0)
        total_bytes += size
        if size > 1_000_000_000:
            size_str = f"{size / 1_073_741_824:.2f} GB"
        elif size > 1_000_000:
            size_str = f"{size / 1_048_576:.1f} MB"
        else:
            size_str = f"{size / 1024:.0f} KB"
        name = f.get("name", "?")
        marker = ""
        vid_id = name.replace("_frames.tar.gz", "").replace("_annotations.tar.gz", "")
        if vid_id in OUR_VIDEOS and "_frames" in name:
            marker = "  ← SurgViVQA"
            our_frames.append(f)
        print(f"  {name:<53} {size_str:>10}  {f.get('download_url', 'N/A')}{marker}")

    if total_bytes > 1_000_000_000:
        total_str = f"{total_bytes / 1_073_741_824:.2f} GB"
    else:
        total_str = f"{total_bytes / 1_048_576:.1f} MB"
    print(f"\nTotal: {len(files)} file(s), {total_str}")

    if our_frames:
        our_total = sum(f.get("size", 0) for f in our_frames)
        print(f"\n--- Files needed for SurgViVQA dataset ({len(our_frames)} video_ids, "
              f"{our_total / 1_073_741_824:.1f} GB total) ---")
        print("\nDownload + extract commands (run from repo root):")
        for f in our_frames:
            vid = f["name"].replace("_frames.tar.gz", "")
            url = f.get("download_url", "")
            print(f"  # {vid}  ({f.get('size',0)/1_073_741_824:.2f} GB)")
            print(f"  wget -O dataset/{f['name']} '{url}'")
            print(f"  mkdir -p dataset/frames/{vid} && tar -xzf dataset/{f['name']} --strip-components=1 -C dataset/frames/{vid}/")
            print()
        print("After extracting, verify:")
        print("  python scripts/download_sample_frames.py --verify --output_dir dataset/frames")
        print("\nTo test viewer with one video first (002-001, 7.08 GB):")
        print("  python scripts/download_sample_frames.py --verify \\")
        print("      --output_dir dataset/frames --video_ids 002-001")


def verify_frames(jsonl_file: str, output_dir: str, video_ids: list[str] | None) -> None:
    if not os.path.exists(jsonl_file):
        print(f"ERROR: JSONL file not found: {jsonl_file}", file=sys.stderr)
        sys.exit(1)

    with open(jsonl_file) as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    if video_ids:
        samples = [s for s in samples if s["video_id"] in video_ids]
        print(f"Filtered to {len(samples)} samples for video_ids: {video_ids}")

    found = 0
    missing = 0
    extensions: dict[str, int] = defaultdict(int)
    patterns: dict[str, int] = defaultdict(int)
    missing_examples: list[str] = []

    for sample in samples:
        for stem in sample["frames"]:
            path = find_frame(output_dir, stem, sample["video_id"])
            if path:
                found += 1
                ext = os.path.splitext(path)[1].lower()
                extensions[ext] += 1
                rel = os.path.relpath(path, output_dir)
                patterns["subfolder/{video_id}" if os.sep in rel else "flat"] += 1
            else:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(stem)

    total = found + missing
    print(f"\nVerification against: {jsonl_file}")
    print(f"  Samples checked  : {len(samples)}")
    print(f"  Frame stems total: {total}")
    if total:
        print(f"  Found            : {found}  ({100 * found / total:.1f}%)")
        print(f"  Missing          : {missing}  ({100 * missing / total:.1f}%)")
    if extensions:
        print(f"  Extension(s)     : {dict(extensions)}")
    if patterns:
        print(f"  Path pattern     : {dict(patterns)}")
    if missing_examples:
        print(f"  Missing examples (first 5): {missing_examples}")
    if missing == 0:
        print("\n✓ All frames accounted for.")
    else:
        print(f"\n⚠  {missing} frames missing. Check --output_dir or download more video_ids.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figshare query + frame verification for REAL-Colon dataset")
    parser.add_argument("--output_dir", default="dataset/frames",
                        help="Base directory where frames are stored (default: dataset/frames)")
    parser.add_argument("--jsonl_file", default="data/in_template.jsonl",
                        help="Annotation JSONL to verify against (default: data/in_template.jsonl)")
    parser.add_argument("--video_ids", nargs="*",
                        help="Only check stems from these video_ids")
    parser.add_argument("--verify", action="store_true",
                        help="Verify local frames instead of querying Figshare API")
    args = parser.parse_args()

    if args.verify:
        verify_frames(args.jsonl_file, args.output_dir, args.video_ids)
    else:
        query_figshare()


if __name__ == "__main__":
    main()
