#!/usr/bin/env python3
"""
Generate a self-contained HTML viewer for SurgViVQA dataset samples.

Requirements: none beyond Python stdlib (Pillow used only for optional image
verification, not embedding).

Usage:
  # Minimal — browse both annotation files, no predictions
  python scripts/generate_viewer_html.py --frames_dir dataset/frames

  # With predictions overlay
  python scripts/generate_viewer_html.py \\
      --frames_dir dataset/frames \\
      --predictions results/qwen3_zeroshot_test.jsonl

  # Restrict to one video_id (faster for initial test)
  python scripts/generate_viewer_html.py \\
      --frames_dir dataset/frames \\
      --video_ids 002-001

Launch:
  python -m http.server 8080
  # open http://localhost:8080/viewer/data_viewer.html
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_frame_url(frames_dir: str, stem: str, video_id: str) -> tuple[str | None, str | None]:
    """
    Return (server_url, abs_path) for the first existing file, or (None, None).
    server_url is relative to repo root (for http.server).
    """
    candidates = [
        (f"{frames_dir}/{stem}.jpg", os.path.join(REPO_ROOT, frames_dir, f"{stem}.jpg")),
        (f"{frames_dir}/{stem}.png", os.path.join(REPO_ROOT, frames_dir, f"{stem}.png")),
        (f"{frames_dir}/{video_id}/{stem}.jpg", os.path.join(REPO_ROOT, frames_dir, video_id, f"{stem}.jpg")),
        (f"{frames_dir}/{video_id}/{stem}.png", os.path.join(REPO_ROOT, frames_dir, video_id, f"{stem}.png")),
    ]
    for url, abs_path in candidates:
        if os.path.exists(abs_path):
            return f"/{url}", abs_path
    return None, None


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_predictions(pred_path: str, test_jsonl_path: str | None) -> dict[tuple[str, str], dict]:
    """
    Return mapping from (question_id, video_id) -> prediction record.
    Requires test_jsonl to resolve which video_id each prediction ID came from.
    Falls back to (question_id, '') if test_jsonl not provided.
    """
    records = load_jsonl(pred_path)
    by_id = {r["question_id"]: r for r in records}

    if test_jsonl_path and os.path.exists(test_jsonl_path):
        test_samples = load_jsonl(test_jsonl_path)
        result = {}
        for ts in test_samples:
            qid = ts["id"]
            if qid in by_id:
                result[(qid, ts["video_id"])] = by_id[qid]
        return result

    # fallback: key by (question_id, '') — will match only samples with no video_id context
    return {(qid, ""): rec for qid, rec in by_id.items()}


def build_sample_list(
    in_samples: list[dict],
    out_samples: list[dict],
    frames_dir: str,
    video_ids: list[str] | None,
    predictions: dict[tuple[str, str], dict],
) -> list[dict]:
    """Merge in/out annotations, resolve frame paths, attach predictions."""
    out_by_id = {s["id"]: s for s in out_samples}
    missing_log: list[str] = []
    result = []

    for s in in_samples:
        if video_ids and s["video_id"] not in video_ids:
            continue

        frame_urls = []
        for stem in s["frames"]:
            url, _ = find_frame_url(frames_dir, stem, s["video_id"])
            if url is None:
                missing_log.append(stem)
            frame_urls.append(url)  # None means missing

        out_s = out_by_id.get(s["id"], {})
        pred = predictions.get((s["id"], s["video_id"])) or predictions.get((s["id"], ""))

        entry: dict = {
            "id": s["id"],
            "video_id": s["video_id"],
            "frame_numbers": s["frame_numbers"],
            "frame_urls": frame_urls,
            "in_question": s["question"],
            "out_question": out_s.get("question", s["question"]),
            "question_type": s["question_type"],
            "in_answer": s["answer"],
            "out_answer": out_s.get("answer", s["answer"]),
            "short_answer": s["short_answer"],
        }
        if pred:
            correct = pred.get("correct")
            if correct is None:
                correct = bool(pred.get("exact_match", 0))
            entry["prediction"] = pred.get("predicted_answer", "")
            entry["correct"] = correct
        result.append(entry)

    if missing_log:
        print(f"  ⚠  {len(missing_log)} frame paths not found locally (gray placeholders shown).")
        print(f"     First 5: {missing_log[:5]}")
    return result


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SurgViVQA Viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1c1f2e; --border: #2d3048;
    --text: #e2e8f0; --muted: #8892a4; --accent: #5b8af8;
    --correct: #22c55e; --wrong: #ef4444; --warn: #f59e0b;
    --mono: 'SF Mono', 'Fira Code', monospace;
  }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
         font-size: 14px; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* ── top bar ── */
  #topbar {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 10px 16px; background: var(--surface);
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  #topbar h1 { font-size: 15px; font-weight: 700; color: var(--accent); white-space: nowrap; }
  .ctrl-group { display: flex; align-items: center; gap: 6px; }
  label { color: var(--muted); font-size: 12px; }
  select, input[type=text] {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font-size: 13px; min-width: 160px;
  }
  select:focus, input:focus { outline: 2px solid var(--accent); border-color: transparent; }
  .toggle-btn {
    background: var(--border); color: var(--text); border: none;
    border-radius: 4px; padding: 5px 12px; cursor: pointer; font-size: 13px;
  }
  .toggle-btn.active { background: var(--accent); color: #fff; }
  #pos-badge {
    margin-left: auto; color: var(--muted); font-size: 12px; white-space: nowrap;
  }
  #acc-badge {
    font-size: 12px; padding: 3px 8px; border-radius: 12px;
    background: #1a2a1a; color: var(--correct); border: 1px solid #2a4a2a;
    white-space: nowrap;
  }

  /* ── main layout ── */
  #main { display: flex; flex: 1; overflow: hidden; }

  /* ── sidebar ── */
  #sidebar {
    width: 220px; flex-shrink: 0; padding: 12px;
    background: var(--surface); border-right: 1px solid var(--border);
    overflow-y: auto; font-size: 13px;
  }
  .sidebar-section { margin-bottom: 16px; }
  .sidebar-section h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                        color: var(--muted); margin-bottom: 8px; }
  .pred-dist { display: flex; flex-direction: column; gap: 4px; }
  .pred-row { display: flex; justify-content: space-between; }
  .pred-val { font-family: var(--mono); color: var(--accent); }
  .pred-count { color: var(--muted); }

  /* ── sample panel ── */
  #content { flex: 1; overflow-y: auto; padding: 16px; }

  /* ── flipbook ── */
  #flipbook-view {
    display: none; position: relative; margin-bottom: 10px;
    background: #0a0c14; border-radius: 6px; border: 2px solid var(--accent);
    overflow: hidden;
  }
  #flipbook-view img {
    width: 100%; max-height: 340px; object-fit: contain; display: block;
  }
  #flipbook-overlay {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,.8));
    padding: 20px 12px 10px;
    display: flex; justify-content: space-between; align-items: flex-end;
  }
  #flipbook-frame-num {
    font-family: var(--mono); font-size: 13px; color: #fff;
  }
  #flipbook-dots {
    display: flex; gap: 5px;
  }
  .fb-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(255,255,255,.3); transition: background .1s;
  }
  .fb-dot.active { background: var(--accent); }

  #flipbook-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0 10px; margin-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .fb-btn {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 5px 14px; cursor: pointer; font-size: 13px;
    min-width: 72px; text-align: center;
  }
  .fb-btn:hover { background: var(--border); }
  .fb-btn.playing { background: #3a2a10; border-color: var(--warn); color: var(--warn); }
  #fps-group { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; }
  #fps-slider { width: 80px; accent-color: var(--accent); }
  #fps-label { font-family: var(--mono); color: var(--text); min-width: 32px; }
  .kb-hint { font-size: 11px; color: var(--muted); margin-left: auto; }
  kbd {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 3px; padding: 1px 5px; font-family: var(--mono);
    font-size: 10px; color: var(--text);
  }

  /* ── frame grid ── */
  #frame-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 8px; margin-bottom: 10px;
  }
  .frame-cell { position: relative; cursor: pointer; }
  .frame-cell img {
    width: 100%; aspect-ratio: 4/3; object-fit: cover;
    border-radius: 4px; border: 2px solid transparent;
    display: block; transition: border-color .1s;
  }
  .frame-cell.fb-active img { border-color: var(--accent); }
  .frame-cell:hover img { border-color: var(--border); }
  .frame-missing {
    width: 100%; aspect-ratio: 4/3; border-radius: 4px;
    border: 1px dashed var(--border); background: #1a1d2e;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    text-align: center; padding: 4px;
  }
  .frame-label {
    position: absolute; bottom: 4px; left: 4px;
    background: rgba(0,0,0,.65); color: #ccc; font-size: 10px;
    font-family: var(--mono); padding: 1px 4px; border-radius: 3px;
  }

  /* ── metadata ── */
  #meta { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    border: 1px solid var(--border); color: var(--muted);
  }
  .badge.qt { border-color: #3b4fa8; color: #93a8f8; background: #1a2040; }
  .badge.vid { border-color: var(--border); }

  /* ── QA section ── */
  #qa-section { background: var(--surface); border-radius: 6px;
                padding: 14px; margin-bottom: 12px; }
  #qa-section .q { font-size: 15px; font-weight: 600; margin-bottom: 10px;
                   line-height: 1.4; }
  .ans-row { display: flex; gap: 12px; align-items: flex-start; }
  .ans-col { flex: 1; }
  .ans-col h4 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                color: var(--muted); margin-bottom: 4px; }
  .ans-col p { line-height: 1.5; }
  .short-chip {
    display: inline-block; margin-top: 6px; padding: 2px 10px;
    border-radius: 10px; font-family: var(--mono); font-size: 12px;
    background: #1e253a; border: 1px solid var(--border); color: var(--accent);
  }

  /* ── prediction section ── */
  #pred-section {
    background: var(--surface); border-radius: 6px;
    padding: 14px; display: none;
  }
  #pred-section h3 { font-size: 11px; text-transform: uppercase;
                     letter-spacing: .06em; color: var(--muted); margin-bottom: 8px; }
  #pred-content { display: flex; gap: 12px; align-items: flex-start; }
  .pred-col { flex: 1; }
  .pred-col h4 { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .result-badge {
    display: inline-block; margin-top: 6px; padding: 3px 12px;
    border-radius: 4px; font-size: 13px; font-weight: 700; letter-spacing: .04em;
  }
  .result-badge.correct { background: #14521e; color: #4ade80; border: 1px solid #22c55e; }
  .result-badge.wrong   { background: #4a1414; color: #f87171; border: 1px solid #ef4444; }

  /* ── nav bar ── */
  #navbar {
    display: flex; align-items: center; gap: 10px; margin-top: 14px;
    padding-top: 14px; border-top: 1px solid var(--border);
  }
  .nav-btn {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 6px 16px; cursor: pointer; font-size: 13px;
  }
  .nav-btn:hover { background: var(--border); }
  .nav-btn:disabled { opacity: .35; cursor: default; }
  #nav-slider { flex: 1; accent-color: var(--accent); }
  #nav-label { color: var(--muted); font-size: 12px; white-space: nowrap; min-width: 120px; text-align: right; }

  /* ── empty state ── */
  #empty { display: none; padding: 40px; text-align: center; color: var(--muted); }
</style>
</head>
<body>

<div id="topbar">
  <h1>SurgViVQA Viewer</h1>

  <div class="ctrl-group">
    <label>Annotation</label>
    <button class="toggle-btn active" id="btn-in" onclick="setTemplate('in')">in_template</button>
    <button class="toggle-btn" id="btn-out" onclick="setTemplate('out')">out_template</button>
  </div>

  <div class="ctrl-group">
    <label>Question type</label>
    <select id="sel-qt" onchange="applyFilters()"></select>
  </div>

  <div class="ctrl-group" id="correct-ctrl" style="display:none">
    <label>Result</label>
    <select id="sel-correct" onchange="applyFilters()">
      <option value="all">All</option>
      <option value="correct">Correct</option>
      <option value="wrong">Wrong</option>
    </select>
  </div>

  <span id="acc-badge" style="display:none"></span>
  <span id="pos-badge">—</span>
</div>

<div id="main">
  <div id="sidebar">
    <div class="sidebar-section" id="pred-dist-section" style="display:none">
      <h3>Prediction distribution</h3>
      <div class="pred-dist" id="pred-dist"></div>
    </div>
    <div class="sidebar-section">
      <h3>Question types</h3>
      <div class="pred-dist" id="qt-dist"></div>
    </div>
  </div>

  <div id="content">
    <div id="empty">No samples match the current filters.</div>

    <!-- flipbook controls -->
    <div id="flipbook-bar">
      <button class="fb-btn" id="fb-play-btn" onclick="toggleFlipbook()">▶ Play</button>
      <div id="fps-group">
        <span>FPS</span>
        <input type="range" id="fps-slider" min="1" max="12" value="4"
               oninput="setFps(+this.value)">
        <span id="fps-label">4</span>
      </div>
      <span class="kb-hint"><kbd>Space</kbd> play/pause &nbsp; <kbd>←</kbd><kbd>→</kbd> navigate</span>
    </div>

    <!-- large single-frame flipbook view (hidden until play) -->
    <div id="flipbook-view">
      <img id="fb-img" src="" alt="flipbook frame">
      <div id="flipbook-overlay">
        <span id="flipbook-frame-num"></span>
        <div id="flipbook-dots"></div>
      </div>
    </div>

    <!-- 2×4 grid overview (always visible) -->
    <div id="frame-grid"></div>

    <div id="meta"></div>

    <div id="qa-section">
      <div class="q" id="question-text"></div>
      <div class="ans-row">
        <div class="ans-col">
          <h4>Ground truth</h4>
          <p id="gt-long"></p>
          <span class="short-chip" id="gt-short"></span>
        </div>
      </div>
    </div>

    <div id="pred-section">
      <h3>Model prediction</h3>
      <div id="pred-content">
        <div class="pred-col">
          <h4>Predicted answer</h4>
          <p id="pred-text"></p>
          <span class="result-badge" id="result-badge"></span>
        </div>
        <div class="pred-col">
          <h4>Ground truth (short)</h4>
          <p id="pred-gt-short"></p>
        </div>
      </div>
    </div>

    <div id="navbar">
      <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">← Prev</button>
      <input type="range" id="nav-slider" min="0" value="0" oninput="navigateTo(+this.value)">
      <button class="nav-btn" id="btn-next" onclick="navigate(1)">Next →</button>
      <span id="nav-label"></span>
    </div>
  </div>
</div>

<script>
const DATA = %%DATA_JSON%%;

let template = 'in';
let filteredIndices = [];
let pos = 0;

// ── flipbook state ──
let fbPlaying = false;
let fbFrame = 0;
let fbTimer = null;
let fbFps = 4;
let fbUrls = [];
let fbNums = [];

function setFps(v) {
  fbFps = v;
  document.getElementById('fps-label').textContent = v;
  if (fbPlaying) { clearInterval(fbTimer); fbTimer = setInterval(fbTick, 1000 / fbFps); }
}

function toggleFlipbook() {
  fbPlaying ? stopFlipbook() : startFlipbook();
}

function startFlipbook() {
  if (!fbUrls.length || fbUrls.every(u => !u)) return; // nothing to play
  fbPlaying = true;
  fbFrame = 0;
  document.getElementById('fb-play-btn').textContent = '⏹ Stop';
  document.getElementById('fb-play-btn').classList.add('playing');
  document.getElementById('flipbook-view').style.display = '';
  fbTick();
  fbTimer = setInterval(fbTick, 1000 / fbFps);
}

function stopFlipbook() {
  fbPlaying = false;
  clearInterval(fbTimer);
  fbTimer = null;
  document.getElementById('fb-play-btn').textContent = '▶ Play';
  document.getElementById('fb-play-btn').classList.remove('playing');
  document.getElementById('flipbook-view').style.display = 'none';
  // clear grid highlights
  document.querySelectorAll('.frame-cell').forEach(c => c.classList.remove('fb-active'));
}

function fbTick() {
  // advance to next valid (non-null) frame, wrap around
  let tried = 0;
  while (!fbUrls[fbFrame] && tried < fbUrls.length) {
    fbFrame = (fbFrame + 1) % fbUrls.length;
    tried++;
  }
  if (!fbUrls[fbFrame]) return; // all missing

  const url = fbUrls[fbFrame];
  const num = fbNums[fbFrame];
  const img = document.getElementById('fb-img');
  img.src = url;
  document.getElementById('flipbook-frame-num').textContent = `Frame ${fbFrame + 1}/8  #${num}`;

  // dots
  document.querySelectorAll('.fb-dot').forEach((d, i) => {
    d.classList.toggle('active', i === fbFrame);
  });

  // highlight grid cell
  document.querySelectorAll('.frame-cell').forEach((c, i) => {
    c.classList.toggle('fb-active', i === fbFrame);
  });

  fbFrame = (fbFrame + 1) % fbUrls.length;
}

// clicking a grid thumbnail seeks flipbook to that frame
function seekFlipbook(i) {
  fbFrame = i;
  if (!fbPlaying) startFlipbook();
  else fbTick();
}

function setTemplate(t) {
  template = t;
  document.getElementById('btn-in').classList.toggle('active', t === 'in');
  document.getElementById('btn-out').classList.toggle('active', t === 'out');
  renderCurrent();
}

function applyFilters() {
  const qt = document.getElementById('sel-qt').value;
  const cr = document.getElementById('sel-correct').value;

  filteredIndices = DATA.samples
    .map((s, i) => i)
    .filter(i => {
      const s = DATA.samples[i];
      if (qt !== '__all__' && s.question_type !== qt) return false;
      if (cr === 'correct' && !s.correct) return false;
      if (cr === 'wrong' && (s.correct === undefined || s.correct === true)) return false;
      return true;
    });

  pos = 0;
  updateSlider();
  updateAccBadge();
  updatePredDist();
  renderCurrent();
}

function navigate(delta) {
  pos = Math.max(0, Math.min(filteredIndices.length - 1, pos + delta));
  updateSlider();
  renderCurrent();
}

function navigateTo(idx) {
  pos = Math.max(0, Math.min(filteredIndices.length - 1, idx));
  renderCurrent();
}

function updateSlider() {
  const sl = document.getElementById('nav-slider');
  sl.max = Math.max(0, filteredIndices.length - 1);
  sl.value = pos;
}

function updateAccBadge() {
  if (!DATA.has_predictions) return;
  const qt = document.getElementById('sel-qt').value;
  const relevant = DATA.samples.filter(s =>
    (qt === '__all__' || s.question_type === qt) && s.correct !== undefined
  );
  if (!relevant.length) { document.getElementById('acc-badge').style.display = 'none'; return; }
  const n_correct = relevant.filter(s => s.correct).length;
  const pct = (100 * n_correct / relevant.length).toFixed(1);
  const badge = document.getElementById('acc-badge');
  badge.textContent = `Accuracy: ${n_correct}/${relevant.length} (${pct}%)`;
  badge.style.display = '';
}

function updatePredDist() {
  if (!DATA.has_predictions) return;
  const relevant = filteredIndices.map(i => DATA.samples[i]).filter(s => s.prediction !== undefined);
  const counts = {};
  relevant.forEach(s => { counts[s.prediction] = (counts[s.prediction] || 0) + 1; });
  const sorted = Object.entries(counts).sort((a,b) => b[1]-a[1]).slice(0, 12);
  const container = document.getElementById('pred-dist');
  container.innerHTML = sorted.map(([v, c]) =>
    `<div class="pred-row"><span class="pred-val">${esc(v)}</span><span class="pred-count">${c}</span></div>`
  ).join('');
  document.getElementById('pred-dist-section').style.display = sorted.length ? '' : 'none';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderCurrent() {
  const wasPlaying = fbPlaying;
  stopFlipbook();

  const empty = document.getElementById('empty');
  const grid = document.getElementById('frame-grid');
  const meta = document.getElementById('meta');

  if (!filteredIndices.length) {
    empty.style.display = '';
    grid.innerHTML = '';
    meta.innerHTML = '';
    document.getElementById('pos-badge').textContent = '0 of 0';
    document.getElementById('nav-label').textContent = '0 of 0';
    document.getElementById('btn-prev').disabled = true;
    document.getElementById('btn-next').disabled = true;
    return;
  }
  empty.style.display = 'none';

  const idx = filteredIndices[pos];
  const s = DATA.samples[idx];

  // stash urls/nums for flipbook
  fbUrls = s.frame_urls;
  fbNums = s.frame_numbers;

  // dots
  document.getElementById('flipbook-dots').innerHTML =
    s.frame_urls.map((_, i) => `<span class="fb-dot" id="fb-dot-${i}"></span>`).join('');

  // position
  const posStr = `Sample ${pos + 1} of ${filteredIndices.length} — ${s.question_type}`;
  document.getElementById('pos-badge').textContent = posStr;
  document.getElementById('nav-label').textContent = `${pos + 1} / ${filteredIndices.length}`;
  document.getElementById('btn-prev').disabled = pos === 0;
  document.getElementById('btn-next').disabled = pos === filteredIndices.length - 1;

  // frame grid — thumbnails, clickable to seek flipbook
  grid.innerHTML = s.frame_urls.map((url, i) => {
    const num = s.frame_numbers[i];
    if (!url) {
      return `<div class="frame-cell"><div class="frame-missing">${esc(s.video_id)}_${num}<br>(not found)</div></div>`;
    }
    return `<div class="frame-cell" onclick="seekFlipbook(${i})">
      <img src="${url}" alt="frame ${num}" onerror="this.parentNode.innerHTML='<div class=\'frame-missing\'>${esc(s.video_id)}_${num}<br>(load error)</div>'">
      <span class="frame-label">#${num}</span>
    </div>`;
  }).join('');

  // meta badges
  meta.innerHTML = `
    <span class="badge qt">${esc(s.question_type)}</span>
    <span class="badge vid">video: ${esc(s.video_id)}</span>
    <span class="badge">id: ${esc(s.id)}</span>
  `;

  // QA
  const q = template === 'in' ? s.in_question : s.out_question;
  const ans = template === 'in' ? s.in_answer : s.out_answer;
  document.getElementById('question-text').textContent = q;
  document.getElementById('gt-long').textContent = ans;
  document.getElementById('gt-short').textContent = s.short_answer;

  // prediction
  const predSection = document.getElementById('pred-section');
  if (s.prediction !== undefined) {
    predSection.style.display = '';
    document.getElementById('pred-text').textContent = s.prediction;
    document.getElementById('pred-gt-short').textContent = s.short_answer;
    const badge = document.getElementById('result-badge');
    badge.textContent = s.correct ? 'CORRECT' : 'WRONG';
    badge.className = 'result-badge ' + (s.correct ? 'correct' : 'wrong');
  } else {
    predSection.style.display = 'none';
  }

  // resume flipbook if it was playing before navigation
  if (wasPlaying) startFlipbook();
}

function populateQtDropdown() {
  const qtCounts = {};
  DATA.samples.forEach(s => { qtCounts[s.question_type] = (qtCounts[s.question_type] || 0) + 1; });
  const sel = document.getElementById('sel-qt');
  sel.innerHTML = `<option value="__all__">All Types (${DATA.samples.length})</option>`;
  Object.entries(qtCounts).sort((a,b) => a[0].localeCompare(b[0])).forEach(([qt, n]) => {
    sel.innerHTML += `<option value="${qt}">${qt} (${n})</option>`;
  });

  // sidebar qt dist
  const qtDist = document.getElementById('qt-dist');
  qtDist.innerHTML = Object.entries(qtCounts).sort((a,b) => a[0].localeCompare(b[0]))
    .map(([qt, n]) => `<div class="pred-row"><span class="pred-val" style="color:var(--muted);font-size:11px">${esc(qt)}</span><span class="pred-count">${n}</span></div>`)
    .join('');
}

function init() {
  if (DATA.has_predictions) {
    document.getElementById('correct-ctrl').style.display = '';
    document.getElementById('acc-badge').style.display = '';
  }
  populateQtDropdown();

  // if predictions loaded, default to lesion_motion_direction + wrong to show the key failure
  if (DATA.has_predictions) {
    const sel = document.getElementById('sel-qt');
    const motionOpt = Array.from(sel.options).find(o => o.value === 'lesion_motion_direction');
    if (motionOpt) {
      sel.value = 'lesion_motion_direction';
      document.getElementById('sel-correct').value = 'wrong';
    }
  }

  applyFilters();
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  { navigate(-1); }
  if (e.key === 'ArrowRight') { navigate(1); }
  if (e.key === ' ') { e.preventDefault(); toggleFlipbook(); }
});

init();
</script>
</body>
</html>
"""


def smoke_test(samples: list[dict]) -> None:
    from collections import Counter
    qt_counts = Counter(s["question_type"] for s in samples)
    print(f"\n  Sample count      : {len(samples)}")
    print(f"  Question types    : {len(qt_counts)}")
    for qt, n in sorted(qt_counts.items()):
        print(f"    {qt:<40} {n}")
    frames_with_url = sum(1 for s in samples for url in s["frame_urls"] if url)
    frames_total = sum(len(s["frame_urls"]) for s in samples)
    print(f"\n  Frames found      : {frames_with_url} / {frames_total}")
    if frames_with_url > 0:
        print("  ✓ At least one image found locally — viewer will render frames.")
    else:
        print("  ⚠  No frames found locally. Download frames first; viewer shows placeholders.")
    pred_samples = sum(1 for s in samples if s.get("prediction") is not None)
    if pred_samples:
        correct = sum(1 for s in samples if s.get("correct"))
        print(f"\n  Predictions loaded: {pred_samples}")
        print(f"  Accuracy          : {correct}/{pred_samples} ({100*correct/pred_samples:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SurgViVQA HTML data viewer")
    parser.add_argument("--frames_dir", default="dataset/frames",
                        help="Frame base dir relative to repo root (default: dataset/frames)")
    parser.add_argument("--in_template", default="data/in_template.jsonl")
    parser.add_argument("--out_template", default="data/out_template.jsonl")
    parser.add_argument("--predictions",
                        help="Optional predictions JSONL (e.g. results/qwen3_zeroshot_test.jsonl)")
    parser.add_argument("--test_jsonl", default="data/test_multivideo.jsonl",
                        help="Test set JSONL with video_id per sample — used to correctly join predictions "
                             "(default: data/test_multivideo.jsonl)")
    parser.add_argument("--video_ids", nargs="*",
                        help="Only include samples from these video_ids")
    parser.add_argument("--out", default="viewer/data_viewer.html",
                        help="Output HTML path (default: viewer/data_viewer.html)")
    args = parser.parse_args()

    # Resolve paths relative to repo root for consistency
    def repo(p: str) -> str:
        return os.path.join(REPO_ROOT, p) if not os.path.isabs(p) else p

    print("Loading annotation files...")
    in_path = repo(args.in_template)
    out_path = repo(args.out_template)
    for p in [in_path, out_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)
    in_samples = load_jsonl(in_path)
    out_samples = load_jsonl(out_path)
    print(f"  in_template : {len(in_samples)} samples")
    print(f"  out_template: {len(out_samples)} samples")

    predictions: dict[str, dict] = {}
    if args.predictions:
        pred_path = repo(args.predictions)
        if not os.path.exists(pred_path):
            print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
            sys.exit(1)
        test_path = repo(args.test_jsonl) if args.test_jsonl else None
        predictions = load_predictions(pred_path, test_path)
        n_test = sum(1 for (_, vid) in predictions if vid)
        print(f"  predictions : {len(predictions)} records from {args.predictions}")
        if test_path and os.path.exists(test_path):
            print(f"  test_jsonl  : {args.test_jsonl} (video_id join enabled)")

    print("\nResolving frame paths...")
    samples = build_sample_list(
        in_samples, out_samples,
        args.frames_dir,
        args.video_ids,
        predictions,
    )
    print(f"  Built {len(samples)} samples")

    print("\nSmoke test:")
    smoke_test(samples)

    data_json = json.dumps({
        "samples": samples,
        "has_predictions": bool(predictions),
    }, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("%%DATA_JSON%%", data_json)

    out_path = repo(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n✓ Viewer written: {args.out}  ({size_kb:.0f} KB)")
    print(f"""
To launch:
  cd {REPO_ROOT}
  python -m http.server 8080
  # open http://localhost:8080/{args.out}

Keyboard shortcuts: ← / → to navigate samples.
""")


if __name__ == "__main__":
    main()
