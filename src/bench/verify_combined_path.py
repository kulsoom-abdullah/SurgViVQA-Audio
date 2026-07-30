#!/usr/bin/env python3
"""
verify_combined_path.py — screening probe for omni-model baselines.

Decides whether a candidate is worth a full evaluation run, without renting a pod for
one. Answers four questions per candidate:

  0. Is decoding DETERMINISTIC?      (control — without it, Q3 is uninterpretable)
  1. Does 8 images + 1 audio clip survive a single prompt without erroring?
  2. Does the rendered prompt leak question text?   (Gate 2, PRE_REGISTRATION §4.2)
  3. IS THE AUDIO PATHWAY ACTUALLY CONNECTED?       (Gate 3, PRE_REGISTRATION §4.4)

Q3 is the reason this file exists: a model can accept audio, run clean, and produce a
plausible accuracy number while the audio pathway is silently dead, in which case the
output is a function of the images and the answer prior alone. That failure already
invalidated one result on this project.

Q0 is why Q3 is trustworthy. If sampling is on, two runs of the SAME audio differ by
chance, "outputs differ" proves nothing, and the probe passes while measuring noise.
Both candidates' model cards use do_sample=True in their examples. We force greedy and
then verify greed took effect.

SCOPE: phi4mm and minicpmo45. Gemma 4 / Nemotron 3 are scoped-not-run.

Usage:
    # 1. build pairs from real artifacts (no GPU needed)
    python verify_combined_path.py --build-pairs \
        --audio-dir data/audio/test --test-manifest data/test_multivideo.jsonl \
        --out-pairs data/probe_pairs.json

    # 2. probe (one venv per model — see PRE_REGISTRATION §11)
    python verify_combined_path.py --model phi4mm     --pairs data/probe_pairs.json \
        --frames data/frames/002-004
    python verify_combined_path.py --model minicpmo45 --pairs data/probe_pairs.json \
        --frames data/frames/002-004
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

N_FRAMES = 8
MAX_NEW_TOKENS = 32
N_PAIRS = 10
SENSITIVITY_THRESHOLD = 0.8          # >= 8/10 pairs must differ
FRAME_EXTS = (".jpg", ".jpeg", ".png")

# Frozen per PRE_REGISTRATION.md §4.6 (Parity-B). Recovered verbatim from
# src/evaluate_checkpoint.py:309 — the exact string the audio_only arm ran under, and
# therefore the regime M_ft = 0.571 was measured in. Adopted rather than invented so that
# no re-measurement of A2 is needed. Carries no question content.
#
# Do NOT "improve" this wording. An earlier invented alternative ("Answer the spoken
# question about these video frames in a few words") collides with the leak gate on 4 of
# the 20 question strings — "frames" and "these" — which would abort 200 of 1000 rows.
NEUTRAL_INSTRUCTION = "Answer the question concisely based on the visual and audio evidence."

# Byte-pin. assert_no_leak() EXCISES this string before checking, so the string itself is a
# blind spot in the automated gate by design — it is audited by eye once under Gate 2.
# The hash makes that blind spot fixed in size: edit the wording and this fails loudly
# instead of silently widening what the gate ignores.
#
# It also guards the extraction hazard found at Stage 0. grep -cF returns 2 on
# src/evaluate_checkpoint.py: the bare string at line 309, and the include_question variant
# at ~306 of which the bare string is a SUFFIX. Pinning bytes rather than a line number
# means a wrong extract cannot pass quietly.
PARITY_B_SHA256 = "19cdcc49804887546caa625547573347cd09b3303bf7143f365e1f58a991f410"
_h = __import__("hashlib").sha256(NEUTRAL_INSTRUCTION.encode()).hexdigest()
assert _h == PARITY_B_SHA256, (
    f"Parity-B instruction changed: {_h} != {PARITY_B_SHA256}. If this was deliberate, "
    "log it in PRE_REGISTRATION.md §10 and re-verify collisions against all 20 question "
    "strings before updating this hash — the earlier invented wording collided on 4 of 20."
)

# Deliberately MINIMAL. An earlier version also excluded "what", "how", "many",
# "visible", "see", "any" — content-bearing in surgical VQA. That reduced
# "how many polyps are visible" to the single token {polyps}, so a template injecting
# "how many are visible" would have passed. Only true function words here.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "be", "been", "with", "from", "this", "that", "it",
}


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def content_tokens(question_text: str) -> list[str]:
    words = re.findall(r"[a-z]{3,}", question_text.lower())
    return sorted({w for w in words if w not in STOPWORDS})


def question_trigrams(question_text: str) -> list[str]:
    """Contiguous 3-grams including function words. Catches injected phrasing whose
    individual words are each innocuous, which single-token matching cannot see."""
    w = _norm(question_text).split()
    return [" ".join(w[i:i + 3]) for i in range(max(0, len(w) - 2))]


def assert_no_leak(rendered: str, question_text: str, tag: str) -> None:
    """Gate 2. Runs against the RENDERED prompt, not the template source.

    The frozen instruction is EXCISED before checking. It is a known constant audited once
    by eye under Gate 2 (PRE_REGISTRATION §4.3), so matching against it produces only false
    positives: "based" is a content token of "How severe is the fluid-based occlusion?" and
    also appears in "based on the visual and audio evidence". Excising the constant first
    keeps the automated check on the part that can actually vary.
    """
    stripped = rendered.replace(NEUTRAL_INSTRUCTION, " ")
    raw, norm = stripped.lower(), _norm(stripped)

    leaked = [t for t in content_tokens(question_text) if t in raw]
    if leaked:
        raise AssertionError(
            f"PROMPT LEAK [{tag}]: question tokens {leaked} in rendered prompt.\n"
            f"--- rendered ---\n{rendered[:1200]}\n----------------"
        )
    grams = [g for g in question_trigrams(question_text) if g in norm]
    if grams:
        raise AssertionError(
            f"PROMPT LEAK [{tag}]: question trigrams {grams} in rendered prompt.\n"
            f"--- rendered ---\n{rendered[:1200]}\n----------------"
        )


# --------------------------------------------------------------------------------------
# Probe-pair construction from real artifacts
# --------------------------------------------------------------------------------------

def k2_key(video_id: str, qid: str, qtype: str) -> str:
    """Mirror of k2_key() in src/evaluate_checkpoint.py: {video_id}_{id}_{question_type}."""
    return f"{video_id}_{qid}_{qtype}"


def build_pairs(audio_dir: Path, test_manifest: Path, out_pairs: Path,
                seed: int = 0) -> None:
    """Pair clips from DIFFERENT question_types so the two spoken questions genuinely differ.

    Keys are constructed FORWARD from data/test_multivideo.jsonl. Filenames are never
    parsed.

    An earlier version split the stem on "_" and read parts[1] as the id. That is wrong:
    the id is itself `qa_NNNNNN`, so `002-004_qa_002496_scope_outside` yielded
    id="qa" and question_type="002496_scope_outside". Two silent failures followed —
    every clip got a unique pseudo-type (destroying the cross-type guarantee, which then
    held only by luck), and `qtext.get("qa")` never hit, so the leak gate would have been
    fed "003353 lesion site" instead of the real question text. A gate checking the wrong
    tokens passes prompts that genuinely leak.

    Reverse-engineering a filename whose fields can each contain the delimiter is not
    fixable by counting fields. Forward construction has no ambiguity, and it verifies
    audio coverage for free.
    """
    rows = [json.loads(l) for l in open(test_manifest)]
    print(f"manifest: {len(rows)} rows from {test_manifest}")

    recs, missing = [], []
    for r in rows:
        key = k2_key(r["video_id"], r["id"], r["question_type"])
        path = audio_dir / f"{key}.mp3"
        if not path.exists():
            missing.append(key)
        recs.append({"path": path, "qid": r["id"], "qtype": r["question_type"],
                     "question": r["question"]})
    if missing:
        raise SystemExit(
            f"{len(missing)} expected clips absent from {audio_dir}; first 5: "
            f"{missing[:5]}. Pair building requires full audio coverage.")
    print(f"audio coverage: {len(recs)}/{len(rows)} clips resolved")

    by_type: dict[str, list[dict]] = {}
    for rec in recs:
        by_type.setdefault(rec["qtype"], []).append(rec)
    types = sorted(by_type)
    print(f"question_types: {len(types)}")
    if len(types) != 20:
        raise SystemExit(f"expected 20 question_types, got {len(types)}: {types}")

    rng = random.Random(seed)
    pairs = []
    for i in range(N_PAIRS):
        ta, tb = rng.sample(types, 2)
        ra, rb = rng.choice(by_type[ta]), rng.choice(by_type[tb])

        # Invariants asserted, not assumed. The old code claimed cross-type pairing while
        # sampling rows; here each is checked per pair and aborts on violation.
        if ra["qtype"] == rb["qtype"]:
            raise SystemExit(f"pair {i}: same question_type {ra['qtype']}")
        if ra["question"] == rb["question"]:
            raise SystemExit(f"pair {i}: identical question text")
        # 1000 paths over ~180 inodes (20 types x 9 voices, hardlinked onto canonical
        # clips), so different paths can be the SAME FILE -> identical output for a
        # trivially correct reason, failing Gate 3b spuriously.
        if ra["path"].stat().st_ino == rb["path"].stat().st_ino:
            raise SystemExit(
                f"pair {i}: {ra['path'].name} and {rb['path'].name} share inode "
                f"{ra['path'].stat().st_ino} despite differing question_type")

        pairs.append({
            "audio_a": str(ra["path"]), "audio_b": str(rb["path"]),
            "type_a": ra["qtype"], "type_b": rb["qtype"],
            "id_a": ra["qid"], "id_b": rb["qid"],
            "text_a": ra["question"], "text_b": rb["question"],
        })

    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    out_pairs.write_text(json.dumps(pairs, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out_pairs}")
    for p_ in pairs:
        print(f"  {p_['type_a']:<26} | {p_['text_a'][:44]}")
        print(f"  {p_['type_b']:<26} | {p_['text_b'][:44]}\n")


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------

@dataclass
class Adapter:
    name: str
    model: object = None
    processor: object = None
    notes: list[str] = field(default_factory=list)

    def load(self) -> None:
        raise NotImplementedError

    def run(self, images: list[Image.Image], audio_path: str) -> tuple[str, str]:
        """-> (rendered_prompt_or_best_approximation, generated_text)"""
        raise NotImplementedError


class Phi4MMAdapter(Adapter):
    """microsoft/Phi-4-multimodal-instruct, remote-code path.

    Documented vision-speech template:
        <|user|><|image_1|>...<|image_N|><|audio_1|><instruction><|end|><|assistant|>

    TRAP: two code paths with DIFFERENT placeholder schemes — this remote-code path
    (<|image_1|>, <|audio_1|>, transformers==4.48.2) and native Phi4MultimodalForCausalLM.
    Mixing them lets a placeholder pass through as literal text with nothing attached.
    """

    MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
    # Verified live 2026-07-29. 12.9 GB repo; shards 5 + 4.95 + 1.2 = 11.15 GB BF16
    # -> 5.58B params. License MIT.
    REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    ATTN_IMPL = "eager"   # card sanctions eager; avoids the flash-attn wheel entirely

    def load(self) -> None:
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        if not transformers.__version__.startswith("4.48"):
            self.notes.append(
                f"WARNING: transformers=={transformers.__version__}; card states 4.48.2. "
                "Remote code breaks on >=4.50. PIN IS TRANSCRIBED FROM CARD, NOT "
                "GPU-VALIDATED."
            )
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, revision=self.REVISION, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, revision=self.REVISION, device_map="cuda",
            torch_dtype="auto", trust_remote_code=True,
            _attn_implementation=self.ATTN_IMPL,
        ).cuda()
        self.gen_cfg = GenerationConfig.from_pretrained(
            self.MODEL_ID, revision=self.REVISION)
        self.notes.append(f"rev {self.REVISION[:7]} | attn {self.ATTN_IMPL}")

    def run(self, images, audio_path):
        import soundfile as sf
        placeholders = "".join(f"<|image_{i + 1}|>" for i in range(len(images)))
        prompt = f"<|user|>{placeholders}<|audio_1|>{NEUTRAL_INSTRUCTION}<|end|><|assistant|>"
        audio, sr = sf.read(audio_path)
        inputs = self.processor(text=prompt, images=images, audios=[(audio, sr)],
                                return_tensors="pt").to("cuda:0")
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  generation_config=self.gen_cfg,
                                  do_sample=False, num_beams=1)
        out = out[:, inputs["input_ids"].shape[1]:]
        return prompt, self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


class MiniCPMo45Adapter(Adapter):
    """openbmb/MiniCPM-o-4_5 — SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B.

    Whisper encoder projected into an LLM: the same design as SurgViVQA-Audio, built by
    a better-resourced team. That is why it is the informative run.

    Verified live 2026-07-29 from the card:
      - transformers==4.51.0 REQUIRED ("other versions may have compatibility issues")
        -> CANNOT share a venv with Phi-4's 4.48.2
      - attn_implementation: "sdpa" or "flash_attention_2". NOT eager — the MiniCPM
        family does not support eager. sdpa is torch-native, no compiled wheel, so it
        satisfies "skip flash-attn" without violating the model's constraint.
      - audio must be 16 kHz mono float ndarray -> librosa.load(sr=16000, mono=True).
        ffmpeg needed for .mp3 decode.
      - instruct AND thinking modes in one model -> enable_thinking=False is mandatory.
      - init_tts=False: we want text out only, and it saves memory.

    Combined path: the card's "Structured Content Input" documents a single content list
    mixing PIL images, a 16 kHz audio ndarray, and text. Multi-image and video-as-frames
    are both documented. The earlier "one image per turn" claim came from an MLX port,
    not the model. Still probed rather than assumed.

    omni_mode: the card's omni examples interleave frame/audio-segment pairs synced on a
    timeline (TDM). Our case is 8 static frames + one complete spoken question, which is
    the plain chat path. --omni-mode probes the alternative.
    """

    MODEL_ID = "openbmb/MiniCPM-o-4_5"
    # Verified live 2026-07-29: repo 20 GB, 58 commits, 12 contributors, Apache-2.0,
    # 9B params BF16 (~18 GB weights + ~2 GB assets — consistent, no discrepancy).
    REVISION = "44151b3"
    ATTN_IMPL = "sdpa"

    def __init__(self, name: str, omni_mode: bool = False):
        super().__init__(name=name)
        self.omni_mode = omni_mode

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if not transformers.__version__.startswith("4.51"):
            self.notes.append(
                f"WARNING: transformers=={transformers.__version__}; card requires "
                "4.51.0 exactly. PIN IS TRANSCRIBED FROM CARD, NOT GPU-VALIDATED."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, revision=self.REVISION, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, revision=self.REVISION, trust_remote_code=True,
            attn_implementation=self.ATTN_IMPL, torch_dtype=torch.bfloat16,
            init_vision=True, init_audio=True, init_tts=False,
        ).eval().cuda()
        self.notes.append(
            f"rev {self.REVISION} | attn {self.ATTN_IMPL} | init_tts=False | "
            f"omni_mode={self.omni_mode}")

    def run(self, images, audio_path):
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        content = list(images) + [audio, NEUTRAL_INSTRUCTION]
        msgs = [{"role": "user", "content": content}]

        kwargs = dict(
            msgs=msgs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,              # greedy — Q0 verifies this took effect
            use_tts_template=False,
            enable_thinking=False,        # mandatory: instruct+thinking in one model
            use_image_id=False,           # per the card's multi-frame example
            max_slice_nums=1,
        )
        if self.omni_mode:
            kwargs["omni_mode"] = True
        text = self.model.chat(**kwargs)

        # Approximation of the rendered prompt. MiniCPM-o assembles the true prompt
        # inside chat(), so we assert on (a) every string we put in, and (b) the
        # tokenizer's rendered template over a text-only skeleton. Stated as an
        # approximation rather than pretended to be the real thing; Gate 3 below is
        # the model-agnostic protection.
        skeleton = [{"role": "user", "content": NEUTRAL_INSTRUCTION}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                skeleton, tokenize=False, add_generation_prompt=True)
        except Exception as e:                                  # noqa: BLE001
            rendered = NEUTRAL_INSTRUCTION
            self.notes.append(f"chat-template render unavailable ({e}); "
                              "Gate 2 covers supplied strings only")
        rendered += "\n[APPROX: string content supplied] " + NEUTRAL_INSTRUCTION
        return rendered, (text or "").strip()


def make_adapter(name: str, omni_mode: bool) -> Adapter:
    if name == "phi4mm":
        return Phi4MMAdapter(name=name)
    if name == "minicpmo45":
        return MiniCPMo45Adapter(name=name, omni_mode=omni_mode)
    raise SystemExit(f"unknown model {name}")


# --------------------------------------------------------------------------------------

def load_frames(frame_dir: Path) -> list[Image.Image]:
    """Frames are .jpg on this project (e.g. 002-004_29052.jpg). Accept png too.

    NOTE: 002-004/ holds 31,818 files. 6,685 is the number of DISTINCT frames referenced
    by the 1000 test rows (8,000 references, 1.20 refs per frame) — not a directory size.
    Do not assert on directory count.
    """
    paths = sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)
    if len(paths) < N_FRAMES:
        raise SystemExit(f"need {N_FRAMES} frames in {frame_dir}, found {len(paths)}")
    if len(paths) > N_FRAMES:
        print(f"  {len(paths)} frames in {frame_dir}; taking first {N_FRAMES} by sorted "
              f"name (one fixed set throughout — Gate 3 tests audio, not correctness)")
    return [Image.open(p).convert("RGB") for p in paths[:N_FRAMES]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pairs", action="store_true")
    ap.add_argument("--audio-dir", type=Path, default=Path("data/audio/test"))
    ap.add_argument("--test-manifest", type=Path,
                    default=Path("data/test_multivideo.jsonl"),
                    help="REQUIRED for pair building. There is no manifest-less fallback: "
                         "the old degraded mode fed question_type strings to the leak gate "
                         "and could activate silently even with a manifest present.")
    ap.add_argument("--out-pairs", type=Path, default=Path("data/probe_pairs.json"))
    ap.add_argument("--model", choices=["phi4mm", "minicpmo45"])
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--frames", type=Path, default=Path("dataset/frames/002-004"),
                    help="frames root is machine-specific: dataset/frames locally, "
                         "data/frames on the pod after rsync. Always pass explicitly.")
    ap.add_argument("--omni-mode", action="store_true",
                    help="minicpmo45 only: probe the interleaved omni path instead")
    ap.add_argument("--out", type=Path, default=Path("artifacts/probe"))
    args = ap.parse_args()

    if args.build_pairs:
        build_pairs(args.audio_dir, args.test_manifest, args.out_pairs)
        return 0
    if not args.model or not args.pairs:
        raise SystemExit("need --model and --pairs (or --build-pairs)")

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(args.pairs.read_text())
    images = load_frames(args.frames)

    adapter = make_adapter(args.model, args.omni_mode)
    print(f"[{args.model}] loading...")
    adapter.load()
    for n in adapter.notes:
        print(f"  ! {n}")

    # ---- Q0: determinism control -----------------------------------------------------
    # Without this, "outputs differ" cannot be attributed to the audio. Both model cards
    # use do_sample=True in their examples; we forced greedy and now verify it landed.
    _, det_a = adapter.run(images, pairs[0]["audio_a"])
    _, det_b = adapter.run(images, pairs[0]["audio_a"])
    deterministic = det_a.strip().lower() == det_b.strip().lower()
    print(f"\n  [Q0] same audio twice: "
          f"{'DETERMINISTIC' if deterministic else 'NONDETERMINISTIC'}")
    if not deterministic:
        print("  FAIL — decoding is not greedy. The audio-sensitivity result below")
        print("  would measure sampling noise, not the audio pathway. Fix do_sample /")
        print("  temperature / seed before interpreting anything.")
        print(f"    run1={det_a[:60]!r}\n    run2={det_b[:60]!r}")
        (args.out / f"{args.model}_probe.json").write_text(json.dumps(
            {"model": args.model, "deterministic": False, "passed": False}, indent=2))
        return 1

    # ---- Q1-Q3 -----------------------------------------------------------------------
    n_differ, rows = 0, []
    for i, pair in enumerate(pairs):
        rendered_a, out_a = adapter.run(images, pair["audio_a"])
        rendered_b, out_b = adapter.run(images, pair["audio_b"])

        assert_no_leak(rendered_a, pair["text_a"], f"{args.model}/pair{i}/a")
        assert_no_leak(rendered_b, pair["text_b"], f"{args.model}/pair{i}/b")

        differs = out_a.strip().lower() != out_b.strip().lower()
        n_differ += differs
        rows.append({"pair": i, "type_a": pair.get("type_a"), "type_b": pair.get("type_b"),
                     "out_a": out_a, "out_b": out_b, "differs": differs})
        print(f"  pair {i}: {'DIFFER' if differs else 'IDENTICAL':9} "
              f"| a={out_a[:40]!r} | b={out_b[:40]!r}")
        if i == 0:
            (args.out / f"{args.model}_rendered_prompt.txt").write_text(rendered_a)

    rate = n_differ / len(pairs)
    passed = rate >= SENSITIVITY_THRESHOLD

    (args.out / f"{args.model}_probe.json").write_text(json.dumps({
        "model": args.model, "revision": getattr(adapter, "REVISION", None),
        "omni_mode": args.omni_mode, "deterministic": True,
        "sensitivity": rate, "passed": passed, "rows": rows}, indent=2))

    print(f"\n[{args.model}] deterministic: yes | audio sensitivity: "
          f"{n_differ}/{len(pairs)} = {rate:.0%}")
    if passed:
        print("  PASS — audio pathway is live. Cleared for full evaluation.")
        print("  Next: read artifacts/probe/*_rendered_prompt.txt by eye (Gate 2).")
        return 0

    print("  FAIL — identical outputs across different questions, with decoding")
    print("  confirmed deterministic. The audio is NOT reaching the model.")
    print("  Any accuracy number from this configuration measures the image prior.")
    print("  Check: code path / placeholder scheme, transformers version, processor")
    print("  dropping the audio kwarg, sample rate (16 kHz mono?), init_audio, adapter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
