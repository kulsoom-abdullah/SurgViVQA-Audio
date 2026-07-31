#!/usr/bin/env python3
"""
omni_adapters.py — shared model adapters and the prompt-leak gate.

Extracted from verify_combined_path.py so the Gate 3 probe and the Stage 5 run harness
generate through the SAME code. If the probe verifies the audio and image pathways via
adapter A and the run measures via adapter B, Gate 3 verified code that is not the code
under measurement — this project's original failure restated: verify one configuration,
publish a number from another.

Three properties follow from the extraction:
  1. Gate 3 verifies the exact generate path the 1000-row run uses.
  2. The Parity-B hash assertion fires once, at shared-module import, so neither tool can
     drift from the frozen instruction independently.
  3. assert_no_leak is one implementation, so Stage 5's per-row assertion is the same
     check the probe passed, not a reimplementation of it.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# Reuse latency_bench's sync() so there is one timing primitive, not two. It is imported
# rather than redefined where possible; the fallback exists because latency_bench imports
# librosa at module level (latency_bench.py:50) and venv-phi4 has soundfile, not librosa.
# The fallback body is identical, so which branch runs cannot change a measurement.
try:                                                        # pragma: no cover
    from latency_bench import sync
    SYNC_SOURCE = "latency_bench.sync"
except Exception:                                           # noqa: BLE001
    def sync() -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    SYNC_SOURCE = "fallback"

#: Which of the two branches above is live, recorded in the probe artifact and the Stage 5
#: manifest. It is fully determined by the venv — latency_bench's module-level
#: `import librosa` means venv-phi4 ALWAYS takes the fallback and venv-minicpm always takes
#: the import — so it cannot vary run to run. It is recorded anyway because "both bodies are
#: identical" is a claim about today's source that a reader should be able to check rather
#: than trust, and it would stop being true silently.
_ = SYNC_SOURCE

#: sha256 of THIS FILE as it sits on disk at import. Emitted into BOTH
#: artifacts/probe/{model}_probe.json and the Stage 5 run manifest. Equal hashes are the
#: proof that Gate 3 verified the same generate path Stage 5 measured; unequal hashes mean
#: the probe result does not transfer to the run, however plausible the numbers look. That
#: is precisely the failure this project already made once — verify one configuration,
#: publish a number from another — reduced here to a string comparison.
ADAPTERS_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

N_FRAMES = 8
MAX_NEW_TOKENS = 32
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

# Byte-pin. assert_no_leak() EXCISES the active instruction before checking, so that string
# is a blind spot in the automated gate by design — it is audited by eye once under Gate 2.
# The hash makes that blind spot fixed in size: edit the wording and this fails loudly
# instead of silently widening what the gate ignores.
#
# It also guards the extraction hazard found at Stage 0. grep -cF returns 2 on
# src/evaluate_checkpoint.py: the bare string at line 309, and the include_question variant
# at ~306 of which the bare string is a SUFFIX. Pinning bytes rather than a line number
# means a wrong extract cannot pass quietly.
PARITY_B_SHA256 = "19cdcc49804887546caa625547573347cd09b3303bf7143f365e1f58a991f410"
_h = hashlib.sha256(NEUTRAL_INSTRUCTION.encode()).hexdigest()
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


def assert_no_leak(rendered: str, question_text: str, tag: str, instruction: str) -> None:
    """Gate 2. Runs against the RENDERED prompt, not the template source.

    The ACTIVE instruction is EXCISED before checking. It is a known constant audited once
    by eye under Gate 2 (PRE_REGISTRATION §4.3), so matching against it produces only false
    positives: "based" is a content token of "How severe is the fluid-based occlusion?" and
    also appears in "based on the visual and audio evidence". Excising it first keeps the
    automated check on the part that can actually vary.

    `instruction` is REQUIRED and has no default, deliberately: a defaulted argument is how
    the manifest-less fallback silently did the wrong thing at Stage 2c. An explicit
    argument cannot. The study now runs one condition, so today the caller always passes
    NEUTRAL_INSTRUCTION — but the argument stays required so a second condition cannot be
    added without confronting this.

    DESIGN ASSUMPTION, and its limit:
    Any condition that adds text to the prompt must pass that exact text as `instruction`.
    The gate cannot distinguish supplied vocabulary from leaked question content;
    correctness depends on the excision matching what was actually sent.

    The cut V1 condition demonstrated this concretely. V1 appended the 26 train answers to
    the instruction, and excising only the V0 constant left that vocabulary in the stripped
    text — where it collided with words appearing in the questions themselves:

        ['nbi']                      <- Are these frames captured using NBI mode?
        ['catheter']                 <- Is the instrument on screen a catheter?
        ['advancing','withdrawing']  <- Is the scope primarily advancing or withdrawing...?

    3 of 20 question types, 150 of 1000 rows, aborting on a false positive. Excising the
    active instruction gave 0 collisions across all 20 — verified before V1 was cut.
    """
    stripped = rendered.replace(instruction, " ")
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
# Adapters
# --------------------------------------------------------------------------------------

@dataclass
class Adapter:
    name: str
    model: object = None
    processor: object = None
    notes: list[str] = field(default_factory=list)

    #: What sits inside the sync()-bracketed timed region. NOT the same across adapters,
    #: so it is recorded in the run manifest rather than left implicit — the two numbers
    #: are not interchangeable and comparing them across models would be wrong.
    TIMING_BRACKET = "unspecified"

    def load(self) -> None:
        raise NotImplementedError

    def run(self, images: list[Image.Image], audio_path: str) -> tuple[str, str, float]:
        """-> (rendered_prompt_or_best_approximation, generated_text, latency_s)

        Latency is measured INSIDE run(), around the generate call, with sync() either
        side. Timing from the caller instead would fold in frame decoding and audio load,
        which are harness costs rather than model costs, and would leave the measurement
        in different code from the generation — the split §5.5 exists to avoid.
        """
        raise NotImplementedError


class Phi4MMAdapter(Adapter):
    """microsoft/Phi-4-multimodal-instruct, remote-code path.

    Documented vision-speech template:
        <|user|><|image_1|>...<|image_N|><|audio_1|><instruction><|end|><|assistant|>

    TRAP: two code paths with DIFFERENT placeholder schemes — this remote-code path
    (<|image_1|>, <|audio_1|>, transformers==4.48.2) and native Phi4MultimodalForCausalLM.
    Mixing them lets a placeholder pass through as literal text with nothing attached.

    Remote-code dependencies beyond PRE_REGISTRATION §11.1: torchvision==0.21.0 (the
    torch 2.6.0 partner) and backoff. Both blocked AutoProcessor.from_pretrained at
    check_imports before any weight loaded. flash_attn is imported only behind
    is_flash_attn_2_available() in vision_siglip_navit.py:334, so its absence is fine and
    the "no flash-attn" constraint is satisfied without patching anything.
    """

    MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
    # Verified live 2026-07-29. 12.9 GB repo; shards 5 + 4.95 + 1.2 = 11.15 GB BF16
    # -> 5.58B params. License MIT.
    REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    ATTN_IMPL = "eager"   # card sanctions eager; avoids the flash-attn wheel entirely
    # Tightest available bracket: the processor call is separate and observable, so only
    # generation is timed. Preprocessing is EXCLUDED.
    TIMING_BRACKET = "model.generate() only; processor/preprocessing excluded"

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
        sync()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  generation_config=self.gen_cfg,
                                  do_sample=False, num_beams=1)
        sync()
        latency_s = time.perf_counter() - t0
        out = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return prompt, text, latency_s


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
    # chat() preprocesses internally — images and the audio ndarray are converted inside
    # the call — so the whole call is the tightest bracket available. This INCLUDES
    # preprocessing, unlike Phi-4's. The two latencies are therefore not comparable to each
    # other; the manifest records which is which so nobody sets them side by side.
    TIMING_BRACKET = "model.chat() whole call; includes internal preprocessing"

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
        sync()
        t0 = time.perf_counter()
        text = self.model.chat(**kwargs)
        sync()
        latency_s = time.perf_counter() - t0

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
        return rendered, (text or "").strip(), latency_s


def make_adapter(name: str, omni_mode: bool) -> Adapter:
    if name == "phi4mm":
        return Phi4MMAdapter(name=name)
    if name == "minicpmo45":
        return MiniCPMo45Adapter(name=name, omni_mode=omni_mode)
    raise SystemExit(f"unknown model {name}")


# --------------------------------------------------------------------------------------
# Frame loading
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


def k2_key(video_id: str, qid: str, qtype: str) -> str:
    """The audio-clip key: {video_id}_{id}_{question_type}. Mirrors k2_key() in
    src/evaluate_checkpoint.py, which is deliberately NOT modified — it is the file that
    produced M_ft = 0.571 and is not under change in this study.

    Bare `id` is not unique corpus-wide; it repeats across videos and splits carrying
    different questions. K2 is collision-free at 3700/3700.

    Three explicit arguments, no row-dict overload. Two shapes would be two ways to build
    the same key, which is the shape of the Stage 2c bug: a key built one way, consumed by
    code assuming the other, failing silently.
    """
    return f"{video_id}_{qid}_{qtype}"


def load_frames_for_row(frames_root: Path, row: dict) -> list[Image.Image]:
    """Load the 8 frames a specific manifest row references."""
    paths = [frames_root / row["video_id"] / f"{f}.jpg" for f in row["frames"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing frames for {row['id']}: {missing[:3]}")
    return [Image.open(p).convert("RGB") for p in paths]
