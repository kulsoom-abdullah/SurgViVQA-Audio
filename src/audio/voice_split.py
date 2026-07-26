"""
Speaker-disjoint train/eval/test voice split for held-out TTS generation.

The split is reproducible from data, not from memory: it parses the committed
roster at data/voices/en_voices_raw.txt (a verbatim capture of
`edge-tts --list-voices | grep "en-"`) and deals voices with a fixed seed.

Speaker-disjointness is the point. A voice that appears in train must never
appear in eval or test, otherwise the "held-out speaker" claim is false. Two
exclusion lists below protect that property; both are explicit, named, and
applied before any dealing.

Usage:
    python -m src.audio.voice_split --print
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

SEED = 20260725

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "data" / "voices" / "en_voices_raw.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "voices" / "voice_split.json"

# Multilingual/expressive variants that are the SAME underlying voice persona as a
# base variant already in the roster. Keeping both would put one speaker on both
# sides of the split, which breaks speaker-disjointness.
COLLAPSED_PERSONAS = {
    "en-US-AndrewMultilingualNeural",  # same persona as en-US-AndrewNeural
    "en-US-AvaMultilingualNeural",     # same persona as en-US-AvaNeural
    "en-US-BrianMultilingualNeural",   # same persona as en-US-BrianNeural
    "en-US-EmmaMultilingualNeural",    # same persona as en-US-EmmaNeural
    "en-IN-NeerjaExpressiveNeural",    # same persona as en-IN-NeerjaNeural
}

# Dropped on quality grounds rather than duplication.
EXCLUDED_VOICES = {
    "en-US-AnaNeural",  # cartoon/child voice, poor face validity for surgical questions
}

# en-AU-WilliamMultilingualNeural is deliberately KEPT: the roster has no
# en-AU-WilliamNeural base twin, so it is not a collapsed persona.

EXPECTED_POOL_SIZE = 41

LOCALE_TO_REGION = {
    "US": "N.America", "CA": "N.America",
    "GB": "BritishIsles", "IE": "BritishIsles",
    "AU": "Oceania", "NZ": "Oceania",
    "IN": "Asia", "HK": "Asia", "SG": "Asia", "PH": "Asia",
    "KE": "Africa", "NG": "Africa", "TZ": "Africa", "ZA": "Africa",
}

SPLITS = ("train", "eval", "test")

# region -> (train, eval, test)
QUOTAS = {
    "N.America": (8, 3, 3),
    "BritishIsles": (5, 1, 1),
    "Oceania": (2, 1, 1),
    "Asia": (4, 2, 2),
    "Africa": (4, 2, 2),
}

# Voices forced into a specific split. Applied BEFORE the deal, decrementing the
# corresponding region/split quota, so the dealt result is never post-edited.
PINNED = {"en-IN-NeerjaNeural": "test"}


@dataclass(frozen=True)
class Voice:
    short_name: str
    gender: str
    locale: str

    @property
    def region(self) -> str:
        return LOCALE_TO_REGION[self.locale]


def parse_roster(path: Path = ROSTER_PATH) -> list[Voice]:
    """Parse the committed roster into Voice records.

    Roster lines are whitespace-aligned columns: Name, Gender, Categories,
    Personalities. Only the first two are load-bearing here.
    """
    if not path.exists():
        raise FileNotFoundError(f"Voice roster not found: {path}")

    voices = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: cannot parse voice line: {raw!r}")
        short_name, gender = parts[0], parts[1]
        segments = short_name.split("-")
        if len(segments) < 3:
            raise ValueError(f"{path}:{lineno}: unexpected voice name: {short_name!r}")
        locale = segments[1]
        if locale not in LOCALE_TO_REGION:
            raise ValueError(f"{path}:{lineno}: locale {locale!r} has no region mapping")
        voices.append(Voice(short_name=short_name, gender=gender, locale=locale))
    return voices


def build_pool(voices: list[Voice]) -> list[Voice]:
    """Apply both exclusion lists and assert the surviving pool size."""
    dropped = COLLAPSED_PERSONAS | EXCLUDED_VOICES
    pool = [v for v in voices if v.short_name not in dropped]

    known = {v.short_name for v in voices}
    missing = dropped - known
    if missing:
        raise AssertionError(f"Exclusion list names voices absent from roster: {sorted(missing)}")

    if len(pool) != EXPECTED_POOL_SIZE:
        raise AssertionError(
            f"Surviving pool is {len(pool)}, expected {EXPECTED_POOL_SIZE}. "
            f"Roster={len(voices)}, excluded={len(dropped)}."
        )
    return pool


def _alternate_by_gender(voices: list[Voice], rng: random.Random) -> list[Voice]:
    """Interleave a region's voices M/F/M/F so sequential dealing spreads genders.

    Starts with the larger gender group so the alternation does not run out early;
    a tie is broken by the seeded RNG, keeping the result reproducible.
    """
    females = sorted([v for v in voices if v.gender == "Female"], key=lambda v: v.short_name)
    males = sorted([v for v in voices if v.gender == "Male"], key=lambda v: v.short_name)
    rng.shuffle(females)
    rng.shuffle(males)

    if len(females) > len(males):
        first, second = females, males
    elif len(males) > len(females):
        first, second = males, females
    else:
        first, second = (females, males) if rng.random() < 0.5 else (males, females)

    out = []
    for i in range(max(len(first), len(second))):
        if i < len(first):
            out.append(first[i])
        if i < len(second):
            out.append(second[i])
    return out


def deal(pool: list[Voice]) -> dict[str, list[Voice]]:
    """Deal the pool into speaker-disjoint splits, honouring pins and quotas."""
    assigned: dict[str, list[Voice]] = {s: [] for s in SPLITS}
    by_name = {v.short_name: v for v in pool}

    # Working copy of the quotas; pins decrement these before any dealing happens.
    quotas = {region: dict(zip(SPLITS, counts)) for region, counts in QUOTAS.items()}

    pinned_names = set()
    for name, split in PINNED.items():
        if split not in SPLITS:
            raise AssertionError(f"PINNED voice {name} targets unknown split {split!r}")
        if name not in by_name:
            raise AssertionError(f"PINNED voice {name} is not in the surviving pool")
        voice = by_name[name]
        if quotas[voice.region][split] < 1:
            raise AssertionError(
                f"PINNED voice {name} cannot fit: {voice.region}/{split} quota is exhausted"
            )
        quotas[voice.region][split] -= 1
        assigned[split].append(voice)
        pinned_names.add(name)

    for region in sorted(QUOTAS):
        remaining = [v for v in pool if v.region == region and v.short_name not in pinned_names]
        need = sum(quotas[region].values())
        if len(remaining) != need:
            raise AssertionError(
                f"{region}: {len(remaining)} voices to deal but quotas total {need}"
            )
        ordered = _alternate_by_gender(remaining, random.Random(f"{SEED}:{region}"))
        cursor = 0
        for split in SPLITS:
            take = quotas[region][split]
            assigned[split].extend(ordered[cursor:cursor + take])
            cursor += take

    return {s: sorted(v, key=lambda x: x.short_name) for s, v in assigned.items()}


def validate(assigned: dict[str, list[Voice]]) -> None:
    """Abort on any violation of the split's defining properties."""
    expected_totals = {"train": 23, "eval": 9, "test": 9}
    for split, want in expected_totals.items():
        got = len(assigned[split])
        if got != want:
            raise AssertionError(f"{split}: {got} voices, expected {want}")

    seen: dict[str, str] = {}
    for split, voices in assigned.items():
        for v in voices:
            if v.short_name in seen:
                raise AssertionError(
                    f"Speaker-disjointness violated: {v.short_name} in both "
                    f"{seen[v.short_name]} and {split}"
                )
            seen[v.short_name] = split

    for split, voices in assigned.items():
        regions = {v.region for v in voices}
        missing = set(QUOTAS) - regions
        if missing:
            raise AssertionError(f"{split}: no voice from region(s) {sorted(missing)}")
        genders = {v.gender for v in voices}
        if len(genders) < 2:
            raise AssertionError(f"{split}: only gender(s) {sorted(genders)} present")

    for name, split in PINNED.items():
        if name not in {v.short_name for v in assigned[split]}:
            raise AssertionError(f"PINNED voice {name} did not land in {split}")


def build() -> dict[str, list[Voice]]:
    return deal(build_pool(parse_roster()))


def write_split(assigned: dict[str, list[Voice]], path: Path = OUTPUT_PATH) -> dict:
    payload = {
        "seed": SEED,
        "roster_path": str(ROSTER_PATH.relative_to(REPO_ROOT)),
        "pool_size": sum(len(v) for v in assigned.values()),
        "collapsed_personas": sorted(COLLAPSED_PERSONAS),
        "excluded_voices": sorted(EXCLUDED_VOICES),
        "pinned": dict(PINNED),
        "quotas": {r: dict(zip(SPLITS, c)) for r, c in QUOTAS.items()},
        "splits": {s: [v.short_name for v in assigned[s]] for s in SPLITS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def print_table(assigned: dict[str, list[Voice]]) -> None:
    print(f"SEED={SEED}   roster={ROSTER_PATH}")
    print(f"pool after exclusions: {sum(len(v) for v in assigned.values())}")
    print()
    header = f"{'region':<14}" + "".join(f"{s:>22}" for s in SPLITS)
    print(header)
    print("-" * len(header))
    for region in sorted(QUOTAS):
        cells = []
        for split in SPLITS:
            vs = [v for v in assigned[split] if v.region == region]
            f = sum(1 for v in vs if v.gender == "Female")
            m = len(vs) - f
            cells.append(f"{len(vs)} ({f}F/{m}M)")
        print(f"{region:<14}" + "".join(f"{c:>22}" for c in cells))
    print("-" * len(header))
    totals = []
    for split in SPLITS:
        vs = assigned[split]
        f = sum(1 for v in vs if v.gender == "Female")
        totals.append(f"{len(vs)} ({f}F/{len(vs)-f}M)")
    print(f"{'TOTAL':<14}" + "".join(f"{t:>22}" for t in totals))
    print()
    for split in SPLITS:
        print(f"{split} ({len(assigned[split])}):")
        for v in assigned[split]:
            pin = "  <- PINNED" if v.short_name in PINNED else ""
            print(f"    {v.short_name:<34} {v.gender:<7} {v.region}{pin}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the speaker-disjoint voice split")
    parser.add_argument("--print", action="store_true", dest="do_print",
                        help="Print the split table")
    parser.add_argument("--write", action="store_true",
                        help=f"Write {OUTPUT_PATH.name}")
    args = parser.parse_args()

    assigned = build()
    validate(assigned)

    if args.write:
        write_split(assigned)
        print(f"wrote {OUTPUT_PATH}")
    if args.do_print or not args.write:
        print_table(assigned)


if __name__ == "__main__":
    main()
