"""
Task 1: Generate dual-source phoneme mapping config.

Design principles:
  - Pure consonants (en_K, en_S, etc.): 1.0 weight — exact match.
  - Vowels: English 0.8 + Chinese 0.2 — close match, small Chinese correction.
  - Composites (jp_r, jp_ts, jp_u, jp_ry): lower init weight or skip entirely
    when no source phoneme is close enough (cos_sim < 0.6).
  - jp_pau/jp_cl: pause mean (silence, no phonetic content needed).

Usage:
    python train/lora_jp/phoneme_mapping.py
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'jp_phoneme_mapping.json')

MAPPING = {
    # ── Special phonemes (silence) ────────────────────────────────────
    "jp_pau": {
        "strategy": "pause_mean",
        "pause_sources": ["<SP>", "<AP>"],
        "noise_std": 0.01
    },
    "jp_cl": {
        "strategy": "pause_mean",
        "pause_sources": ["<SP>"],
        "noise_std": 0.01
    },

    # ── Vowels (English primary + Chinese secondary) ──────────────────
    "jp_a": {
        "sources": [
            {"phone": "en_AA1", "weight": 0.8},
            {"phone": "zh_a1",  "weight": 0.2}
        ]
    },
    "jp_i": {
        "sources": [
            {"phone": "en_IY1", "weight": 0.8},
            {"phone": "zh_yi1", "weight": 0.2}
        ]
    },
    # JP /ɯ/ = unrounded back vowel — no close English match
    # Use Chinese wu (closest) as primary, English UH as secondary
    "jp_u": {
        "sources": [
            {"phone": "zh_wu1", "weight": 0.6},
            {"phone": "en_UH1", "weight": 0.4}
        ],
        "init_weight": 0.5,
        "note": "JP u is unrounded; weak match, reduced init weight"
    },
    "jp_e": {
        "sources": [
            {"phone": "en_EH1", "weight": 0.8},
            {"phone": "zh_e1",  "weight": 0.2}
        ]
    },
    "jp_o": {
        "sources": [
            {"phone": "en_OW1", "weight": 0.8},
            {"phone": "zh_o1",  "weight": 0.2}
        ]
    },

    # ── Pure consonants (English exact match) ─────────────────────────
    "jp_k": {"sources": [{"phone": "en_K", "weight": 1.0}]},
    "jp_t": {"sources": [{"phone": "en_T", "weight": 1.0}]},
    "jp_p": {"sources": [{"phone": "en_P", "weight": 1.0}]},
    "jp_g": {"sources": [{"phone": "en_G", "weight": 1.0}]},
    "jp_d": {"sources": [{"phone": "en_D", "weight": 1.0}]},
    "jp_b": {"sources": [{"phone": "en_B", "weight": 1.0}]},
    "jp_s": {"sources": [{"phone": "en_S", "weight": 1.0}]},
    "jp_z": {"sources": [{"phone": "en_Z", "weight": 1.0}]},
    "jp_h": {"sources": [{"phone": "en_HH", "weight": 1.0}]},
    "jp_n": {"sources": [{"phone": "en_N", "weight": 1.0}]},
    "jp_m": {"sources": [{"phone": "en_M", "weight": 1.0}]},
    "jp_w": {"sources": [{"phone": "en_W", "weight": 1.0}]},
    "jp_y": {"sources": [{"phone": "en_Y", "weight": 1.0}]},
    "jp_ch": {"sources": [{"phone": "en_CH", "weight": 1.0}]},
    "jp_j": {"sources": [{"phone": "en_JH", "weight": 1.0}]},
    "jp_sh": {"sources": [{"phone": "en_SH", "weight": 1.0}]},

    # ── Fricatives with known phonetic gap ────────────────────────────
    # JP /ɸ/ (bilabial) vs EN /f/ (labiodental) — close enough in embedding space
    "jp_f": {"sources": [{"phone": "en_F", "weight": 1.0}]},

    # ── Composites: reduced init weight or skip ───────────────────────
    # JP /ɾ/ = alveolar tap — no close English match
    # en_L (lateral) is closest at 0.72, but tap ≠ lateral
    # Use en_L only, reduce weight so model can learn from data
    "jp_r": {
        "sources": [
            {"phone": "en_L", "weight": 1.0}
        ],
        "init_weight": 0.5,
        "note": "JP r is a tap; en_L is closest but not close. Reduced init weight."
    },
    # JP /ts/ = alveolar affricate — en_T+en_S blend at 0.72/0.68
    # Use en_T as primary (closer), reduced weight
    "jp_ts": {
        "sources": [
            {"phone": "en_T", "weight": 0.6},
            {"phone": "en_S", "weight": 0.4}
        ],
        "init_weight": 0.5,
        "note": "JP ts is an affricate; blend doesn't match well. Reduced init weight."
    },

    # ── Palatalized consonants (C + y glide) ──────────────────────────
    "jp_ky": {
        "sources": [
            {"phone": "en_K", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
    "jp_gy": {
        "sources": [
            {"phone": "en_G", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
    "jp_ny": {
        "sources": [
            {"phone": "en_N", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
    "jp_hy": {
        "sources": [
            {"phone": "en_HH", "weight": 0.6},
            {"phone": "en_Y",  "weight": 0.4}
        ]
    },
    "jp_my": {
        "sources": [
            {"phone": "en_M", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
    # JP /ɾʲ/ = palatalized tap — weakest composite (best match en_Y at 0.64)
    # Skip weighted init, let fallback handle it
    "jp_ry": {
        "sources": [
            {"phone": "en_Y", "weight": 1.0}
        ],
        "init_weight": 0.3,
        "note": "JP ry is palatalized tap; no good source. Minimal init from en_Y."
    },
    "jp_py": {
        "sources": [
            {"phone": "en_P", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
    "jp_by": {
        "sources": [
            {"phone": "en_B", "weight": 0.6},
            {"phone": "en_Y", "weight": 0.4}
        ]
    },
}


def validate_mapping():
    """Check all weights sum to 1.0 and return any issues."""
    issues = []
    for jp_ph, entry in MAPPING.items():
        if entry.get("strategy") in ("pause_mean", "fallback_mean"):
            continue
        sources = entry.get("sources", [])
        total = sum(s["weight"] for s in sources)
        if abs(total - 1.0) > 1e-6:
            issues.append(f"{jp_ph}: weights sum to {total:.4f}, expected 1.0")
    return issues


def main():
    issues = validate_mapping()
    if issues:
        print("WARNING: Mapping issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All weight sums validated OK.")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(MAPPING, f, ensure_ascii=False, indent=2)

    n_weighted = sum(1 for v in MAPPING.values() if v.get("strategy") != "pause_mean")
    n_pause = sum(1 for v in MAPPING.values() if v.get("strategy") == "pause_mean")
    n_reduced = sum(1 for v in MAPPING.values() if v.get("init_weight") is not None)
    print(f"\nMapping saved to: {OUTPUT_PATH}")
    print(f"  Total JP phonemes: {len(MAPPING)}")
    print(f"  Full init: {n_weighted - n_reduced}")
    print(f"  Reduced init: {n_reduced}")
    print(f"  Pause mean: {n_pause}")


if __name__ == '__main__':
    main()
