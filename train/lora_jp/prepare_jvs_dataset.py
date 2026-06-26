"""
Prepare JVS-MuSiC corpus for Japanese LoRA fine-tuning.

JVS-MuSiC provides ONLY audio (no .lab, no .mid, no lyrics).
This script:
  1. Uses ROSVOT NoteTranscriber to transcribe each wav -> note timing + pitch
  2. Uses known public-domain lyrics (jvs_lyrics.json) + jp_g2p.py for phonemes
  3. Aligns phoneme syllables to ROSVOT notes by count (1 syllable per note,
     matching inference-side preprocessing.js behavior)
  4. Outputs metadata.json + f0.npy compatible with dataset.py

Output format is identical to PJS prepare_dataset.py:
  - One segment per wav with merged 'jp_t-a' style phoneme tokens
  - note_dur (seconds), note_pitch (MIDI), note_type (0=note, 1=SP)
  - f0.npy extracted via RMVPE

Usage:
    python train/lora_jp/prepare_jvs_dataset.py --max-speakers 2 --dry-run
    python train/lora_jp/prepare_jvs_dataset.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOULX_ROOT = Path(__file__).resolve().parents[2]  # SoulX-Singer/
LORA_JP_DIR = Path(__file__).resolve().parent     # SoulX-Singer/train/lora_jp/
JVS_ROOT = SOULX_ROOT / "datasets" / "jvs_music_ver1"
LYRICS_FILE = LORA_JP_DIR / "jvs_lyrics.json"
OUTPUT_DIR = LORA_JP_DIR / "dataset_jvs"

PRETRAINED = SOULX_ROOT / "pretrained_models" / "SoulX-Singer-Preprocess"
ROSVOT_DIR = PRETRAINED / "rosvot"
RMVPE_PATH = PRETRAINED / "rmvpe" / "rmvpe.pt"

# Add SoulX-Singer to sys.path so we can import preprocess modules
if str(SOULX_ROOT) not in sys.path:
    sys.path.insert(0, str(SOULX_ROOT))
# Add lora_jp dir for jp_g2p import
if str(LORA_JP_DIR) not in sys.path:
    sys.path.insert(0, str(LORA_JP_DIR))

from jp_g2p import japanese_g2p, syllabify_jp, lyrics_to_jp_tokens


# ---------------------------------------------------------------------------
# Singer info parsing
# ---------------------------------------------------------------------------
def load_singer_info() -> List[Dict[str, str]]:
    """Parse singer_info.txt into a list of speaker dicts.

    Format (space-separated columns):
      singer gender key_group tempo key_singer unique_song_name
    e.g. 'jvs001 m mid 92 7 はとぽっぽ'

    The common song 'katatsumuri' is sung by all speakers as 'song_common'.
    """
    info_path = JVS_ROOT / "singer_info.txt"
    speakers = []
    with open(info_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header line (first line: 'singer gender key_group tempo key_singer unique_song_name')
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        speaker_id = parts[0].strip()
        # unique_song_name is the last field (index 5)
        unique_song = parts[5].strip()
        speakers.append({
            "speaker_id": speaker_id,
            "unique_song": unique_song,
        })
    return speakers


def get_wav_path(speaker_id: str, song_type: str) -> Optional[Path]:
    """Get the wav file path for a speaker and song type.

    JVS-MuSiC layout:
      song_common/wav/: raw.wav, modified.wav, modified_grouped.wav
      song_unique/wav/: raw.wav only (no pitch-corrected versions)

    Preference order: modified_grouped.wav > modified.wav > raw.wav
    """
    wav_dir = JVS_ROOT / speaker_id / f"song_{song_type}" / "wav"
    for name in ("modified_grouped.wav", "modified.wav", "raw.wav"):
        wav = wav_dir / name
        if wav.exists():
            return wav
    return None


# ---------------------------------------------------------------------------
# Phoneme-to-note alignment
# ---------------------------------------------------------------------------
def align_syllables_to_notes(
    syllable_tokens: List[str],
    note_durs: List[float],
    note_pitches: List[int],
) -> Tuple[List[str], List[float], List[int], List[int]]:
    """Align phoneme syllables to ROSVOT notes.

    Strategy (matching inference preprocessing.js):
    - One syllable token per note
    - If #syllables == #notes: direct 1:1 mapping
    - If #syllables > #notes: split extra syllables across longest notes
    - If #syllables < #notes: merge extra notes into adjacent syllables
    - Silent notes (pitch=0) become <SP>

    Returns: (phonemes, note_dur, note_pitch, note_type)
    """
    n_syl = len(syllable_tokens)
    n_notes = len(note_durs)

    if n_notes == 0:
        return [], [], [], []

    phonemes_out: List[str] = []
    dur_out: List[float] = []
    pitch_out: List[int] = []
    type_out: List[int] = []

    # Case: counts match - simple 1:1
    if n_syl == n_notes:
        for i in range(n_notes):
            if note_pitches[i] == 0:
                phonemes_out.append("<SP>")
                type_out.append(1)
            else:
                phonemes_out.append(syllable_tokens[i])
                type_out.append(0)
            dur_out.append(note_durs[i])
            pitch_out.append(int(note_pitches[i]))
        return phonemes_out, dur_out, pitch_out, type_out

    # Case: more notes than syllables - merge extra notes into nearest syllable
    if n_syl < n_notes:
        ratio = n_notes / n_syl
        note_idx = 0
        for syl_idx in range(n_syl):
            if syl_idx == n_syl - 1:
                end_idx = n_notes
            else:
                end_idx = min(n_notes, int((syl_idx + 1) * ratio))

            merged_dur = sum(note_durs[note_idx:end_idx])
            pitch = 0
            for p in note_pitches[note_idx:end_idx]:
                if p > 0:
                    pitch = int(p)
                    break

            if pitch == 0:
                phonemes_out.append("<SP>")
                type_out.append(1)
            else:
                phonemes_out.append(syllable_tokens[syl_idx])
                type_out.append(0)

            dur_out.append(merged_dur)
            pitch_out.append(pitch)
            note_idx = end_idx

        return phonemes_out, dur_out, pitch_out, type_out

    # Case: more syllables than notes - split notes to cover extra syllables
    if n_syl > n_notes:
        note_order = sorted(range(n_notes), key=lambda i: note_durs[i], reverse=True)
        syllables_per_note = [1] * n_notes
        extra = n_syl - n_notes
        for idx in note_order[:extra]:
            syllables_per_note[idx] += 1

        syl_idx = 0
        for note_idx in range(n_notes):
            count = syllables_per_note[note_idx]
            sub_dur = note_durs[note_idx] / count
            pitch = int(note_pitches[note_idx])

            for _ in range(count):
                if pitch == 0:
                    phonemes_out.append("<SP>")
                    type_out.append(1)
                elif syl_idx < n_syl:
                    phonemes_out.append(syllable_tokens[syl_idx])
                    type_out.append(0)
                    syl_idx += 1
                else:
                    phonemes_out.append("<SP>")
                    type_out.append(1)
                dur_out.append(sub_dur)
                pitch_out.append(pitch)

        return phonemes_out, dur_out, pitch_out, type_out

    return phonemes_out, dur_out, pitch_out, type_out


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_speaker(
    speaker: Dict[str, str],
    lyrics_data: Dict,
    note_transcriber,
    f0_extractor,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Process both songs (common + unique) for one speaker.

    Returns list of metadata dicts (one per song).
    """
    results = []
    speaker_id = speaker["speaker_id"]

    songs = [
        ("common", lyrics_data["common_song"]["lyrics"], lyrics_data["common_song"]["song_name"]),
        ("unique", lyrics_data["unique_songs"].get(speaker["unique_song"], ""), speaker["unique_song"]),
    ]

    for song_type, lyrics, song_name in songs:
        if not lyrics:
            if verbose:
                print(f"  [{speaker_id}/{song_type}] SKIP: no lyrics for '{song_name}'")
            continue

        wav_path = get_wav_path(speaker_id, song_type)
        if wav_path is None or not wav_path.exists():
            if verbose:
                print(f"  [{speaker_id}/{song_type}] SKIP: wav not found")
            continue

        # G2P: lyrics -> syllable tokens
        syllable_tokens = lyrics_to_jp_tokens(lyrics)
        if not syllable_tokens:
            if verbose:
                print(f"  [{speaker_id}/{song_type}] SKIP: G2P produced no tokens")
            continue

        item_name = f"{speaker_id}_{song_type}"
        if verbose:
            print(f"  [{item_name}] lyrics='{song_name}' syllables={len(syllable_tokens)} wav={wav_path.name}")

        if dry_run:
            print(f"    tokens (first 10): {syllable_tokens[:10]}")
            print(f"    token count: {len(syllable_tokens)}")
            results.append({
                "index": item_name,
                "language": "Japanese",
                "dry_run": True,
                "syllable_count": len(syllable_tokens),
                "tokens_sample": syllable_tokens[:10],
            })
            continue

        # --- F0 extraction ---
        import librosa
        wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
        if verbose:
            print(f"    extracting f0 (sr={sr}, len={len(wav)/sr:.1f}s)...")
        f0 = f0_extractor.extract(wav, sr)
        f0_out_path = output_dir / f"{item_name}_f0.npy"
        np.save(f0_out_path, f0)

        # --- Note transcription (ROSVOT) ---
        item = {
            "item_name": item_name,
            "wav_fn": str(wav_path),
        }
        rosvot_out = note_transcriber.process(item, apply_rwbd=True, verbose=False)

        note_pitches = rosvot_out.get("note_pitch", [])
        note_durs = rosvot_out.get("note_dur", [])
        note_types = rosvot_out.get("note_type", [])

        if not note_durs:
            if verbose:
                print(f"    SKIP: ROSVOT detected 0 notes")
            continue

        if verbose:
            print(f"    ROSVOT: {len(note_durs)} notes detected")

        # --- Align phonemes to notes ---
        phonemes, dur, pitch, ntype = align_syllables_to_notes(
            syllable_tokens, note_durs, note_pitches
        )

        # --- Save segment wav ---
        seg_wav_path = output_dir / f"{item_name}.wav"
        import soundfile as sf
        sf.write(str(seg_wav_path), wav, sr)

        meta = {
            "index": item_name,
            "language": "Japanese",
            "time": [0, int(len(wav) / sr * 1000)],
            "duration": " ".join(f"{d:.2f}" for d in dur),
            "text": " ".join(phonemes),
            "phoneme": " ".join(phonemes),
            "note_pitch": " ".join(map(str, pitch)),
            "note_type": " ".join(map(str, ntype)),
            "f0": " ".join(f"{x:.1f}" for x in f0),
            "wav_fn": str(seg_wav_path),
            "speaker_id": speaker_id,
            "song_type": song_type,
            "song_name": song_name,
        }
        results.append(meta)

        if verbose:
            print(f"    DONE: {len(phonemes)} phonemes, dur={sum(dur):.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare JVS-MuSiC for LoRA JP training")
    parser.add_argument("--max-speakers", type=int, default=0,
                        help="Max speakers to process (0 = all 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only verify G2P + lyrics, skip model inference")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for processed data")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load lyrics ---
    with open(LYRICS_FILE, "r", encoding="utf-8") as f:
        lyrics_data = json.load(f)

    # --- Load singer info ---
    speakers = load_singer_info()
    if args.max_speakers > 0:
        speakers = speakers[:args.max_speakers]

    print(f"JVS-MuSiC preprocessing: {len(speakers)} speakers, dry_run={args.dry_run}")
    print(f"Output: {output_dir}")

    all_metadata: List[Dict[str, Any]] = []

    if args.dry_run:
        for spk in speakers:
            meta = process_speaker(spk, lyrics_data, None, None, output_dir,
                                   dry_run=True, verbose=args.verbose)
            all_metadata.extend(meta)
        print(f"\n[Dry run] Processed {len(all_metadata)} segments (G2P only)")
    else:
        print("Loading ROSVOT note transcriber...")
        from preprocess.tools.note_transcription.model import NoteTranscriber

        rosvot_ckpt = str(ROSVOT_DIR / "rosvot" / "checkpoint_08000000.pth.tar")
        rwbd_ckpt = str(ROSVOT_DIR / "rwbd" / "checkpoint_00900000.pth.tar")

        note_transcriber = NoteTranscriber(
            rosvot_model_path=rosvot_ckpt,
            rwbd_model_path=rwbd_ckpt,
            device=args.device,
            verbose=args.verbose,
        )

        print("Loading RMVPE f0 extractor...")
        from preprocess.tools.f0_extraction import F0Extractor
        f0_extractor = F0Extractor(
            f0_extractor="rmvpe",
            rmvpe_path=str(RMVPE_PATH),
            device=args.device,
        )

        t0 = time.time()
        for i, spk in enumerate(speakers):
            if args.verbose:
                print(f"\n[{i+1}/{len(speakers)}] {spk['speaker_id']} ({spk['unique_song']})")
            meta = process_speaker(spk, lyrics_data, note_transcriber, f0_extractor,
                                   output_dir, dry_run=False, verbose=args.verbose)
            all_metadata.extend(meta)

            import torch
            torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"\nProcessed {len(all_metadata)} segments in {dt:.1f}s")

    # --- Save metadata.json ---
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")

    # --- Verify ---
    print("\n=== Verification ===")
    jp_phoneme_set = set()
    for meta in all_metadata:
        if meta.get("dry_run"):
            # In dry-run, collect from tokens_sample
            for tok in meta.get("tokens_sample", []):
                if tok.startswith("jp_"):
                    jp_phoneme_set.add(tok)
            continue
        phonemes = meta.get("phoneme", "").split()
        for ph in phonemes:
            if ph.startswith("jp_"):
                jp_phoneme_set.add(ph)

    phone_set_path = LORA_JP_DIR / "jp_phone_set.json"
    with open(phone_set_path, "r", encoding="utf-8") as f:
        ref_phones = set(json.load(f))

    unknown = jp_phoneme_set - ref_phones
    if unknown:
        print(f"  WARNING: {len(unknown)} phonemes NOT in reference phone set: {sorted(unknown)}")
    else:
        print(f"  OK: all {len(jp_phoneme_set)} jp_* phonemes are in reference phone set")

    print(f"  Total segments: {len(all_metadata)}")
    print(f"  Unique jp_* phonemes used: {sorted(jp_phoneme_set)}")


if __name__ == "__main__":
    main()
