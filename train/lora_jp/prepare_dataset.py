"""
Prepare training dataset from PJS Corpus for LoRA fine-tuning.

Converts PJS Corpus (lab + wav + MIDI) to metadata JSON format
compatible with DataProcessor.process().

Usage:
    python train/lora_jp/prepare_dataset.py \
        --corpus_dir pretrained_models/SoulX-Singer/assets/LoRA-JP/PJS_corpus_ver1.1 \
        --output_dir train/lora_jp/dataset \
        --sample_rate 24000
"""

import os
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import mido
except ImportError:
    mido = None

try:
    import parselmouth
except ImportError:
    parselmouth = None


# Phoneme mapping: lab phoneme -> jp_* prefixed phoneme
PHONEME_MAP = {
    'pau': 'jp_pau',
    'a': 'jp_a',
    'i': 'jp_i',
    'u': 'jp_u',
    'e': 'jp_e',
    'o': 'jp_o',
    'k': 'jp_k',
    's': 'jp_s',
    't': 'jp_t',
    'n': 'jp_n',
    'h': 'jp_h',
    'm': 'jp_m',
    'r': 'jp_r',
    'w': 'jp_w',
    'y': 'jp_y',
    'g': 'jp_g',
    'z': 'jp_z',
    'd': 'jp_d',
    'b': 'jp_b',
    'p': 'jp_p',
    'f': 'jp_f',
    'j': 'jp_j',
    'ch': 'jp_ch',
    'sh': 'jp_sh',
    'ts': 'jp_ts',
    'ky': 'jp_ky',
    'gy': 'jp_gy',
    'ny': 'jp_ny',
    'hy': 'jp_hy',
    'my': 'jp_my',
    'ry': 'jp_ry',
    'py': 'jp_py',
    'by': 'jp_by',
    'cl': 'jp_cl',
    'I': 'jp_a',    # map uppercase variants
    'N': 'jp_n',
    'O': 'jp_o',
    'U': 'jp_u',
    'xx': 'jp_pau',  # unknown -> pause
}

# Consonants (for note_type determination)
CONSONANTS = {
    'k', 's', 't', 'n', 'h', 'm', 'r', 'w', 'y', 'g', 'z', 'd', 'b', 'p',
    'f', 'j', 'ch', 'sh', 'ts', 'ky', 'gy', 'ny', 'hy', 'my', 'ry', 'py', 'by', 'cl',
}


def parse_lab_file(lab_path: str) -> List[Dict]:
    """Parse .lab file. Each line: start_time_100ns end_time_100ns phoneme

    PJS corpus uses HTK-style timestamps in 100ns units.
    """
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start_100ns = int(parts[0])
            end_100ns = int(parts[1])
            phoneme = parts[2]
            segments.append({
                'start': start_100ns / 1e7,  # convert 100ns units to seconds
                'end': end_100ns / 1e7,
                'phoneme': phoneme,
            })
    return segments


def parse_midi_file(midi_path: str) -> List[Dict]:
    """Parse MIDI file and extract note events."""
    if mido is None:
        raise ImportError("mido is required for MIDI parsing. Install: pip install mido")

    notes = []
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Extract tempo from meta messages (default: 500000 us/beat = 120 BPM)
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    # Collect all note events with absolute time (in ticks)
    events = []
    for track in mid.tracks:
        absolute_time = 0
        for msg in track:
            absolute_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'type': 'on',
                    'time': absolute_time,
                    'note': msg.note,
                    'velocity': msg.velocity,
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({
                    'type': 'off',
                    'time': absolute_time,
                    'note': msg.note,
                })

    # Match note_on with note_off
    active_notes = {}
    for event in sorted(events, key=lambda x: x['time']):
        if event['type'] == 'on':
            active_notes[event['note']] = event
        elif event['type'] == 'off' and event['note'] in active_notes:
            on_event = active_notes.pop(event['note'])
            start_sec = mido.tick2second(on_event['time'], ticks_per_beat, tempo)
            end_sec = mido.tick2second(event['time'], ticks_per_beat, tempo)
            if end_sec > start_sec:
                notes.append({
                    'start': start_sec,
                    'end': end_sec,
                    'pitch': event['note'],
                    'velocity': on_event['velocity'],
                })

    return sorted(notes, key=lambda x: x['start'])


def extract_f0_parselmouth(wav_path: str, sr: int = 24000) -> np.ndarray:
    """Extract F0 using parselmouth (Praat)."""
    if parselmouth is None:
        raise ImportError("parselmouth is required. Install: pip install praat-parselmouth")

    y, orig_sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]

    # Resample if needed
    if orig_sr != sr:
        import scipy.signal
        y = scipy.signal.resample(y, int(len(y) * sr / orig_sr))

    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=480 / sr, pitch_floor=32.7031956625, pitch_ceiling=1000.0)

    f0 = pitch.selected_array['frequency']
    return f0.astype(np.float32)


def lab_phonemes_to_notes(
    lab_segments: List[Dict],
    midi_notes: List[Dict],
    sample_rate: int = 24000,
    hop_size: int = 480,
) -> Tuple[List[str], List[float], List[int], List[int]]:
    """
    Align lab phonemes to MIDI notes.

    For each MIDI note, include all matching phonemes (consonant + vowel).
    Consonants get a short duration, vowels get the remaining duration.
    Returns: (phonemes, durations, note_pitches, note_types)
    """
    phonemes = []
    durations = []
    note_pitches = []
    note_types = []

    for note in midi_notes:
        note_start = note['start']
        note_end = note['end']
        note_pitch = note['pitch']
        note_duration = note_end - note_start

        # Find lab phonemes within this note's time range
        matching_phonemes = []
        for seg in lab_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            if seg_start < note_end and seg_end > note_start:
                matching_phonemes.append(seg)

        if not matching_phonemes:
            phonemes.append('jp_a')
            durations.append(note_duration)
            note_pitches.append(note_pitch)
            note_types.append(2)  # normal note
            continue

        # Split into consonants and vowels
        consonants = []
        vowels = []
        for seg in matching_phonemes:
            raw_ph = seg['phoneme']
            jp_ph = PHONEME_MAP.get(raw_ph, f'jp_{raw_ph}')
            seg_start_in_note = max(seg['start'], note_start)
            seg_end_in_note = min(seg['end'], note_end)
            seg_duration = max(0.001, seg_end_in_note - seg_start_in_note)

            if raw_ph in CONSONANTS or raw_ph == 'cl':
                consonants.append({'ph': jp_ph, 'dur': seg_duration})
            else:
                vowels.append({'ph': jp_ph, 'dur': seg_duration})

        # If no vowels found, use the first phoneme
        if not vowels:
            raw_ph = matching_phonemes[0]['phoneme']
            jp_ph = PHONEME_MAP.get(raw_ph, f'jp_{raw_ph}')
            phonemes.append(jp_ph)
            durations.append(note_duration)
            note_pitches.append(note_pitch)
            note_types.append(2 if raw_ph != 'pau' else 1)  # normal note or SP
            continue

        # Total consonant and vowel durations from lab timing
        total_cons_dur = sum(c['dur'] for c in consonants)
        total_vowel_dur = sum(v['dur'] for v in vowels)
        total_lab_dur = total_cons_dur + total_vowel_dur

        # Scale durations to fit the note duration
        if total_lab_dur > 0:
            scale = note_duration / total_lab_dur
        else:
            scale = 1.0

        # Minimum duration: 3 mel frames to prevent DataProcessor overlap bug.
        # preprocess() expands each phoneme to BOW+phoneme+EOW (3 tokens).
        # With only 2 frames, BOW+EOW consume all frames and the actual
        # phoneme gets 0 frames, causing it to be skipped in mel2note.
        min_dur = 3 * hop_size / sample_rate

        # Add consonant phonemes (each gets its proportional lab duration)
        for c in consonants:
            phonemes.append(c['ph'])
            durations.append(max(c['dur'] * scale, min_dur))
            note_pitches.append(note_pitch)
            note_types.append(2)  # consonant = normal note (type 2)

        # Add vowel phonemes
        for v in vowels:
            phonemes.append(v['ph'])
            durations.append(max(v['dur'] * scale, min_dur))
            note_pitches.append(note_pitch)
            note_types.append(2)  # vowel = normal note (type 2)

    return phonemes, durations, note_pitches, note_types


def process_one_sample(
    sample_dir: str,
    sample_id: str,
    output_dir: str,
    sample_rate: int = 24000,
    lab_dir: str = None,
) -> Optional[Dict]:
    """Process one PJS sample (wav + lab + midi)."""
    wav_path = os.path.join(sample_dir, f'{sample_id}_song.wav')
    # Use corrected lab if available, otherwise fall back to original
    if lab_dir:
        lab_path = os.path.join(lab_dir, f'{sample_id}.lab')
        if not os.path.exists(lab_path):
            lab_path = os.path.join(sample_dir, f'{sample_id}.lab')
    else:
        lab_path = os.path.join(sample_dir, f'{sample_id}.lab')
    midi_path = os.path.join(sample_dir, f'{sample_id}.mid')

    if not os.path.exists(wav_path):
        print(f'  Warning: wav not found: {wav_path}')
        return None
    if not os.path.exists(lab_path):
        print(f'  Warning: lab not found: {lab_path}')
        return None

    # Parse inputs
    lab_segments = parse_lab_file(lab_path)
    midi_notes = []
    if os.path.exists(midi_path) and mido is not None:
        try:
            midi_notes = parse_midi_file(midi_path)
        except Exception as e:
            print(f'  Warning: MIDI parse failed for {sample_id}: {e}')

    if not midi_notes:
        # Fallback: use lab segments as notes (each phoneme = one note)
        print(f'  Warning: No MIDI notes for {sample_id}, using lab segments as notes')
        for seg in lab_segments:
            if seg['phoneme'] == 'pau':
                continue
            jp_ph = PHONEME_MAP.get(seg['phoneme'], f'jp_{seg["phoneme"]}')
            duration = seg['end'] - seg['start']
            if duration <= 0:
                continue
            # Estimate pitch from phoneme (default to A4=69)
            midi_notes.append({
                'start': seg['start'],
                'end': seg['end'],
                'pitch': 69,
                'velocity': 80,
            })

    if not midi_notes:
        print(f'  Warning: No usable data for {sample_id}')
        return None

    # Align phonemes to notes
    phonemes, durations, note_pitches, note_types = lab_phonemes_to_notes(
        lab_segments, midi_notes, sample_rate
    )

    if not phonemes:
        print(f'  Warning: No phonemes after alignment for {sample_id}')
        return None

    # Extract F0 (required for training)
    f0 = None
    if parselmouth is not None:
        try:
            f0 = extract_f0_parselmouth(wav_path, sample_rate)
        except Exception as e:
            print(f'  Warning: F0 extraction failed for {sample_id}: {e}')
    else:
        print(f'  Warning: parselmouth not available, skipping F0 for {sample_id}')

    # Resample wav to target sample_rate
    y, orig_sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    if orig_sr != sample_rate:
        import scipy.signal
        y = scipy.signal.resample(y, int(len(y) * sample_rate / orig_sr))

    # Save resampled wav
    resampled_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(resampled_dir, exist_ok=True)
    resampled_path = os.path.join(resampled_dir, f'{sample_id}_song.wav')
    sf.write(resampled_path, y, sample_rate)

    # Build metadata
    metadata = {
        'phoneme': ' '.join(phonemes),
        'duration': ' '.join(f'{d:.6f}' for d in durations),
        'note_pitch': ' '.join(str(p) for p in note_pitches),
        'note_type': ' '.join(str(t) for t in note_types),
    }
    if f0 is not None and len(f0) > 0:
        metadata['f0'] = ' '.join(f'{v:.2f}' for v in f0.tolist())

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare PJS Corpus for LoRA training')
    parser.add_argument('--corpus_dir', type=str,
                        default='pretrained_models/SoulX-Singer/assets/LoRA-JP/PJS_corpus_ver1.1',
                        help='Path to PJS corpus directory')
    parser.add_argument('--lab_dir', type=str,
                        default='pretrained_models/SoulX-Singer/assets/LoRA-JP/lab',
                        help='Path to corrected lab files (overrides corpus lab)')
    parser.add_argument('--output_dir', type=str,
                        default='train/lora_jp/dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Target sample rate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process (for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all PJS sample directories
    corpus_dir = Path(args.corpus_dir)
    sample_dirs = sorted([
        d for d in corpus_dir.iterdir()
        if d.is_dir() and d.name.startswith('pjs')
    ])

    if args.max_samples:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f'Found {len(sample_dirs)} samples in {corpus_dir}')

    all_metadata = []
    for i, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name
        print(f'[{i+1}/{len(sample_dirs)}] Processing {sample_id}...')

        metadata = process_one_sample(
            str(sample_dir), sample_id, args.output_dir, args.sample_rate,
            lab_dir=args.lab_dir
        )
        if metadata is not None:
            all_metadata.append(metadata)

    # Save metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f'\nProcessed {len(all_metadata)}/{len(sample_dirs)} samples')
    print(f'Metadata saved to: {metadata_path}')
    print(f'Resampled wavs saved to: {os.path.join(args.output_dir, "wavs")}')


if __name__ == '__main__':
    main()
