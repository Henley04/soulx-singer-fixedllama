"""Diagnose training data alignment.

Checks:
1. mel2note max value vs phoneme length (off-by-one?)
2. f0 length vs mel2note length
3. waveform length vs mel2note * hop_size
4. target_mel length vs mel_feat length
5. Are phoneme IDs valid (in phone2idx)?
"""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from train.lora_jp.dataset import JpLoRADataset, collate_fn
from torch.utils.data import DataLoader

phoneset_path = 'train/lora_jp/jp_phone_set.json'
with open(phoneset_path, 'r', encoding='utf-8') as f:
    phone_list = json.load(f)
phone2idx = {ph: i for i, ph in enumerate(phone_list)}

dataset = JpLoRADataset(
    metadata_path='train/lora_jp/dataset/metadata.json',
    wav_dir='train/lora_jp/dataset/wavs',
    phoneset_path=phoneset_path,
    sample_rate=24000,
    hop_size=480,
    device='cpu',
)

print("=" * 70)
print("Diagnosing first 3 samples")
print("=" * 70)

for idx in range(3):
    print(f"\n--- Sample {idx} ---")
    item = dataset[idx]
    if item is None:
        print("  FAILED to load")
        continue

    phoneme = item['phoneme']  # [1, T_ph]
    note_pitch = item['note_pitch']  # [1, T_ph]
    note_type = item['note_type']  # [1, T_ph]
    mel2note = item['mel2note']  # [1, T_mel]
    f0 = item.get('f0')  # [1, T_mel] or None
    waveform = item.get('waveform')  # [1, T_audio] or None

    print(f"  phoneme shape: {phoneme.shape}")
    print(f"  note_pitch shape: {note_pitch.shape}")
    print(f"  note_type shape: {note_type.shape}")
    print(f"  mel2note shape: {mel2note.shape}")
    print(f"  f0 shape: {f0.shape if f0 is not None else None}")
    print(f"  waveform shape: {waveform.shape if waveform is not None else None}")

    # Check 1: mel2note max vs phoneme length
    ph_len = phoneme.shape[1]
    mel2note_max = mel2note.max().item()
    print(f"\n  Check 1: mel2note alignment")
    print(f"    phoneme length: {ph_len}")
    print(f"    mel2note max: {mel2note_max}")
    print(f"    max < ph_len: {mel2note_max < ph_len}")
    if mel2note_max >= ph_len:
        print(f"    *** ERROR: mel2note max ({mel2note_max}) >= phoneme length ({ph_len})!")
        print(f"    *** This will cause expand_states to read out-of-bounds (clamped)")

    # Check 2: f0 length vs mel2note length
    if f0 is not None:
        f0_len = f0.shape[1]
        mel_len = mel2note.shape[1]
        print(f"\n  Check 2: f0 vs mel2note length")
        print(f"    f0 length: {f0_len}")
        print(f"    mel2note length: {mel_len}")
        print(f"    Match: {f0_len == mel_len}")
        nonzero_f0 = (f0 > 0).sum().item()
        print(f"    Non-zero f0 frames: {nonzero_f0}/{f0_len} ({nonzero_f0/f0_len*100:.1f}%)")

    # Check 3: waveform length vs mel2note * hop_size
    if waveform is not None:
        expected_audio_len = mel2note.shape[1] * 480
        audio_len = waveform.shape[1]
        print(f"\n  Check 3: waveform length")
        print(f"    waveform length: {audio_len}")
        print(f"    expected (mel2note * hop): {expected_audio_len}")
        print(f"    Match: {audio_len == expected_audio_len}")

    # Check 4: phoneme ID validity
    print(f"\n  Check 4: phoneme ID validity")
    ph_ids = phoneme.squeeze(0).tolist()
    invalid = [pid for pid in ph_ids if pid < 0 or pid >= len(phone_list)]
    print(f"    Invalid IDs: {len(invalid)}")
    # Show first few phonemes with their names
    for i in range(min(10, len(ph_ids))):
        name = phone_list[ph_ids[i]] if 0 <= ph_ids[i] < len(phone_list) else 'OOB'
        print(f"    [{i}] id={ph_ids[i]} name={name}")

    # Check 5: target_mel will match mel_feat
    print(f"\n  Check 5: mel_feat vs target_mel (computed from waveform)")
    if waveform is not None:
        # Simulate what train_staged does
        from omegaconf import OmegaConf
        config = OmegaConf.load('soulxsinger/config/soulxsinger.yaml')
        from soulxsinger.models.soulxsinger import SoulXSinger
        # Don't load full model (slow), just check mel transform
        # target_mel = model.mel(waveform.float())
        # mel_feat length = mel2note.shape[1]
        # target_mel length depends on waveform length
        # model.mel produces ceil(waveform_len / hop_size) frames
        expected_mel_frames = waveform.shape[1] // 480
        mel_feat_frames = mel2note.shape[1]
        print(f"    mel_feat frames (from mel2note): {mel_feat_frames}")
        print(f"    target_mel frames (from waveform//hop): {expected_mel_frames}")
        print(f"    Match: {mel_feat_frames == expected_mel_frames}")
        if mel_feat_frames != expected_mel_frames:
            print(f"    *** MISMATCH! train_staged does target_mel[:, :mel_feat.shape[1], :]")
            print(f"    *** This truncates target_mel, potentially losing data")

print("\n" + "=" * 70)
print("Diagnosis complete")
print("=" * 70)
