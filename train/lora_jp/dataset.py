"""
PyTorch Dataset for LoRA fine-tuning on PJS Corpus.

Loads preprocessed metadata and wav files, converts to model input format.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.utils.data_processor import DataProcessor


class JpLoRADataset(Dataset):
    """
    Dataset for Japanese LoRA fine-tuning.

    Each item is a training sample with:
    - phoneme: token IDs [1, T_ph]
    - note_pitch: pitch values [1, T_ph]
    - note_type: type values [1, T_ph]
    - mel2note: frame-to-token alignment [1, T_mel]
    - f0: F0 values [1, T_mel] (optional)
    - waveform: audio waveform [1, T_audio] (optional)
    """

    def __init__(
        self,
        metadata_path: str,
        wav_dir: str,
        phoneset_path: str,
        sample_rate: int = 24000,
        hop_size: int = 480,
        device: str = 'cpu',
        max_frames: int = 2000,
    ):
        """
        Args:
            metadata_path: Path to metadata.json (list of metadata dicts)
            wav_dir: Directory containing resampled wav files
            phoneset_path: Path to jp_phone_set.json
            sample_rate: Audio sample rate
            hop_size: Hop size for mel spectrogram
            device: Device for tensor computation
            max_frames: Maximum number of mel frames per sample
        """
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.device = device
        self.max_frames = max_frames

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.data_processor = DataProcessor(
            hop_size=hop_size,
            sample_rate=sample_rate,
            phoneset_path=phoneset_path,
            device=device,
        )

        # Build a deterministic sample_id -> wav_path mapping at __init__ time.
        # os.listdir order is not guaranteed across platforms/runs, so we sort
        # and match by sample_id (the "id" field in metadata) when possible,
        # falling back to sorted index order for legacy metadata without ids.
        self.wav_path_map = {}
        if os.path.isdir(wav_dir):
            wav_files = sorted(
                f for f in os.listdir(wav_dir)
                if f.endswith('.wav') and f.startswith('pjs')
            )
            # Try to match by sample_id (e.g., "pjs001" -> "pjs001.wav")
            id_to_wav = {}
            for wf in wav_files:
                stem = os.path.splitext(wf)[0]
                id_to_wav[stem] = wf

            for i, meta in enumerate(self.metadata):
                sid = meta.get("id") or meta.get("sample_id")
                if sid and sid in id_to_wav:
                    self.wav_path_map[i] = os.path.join(wav_dir, id_to_wav[sid])
                elif i < len(wav_files):
                    # Fallback: sorted-index match (deterministic)
                    self.wav_path_map[i] = os.path.join(wav_dir, wav_files[i])
            print(f'[JpLoRADataset] Matched {len(self.wav_path_map)}/{len(self.metadata)} wav files')
        else:
            print(f'[JpLoRADataset] WARNING: wav_dir does not exist: {wav_dir}')

        print(f'[JpLoRADataset] Loaded {len(self.metadata)} samples')

    def __len__(self):
        return len(self.metadata)

    def _ensure_min_duration(self, meta):
        """Ensure each phoneme has at least 2 mel frames duration.

        Prevents DataProcessor.preprocess() overlap bug where multiple
        phonemes map to the same mel frame, causing phonemes to be skipped
        in the mel2note alignment.
        """
        durations = [float(x) for x in meta["duration"].split()]
        min_dur = 3 * self.hop_size / self.sample_rate  # 3 frames = 0.06s

        if all(d >= min_dur for d in durations):
            return

        # Set minimum without rescaling — slightly longer total is fine
        adjusted = [max(d, min_dur) for d in durations]
        meta["duration"] = ' '.join(f'{d:.6f}' for d in adjusted)

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        import copy
        meta = copy.deepcopy(self.metadata[idx])

        try:
            # Use pre-built wav_path_map (deterministic, built at __init__)
            wav_path = self.wav_path_map.get(idx)
            if wav_path is None or not os.path.exists(wav_path):
                return None

            # Fix durations that are too short (< 1 mel frame) to prevent
            # DataProcessor.preprocess() overlap bug where mel2note index
            # goes out of bounds
            self._ensure_min_duration(meta)

            # Use DataProcessor to convert to model format
            item = self.data_processor.process(meta, wav_path)

            # Align mel2note and f0 lengths (consonants may cause mismatches)
            if item.get('f0') is not None and item['mel2note'] is not None:
                min_len = min(item['mel2note'].shape[1], item['f0'].shape[1])
                item['mel2note'] = item['mel2note'][:, :min_len]
                item['f0'] = item['f0'][:, :min_len]

            # Truncate if too long
            if item['mel2note'].shape[1] > self.max_frames:
                item['mel2note'] = item['mel2note'][:, :self.max_frames]
                if item['f0'] is not None:
                    item['f0'] = item['f0'][:, :self.max_frames]
                if 'waveform' in item:
                    item['waveform'] = item['waveform'][:, :self.max_frames * self.hop_size]

            return item

        except Exception as e:
            print(f'[JpLoRADataset] Error loading sample {idx}: {e}')
            return None


def collate_fn(batch):
    """Custom collate function that filters out None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Pad sequences to the same length within the batch
    max_ph_len = max(item['phoneme'].shape[1] for item in batch)
    max_mel_len = max(item['mel2note'].shape[1] for item in batch)

    padded_batch = []
    for item in batch:
        padded = {}
        for key, val in item.items():
            if val is None:
                padded[key] = None
                continue
            if key == 'phoneme' or key == 'note_pitch' or key == 'note_type':
                # Pad along sequence dimension
                pad_len = max_ph_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_ph_len]
            elif key == 'mel2note' or key == 'f0':
                pad_len = max_mel_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_mel_len]
            elif key == 'waveform':
                max_audio_len = max_mel_len * 480
                pad_len = max_audio_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_audio_len]
            else:
                padded[key] = val
        padded_batch.append(padded)

    # Stack into batch
    result = {}
    for key in padded_batch[0]:
        vals = [item[key] for item in padded_batch if item[key] is not None]
        if vals and all(v is not None for v in vals):
            result[key] = torch.cat(vals, dim=0)
        else:
            result[key] = None

    return result
