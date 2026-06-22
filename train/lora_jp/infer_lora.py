"""
Validate LoRA fine-tuned model with Japanese inference.

Loads base model + LoRA weights and runs inference on Japanese samples.

Usage:
    python train/lora_jp/infer_lora.py \
        --model_path pretrained_models/SoulX-Singer/model.pt \
        --lora_checkpoint outputs/lora_jp/lora_jp_latest.pt \
        --config soulxsinger/config/soulxsinger.yaml \
        --phoneset_path train/lora_jp/jp_phone_set.json \
        --dataset_dir train/lora_jp/dataset \
        --output_dir outputs/lora_jp/inference \
        --device cuda
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor

JP_PHONEME_START = 2820
JP_PHONEME_COUNT = 34
EMBED_DIM = 512


def load_model_with_lora(model_path, lora_checkpoint_path, config, device='cuda'):
    """Load base model and apply LoRA weights."""
    from train.lora_jp.train_lora import LoRABlockAdapter

    # Load base model
    model = SoulXSinger(config)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    print(f'Base model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')

    # Load LoRA checkpoint
    lora_checkpoint = torch.load(lora_checkpoint_path, map_location='cpu', weights_only=False)

    # Add LoRA adapters to model
    lora_adapters = nn.ModuleList()
    for i in range(4):
        adapter = LoRABlockAdapter(dim=EMBED_DIM, r=16, alpha=32, dropout=0.0)
        lora_adapters.append(adapter)
    model.lora_adapters = lora_adapters

    # Load adapter weights
    if 'lora_adapters_state_dict' in lora_checkpoint:
        model.lora_adapters.load_state_dict(lora_checkpoint['lora_adapters_state_dict'])
        print('Applied LoRA adapters from checkpoint')
    else:
        print('Warning: No lora_adapters_state_dict in checkpoint')

    model = model.to(device)
    model.eval()
    print(f'Model ready on {device}')

    return model


@torch.no_grad()
def run_inference(model, data_processor, metadata, wav_path, output_path, device='cuda'):
    """Run inference on a single sample."""
    item = data_processor.process(metadata, wav_path)

    # Move to device
    for k, v in item.items():
        if v is not None:
            item[k] = v.to(device)

    # Build inference meta (same format as SoulXSinger.infer)
    meta = {
        'target': {
            'phoneme': item['phoneme'],
            'mel2note': item['mel2note'],
            'note_type': item['note_type'],
            'note_pitch': item.get('note_pitch'),
            'f0': item.get('f0'),
        },
        'prompt': {
            'waveform': item.get('waveform'),
            'phoneme': item['phoneme'][:, :1],  # Use first token as prompt
            'mel2note': item['mel2note'][:, :1],
            'note_type': item['note_type'][:, :1],
            'note_pitch': item['note_pitch'][:, :1] if item.get('note_pitch') is not None else None,
            'f0': item['f0'][:, :1] if item.get('f0') is not None else None,
        }
    }

    # Run inference
    with torch.no_grad():
        generated_audio = model.infer(
            meta,
            n_steps=32,
            cfg=3,
            control='melody',
        )

    # Save audio
    audio_np = generated_audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_np, 24000)
    print(f'  Audio saved: {output_path} ({len(audio_np)/24000:.2f}s)')

    return audio_np


def main():
    parser = argparse.ArgumentParser(description='Validate LoRA model with Japanese inference')
    parser.add_argument('--model_path', type=str,
                        default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--lora_checkpoint', type=str,
                        default='outputs/lora_jp/lora_jp_latest.pt')
    parser.add_argument('--config', type=str,
                        default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--phoneset_path', type=str,
                        default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--dataset_dir', type=str,
                        default='train/lora_jp/dataset')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/lora_jp/inference')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to inference')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = OmegaConf.load(args.config)

    # Load model with LoRA
    print('Loading model with LoRA...')
    model = load_model_with_lora(args.model_path, args.lora_checkpoint, config, args.device)

    # Load data processor
    data_processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=args.phoneset_path,
        device=args.device,
    )

    # Load metadata
    metadata_path = os.path.join(args.dataset_dir, 'metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    wav_dir = os.path.join(args.dataset_dir, 'wavs')
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])

    print(f'\nRunning inference on {args.num_samples} samples...')

    for i in range(min(args.num_samples, len(all_metadata), len(wav_files))):
        print(f'\n[{i+1}/{args.num_samples}] {wav_files[i]}')
        wav_path = os.path.join(wav_dir, wav_files[i])
        output_path = os.path.join(args.output_dir, f'jp_inferred_{i:03d}.wav')

        try:
            run_inference(model, data_processor, all_metadata[i], wav_path, output_path, args.device)
        except Exception as e:
            print(f'  Error: {e}')
            import traceback
            traceback.print_exc()

    print(f'\nInference complete. Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
