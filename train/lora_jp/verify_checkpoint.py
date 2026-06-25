"""
Verify trained checkpoint quality.

Usage:
    python train/lora_jp/verify_checkpoint.py --checkpoint outputs/lora_jp/best/latest.pt
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 33  # jp_pau removed; 33 JP phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='outputs/lora_jp/best/latest.pt')
    parser.add_argument('--base_model', default='pretrained_models/SoulX-Singer/model.pt')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f'ERROR: Checkpoint not found: {args.checkpoint}')
        return

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f'  Keys: {list(ckpt.keys())}')
    print(f'  Epoch: {ckpt.get("epoch", "?")}, Loss: {ckpt.get("loss", "?"):.4f}')

    # Load base model embedding for comparison
    print(f'\nLoading base model: {args.base_model}')
    base_ckpt = torch.load(args.base_model, map_location='cpu', weights_only=False)
    base_sd = base_ckpt.get('state_dict', base_ckpt)
    base_embed = base_sd['note_text_encoder.weight']
    print(f'  Base embedding: {base_embed.shape}, std={base_embed.std():.4f}')

    # Get fine-tuned embedding
    if 'embed_state_dict' in ckpt:
        ft_embed = ckpt['embed_state_dict']['weight']
        print(f'\nFull embedding (embed_state_dict): {ft_embed.shape}')
        if ft_embed.shape[0] == base_embed.shape[0]:
            print(f'  WARNING: Same size as base ({ft_embed.shape[0]}). Embedding was NOT extended during training!')
    else:
        jp_embed = ckpt['jp_embed']
        ft_embed = torch.cat([base_embed, jp_embed], dim=0)
        print(f'\nConcatenated embedding: {ft_embed.shape}')

    # Compare base rows (should be close but not identical)
    base_diff = (ft_embed[:JP_PHONEME_START] - base_embed[:JP_PHONEME_START])
    print(f'\n--- Base phoneme rows (0-{JP_PHONEME_START-1}) ---')
    print(f'  Diff from original: mean={base_diff.mean():.6f}, std={base_diff.std():.6f}')
    print(f'  Max diff: {base_diff.abs().max():.4f}')
    if base_diff.abs().max() < 0.001:
        print(f'  Status: BARELY CHANGED (embedding not trained)')
    elif base_diff.abs().max() < 0.1:
        print(f'  Status: SLIGHTLY ADAPTED (good, preserves base behavior)')
    else:
        print(f'  Status: SIGNIFICANTLY CHANGED (may affect non-JP quality)')

    # Check JP rows
    jp_rows = ft_embed[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT]
    print(f'\n--- JP phoneme rows ({JP_PHONEME_START}-{JP_PHONEME_START+JP_PHONEME_COUNT-1}) ---')
    print(f'  Shape: {jp_rows.shape}')
    print(f'  Mean: {jp_rows.mean():.6f}')
    print(f'  Std: {jp_rows.std():.6f}')
    print(f'  Range: [{jp_rows.min():.4f}, {jp_rows.max():.4f}]')

    init_std = 0.02  # initialization std
    if jp_rows.std() < init_std * 1.5:
        print(f'  Status: NEAR ZERO (barely changed from init std={init_std})')
    elif jp_rows.std() < 0.1:
        print(f'  Status: WEAK (std={jp_rows.std():.4f}, started at {init_std}, needs more training)')
    elif jp_rows.std() < 0.5:
        print(f'  Status: MODERATE (std={jp_rows.std():.4f}, partially trained)')
    else:
        print(f'  Status: WELL TRAINED (std={jp_rows.std():.4f}, comparable to base std={base_embed.std():.4f})')

    # Per-phoneme analysis
    print(f'\n--- Per-phoneme norms ---')
    for i in range(JP_PHONEME_COUNT):
        ph_norm = jp_rows[i].norm().item()
        ph_std = jp_rows[i].std().item()
        ph_name = f'jp_{i}'  # We don't have names here, just indices
        bar = '#' * min(40, int(ph_norm * 2))
        if i < 10 or ph_norm > 1.0 or ph_norm < 0.01:
            print(f'  [{JP_PHONEME_START+i:4d}] norm={ph_norm:6.3f} std={ph_std:6.4f} {bar}')

    # Preflow analysis
    if 'preflow_state_dict' in ckpt:
        pf_sd = ckpt['preflow_state_dict']
        print(f'\n--- Preflow ---')
        print(f'  Keys: {len(pf_sd)}')
        # Compare first block with original
        if '0.dwconv.weight' in pf_sd:
            ft_w = pf_sd['0.dwconv.weight']
            print(f'  Block 0 dwconv: shape={ft_w.shape}, std={ft_w.float().std():.4f}')

    # F0 encoder analysis
    if 'f0_encoder_state_dict' in ckpt:
        f0_sd = ckpt['f0_encoder_state_dict']
        print(f'\n--- F0 Encoder ---')
        print(f'  Keys: {len(f0_sd)}')

    # Overall verdict
    print(f'\n{"="*50}')
    print(f'VERDICT:')
    if jp_rows.std() >= 0.5 and base_diff.abs().max() < 0.5:
        print(f'  GOOD: JP embeddings well-trained, base behavior preserved')
    elif jp_rows.std() >= 0.5:
        print(f'  OK: JP embeddings trained, but base rows changed significantly')
    elif jp_rows.std() >= 0.1:
        print(f'  MODERATE: JP embeddings partially trained (std={jp_rows.std():.4f}), consider more epochs or higher lr')
    elif jp_rows.std() >= 0.03:
        print(f'  WEAK: JP embeddings barely trained (std={jp_rows.std():.4f}), need higher embedding lr')
    else:
        print(f'  FAILED: JP embeddings not trained, check training code')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
