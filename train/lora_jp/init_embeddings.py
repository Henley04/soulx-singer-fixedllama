"""
Task 2: Embedding table initialization with scale calibration.

Loads pretrained model, extends embedding to include JP phonemes,
initializes them from the English-source mapping,
applies L2 normalization + global std matching.

Usage:
    python train/lora_jp/init_embeddings.py \
        --model_path pretrained_models/SoulX-Singer/model.pt \
        --mapping train/lora_jp/jp_phoneme_mapping.json \
        --phoneset train/lora_jp/jp_phone_set.json \
        --output outputs/lora_jp/init_embed.pt \
        --target_std 0.9
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 33  # jp_pau removed; 33 JP phonemes (jp_a..jp_cl)
EMBED_DIM = 512


def add_orthogonal_noise(vec, noise_scale=0.08):
    """Add a tiny orthogonal perturbation to break exact copies.

    Generates a random vector, subtracts its projection onto vec,
    then scales the orthogonal component to noise_scale * ||vec||.
    This ensures the perturbation is perpendicular to vec,
    preserving direction while breaking cosine_sim = 1.0.

    noise_scale=0.08 (8%) gives initial cos_sim ≈ 0.9968, which is
    below the collapse threshold (0.998) and gives decouple loss room
    to push cos down to the target 0.90 during training.
    """
    rand = torch.randn_like(vec)
    # Subtract projection onto vec: rand_orth = rand - (rand·v / ||v||²) * v
    dot = torch.dot(rand, vec)
    norm_sq = torch.dot(vec, vec)
    rand_orth = rand - (dot / (norm_sq + 1e-8)) * vec
    # Scale to noise_scale * ||vec||
    rand_orth_norm = rand_orth.norm()
    if rand_orth_norm < 1e-8:
        return vec  # degenerate, skip
    perturbation = rand_orth / rand_orth_norm * vec.norm() * noise_scale
    return vec + perturbation


def load_phone2idx(phoneset_path):
    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phones = json.load(f)
    return {ph: idx for idx, ph in enumerate(phones)}


def load_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_jp_embeddings(embed_weight, mapping, phone2idx):
    """Initialize JP phoneme embeddings from the source mapping.

    Two-pass approach:
      Pass 1: Compute weighted sums for all phonemes, store temporarily.
      Pass 2: For phonemes with init_weight < 1.0, blend with fallback_mean.

    Returns: (initialized_weight, log_dict)
    """
    initialized = {}
    # Temporary storage for pass 1 results: jp_idx -> (weighted_sum, init_weight, details)
    pending = {}

    # ── Pass 1: compute weighted sums ────────────────────────────────
    for jp_phone, entry in mapping.items():
        if jp_phone not in phone2idx:
            initialized[jp_phone] = {"status": "failed", "reason": "not in phone2idx"}
            continue
        jp_idx = phone2idx[jp_phone]

        if entry.get("strategy") == "pause_mean":
            pause_indices = []
            for src in entry.get("pause_sources", []):
                if src in phone2idx:
                    pause_indices.append(phone2idx[src])
            if pause_indices:
                mean_vec = embed_weight[pause_indices].mean(dim=0)
                noise = torch.randn_like(mean_vec) * entry.get("noise_std", 0.01)
                embed_weight[jp_idx] = mean_vec + noise
                initialized[jp_phone] = {"status": "pause_mean", "sources": len(pause_indices)}
            else:
                initialized[jp_phone] = {"status": "failed", "reason": "no pause sources found"}

        else:
            sources = entry.get("sources", [])
            weighted_sum = torch.zeros(EMBED_DIM)
            total_weight = 0.0
            source_details = []

            for src in sources:
                src_phone = src["phone"]
                src_weight = src["weight"]
                if src_phone in phone2idx:
                    src_idx = phone2idx[src_phone]
                    weighted_sum += embed_weight[src_idx] * src_weight
                    total_weight += src_weight
                    source_details.append(f"{src_phone}({src_weight})")
                else:
                    fallback_key = f"fallback_{src_phone[:2]}"
                    fallback_phone = entry.get(fallback_key)
                    if fallback_phone and fallback_phone in phone2idx:
                        src_idx = phone2idx[fallback_phone]
                        weighted_sum += embed_weight[src_idx] * src_weight
                        total_weight += src_weight
                        source_details.append(f"{fallback_phone}({src_weight},fallback)")

            if total_weight > 0:
                if abs(total_weight - 1.0) > 1e-6:
                    weighted_sum = weighted_sum / total_weight
                init_weight = entry.get("init_weight", 1.0)
                pending[jp_phone] = {
                    "jp_idx": jp_idx,
                    "weighted_sum": weighted_sum,
                    "init_weight": init_weight,
                    "source_details": source_details,
                }
            else:
                initialized[jp_phone] = {"status": "failed", "reason": "no valid sources"}

    # ── Compute fallback mean from full-weight phonemes ──────────────
    full_weight_indices = [
        pending[ph]["jp_idx"] for ph in pending
        if pending[ph]["init_weight"] >= 1.0
    ]
    # Also include pause_mean phonemes
    for ph, info in initialized.items():
        if info.get("status") == "pause_mean" and ph in phone2idx:
            full_weight_indices.append(phone2idx[ph])

    fallback_mean = None
    if full_weight_indices:
        fallback_mean = embed_weight[full_weight_indices].mean(dim=0)

    # ── Pass 2: apply init_weight blending ───────────────────────────
    for jp_phone, pdata in pending.items():
        jp_idx = pdata["jp_idx"]
        weighted_sum = pdata["weighted_sum"]
        init_weight = pdata["init_weight"]

        if init_weight >= 1.0:
            # Full initialization from source — add orthogonal noise to break
            # exact copies (cosine_sim would be 1.0 otherwise, preventing learning).
            # noise_scale=0.5 brings initial cos from 1.0 to ~0.894 (1/sqrt(1+0.5²)),
            # which is close to the decouple_loss target (0.85). With the previous
            # 0.12 scale, cos stayed at ~0.993 and the decouple gradient was ~0
            # (d(cos)/dw → 0 near cos=1), so JP phonemes never separated from EN
            # sources — causing phoneme collapse and吞字/混淆.
            has_single_source = len(pending[jp_phone].get("source_details", [])) <= 1
            if has_single_source:
                embed_weight[jp_idx] = add_orthogonal_noise(weighted_sum, noise_scale=0.5)
            else:
                embed_weight[jp_idx] = weighted_sum
            initialized[jp_phone] = {
                "status": "ok",
                "sources": pdata["source_details"],
                "init_weight": 1.0,
                "norm": weighted_sum.norm().item()
            }
        elif fallback_mean is not None:
            # Blend: init_weight * source + (1 - init_weight) * fallback
            blended = init_weight * weighted_sum + (1 - init_weight) * fallback_mean
            embed_weight[jp_idx] = blended + torch.randn_like(blended) * 0.01
            initialized[jp_phone] = {
                "status": "reduced_init",
                "sources": pdata["source_details"],
                "init_weight": init_weight,
                "norm": embed_weight[jp_idx].norm().item()
            }
        else:
            # No fallback available, use source at reduced weight
            embed_weight[jp_idx] = weighted_sum * init_weight
            initialized[jp_phone] = {
                "status": "reduced_init_no_fallback",
                "sources": pdata["source_details"],
                "init_weight": init_weight,
                "norm": embed_weight[jp_idx].norm().item()
            }

    # ── Final fallback for any remaining failed phonemes ─────────────
    failed = [ph for ph, info in initialized.items() if info.get("status") == "failed"]
    if failed and fallback_mean is not None:
        for jp_phone in failed:
            if jp_phone in phone2idx:
                jp_idx = phone2idx[jp_phone]
                embed_weight[jp_idx] = fallback_mean + torch.randn_like(fallback_mean) * 0.01
                initialized[jp_phone] = {
                    "status": "fallback_mean",
                    "norm": embed_weight[jp_idx].norm().item()
                }

    return embed_weight, initialized


def l2_normalize_rows(embed_weight):
    """L2 normalize each row to unit norm."""
    norms = embed_weight.norm(dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    return embed_weight / norms


def scale_to_target_std(embed_weight, target_std=0.9):
    """Scale entire table so global std matches target."""
    actual_std = embed_weight.std().item()
    if actual_std < 1e-8:
        print("WARNING: Embedding std is near zero, cannot scale")
        return embed_weight, 1.0
    scale_factor = target_std / actual_std
    return embed_weight * scale_factor, scale_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--mapping', default='train/lora_jp/jp_phoneme_mapping.json')
    parser.add_argument('--phoneset', default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--output', default='outputs/lora_jp/init_embed.pt')
    parser.add_argument('--target_std', type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load phone set
    phone2idx = load_phone2idx(args.phoneset)
    print(f"Phone set: {len(phone2idx)} entries")

    # Load mapping
    mapping = load_mapping(args.mapping)
    print(f"Mapping: {len(mapping)} JP phonemes")

    # Load base model embedding
    print(f"\nLoading base model: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    base_sd = ckpt.get('state_dict', ckpt)
    base_embed = base_sd['note_text_encoder.weight'].clone()
    print(f"  Base embedding: {base_embed.shape}, std={base_embed.std():.4f}")

    # Extend to include JP phonemes
    full_size = JP_PHONEME_START + JP_PHONEME_COUNT
    if base_embed.shape[0] < full_size:
        new_embed = torch.zeros(full_size, EMBED_DIM)
        new_embed[:base_embed.shape[0]] = base_embed
        base_embed = new_embed
    print(f"  Extended to: {base_embed.shape}")

    # Initialize JP embeddings from mapping
    print("\nInitializing JP embeddings from English-source mapping...")
    embed_weight, init_log = initialize_jp_embeddings(base_embed, mapping, phone2idx)

    # Count statuses
    ok_count = sum(1 for v in init_log.values() if v.get("status") in ("ok", "pause_mean"))
    fallback_count = sum(1 for v in init_log.values() if v.get("status") == "fallback_mean")
    failed_count = sum(1 for v in init_log.values() if v.get("status") == "failed")
    print(f"  OK: {ok_count}, Fallback: {fallback_count}, Failed: {failed_count}")

    # Step 1: Gently scale JP rows to match base embedding mean norm.
    # We do NOT L2-normalize because that would destroy the natural norm
    # variation inherited from source embeddings. Source blends already
    # have norms close to base embeddings (~22.5) since sources ARE base
    # embeddings. We only apply a gentle correction if the mean norms
    # differ by more than 10%.
    print("\nScaling JP rows to match base mean norm (gentle correction)...")
    base_rows = embed_weight[:JP_PHONEME_START]
    base_mean_norm = base_rows.norm(dim=1).mean().item()
    jp_rows = embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT]
    jp_mean_norm = jp_rows.norm(dim=1).mean().item()
    print(f"  Base mean norm: {base_mean_norm:.4f}")
    print(f"  JP mean norm (pre-scale): {jp_mean_norm:.4f}")

    if jp_mean_norm > 1e-8:
        ratio = base_mean_norm / jp_mean_norm
        # Only scale if deviation > 10%; otherwise preserve source blend norms
        if abs(ratio - 1.0) > 0.10:
            jp_scale = ratio
            embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT] *= jp_scale
            print(f"  Applied scale factor: {jp_scale:.4f} (deviation was {abs(ratio-1.0)*100:.1f}%)")
        else:
            jp_scale = 1.0
            print(f"  No scaling needed (deviation {abs(ratio-1.0)*100:.1f}% < 10%)")
    else:
        jp_scale = 1.0
        print(f"  WARNING: JP mean norm near zero, skipping scale")

    jp_rows = embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT]
    base_std = base_rows.std().item()
    jp_std = jp_rows.std().item()
    print(f"  Base std: {base_std:.4f}, JP std: {jp_std:.4f}")
    print(f"  Final JP mean norm: {jp_rows.norm(dim=1).mean():.4f}")
    print(f"  JP norm range: [{jp_rows.norm(dim=1).min():.4f}, {jp_rows.norm(dim=1).max():.4f}]")

    # Verify no zero vectors
    zero_count = (jp_rows.norm(dim=1) < 1e-6).sum().item()
    if zero_count > 0:
        print(f"  WARNING: {zero_count} JP phonemes have near-zero norm!")
    else:
        print(f"  All JP phonemes have non-zero vectors.")

    # Save
    save_dict = {
        'embed_weight': embed_weight,
        'jp_phoneme_start': JP_PHONEME_START,
        'jp_phoneme_count': JP_PHONEME_COUNT,
        'target_std': args.target_std,
        'jp_scale': jp_scale,
        'base_mean_norm': base_mean_norm,
        'init_log': init_log,
    }
    torch.save(save_dict, args.output)
    print(f"\nSaved to: {args.output}")
    print(f"  Shape: {embed_weight.shape}")
    print(f"  Size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

    # Print per-phoneme details
    print("\n--- Per-phoneme initialization details ---")
    for jp_ph, details in init_log.items():
        status = details.get("status", "?")
        norm = details.get("norm", jp_rows[0].norm().item())  # fallback
        if status == "ok":
            sources = ", ".join(details.get("sources", []))
            print(f"  {jp_ph:10s} OK   norm={norm:.3f}  sources=[{sources}]")
        elif status == "reduced_init":
            sources = ", ".join(details.get("sources", []))
            iw = details.get("init_weight", "?")
            print(f"  {jp_ph:10s} REDUCED(norm={norm:.3f}, init_weight={iw})  sources=[{sources}]")
        elif status == "pause_mean":
            print(f"  {jp_ph:10s} PAUSE norm={norm:.3f}")
        elif status == "fallback_mean":
            print(f"  {jp_ph:10s} FALLBACK norm={norm:.3f}")
        else:
            print(f"  {jp_ph:10s} FAIL  {details.get('reason', '')}")


if __name__ == '__main__':
    main()
