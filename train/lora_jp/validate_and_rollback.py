"""
Task 5: Automatic validation and rollback decision logic.

Evaluates checkpoint quality using three metrics:
  1. Embedding std (healthy: 0.5-1.5)
  2. Avg predicted frames per phoneme (target: 6-10)
  3. Validation loss

Triggers rollback if any danger threshold is crossed.
Optionally synthesizes test sentences for subjective evaluation.

Usage:
    python train/lora_jp/validate_and_rollback.py \
        --checkpoint outputs/lora_jp/stage1/best.pt \
        --model_path pretrained_models/SoulX-Singer/model.pt \
        --config soulxsinger/config/soulxsinger.yaml
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.models.soulxsinger import SoulXSinger
from train.lora_jp.dataset import JpLoRADataset, collate_fn

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 33  # jp_pau removed; 33 JP phonemes
EMBED_DIM = 512

# Thresholds
EMBED_STD_MIN = 0.5
EMBED_STD_MAX = 1.5
LOSS_INFLATE_MAX = 2.0  # 200% of baseline
# Collapse threshold: cos > 0.998 means embeddings are nearly identical to
# EN source (init noise was 0.08, so initial cos ~0.9968). Anything above
# 0.998 after training means decouple loss failed to push them apart.
# The training target is cos < 0.90, but 0.90-0.998 is "not yet collapsed"
# (just slow to decouple), while > 0.998 is "fully collapsed".
COS_SIM_COLLAPSE_THRESHOLD = 0.998


def load_jp_to_en_source_indices(mapping_path, phoneset_path):
    """Build JP->EN source index map from config files.

    Same logic as train_staged.py — single-source phonemes only.
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    jp_to_en = {}
    for jp_name, entry in mapping.items():
        if entry.get("strategy") == "pause_mean":
            continue
        if jp_name not in phone2idx:
            continue
        jp_idx = phone2idx[jp_name]
        jp_offset = jp_idx - JP_PHONEME_START
        if jp_offset < 0 or jp_offset >= JP_PHONEME_COUNT:
            continue
        sources = entry.get("sources", [])
        if len(sources) == 1:
            src_phone = sources[0]["phone"]
            if src_phone in phone2idx:
                jp_to_en[jp_offset] = phone2idx[src_phone]
    return jp_to_en


def compute_embedding_stats(checkpoint, jp_to_en=None):
    """Compute JP embedding statistics including cosine similarity with EN sources."""
    jp_embed = checkpoint.get('jp_embed')
    embed_weight = checkpoint.get('embed_state_dict', {}).get('weight')
    if jp_embed is None and embed_weight is not None:
        jp_embed = embed_weight[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT]

    if jp_embed is None:
        return {"error": "No JP embedding found in checkpoint"}

    result = {
        "jp_std": jp_embed.std().item(),
        "jp_mean": jp_embed.mean().item(),
        "jp_min": jp_embed.min().item(),
        "jp_max": jp_embed.max().item(),
        "jp_mean_norm": jp_embed.norm(dim=1).mean().item(),
        "jp_min_norm": jp_embed.norm(dim=1).min().item(),
        "jp_max_norm": jp_embed.norm(dim=1).max().item(),
        "zero_vectors": (jp_embed.norm(dim=1) < 1e-6).sum().item(),
    }

    # Compute cosine similarity with EN sources (passed in via jp_to_en)
    if embed_weight is not None and jp_to_en:
        sims = []
        for jp_offset, en_idx in jp_to_en.items():
            jp_idx = JP_PHONEME_START + jp_offset
            if jp_idx < embed_weight.shape[0] and en_idx < embed_weight.shape[0]:
                sim = F.cosine_similarity(
                    embed_weight[jp_idx:jp_idx+1],
                    embed_weight[en_idx:en_idx+1]
                ).item()
                sims.append(sim)
        if sims:
            result["jp_en_cos_avg"] = sum(sims) / len(sims)
            result["jp_en_cos_min"] = min(sims)
            result["jp_en_cos_max"] = max(sims)
            result["jp_en_cos_collapsed"] = sum(1 for s in sims if s > COS_SIM_COLLAPSE_THRESHOLD)

    return result


@torch.no_grad()
def compute_val_metrics(model, dataloader, device):
    """Compute validation flow-matching loss and avg frames per phoneme."""
    model.eval()
    total_loss = 0
    total_frames = 0
    total_phonemes = 0
    n = 0

    for batch in dataloader:
        if batch is None:
            continue
        waveform = batch.get('waveform')
        if waveform is None:
            continue
        try:
            note_text = batch['phoneme'].to(device)
            note_pitch = batch['note_pitch'].to(device)
            note_type = batch['note_type'].to(device)
            mel2note = batch['mel2note'].to(device)
            mel_lens = batch['mel_len'].to(device)
            f0 = batch.get('f0')
            if f0 is not None:
                f0 = f0.to(device)
            waveform = waveform.to(device)

            features = (model.note_text_encoder(note_text) +
                        model.note_pitch_encoder(note_pitch) +
                        model.note_type_encoder(note_type))
            features = model.preflow(features)
            mel_feat = model.expand_states(features, mel2note)

            if f0 is not None and f0.shape[1] > 0:
                f0_coarse = model.f0_to_coarse(f0)
                mel_feat = mel_feat + model.f0_encoder(f0_coarse)[:, :mel_feat.shape[1], :]

            target_mel = model.mel(waveform.float())
            T = min(target_mel.shape[1], mel_feat.shape[1])
            target_mel = target_mel[:, :T, :]
            mel_feat = mel_feat[:, :T, :]

            x_mask = (torch.arange(T, device=device).unsqueeze(0)
                      < mel_lens.unsqueeze(1).clamp(max=T)).float()

            noise, x, flow_pred, final_mask, prompt_len = model.cfm_decoder(
                target_mel, x_mask, mel_feat, is_prompt=None)

            sigma = model.cfm_decoder.model.sigma
            flow_target = x - (1 - sigma) * noise
            loss = ((flow_pred - flow_target) ** 2 * final_mask).sum() \
                   / final_mask.sum().clamp(min=1)
            total_loss += loss.item()

            # Use unique non-PAD phoneme IDs for meaningful ratio
            unique_ph = (note_text.unique() > 0).sum().item()
            n_frames = mel2note.shape[1]
            total_frames += n_frames
            total_phonemes += max(unique_ph, 1)
            n += 1
        except RuntimeError as e:
            if 'device' in str(e).lower() or 'cuda' in str(e).lower():
                print(f'  FATAL: Device error during validation: {e}')
                raise
            print(f'  Val error: {e}')

    avg_loss = total_loss / max(n, 1)
    avg_frames = total_frames / max(total_phonemes, 1)
    return avg_loss, avg_frames


def evaluate_checkpoint(checkpoint_path, model_path, config_path, dataset_metadata,
                        dataset_wav_dir, phoneset_path, device='cuda'):
    """Run full validation on a checkpoint.

    Returns: dict with metrics and rollback decision.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'

    config = OmegaConf.load(config_path)

    # Load base model
    model = SoulXSinger(config)
    base_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=True)

    # Apply checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'preflow_state_dict' in ckpt:
        # Filter out LayerNorm keys if present (from old checkpoints)
        pf_sd = {k: v for k, v in ckpt['preflow_state_dict'].items()
                 if not k.startswith('norm.')}
        model.preflow.load_state_dict(pf_sd, strict=False)

    if 'embed_state_dict' in ckpt:
        ft_weight = ckpt['embed_state_dict']['weight']
        if ft_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
            new_emb = nn.Embedding(ft_weight.shape[0], EMBED_DIM)
            new_emb.weight.data = ft_weight
            model.note_text_encoder = new_emb
        else:
            model.note_text_encoder.weight.data[:ft_weight.shape[0]] = ft_weight

    if 'pitch_encoder_state_dict' in ckpt:
        pe_weight = ckpt['pitch_encoder_state_dict']['weight']
        model.note_pitch_encoder.weight.data[:pe_weight.shape[0]] = pe_weight

    # Restore cond_emb (trained with flow-matching loss)
    if 'cond_emb_state_dict' in ckpt:
        model.cfm_decoder.model.cond_emb.load_state_dict(
            ckpt['cond_emb_state_dict'])

    model = model.to(device)

    # Build JP->EN source map from config (same logic as train_staged.py)
    mapping_path = os.path.join(os.path.dirname(__file__), 'jp_phoneme_mapping.json')
    jp_to_en = load_jp_to_en_source_indices(mapping_path, phoneset_path)

    # Embedding stats
    embed_stats = compute_embedding_stats(ckpt, jp_to_en=jp_to_en)

    # Validation metrics — use a held-out val split, NOT the training set.
    # We reuse train_staged.split_train_val with the same seed so the val
    # split here matches the one used during training.
    from train.lora_jp.train_staged import split_train_val
    full_dataset = JpLoRADataset(
        metadata_path=dataset_metadata,
        wav_dir=dataset_wav_dir,
        phoneset_path=phoneset_path,
        sample_rate=config.audio.sample_rate,
        hop_size=config.audio.hop_size,
    )
    _, val_dataset = split_train_val(full_dataset, val_ratio=0.1, seed=42)
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    val_loss, avg_frames = compute_val_metrics(model, dataloader, device)

    return {
        "checkpoint": checkpoint_path,
        "epoch": ckpt.get("epoch", "?"),
        "phase": ckpt.get("phase", "?"),
        "val_loss": val_loss,
        "avg_frames_per_phoneme": avg_frames,
        "embed_stats": embed_stats,
    }


def make_rollback_decision(metrics, baseline_loss=None, phase=0):
    """Decide whether to rollback based on metrics.

    Returns: (should_rollback: bool, reasons: list[str])
    """
    reasons = []
    es = metrics.get("embed_stats", {})

    # Check embedding std
    jp_std = es.get("jp_std", 0)
    if jp_std < EMBED_STD_MIN:
        reasons.append(f"Embedding std {jp_std:.4f} < {EMBED_STD_MIN} (prior forgetting)")
    elif jp_std > EMBED_STD_MAX:
        reasons.append(f"Embedding std {jp_std:.4f} > {EMBED_STD_MAX} (divergence)")

    # Check zero vectors
    if es.get("zero_vectors", 0) > 0:
        reasons.append(f"{es['zero_vectors']} zero vectors in JP embeddings")

    # Check cosine similarity collapse (only for phases where embeddings are unfrozen)
    if phase >= 2:
        collapsed = es.get("jp_en_cos_collapsed", 0)
        if collapsed > 0:
            reasons.append(f"{collapsed} JP phonemes collapsed onto EN source (cos > {COS_SIM_COLLAPSE_THRESHOLD})")

    # Note: avg_frames_per_phoneme is a dataset characteristic (fixed mel2note
    # alignment), not a model output. It's logged for diagnostics only, not
    # used as a rollback trigger.

    # Check loss inflation
    val_loss = metrics.get("val_loss", 0)
    if baseline_loss and baseline_loss > 0:
        inflation = val_loss / baseline_loss
        if inflation > LOSS_INFLATE_MAX:
            reasons.append(f"Val loss {val_loss:.4f} is {inflation:.1f}x baseline {baseline_loss:.4f}")

    return len(reasons) > 0, reasons


def load_baseline_loss(output_dir):
    """Load the baseline loss from Phase 1 first checkpoint."""
    stage1_dir = os.path.join(output_dir, 'stage1')
    if os.path.exists(os.path.join(stage1_dir, 'best.pt')):
        ckpt = torch.load(os.path.join(stage1_dir, 'best.pt'),
                          map_location='cpu', weights_only=False)
        return ckpt.get('loss', None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model_path', default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--config', default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--phoneset_path', default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--dataset_metadata', default='train/lora_jp/dataset/metadata.json')
    parser.add_argument('--dataset_wav_dir', default='train/lora_jp/dataset/wavs')
    parser.add_argument('--output_dir', default='outputs/lora_jp')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--baseline_loss', type=float, default=None)
    parser.add_argument('--synthesize', action='store_true', help='Synthesize test sentences')
    args = parser.parse_args()

    print(f"Validating checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Run evaluation
    metrics = evaluate_checkpoint(
        args.checkpoint, args.model_path, args.config,
        args.dataset_metadata, args.dataset_wav_dir, args.phoneset_path,
        args.device
    )

    # Print results
    print(f"\nEpoch: {metrics['epoch']}, Phase: {metrics['phase']}")
    print(f"Validation loss: {metrics['val_loss']:.4f}")
    print(f"Avg frames/phoneme: {metrics['avg_frames_per_phoneme']:.1f} (dataset characteristic, informational)")

    es = metrics['embed_stats']
    print(f"\n--- Embedding Stats ---")
    print(f"  JP std: {es['jp_std']:.4f}")
    print(f"  JP mean: {es['jp_mean']:.6f}")
    print(f"  JP range: [{es['jp_min']:.4f}, {es['jp_max']:.4f}]")
    print(f"  JP mean norm: {es['jp_mean_norm']:.4f}")
    print(f"  JP norm range: [{es['jp_min_norm']:.4f}, {es['jp_max_norm']:.4f}]")
    print(f"  Zero vectors: {es['zero_vectors']}")
    if 'jp_en_cos_avg' in es:
        print(f"  JP-EN cosine: avg={es['jp_en_cos_avg']:.4f} min={es['jp_en_cos_min']:.4f} max={es['jp_en_cos_max']:.4f}")
        collapsed = es.get('jp_en_cos_collapsed', 0)
        if collapsed > 0:
            print(f"  WARNING: {collapsed} phonemes collapsed onto EN source (cos > {COS_SIM_COLLAPSE_THRESHOLD})")

    # Rollback decision
    baseline = args.baseline_loss or load_baseline_loss(args.output_dir)
    should_rollback, reasons = make_rollback_decision(metrics, baseline, phase=metrics.get('phase', 0))

    print(f"\n{'='*60}")
    if should_rollback:
        print("ROLLBACK TRIGGERED:")
        for r in reasons:
            print(f"  - {r}")
        print("\nAction: Switch to standard normal init (std=0.02), restart Phase 1")
    else:
        print("VALIDATION PASSED")
        if es['jp_std'] >= EMBED_STD_MIN and es['jp_std'] <= EMBED_STD_MAX:
            print(f"  Embedding std: OK ({es['jp_std']:.4f})")
        print(f"  Val loss: {metrics['val_loss']:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, 'validation_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "metrics": {k: v for k, v in metrics.items() if k != "embed_stats"},
            "embed_stats": es,
            "rollback": should_rollback,
            "reasons": reasons,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Synthesis check (optional)
    if args.synthesize:
        print(f"\n--- Synthesis Check ---")
        synthesize_test_sentences(args.checkpoint, args.model_path, args.config,
                                  args.phoneset_path, args.output_dir, args.device)


def synthesize_test_sentences(checkpoint_path, model_path, config_path,
                               phoneset_path, output_dir, device='cuda'):
    """Print test sentences for manual evaluation in the app.

    The synthesis check requires the full diffusion pipeline which is only
    available in the ONNX inference runtime (SXSEditor app). Instead of
    trying to replicate it in PyTorch, we print the test sentences so the
    user can enter them in the app and listen.
    """
    test_sentences = [
        "わ (wa) — should sound like semivowel w + a",
        "か (ka) — should sound like k + a",
        "ら (ra) — should sound like tap r + a",
        "さくら (sakura) — s a k u r a",
        "はる (haru) — h a r u",
    ]

    print("\nTest sentences for manual evaluation in SXSEditor:")
    print("Enter these as lyrics in the fragment editor and listen:")
    for i, sent in enumerate(test_sentences):
        print(f"  {i+1}. {sent}")
    print("\nCompare: 'わ' should NOT sound like 'か'.")
    print("If they sound identical, the embedding initialization is still wrong.")


if __name__ == '__main__':
    main()
