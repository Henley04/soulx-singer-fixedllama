"""
Task 3+4: Staged training with LayerNorm at preflow input.

Three-phase training:
  Phase 1 (Warmup): Freeze embedding, train preflow (15 epochs)
  Phase 2 (Embed FT): Unfreeze embedding with lower LR (to epoch 40)
  Phase 3 (Joint): Full fine-tune with duration monitoring (to epoch 80)

Precision: bfloat16 autocast (no NVFP4). NVFP4 weight-only quantization was
removed because (a) it only accelerates forward matmul, not backward dgrad,
so the training speedup was marginal, and (b) it introduced ~9.5% relative
RMSE precision loss that degraded gradient signal quality. With bf16 autocast,
both forward and backward run at full bf16 precision with native kernels —
no GradScaler needed (bf16 has fp32-equivalent exponent range).

Gradient checkpointing is enabled on the 22-layer DiffLlama (diff_estimator)
to trade ~20% extra forward compute for large activation-memory savings,
enabling larger batch sizes.

The training objective is the actual cfm_decoder flow-matching loss (the same
objective used at inference time via `reverse_diffusion`), NOT a proxy
MelProjection MSE loss. This keeps the training objective aligned with
inference and is the root-cause fix for unintelligible Japanese pronunciation.

Usage:
    # Phase 1
    python train/lora_jp/train_staged.py --phase 1 \
        --init_embed outputs/lora_jp/init_embed.pt \
        --model_path pretrained_models/SoulX-Singer/model.pt

    # Phase 2
    python train/lora_jp/train_staged.py --phase 2 \
        --resume outputs/lora_jp/stage1/best.pt

    # Phase 3
    python train/lora_jp/train_staged.py --phase 3 \
        --resume outputs/lora_jp/stage2/best.pt
"""

import os
import sys
import json
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.models.soulxsinger import SoulXSinger
from train.lora_jp.dataset import JpLoRADataset, collate_fn

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 33  # jp_pau removed; jp_a..jp_cl now 33 phonemes
EMBED_DIM = 512

# Phase configs
PHASE_CONFIGS = {
    1: {"epochs": 15, "freeze_embed": True,  "embed_lr_ratio": 0.0, "loss_threshold": None, "decouple": True},
    2: {"epochs": 40, "freeze_embed": False, "embed_lr_ratio": 0.2, "loss_threshold": None, "decouple": True},
    3: {"epochs": 80, "freeze_embed": False, "embed_lr_ratio": 1.0, "loss_threshold": None, "decouple": False},
}


def load_jp_to_en_source(mapping_path):
    """Build JP->EN source index map from jp_phoneme_mapping.json.

    For each JP phoneme with a single EN source, record (jp_offset, en_idx).
    Only single-source phonemes are included because decoupling loss compares
    one-to-one. Multi-source blends don't have a single EN source to decouple from.

    Returns: dict {jp_offset: en_phone_name}
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Load phone_set to get indices
    phoneset_path = os.path.join(os.path.dirname(__file__), 'jp_phone_set.json')
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
        # Only include single-source phonemes for clean decoupling
        if len(sources) == 1:
            src_phone = sources[0]["phone"]
            if src_phone in phone2idx:
                jp_to_en[jp_offset] = src_phone
    return jp_to_en, phone2idx


def setup_phase(model, phase, init_embed_path=None, resume_path=None, mapping_path=None):
    """Configure model for the given training phase.

    Returns: (model, checkpoint_info)
    """
    config = PHASE_CONFIGS[phase]
    ckpt_info = {"phase": phase, "epoch_start": 1}

    # Load initialization or resume checkpoint
    if init_embed_path and phase == 1:
        print(f"  Loading init embeddings from: {init_embed_path}")
        init_data = torch.load(init_embed_path, map_location='cpu', weights_only=False)
        embed_weight = init_data['embed_weight']
        target_device = model.note_text_encoder.weight.device
        embed_weight = embed_weight.to(target_device)
        if embed_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
            new_emb = nn.Embedding(embed_weight.shape[0], EMBED_DIM)
            new_emb.weight.data = embed_weight
            model.note_text_encoder = new_emb
        else:
            model.note_text_encoder.weight.data[:embed_weight.shape[0]] = embed_weight
        ckpt_info["init_source"] = init_embed_path

    elif resume_path:
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)

        # Restore preflow
        if 'preflow_state_dict' in ckpt:
            model.preflow.load_state_dict(ckpt['preflow_state_dict'], strict=False)

        # Restore embedding
        if 'embed_state_dict' in ckpt:
            ft_weight = ckpt['embed_state_dict']['weight']
            ft_weight = ft_weight.to(model.note_text_encoder.weight.device)
            if ft_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
                new_emb = nn.Embedding(ft_weight.shape[0], EMBED_DIM)
                new_emb.weight.data = ft_weight
                model.note_text_encoder = new_emb
            else:
                model.note_text_encoder.weight.data[:ft_weight.shape[0]] = ft_weight

        # Restore cond_emb (Linear(512, 1024) inside cfm_decoder that projects
        # decoder_inp features to the diff_estimator hidden size).
        # Note: old checkpoints produced by the MelProjection variant do not
        # contain this key; that's fine — we just skip restoration.
        if 'cond_emb_state_dict' in ckpt:
            model.cfm_decoder.model.cond_emb.load_state_dict(ckpt['cond_emb_state_dict'])
            print("  Restored cond_emb from checkpoint")

        ckpt_info["epoch_start"] = ckpt.get("epoch", 0) + 1
        ckpt_info["prev_loss"] = ckpt.get("loss", float('inf'))

    # Configure freeze state — freeze everything first, then selectively unfreeze.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze preflow
    for param in model.preflow.parameters():
        param.requires_grad = True

    # Unfreeze cond_emb (Linear(512, 1024) inside cfm_decoder). This is the
    # only trainable part of the cfm_decoder; it projects decoder_inp features
    # to the diff_estimator hidden size, so adapting it for JP is critical.
    for param in model.cfm_decoder.model.cond_emb.parameters():
        param.requires_grad = True

    # pitch_encoder is ALWAYS frozen — pitch is a MIDI index with no
    # language-specific semantic. Base model trained on 42000+ hours of
    # CN/EN/YUE data already covers MIDI 0-255 uniformly (verified: all
    # 256 rows have norm ~22.5). JP LoRA's PJS data only covers MIDI 36-72,
    # so fine-tuning pitch_encoder would (a) distort base's uniform pitch
    # representation and (b) be discarded at ONNX export time anyway.
    # See: export_onnx.py does NOT export note_pitch_encoder.
    print("  Pitch encoder: FROZEN (shared with base model)")

    # cfm_decoder diff_estimator (22-layer DiffLlama) is FROZEN. Gradients
    # flow THROUGH it to reach preflow + embedding + cond_emb.
    print("  cfm_decoder.diff_estimator: FROZEN")
    print("  cfm_decoder.cond_emb: UNFROZEN")

    # Unfreeze embedding if not frozen
    if not config["freeze_embed"]:
        model.note_text_encoder.weight.requires_grad = True
        print("  Text encoder embedding: UNFROZEN")
    else:
        print("  Text encoder embedding: FROZEN")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    return model, ckpt_info


def build_optimizer(model, phase, base_lr):
    """Build optimizer with per-group learning rates for the given phase.

    pitch_encoder is intentionally excluded — it's always frozen (see
    setup_phase). Only preflow, cfm_decoder.cond_emb, and (when unfrozen)
    text encoder embedding are trained. The diff_estimator stays frozen but
    is still part of the forward graph so gradients reach cond_emb.
    """
    config = PHASE_CONFIGS[phase]
    param_groups = [
        {'params': model.preflow.parameters(), 'lr': base_lr},
        {'params': model.cfm_decoder.model.cond_emb.parameters(), 'lr': base_lr},
    ]
    if not config["freeze_embed"]:
        embed_lr = base_lr * config["embed_lr_ratio"]
        param_groups.append({
            'params': model.note_text_encoder.parameters(),
            'lr': embed_lr
        })
        print(f"  Optimizer: preflow+cond_emb lr={base_lr}, embed lr={embed_lr}")
    else:
        print(f"  Optimizer: preflow+cond_emb lr={base_lr}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, writer,
                    phase=1, use_amp=True, jp_to_en_indices=None,
                    amp_dtype=torch.bfloat16):
    model.train()
    total_loss = 0
    total_recon = 0
    total_decouple = 0
    n = 0
    use_cuda_amp = use_amp and device.startswith('cuda')

    for bi, batch in enumerate(dataloader):
        if batch is None:
            continue
        # Skip batches without waveform — flow-matching loss requires a target mel.
        waveform = batch.get('waveform')
        if waveform is None:
            continue

        optimizer.zero_grad()
        try:
            with torch.amp.autocast('cuda', enabled=use_cuda_amp, dtype=amp_dtype):
                note_text = batch['phoneme'].to(device)
                note_pitch = batch['note_pitch'].to(device)
                note_type = batch['note_type'].to(device)
                mel2note = batch['mel2note'].to(device)
                mel_lens = batch['mel_len'].to(device)  # (B,)
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
                    f0_enc = model.f0_encoder(f0_coarse)
                    mel_feat = mel_feat + f0_enc[:, :mel_feat.shape[1], :]

                # Target mel from waveform (normalized: (x - mean) / sqrt(var))
                target_mel = model.mel(waveform.float())  # (B, T_mel, 128)

                # Align lengths between target_mel and mel_feat
                T = min(target_mel.shape[1], mel_feat.shape[1])
                target_mel = target_mel[:, :T, :]
                mel_feat = mel_feat[:, :T, :]

                # Build x_mask from mel_lens: 1 for valid frames, 0 for padding.
                x_mask = (torch.arange(T, device=device).unsqueeze(0)
                          < mel_lens.unsqueeze(1).clamp(max=T)).float()  # (B, T)

                # Flow-matching loss via cfm_decoder forward. is_prompt=None
                # triggers random prompt-length sampling (with 20% CFG drop),
                # which is correct for few-shot diffusion training.
                noise, x, flow_pred, final_mask, prompt_len = model.cfm_decoder(
                    target_mel, x_mask, mel_feat, is_prompt=None)

                sigma = model.cfm_decoder.model.sigma
                flow_target = x - (1 - sigma) * noise
                recon_loss = ((flow_pred - flow_target) ** 2 * final_mask).sum() \
                             / final_mask.sum().clamp(min=1)

                # Decoupling loss: prevent JP embeddings from collapsing onto EN sources.
                # Only applies when embeddings are unfrozen (phase >= 2).
                # Uses direct cosine penalty (no margin) so the gradient is always
                # pushing cos down as long as cos > 0. Target: cos < 0.85.
                decouple_loss = torch.tensor(0.0, device=device)
                if (PHASE_CONFIGS[phase].get("decouple", True)
                        and not PHASE_CONFIGS[phase]["freeze_embed"]
                        and jp_to_en_indices is not None
                        and jp_to_en_indices):
                    embed_weight = model.note_text_encoder.weight
                    for jp_offset, en_idx in jp_to_en_indices.items():
                        jp_idx = JP_PHONEME_START + jp_offset
                        if jp_idx < embed_weight.shape[0] and en_idx < embed_weight.shape[0]:
                            cos_sim = F.cosine_similarity(
                                embed_weight[jp_idx:jp_idx+1],
                                embed_weight[en_idx:en_idx+1]
                            )
                            # Direct cos penalty (always active when cos > 0.85 target).
                            # Squared so gradient grows for high cos values.
                            decouple_loss = decouple_loss + F.relu(cos_sim - 0.85) ** 2
                    # Note: at cos≈0.993, the gradient d(cos)/dw → 0, so decouple
                    # cannot start moving embeddings from cos≈1 by itself. It needs
                    # the flow-matching gradient through diff_estimator to first
                    # break the symmetry. Phase 3 disables decouple entirely and
                    # relies on flow-matching signal alone.

                loss = recon_loss + decouple_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_decouple += decouple_loss.item() if decouple_loss.item() > 0 else 0
            n += 1
            writer.add_scalar('train/loss_step', loss.item(), epoch * 1000 + bi)
            writer.add_scalar('train/recon_step', recon_loss.item(), epoch * 1000 + bi)
            if bi % 10 == 0:
                print(f'  Epoch {epoch} [{bi}] loss={loss.item():.4f} '
                      f'(flow={recon_loss.item():.4f} decouple={decouple_loss.item():.4f})')
        except RuntimeError as e:
            if 'device' in str(e).lower() or 'cuda' in str(e).lower():
                print(f'  FATAL: Device error at batch {bi}: {e}')
                raise
            print(f'  Error batch {bi}: {e}')
            import traceback
            traceback.print_exc()

    avg = total_loss / max(n, 1)
    avg_recon = total_recon / max(n, 1)
    avg_decouple = total_decouple / max(n, 1)
    writer.add_scalar('train/loss_epoch', avg, epoch)
    writer.add_scalar('train/recon_epoch', avg_recon, epoch)
    writer.add_scalar('train/decouple_epoch', avg_decouple, epoch)
    return avg


def compute_jp_en_cosine_stats(model, jp_to_en_indices):
    """Compute cosine similarity between JP embeddings and their EN sources."""
    if not jp_to_en_indices:
        return 0.0, 0.0, 0.0
    embed = model.note_text_encoder.weight.data
    sims = []
    for jp_offset, en_idx in jp_to_en_indices.items():
        jp_idx = JP_PHONEME_START + jp_offset
        if jp_idx < embed.shape[0] and en_idx < embed.shape[0]:
            sim = F.cosine_similarity(
                embed[jp_idx:jp_idx+1], embed[en_idx:en_idx+1]
            ).item()
            sims.append(sim)
    if not sims:
        return 0.0, 0.0, 0.0
    return sum(sims) / len(sims), min(sims), max(sims)


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer, amp_dtype=torch.bfloat16):
    model.eval()
    total_loss = 0
    n = 0

    for batch in dataloader:
        if batch is None:
            continue
        # Skip batches without waveform — flow-matching loss requires a target mel.
        waveform = batch.get('waveform')
        if waveform is None:
            continue
        try:
            note_text = batch['phoneme'].to(device)
            note_pitch = batch['note_pitch'].to(device)
            note_type = batch['note_type'].to(device)
            mel2note = batch['mel2note'].to(device)
            mel_lens = batch['mel_len'].to(device)  # (B,)
            f0 = batch.get('f0')
            if f0 is not None:
                f0 = f0.to(device)
            waveform = waveform.to(device)

            with torch.amp.autocast('cuda', enabled=device.startswith('cuda'),
                                    dtype=amp_dtype):
                features = (model.note_text_encoder(note_text) +
                            model.note_pitch_encoder(note_pitch) +
                            model.note_type_encoder(note_type))
                features = model.preflow(features)
                mel_feat = model.expand_states(features, mel2note)

                if f0 is not None and f0.shape[1] > 0:
                    f0_coarse = model.f0_to_coarse(f0)
                    mel_feat = mel_feat + model.f0_encoder(f0_coarse)[:, :mel_feat.shape[1], :]

                target_mel = model.mel(waveform.float())  # (B, T_mel, 128)

                T = min(target_mel.shape[1], mel_feat.shape[1])
                target_mel = target_mel[:, :T, :]
                mel_feat = mel_feat[:, :T, :]

                x_mask = (torch.arange(T, device=device).unsqueeze(0)
                          < mel_lens.unsqueeze(1).clamp(max=T)).float()  # (B, T)

                noise, x, flow_pred, final_mask, prompt_len = model.cfm_decoder(
                    target_mel, x_mask, mel_feat, is_prompt=None)

            sigma = model.cfm_decoder.model.sigma
            flow_target = x - (1 - sigma) * noise
            loss = ((flow_pred - flow_target) ** 2 * final_mask).sum() \
                   / final_mask.sum().clamp(min=1)
            total_loss += loss.item()
            n += 1
        except RuntimeError as e:
            if 'device' in str(e).lower() or 'cuda' in str(e).lower():
                print(f'  FATAL: Device error during validation: {e}')
                raise
            print(f'  Val error: {e}')

    avg_loss = total_loss / max(n, 1)
    writer.add_scalar('val/loss_epoch', avg_loss, epoch)
    return avg_loss


def save_checkpoint(model, epoch, loss, output_dir, phase):
    os.makedirs(output_dir, exist_ok=True)
    embed_weight = model.note_text_encoder.weight.data.clone()

    ckpt = {
        'epoch': epoch,
        'loss': loss,
        'phase': phase,
        'preflow_state_dict': model.preflow.state_dict(),
        'embed_state_dict': {'weight': embed_weight},
        'jp_embed': embed_weight[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT].clone(),
        'cond_emb_state_dict': model.cfm_decoder.model.cond_emb.state_dict(),
    }

    path = os.path.join(output_dir, f'epoch{epoch:03d}.pt')
    latest_path = os.path.join(output_dir, 'best.pt')
    torch.save(ckpt, path)
    torch.save(ckpt, latest_path)

    jp_std = ckpt['jp_embed'].std().item()
    print(f'  Saved phase {phase} epoch {epoch} (jp_std={jp_std:.4f})')
    return path


def split_train_val(dataset, val_ratio=0.1, seed=42):
    """Split dataset into train and validation subsets.

    Uses fixed seed for reproducibility. Returns (train_dataset, val_dataset).
    """
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    print(f'  Split: {n_train} train / {n_val} val (seed={seed})')
    return train_subset, val_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--model_path', default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--config', default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--phoneset_path', default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--mapping_path', default='train/lora_jp/jp_phoneme_mapping.json')
    parser.add_argument('--dataset_metadata', default='train/lora_jp/dataset/metadata.json')
    parser.add_argument('--dataset_wav_dir', default='train/lora_jp/dataset/wavs')
    parser.add_argument('--output_dir', default='outputs/lora_jp')
    parser.add_argument('--init_embed', default=None, help='Path to init_embed.pt for Phase 1')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=None, help='Override phase epoch count')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split')
    parser.add_argument('--grad_checkpoint', action='store_true', default=True,
                        help='Enable gradient checkpointing on diff_estimator (default)')
    parser.add_argument('--no-grad-checkpoint', dest='grad_checkpoint', action='store_false',
                        help='Disable gradient checkpointing')
    args = parser.parse_args()

    # Validate device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"WARNING: --device={args.device} requested but CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    if args.device.startswith('cuda'):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: Training on CPU — this will be very slow.")

    phase_config = PHASE_CONFIGS[args.phase]
    total_epochs = args.epochs or phase_config["epochs"]
    stage_dir = os.path.join(args.output_dir, f'stage{args.phase}')
    os.makedirs(stage_dir, exist_ok=True)

    config = OmegaConf.load(args.config)

    # Build JP->EN source index map from mapping config
    jp_to_en_names, phone2idx = load_jp_to_en_source(args.mapping_path)
    jp_to_en_indices = {}
    for jp_offset, en_name in jp_to_en_names.items():
        if en_name in phone2idx:
            jp_to_en_indices[jp_offset] = phone2idx[en_name]
    print(f"  JP->EN decoupling map: {len(jp_to_en_indices)} entries")

    # Determine precision: bf16 autocast (no NVFP4, no precision loss).
    # bf16 has fp32-equivalent exponent range, so no GradScaler is needed.
    use_cuda_amp = args.device.startswith('cuda')
    amp_dtype = torch.bfloat16 if use_cuda_amp else torch.float32
    print(f"[Precision] bf16 autocast (no NVFP4, no precision loss)")

    # Load base model
    print(f'[Phase {args.phase}] Loading base model...')
    model = SoulXSinger(config)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)

    # Move to device
    model = model.to(args.device)

    # Enable gradient checkpointing on the frozen 22-layer DiffLlama to save
    # activation memory. This allows larger batch sizes during fine-tuning.
    # use_reentrant=False in llama.py supports the cond_embedding kwarg.
    if args.grad_checkpoint and hasattr(model.cfm_decoder.model.diff_estimator, 'gradient_checkpointing'):
        model.cfm_decoder.model.diff_estimator.gradient_checkpointing = True
        print("[GradCheckpoint] Enabled on diff_estimator (22 layers)")
    else:
        print("[GradCheckpoint] Disabled")

    # Setup phase (also handles resume, including cond_emb restoration).
    # setup_phase freezes everything, then unfreezes preflow + cond_emb
    # (and optionally the text encoder embedding). The cfm_decoder
    # diff_estimator stays FROZEN — gradients flow through it to reach
    # preflow + embedding + cond_emb.
    print(f'[Phase {args.phase}] Setting up...')
    model, ckpt_info = setup_phase(model, args.phase, args.init_embed, args.resume)
    epoch_start = ckpt_info.get("epoch_start", 1)

    # vocoder is never used in training — keep it frozen to save VRAM.
    for param in model.vocoder.parameters():
        param.requires_grad = False

    # Optimizer (after model is on device, so optimizer states are on GPU from the start).
    # Only trainable params are included; quantized frozen weights are excluded.
    optimizer = build_optimizer(model, args.phase, args.lr)

    # GradScaler: bf16 has fp32-equivalent exponent range, no scaling needed.
    # GradScaler(enabled=False) is a no-op passthrough for loss.backward().
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    # Scheduler
    steps_per_epoch = 50  # approximate
    def lr_lambda(step):
        warmup = min(200, steps_per_epoch * 2)
        if step < warmup:
            return max(0.01, step / warmup)
        progress = (step - warmup) / max(1, total_epochs * steps_per_epoch - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dataset with train/val split
    print('[Phase {}] Loading dataset...'.format(args.phase))
    full_dataset = JpLoRADataset(
        metadata_path=args.dataset_metadata,
        wav_dir=args.dataset_wav_dir,
        phoneset_path=args.phoneset_path,
        sample_rate=config.audio.sample_rate,
        hop_size=config.audio.hop_size,
    )
    train_dataset, val_dataset = split_train_val(full_dataset, args.val_ratio, args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    writer = SummaryWriter(log_dir=os.path.join(stage_dir, 'runs'))

    print(f'\n=== Phase {args.phase}: epochs {epoch_start}-{total_epochs}, '
          f'lr={args.lr}, batch={args.batch_size} ===\n')

    best_loss = float('inf')
    baseline_loss = ckpt_info.get("prev_loss", None)

    for epoch in range(epoch_start, total_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, scaler,
                                   args.device, epoch, writer, phase=args.phase,
                                   jp_to_en_indices=jp_to_en_indices,
                                   amp_dtype=amp_dtype)
        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{total_epochs} loss={avg_loss:.4f} time={elapsed:.1f}s')

        # Log embedding stats every 5 epochs
        if epoch % 5 == 0:
            jp_embed = model.note_text_encoder.weight.data[
                JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT]
            jp_std = jp_embed.std().item()
            jp_norm = jp_embed.norm(dim=1).mean().item()
            avg_cos, min_cos, max_cos = compute_jp_en_cosine_stats(model, jp_to_en_indices)
            print(f'  [Embed] JP std={jp_std:.4f}, norm={jp_norm:.3f}, '
                  f'JP-EN cos: avg={avg_cos:.4f} min={min_cos:.4f} max={max_cos:.4f}')
            writer.add_scalar('embed/jp_std', jp_std, epoch)
            writer.add_scalar('embed/jp_mean_norm', jp_norm, epoch)
            writer.add_scalar('embed/jp_en_cos_avg', avg_cos, epoch)
            writer.add_scalar('embed/jp_en_cos_min', min_cos, epoch)

        # Validation (uses separate val_dataset, not training set)
        if epoch % args.eval_every == 0:
            val_loss = validate(model, val_dataloader, args.device, epoch, writer,
                                 amp_dtype=amp_dtype)
            print(f'  val_loss={val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, epoch, val_loss, stage_dir, args.phase)

        elif epoch % args.save_every == 0:
            save_checkpoint(model, epoch, avg_loss, stage_dir, args.phase)

        scheduler.step()

    writer.close()
    print(f'\nPhase {args.phase} complete. Best val loss: {best_loss:.4f}')
    print(f'Checkpoints in: {stage_dir}/')


if __name__ == '__main__':
    main()
