"""
Fine-tuning for Japanese phoneme support in SoulX-Singer.

Strategy (encoder-only, full embedding):
- note_text_encoder: full embedding fine-tuning (3034 × 512 = 1.55M params)
- preflow: full fine-tuning (4 ConvNeXtV2Blocks, ~4.2M params)
- f0_encoder: fine-tuning (~0.13M params)
- Total trainable: ~5.9M params

The cfm_decoder (diffusion step) is NOT fine-tuned because it operates on continuous features.

Usage:
    python train/lora_jp/train_lora.py \
        --model_path pretrained_models/SoulX-Singer/model.pt \
        --config soulxsinger/config/soulxsinger.yaml \
        --output_dir outputs/lora_jp
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.models.soulxsinger import SoulXSinger
from train.lora_jp.dataset import JpLoRADataset, collate_fn

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 34
EMBED_DIM = 512

# JP phoneme -> English phoneme(s) for embedding initialization.
# Using similar English phonemes gives the optimizer a phonetically
# meaningful starting point instead of random noise.
JP_INIT_MAP = {
    'jp_pau': ['<SP>'],
    'jp_a':   ['en_AA1'],
    'jp_i':   ['en_IY1'],
    'jp_u':   ['en_UW'],
    'jp_e':   ['en_EH1'],
    'jp_o':   ['en_OW1'],
    'jp_k':   ['en_K'],
    'jp_s':   ['en_S'],
    'jp_t':   ['en_T'],
    'jp_n':   ['en_N'],
    'jp_h':   ['en_HH'],
    'jp_m':   ['en_M'],
    'jp_r':   ['en_R'],
    'jp_w':   ['en_UW'],      # Japanese w is a quick rounded glide, not English W
    'jp_y':   ['en_Y'],
    'jp_g':   ['en_G'],
    'jp_z':   ['en_Z'],
    'jp_d':   ['en_D'],
    'jp_b':   ['en_B'],
    'jp_p':   ['en_P'],
    'jp_f':   ['en_F'],
    'jp_j':   ['en_JH'],
    'jp_ch':  ['en_CH'],
    'jp_sh':  ['en_SH'],
    'jp_ts':  ['en_T', 'en_S'],
    'jp_ky':  ['en_K', 'en_Y'],
    'jp_gy':  ['en_G', 'en_Y'],
    'jp_ny':  ['en_N', 'en_Y'],
    'jp_hy':  ['en_HH', 'en_Y'],
    'jp_my':  ['en_M', 'en_Y'],
    'jp_ry':  ['en_R', 'en_Y'],
    'jp_py':  ['en_P', 'en_Y'],
    'jp_by':  ['en_B', 'en_Y'],
    'jp_cl':  ['<SP>'],
}


def extend_embedding(model):
    """Extend note_text_encoder with JP phoneme embedding rows.

    Initializes JP embeddings from similar English phonemes instead of
    random values, giving the optimizer a phonetically meaningful start.
    """
    emb = model.note_text_encoder
    if emb.weight.shape[0] < JP_PHONEME_START + JP_PHONEME_COUNT:
        new_emb = nn.Embedding(JP_PHONEME_START + JP_PHONEME_COUNT, emb.weight.shape[1])
        new_emb.weight.data[:emb.weight.shape[0]] = emb.weight.data
        model.note_text_encoder = new_emb

    # Build phoneme name -> index map
    phone_set_path = os.path.join(os.path.dirname(__file__), 'jp_phone_set.json')
    with open(phone_set_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    # Initialize JP embeddings from similar phonemes
    with torch.no_grad():
        jp_start = JP_PHONEME_START
        jp_phonemes = phone_list[jp_start:jp_start + JP_PHONEME_COUNT]
        for i, jp_ph in enumerate(jp_phonemes):
            src_phones = JP_INIT_MAP.get(jp_ph, [])
            if src_phones:
                src_indices = [phone2idx[ph] for ph in src_phones if ph in phone2idx]
                if src_indices:
                    model.note_text_encoder.weight.data[jp_start + i] = \
                        model.note_text_encoder.weight.data[src_indices].mean(dim=0)
                    continue
            # Fallback: small random
            nn.init.normal_(model.note_text_encoder.weight.data[jp_start + i], std=0.02)

    print(f'[Embedding] Extended to {model.note_text_encoder.weight.shape[0]} rows, JP at [{jp_start}:{jp_start+JP_PHONEME_COUNT}]')
    print(f'[Embedding] JP embeddings initialized from English phonemes')


class MelProjection(nn.Module):
    """Project encoder features (512) to mel space (128) for loss computation."""
    def __init__(self, enc_dim=EMBED_DIM, mel_dim=128):
        super().__init__()
        self.proj = nn.Linear(enc_dim, mel_dim)

    def forward(self, x):
        return self.proj(x)


def train_one_epoch(model, proj, dataloader, optimizer, scheduler, device, epoch, writer, use_amp=True):
    model.train()
    proj.train()
    total_loss = 0
    n = 0
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for bi, batch in enumerate(dataloader):
        if batch is None:
            continue
        optimizer.zero_grad()
        try:
            with torch.amp.autocast('cuda', enabled=use_amp):
                note_text = batch['phoneme'].to(device)
                note_pitch = batch['note_pitch'].to(device)
                note_type = batch['note_type'].to(device)
                mel2note = batch['mel2note'].to(device)
                f0 = batch.get('f0')
                if f0 is not None:
                    f0 = f0.to(device)
                waveform = batch.get('waveform')
                if waveform is not None:
                    waveform = waveform.to(device)

                # Encoder forward pass
                features = (model.note_text_encoder(note_text) +
                            model.note_pitch_encoder(note_pitch) +
                            model.note_type_encoder(note_type))
                features = model.preflow(features)
                mel_feat = model.expand_states(features, mel2note)

                # Add F0 encoding if available
                if f0 is not None and f0.shape[1] > 0:
                    f0_coarse = model.f0_to_coarse(f0)
                    f0_enc = model.f0_encoder(f0_coarse)
                    mel_feat = mel_feat + f0_enc[:, :mel_feat.shape[1], :]

                # Target mel spectrogram
                if waveform is not None:
                    target_mel = model.mel(waveform.float())
                    target_mel = target_mel[:, :mel_feat.shape[1], :]
                else:
                    target_mel = torch.zeros(mel_feat.shape[0], mel_feat.shape[1], 128, device=device)

                # Direct reconstruction loss
                projected = proj(mel_feat[:, :target_mel.shape[1], :])
                recon_loss = F.mse_loss(projected, target_mel)

                # Variance regularization: push JP embeddings toward base embedding scale
                jp_embed = model.note_text_encoder.weight[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT]
                jp_std = jp_embed.std(dim=1).mean()
                target_std = 1.0  # match base embedding scale (std≈1.0)
                var_loss = F.relu(target_std - jp_std) ** 2 * 500

                loss = recon_loss + var_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(proj.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(proj.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

            total_loss += loss.item()
            n += 1
            writer.add_scalar('train/loss_step', loss.item(), epoch * 1000 + bi)
            if bi % 10 == 0:
                print(f'  Epoch {epoch} [{bi}] loss={loss.item():.4f} (recon={recon_loss.item():.4f} var={var_loss.item():.4f})')
        except Exception as e:
            print(f'  Error batch {bi}: {e}')
            import traceback
            traceback.print_exc()

    avg = total_loss / max(n, 1)
    writer.add_scalar('train/loss_epoch', avg, epoch)
    if scheduler:
        scheduler.step()
    return avg


@torch.no_grad()
def validate(model, proj, dataloader, device, epoch, writer):
    model.eval()
    proj.eval()
    total_loss = 0
    n = 0
    for batch in dataloader:
        if batch is None:
            continue
        try:
            note_text = batch['phoneme'].to(device)
            note_pitch = batch['note_pitch'].to(device)
            note_type = batch['note_type'].to(device)
            mel2note = batch['mel2note'].to(device)
            f0 = batch.get('f0')
            if f0 is not None:
                f0 = f0.to(device)
            waveform = batch.get('waveform')
            if waveform is not None:
                waveform = waveform.to(device)

            features = (model.note_text_encoder(note_text) +
                        model.note_pitch_encoder(note_pitch) +
                        model.note_type_encoder(note_type))
            features = model.preflow(features)
            mel_feat = model.expand_states(features, mel2note)

            if f0 is not None and f0.shape[1] > 0:
                f0_coarse = model.f0_to_coarse(f0)
                mel_feat = mel_feat + model.f0_encoder(f0_coarse)[:, :mel_feat.shape[1], :]

            if waveform is not None:
                target_mel = model.mel(waveform.float())[:, :mel_feat.shape[1], :]
            else:
                target_mel = torch.zeros(mel_feat.shape[0], mel_feat.shape[1], 128, device=device)

            projected = proj(mel_feat[:, :target_mel.shape[1], :])
            loss = F.mse_loss(projected, target_mel)
            total_loss += loss.item()
            n += 1
        except Exception as e:
            print(f'  Val error: {e}')
    avg = total_loss / max(n, 1)
    writer.add_scalar('val/loss_epoch', avg, epoch)
    return avg


def save_checkpoint(model, proj, epoch, loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    embed_weight = model.note_text_encoder.weight.data.clone()
    ckpt = {
        'epoch': epoch,
        'loss': loss,
        'preflow_state_dict': {k: v.clone() for k, v in model.preflow.state_dict().items()},
        'embed_state_dict': {'weight': embed_weight},
        'jp_embed': embed_weight[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT].clone(),
        'proj_state_dict': proj.state_dict(),
    }
    torch.save(ckpt, os.path.join(output_dir, f'epoch{epoch:03d}.pt'))
    torch.save(ckpt, os.path.join(output_dir, 'latest.pt'))
    jp_std = ckpt['jp_embed'].std().item()
    print(f'  Saved epoch {epoch} (embed={embed_weight.shape}, jp_std={jp_std:.4f})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--config', default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--phoneset_path', default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--dataset_metadata', default='train/lora_jp/dataset/metadata.json')
    parser.add_argument('--dataset_wav_dir', default='train/lora_jp/dataset/wavs')
    parser.add_argument('--output_dir', default='outputs/lora_jp')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = OmegaConf.load(args.config)

    print('[1/4] Loading base model...')
    model = SoulXSinger(config)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)

    print('[2/4] Extending embedding...')
    extend_embedding(model)
    print(f'  Embedding shape: {model.note_text_encoder.weight.shape}')

    # Freeze everything, then selectively unfreeze
    print('[3/4] Setting up trainable parameters...')
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the FULL embedding (not just JP rows)
    # NOTE: This also modifies base embeddings. The export script restores
    # base embeddings from the base model to prevent corruption.
    model.note_text_encoder.weight.requires_grad = True

    # Unfreeze preflow
    for param in model.preflow.parameters():
        param.requires_grad = True

    # Do NOT unfreeze f0_encoder — it's not exported to ONNX,
    # so fine-tuning it would cause train/inference mismatch

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = sum(p.numel() for p in model.note_text_encoder.parameters())
    preflow_params = sum(p.numel() for p in model.preflow.parameters())
    print(f'  Embedding: {embed_params/1e6:.2f}M ({model.note_text_encoder.weight.shape})')
    print(f'  Preflow: {preflow_params/1e6:.2f}M')
    print(f'  Total trainable: {trainable/1e6:.2f}M')

    # Projection head
    proj = MelProjection().to(args.device)

    # Optimizer with per-group learning rates
    # Embedding needs much higher lr — it has 3034 rows but only 34 new ones get
    # meaningful gradient signal from the preflow loss. The new rows start at std=0.02
    # and need to reach std≈1.0 to match the base embeddings.
    print(f'  Optimizer embedding ref shape: {model.note_text_encoder.weight.shape}')
    optimizer = torch.optim.AdamW([
        {'params': model.note_text_encoder.parameters(), 'lr': args.lr * 20},
        {'params': model.preflow.parameters(), 'lr': args.lr},
        {'params': proj.parameters(), 'lr': args.lr},
    ], weight_decay=0.01)

    def lr_lambda(step):
        warmup = 200
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.epochs * 50 - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print('[4/4] Loading dataset...')
    dataset = JpLoRADataset(
        metadata_path=args.dataset_metadata,
        wav_dir=args.dataset_wav_dir,
        phoneset_path=args.phoneset_path,
        sample_rate=config.audio.sample_rate,
        hop_size=config.audio.hop_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    model = model.to(args.device)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs'))

    print(f'\n=== Training {args.epochs} epochs, batch={args.batch_size}, AMP={args.use_amp} ===')
    print(f'    lr={args.lr}, device={args.device}\n')
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, proj, dataloader, optimizer, scheduler,
                                   args.device, epoch, writer, use_amp=args.use_amp)

        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} time={elapsed:.1f}s')

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, proj, epoch, avg_loss, args.output_dir)

        # Log JP embedding progress every 20 epochs
        if epoch % 20 == 0:
            jp_std = model.note_text_encoder.weight.data[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT].std().item()
            jp_norm = model.note_text_encoder.weight.data[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT].norm(dim=1).mean().item()
            print(f'  [Embed] JP std={jp_std:.4f}, mean_norm={jp_norm:.3f}')

        if epoch % args.eval_every == 0:
            val_loss = validate(model, proj, dataloader, args.device, epoch, writer)
            print(f'  val={val_loss:.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, proj, epoch, val_loss, os.path.join(args.output_dir, 'best'))

    writer.close()
    print(f'\nDone. Best val loss: {best_loss:.4f}')
    print(f'Checkpoint saved to: {args.output_dir}/best/')


if __name__ == '__main__':
    main()
