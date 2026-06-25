"""
Synthesize Japanese test sentences using fine-tuned JP LoRA checkpoint.

Uses the FULL inference pipeline (cfm_decoder.reverse_diffusion + vocoder),
matching how SXSEditor actually generates audio. Requires a prompt audio
to guide the diffusion model (few-shot synthesis).

Usage:
    python train/lora_jp/synthesize.py \\
        --checkpoint outputs/lora_jp/stage3/best.pt \\
        --model_path pretrained_models/SoulX-Singer/model.pt \\
        --prompt_wav example/audio/zh_prompt.wav \\
        --prompt_meta example/audio/zh_prompt.json \\
        --output_dir outputs/lora_jp/synthesis
"""

import os
import sys
import json
import copy
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor
from train.lora_jp.train_staged import JP_PHONEME_START, JP_PHONEME_COUNT


# Japanese test sentences: (lyric_phonemes, note_pitches, note_types, description)
# Phonemes use jp_ prefix; pauses use <SP>; each entry is one note.
TEST_SENTENCES = [
    {
        "desc": "sa-ku-ra (sakura, cherry blossom)",
        "phonemes": ["jp_s", "jp_a", "jp_k", "jp_u", "jp_r", "jp_a"],
        "pitches": [60, 60, 62, 62, 64, 64],  # C4 C4 D4 D4 E4 E4
        "durations": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        "types": [2, 2, 2, 2, 2, 2],
    },
    {
        "desc": "a-i-u-e-o (vowels)",
        "phonemes": ["jp_a", "jp_i", "jp_u", "jp_e", "jp_o"],
        "pitches": [60, 62, 64, 65, 67],
        "durations": [0.4, 0.4, 0.4, 0.4, 0.4],
        "types": [2, 2, 2, 2, 2],
    },
    {
        "desc": "ka-ki-ku-ke-ko (k + vowels)",
        "phonemes": ["jp_k", "jp_a", "jp_k", "jp_i", "jp_k", "jp_u", "jp_k", "jp_e", "jp_k", "jp_o"],
        "pitches": [60, 60, 62, 62, 64, 64, 65, 65, 67, 67],
        "durations": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "types": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    },
    {
        "desc": "sa-ku-ra with pause",
        "phonemes": ["jp_s", "jp_a", "jp_k", "jp_u", "<SP>", "jp_r", "jp_a"],
        "pitches": [60, 60, 62, 62, 0, 64, 64],
        "durations": [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.3],
        "types": [2, 2, 2, 2, 1, 2, 2],
    },
    {
        "desc": "ha-ru (spring)",
        "phonemes": ["jp_h", "jp_a", "jp_r", "jp_u"],
        "pitches": [64, 64, 62, 62],
        "durations": [0.3, 0.3, 0.3, 0.3],
        "types": [2, 2, 2, 2],
    },
]


def build_target_meta(sentence, phoneset_path, sample_rate=24000, hop_size=480):
    """Build target metadata dict for a test sentence."""
    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    phonemes = sentence["phonemes"]
    pitches = sentence["pitches"]
    durations = sentence["durations"]
    types = sentence["types"]

    # Verify all phonemes exist in phone_set
    for ph in phonemes:
        if ph not in phone2idx:
            print(f"  ERROR: phoneme '{ph}' not in phone_set!")
            return None

    meta = {
        "phoneme": " ".join(phonemes),
        "duration": " ".join(f"{d:.6f}" for d in durations),
        "note_pitch": " ".join(str(p) for p in pitches),
        "note_type": " ".join(str(t) for t in types),
    }
    return meta


def load_fine_tuned_model(model_path, checkpoint_path, config_path, device='cuda'):
    """Load base model and apply JP LoRA checkpoint."""
    config = OmegaConf.load(config_path)
    model = SoulXSinger(config)

    # Load base model
    print(f"Loading base model: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)

    # Apply JP LoRA checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Applying JP LoRA: {checkpoint_path}")
        ft_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Restore preflow
        if 'preflow_state_dict' in ft_ckpt:
            model.preflow.load_state_dict(ft_ckpt['preflow_state_dict'], strict=False)
            print("  preflow: loaded")

        # Restore embedding (extended to 3033 rows)
        if 'embed_state_dict' in ft_ckpt:
            ft_weight = ft_ckpt['embed_state_dict']['weight']
            # Extend embedding if needed
            if ft_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
                new_emb = torch.nn.Embedding(ft_weight.shape[0], model.note_text_encoder.weight.shape[1])
                new_emb.weight.data = ft_weight
                model.note_text_encoder = new_emb
            else:
                model.note_text_encoder.weight.data[:ft_weight.shape[0]] = ft_weight
            print(f"  embedding: loaded (shape={model.note_text_encoder.weight.shape})")

        # Restore cond_emb (Linear inside cfm_decoder that projects decoder_inp
        # to diff_estimator hidden size). Trained with flow-matching loss.
        if 'cond_emb_state_dict' in ft_ckpt:
            model.cfm_decoder.model.cond_emb.load_state_dict(
                ft_ckpt['cond_emb_state_dict'])
            print("  cond_emb: loaded")
        else:
            print("  cond_emb: NOT in checkpoint (old checkpoint), using base")
    else:
        print("  No JP checkpoint, using base model only")

    model = model.to(device)
    model.eval()
    return model


def load_prompt(prompt_wav, prompt_meta_path, phoneset_path, sample_rate=24000, hop_size=480, device='cpu', prompt_idx=0):
    """Load prompt audio and metadata."""
    with open(prompt_meta_path, 'r', encoding='utf-8') as f:
        prompt_meta = json.load(f)
    if isinstance(prompt_meta, list):
        prompt_meta = prompt_meta[prompt_idx]

    # Use CPU for processing to avoid CUDA assert from OOB phoneme IDs
    processor = DataProcessor(
        hop_size=hop_size,
        sample_rate=sample_rate,
        phoneset_path=phoneset_path,
        device='cpu',
    )
    item = processor.process(prompt_meta, prompt_wav)
    return item


def synthesize_sentence(model, prompt_item, target_meta, phoneset_path,
                         sample_rate=24000, hop_size=480, device='cuda',
                         n_steps=32, cfg=3.0, use_fp16=True):
    """Synthesize one sentence using full diffusion pipeline."""
    # Use CPU DataProcessor to avoid CUDA assert from OOB phoneme IDs
    processor = DataProcessor(
        hop_size=hop_size,
        sample_rate=sample_rate,
        phoneset_path=phoneset_path,
        device='cpu',
    )

    # Process target
    target_item = processor.process(target_meta, None)
    if target_item is None:
        return None

    # Build inference meta (matching model.infer format)
    pt_note_text = prompt_item['phoneme'].to(device)
    pt_note_pitch = prompt_item['note_pitch'].to(device)
    pt_note_type = prompt_item['note_type'].to(device)
    pt_mel2note = prompt_item['mel2note'].to(device)
    pt_f0 = prompt_item.get('f0')
    if pt_f0 is not None:
        pt_f0 = pt_f0.to(device)
    else:
        pt_f0 = torch.zeros_like(pt_mel2note).float().to(device)
    pt_wav = prompt_item['waveform'].to(device)

    gt_note_text = target_item['phoneme'].to(device)
    gt_note_pitch = target_item['note_pitch'].to(device)
    gt_note_type = target_item['note_type'].to(device)
    gt_mel2note = target_item['mel2note'].to(device)
    gt_f0 = target_item.get('f0')
    if gt_f0 is not None:
        gt_f0 = gt_f0.to(device)
    else:
        gt_f0 = torch.zeros_like(gt_mel2note).float().to(device)

    # Sanity check: all phoneme IDs must be < embedding vocab size
    vocab_size = model.note_text_encoder.weight.shape[0]
    pt_max = pt_note_text.max().item()
    gt_max = gt_note_text.max().item()
    if pt_max >= vocab_size or gt_max >= vocab_size:
        print(f"  SKIP: phoneme ID out of bounds (pt_max={pt_max}, gt_max={gt_max}, vocab={vocab_size})")
        return None

    use_fp16 = use_fp16 and device.startswith('cuda')

    with torch.no_grad():
        # Compute prompt mel
        pt_mel = model.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)
        if use_fp16:
            pt_mel = pt_mel.half()
            pt_f0 = pt_f0.half()
            gt_f0 = gt_f0.half()

        len_prompt = pt_note_pitch.shape[1]
        len_prompt_mel = pt_f0.shape[1]

        # Concatenate prompt + target
        note_pitch = torch.cat([pt_note_pitch, gt_note_pitch], 1)
        note_text = torch.cat([pt_note_text, gt_note_text], 1)
        note_type = torch.cat([pt_note_type, gt_note_type], 1)
        mel2note = torch.cat([pt_mel2note, gt_mel2note + len_prompt], 1)

        f0_course_pt = model.f0_to_coarse(pt_f0)
        f0_course_gt = model.f0_to_coarse(gt_f0, f0_shift=0)
        f0_course = torch.cat([f0_course_pt, f0_course_gt], 1)

        # Clamp pitch to valid range
        note_pitch = torch.clamp(note_pitch, 0, 255)

        from soulxsinger.models.soulxsinger import _autocast_if
        with _autocast_if(use_fp16):
            features = (model.note_pitch_encoder(note_pitch) +
                        model.note_type_encoder(note_type) +
                        model.note_text_encoder(note_text))
            features = model.preflow(features)
            features = model.expand_states(features, mel2note)
            features = features + model.f0_encoder(f0_course)

            gt_decoder_inp = features[:, len_prompt_mel:, :]
            pt_decoder_inp = features[:, :len_prompt_mel, :]

            print(f"    pt_mel: {pt_mel.shape}, pt_decoder_inp: {pt_decoder_inp.shape}, "
                  f"gt_decoder_inp: {gt_decoder_inp.shape}")

            generated_mel = model.cfm_decoder.reverse_diffusion(
                pt_mel,
                pt_decoder_inp,
                gt_decoder_inp,
                n_timesteps=n_steps,
                cfg=cfg
            )
            print(f"    generated_mel: {generated_mel.shape}, range=[{generated_mel.min():.3f}, {generated_mel.max():.3f}]")

            generated_audio = model.vocoder(generated_mel.transpose(1, 2)[0:1, ...]).float()
            print(f"    generated_audio: {generated_audio.shape}, range=[{generated_audio.min():.3f}, {generated_audio.max():.3f}]")

    # Squeeze to 1D for soundfile
    audio_np = generated_audio.squeeze().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np[0]  # take first channel
    return audio_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/lora_jp/stage3/best.pt',
                        help='JP LoRA checkpoint (empty = base only)')
    parser.add_argument('--model_path', type=str, default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--config', type=str, default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--phoneset_path', type=str, default='train/lora_jp/jp_phone_set.json')
    parser.add_argument('--prompt_wav', type=str, default='train/lora_jp/dataset/wavs/pjs001_song.wav',
                        help='Prompt audio for few-shot diffusion')
    parser.add_argument('--prompt_meta', type=str, default='train/lora_jp/dataset/metadata.json',
                        help='Prompt metadata (JSON file or metadata.json list)')
    parser.add_argument('--prompt_idx', type=int, default=0,
                        help='If prompt_meta is a list, use this index')
    parser.add_argument('--output_dir', type=str, default='outputs/lora_jp/synthesis')
    parser.add_argument('--n_steps', type=int, default=32, help='Diffusion steps')
    parser.add_argument('--cfg', type=float, default=3.0, help='Classifier-free guidance')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_fp16', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.use_fp16 = False

    # Load model
    model = load_fine_tuned_model(args.model_path, args.checkpoint, args.config, args.device)
    print(f"Model loaded. embedding shape: {model.note_text_encoder.weight.shape}")

    # Load prompt
    print(f"\nLoading prompt: {args.prompt_wav}")
    prompt_item = load_prompt(args.prompt_wav, args.prompt_meta, args.phoneset_path,
                               device=args.device, prompt_idx=args.prompt_idx)
    print(f"  prompt phoneme: {prompt_item['phoneme'].shape}")
    print(f"  prompt mel2note: {prompt_item['mel2note'].shape}")
    print(f"  prompt waveform: {prompt_item['waveform'].shape}")

    # Synthesize each test sentence
    print(f"\n{'='*60}")
    print(f"Synthesizing {len(TEST_SENTENCES)} test sentences")
    print(f"{'='*60}")

    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"\n[{i+1}/{len(TEST_SENTENCES)}] {sentence['desc']}")
        target_meta = build_target_meta(sentence, args.phoneset_path)
        if target_meta is None:
            continue

        try:
            audio = synthesize_sentence(
                model, prompt_item, target_meta, args.phoneset_path,
                device=args.device, n_steps=args.n_steps, cfg=args.cfg,
                use_fp16=args.use_fp16
            )

            if audio is None:
                print("  FAILED: no audio generated")
                continue

            # Check for silence
            max_amp = np.abs(audio).max()
            rms = np.sqrt(np.mean(audio ** 2))
            print(f"  Audio stats: max_amp={max_amp:.4f}, rms={rms:.4f}, len={len(audio)}")

            if max_amp < 0.001:
                print("  WARNING: Audio is near-silent!")
            else:
                print("  Audio has content.")

            out_path = os.path.join(args.output_dir, f"test_{i+1:02d}.wav")
            sf.write(out_path, audio, 24000)
            print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Synthesis complete. Files in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
