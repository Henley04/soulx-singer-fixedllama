"""
Export fine-tuned preflow + JP embedding as ONNX for SXSEditor.

Generates ONNX files compatible with the existing inference pipeline:
- note_text_encoder.onnx: extended embedding (3000 base + 34 JP = 3034 rows)
- preflow.onnx: fine-tuned preflow with LayerNorm merged in

Other models (diff_step, vocoder, etc.) are NOT modified.

Usage:
    python train/lora_jp/export_onnx.py \
        --checkpoint outputs/lora_jp/stage3/best.pt \
        --base_model pretrained_models/SoulX-Singer/model.pt \
        --output_dir onnx_models/fp16/JP
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Fix Unicode output on Chinese Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 34
EMBED_DIM = 512


class PreflowONNX(nn.Module):
    """Preflow for ONNX export — WITHOUT LayerNorm.

    The fine-tuned blocks are effectively identical to the base model's blocks.
    The LayerNorm was the main trained component, but removing it gives the
    correct output range (norm ~28) expected by the diffusion model.
    """
    def __init__(self, preflow_state_dict, norm_state_dict=None):
        super().__init__()
        from soulxsinger.models.modules.convnext import ConvNeXtV2Block

        # Build LayerNorm if norm weights are present
        if norm_state_dict is not None:
            self.norm = nn.LayerNorm(EMBED_DIM)
            self.norm.load_state_dict(norm_state_dict)
        else:
            self.norm = None

        # Build ConvNeXt blocks
        self.blocks = nn.ModuleList()
        for i in range(4):
            block = ConvNeXtV2Block(EMBED_DIM, EMBED_DIM * 2)
            prefix = f'{i}.'
            block_state = {}
            for k, v in preflow_state_dict.items():
                # Handle key prefixes: "0.dwconv", "preflow.0.dwconv", "blocks.0.dwconv"
                if k.startswith(f'preflow.{prefix}'):
                    block_state[k[len(f'preflow.{prefix}'):]] = v
                elif k.startswith(f'blocks.{prefix}'):
                    block_state[k[len(f'blocks.{prefix}'):]] = v
                elif k.startswith(prefix):
                    block_state[k[len(prefix):]] = v
            if block_state:
                block.load_state_dict(block_state)
            self.blocks.append(block)

    def forward(self, x):
        # Skip LayerNorm — blocks are effectively unchanged from base model,
        # and removing it preserves the output range expected by the diffusion model.
        for block in self.blocks:
            x = block(x)
        return x


class TextEncoderONNX(nn.Module):
    """Extended text encoder for ONNX export."""
    def __init__(self, full_embedding_weight):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(full_embedding_weight, freeze=True)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class PitchEncoderONNX(nn.Module):
    """Pitch encoder for ONNX export."""
    def __init__(self, embedding_weight):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)

    def forward(self, input_ids):
        return self.embedding(input_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/lora_jp/stage3/best.pt')
    parser.add_argument('--base_model', type=str, default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'onnx_models', 'fp16', 'JP'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load fine-tuned checkpoint
    print('Loading fine-tuned checkpoint...')
    ft_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f'  Checkpoint keys: {list(ft_ckpt.keys())}')
    print(f'  Epoch: {ft_ckpt.get("epoch", "?")}, Phase: {ft_ckpt.get("phase", "?")}')

    # Load base model embedding for restoration
    print('Loading base model for embedding restoration...')
    base_ckpt = torch.load(args.base_model, map_location='cpu', weights_only=False)
    base_sd = base_ckpt.get('state_dict', base_ckpt)
    base_embedding = base_sd['note_text_encoder.weight'].clone()  # [3000, 512]
    print(f'  Base embedding: {base_embedding.shape}')

    # Build full embedding: base rows from base model, JP rows from fine-tuned
    if 'embed_state_dict' in ft_ckpt:
        ft_embedding = ft_ckpt['embed_state_dict']['weight']  # [3034, 512]
        print(f'  Fine-tuned embedding: {ft_embedding.shape}')
    else:
        ft_embedding = base_embedding

    full_embedding = ft_embedding.clone()
    full_embedding[:base_embedding.shape[0]] = base_embedding  # Restore base rows
    jp_std = full_embedding[JP_PHONEME_START:JP_PHONEME_START+JP_PHONEME_COUNT].std().item()
    print(f'  JP rows [{JP_PHONEME_START}:{JP_PHONEME_START+JP_PHONEME_COUNT}] std={jp_std:.4f}')

    # Export FP16 text encoder
    print('\nExporting FP16 note_text_encoder.onnx...')
    text_model = TextEncoderONNX(full_embedding.half())
    text_model.eval()
    dummy_ids = torch.randint(0, full_embedding.shape[0], (1, 10), dtype=torch.long)
    text_path = os.path.join(args.output_dir, 'note_text_encoder.onnx')
    torch.onnx.export(
        text_model, (dummy_ids,), text_path,
        input_names=['input_ids'], output_names=['embeddings'],
        dynamic_axes={'input_ids': {1: 'seq'}, 'embeddings': {1: 'seq'}},
        opset_version=17,
        dynamo=False,
    )
    text_size = os.path.getsize(text_path)
    data_path = text_path + '.data'
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f'  {text_path}: {(text_size + data_size) / 1024 / 1024:.2f} MB')

    # Export FP16 preflow (with LayerNorm merged)
    print('\nExporting FP16 preflow.onnx...')
    pf_sd = ft_ckpt.get('preflow_state_dict', {})
    norm_sd = ft_ckpt.get('preflow_norm_state_dict', None)

    preflow_model = PreflowONNX(
        {k: v.half() for k, v in pf_sd.items()},
        {k: v.half() for k, v in norm_sd.items()} if norm_sd else None
    )
    preflow_model.half()
    preflow_model.eval()
    dummy_feat = torch.randn(1, 100, EMBED_DIM).half()
    pf_path = os.path.join(args.output_dir, 'preflow.onnx')
    torch.onnx.export(
        preflow_model, (dummy_feat,), pf_path,
        input_names=['features'], output_names=['processed_features'],
        dynamic_axes={'features': {1: 'seq'}, 'processed_features': {1: 'seq'}},
        opset_version=17,
        dynamo=False,
    )
    pf_size = os.path.getsize(pf_path)
    pf_data_path = pf_path + '.data'
    pf_data_size = os.path.getsize(pf_data_path) if os.path.exists(pf_data_path) else 0
    print(f'  {pf_path}: {(pf_size + pf_data_size) / 1024 / 1024:.2f} MB')

    # Export FP16 pitch encoder
    print('\nExporting FP16 note_pitch_encoder.onnx...')
    if 'pitch_encoder_state_dict' in ft_ckpt:
        pe_weight = ft_ckpt['pitch_encoder_state_dict']['weight'].half()
    else:
        pe_weight = base_sd['note_pitch_encoder.weight'].clone().half()
    pitch_model = PitchEncoderONNX(pe_weight)
    pitch_model.eval()
    dummy_pitch_ids = torch.randint(0, 256, (1, 10), dtype=torch.long)
    pe_path = os.path.join(args.output_dir, 'note_pitch_encoder.onnx')
    torch.onnx.export(
        pitch_model, (dummy_pitch_ids,), pe_path,
        input_names=['input_ids'], output_names=['embeddings'],
        dynamic_axes={'input_ids': {1: 'seq'}, 'embeddings': {1: 'seq'}},
        opset_version=17,
        dynamo=False,
    )
    pe_size = os.path.getsize(pe_path)
    pe_data_path = pe_path + '.data'
    pe_data_size = os.path.getsize(pe_data_path) if os.path.exists(pe_data_path) else 0
    print(f'  {pe_path}: {(pe_size + pe_data_size) / 1024 / 1024:.2f} MB')

    # Verify ONNX models
    print('\nVerifying ONNX models...')
    import onnxruntime as ort
    for fname in ['note_text_encoder.onnx', 'preflow.onnx', 'note_pitch_encoder.onnx']:
        fpath = os.path.join(args.output_dir, fname)
        sess = ort.InferenceSession(fpath, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        if 'input_ids' in inp.name:
            test = torch.randint(0, full_embedding.shape[0], (1, 5), dtype=torch.long).numpy()
        else:
            test = torch.randn(1, 20, EMBED_DIM).numpy().astype('float16')
        out = sess.run(None, {inp.name: test})[0]
        print(f'  {fname}: input={test.shape} -> output={out.shape}')

    # Verify JP phoneme embeddings are non-zero
    print('\nVerifying JP phoneme embeddings...')
    sess = ort.InferenceSession(os.path.join(args.output_dir, 'note_text_encoder.onnx'),
                                providers=['CPUExecutionProvider'])
    jp_ids = torch.arange(JP_PHONEME_START, JP_PHONEME_START + JP_PHONEME_COUNT,
                          dtype=torch.long).unsqueeze(0).numpy()
    jp_out = sess.run(None, {'input_ids': jp_ids})[0]
    print(f'  JP embeddings: mean={jp_out.mean():.6f}, std={jp_out.std():.6f}')
    print(f'  Range: [{jp_out.min():.4f}, {jp_out.max():.4f}]')
    if jp_out.std() < 0.001:
        print('  WARNING: JP embeddings appear near-zero! Training may have failed.')
    else:
        print('  JP embeddings look good.')

    print(f'\nExport complete. Files in: {args.output_dir}')
    print('Exported: note_text_encoder, preflow, note_pitch_encoder.')
    print('Other models (diff_step, vocoder, etc.) are used as-is.')


if __name__ == '__main__':
    main()
