"""
Export fine-tuned preflow + JP embedding as ONNX for SXSEditor.

Generates ONNX files compatible with the existing inference pipeline:
- note_text_encoder.onnx: extended embedding (3000 base + 33 JP = 3033 rows)
- preflow.onnx: fine-tuned preflow blocks (LayerNorm removed for ONNX)

note_pitch_encoder is NOT exported — JP LoRA shares the base model's pitch
encoder because pitch is a MIDI index (0-127) with no language-specific
semantic. Exporting a JP-specific pitch_encoder would cause train/inference
mismatch when switching languages.

Other models (diff_step, vocoder, etc.) are NOT modified.

Dequantization: if the checkpoint contains NVFP4 tensors (from true NVFP4
fine-tuning), they are automatically dequantized to fp16 before export.

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
JP_PHONEME_COUNT = 33  # jp_pau removed; 33 JP phonemes (jp_a..jp_cl)
EMBED_DIM = 512


def dequantize_nvfp4_tensor(tensor, target_dtype=torch.float16):
    """Dequantize an NVFP4Tensor to fp16/fp32 if needed.

    Returns the original tensor unchanged if it's not NVFP4.
    """
    if hasattr(tensor, 'dequantize') and 'NVFP4' in type(tensor).__name__:
        return tensor.dequantize(target_dtype)
    return tensor


def dequantize_state_dict(sd, target_dtype=torch.float16):
    """Dequantize all NVFP4 tensors in a state dict."""
    return {k: dequantize_nvfp4_tensor(v, target_dtype)
            if isinstance(v, torch.Tensor) else v
            for k, v in sd.items()}


class PreflowONNX(nn.Module):
    """Preflow for ONNX export — WITHOUT LayerNorm.

    SXSEditor inference (preprocessing.js) feeds raw textEmb+pitchEmb+typeEmb
    directly into preflow without LayerNorm, so the ONNX model must match:
    no LayerNorm at the input. The 4 ConvNeXtV2Blocks are the trained
    components that adapt features for the JP distribution.
    """
    def __init__(self, preflow_state_dict):
        super().__init__()
        from soulxsinger.models.modules.convnext import ConvNeXtV2Block

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
        # No LayerNorm — matches SXSEditor preprocessing.js inference path
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


class CondEmbONNX(nn.Module):
    """Wrapper for the fine-tuned cond_emb (Linear 512->1024).

    SXSEditor inference (preprocessing.js) calls cond_emb.onnx with
    input name 'cond_code' (float, [1, T, 512]) and expects output
    'cond_embedding' ([1, T, 1024]). This wrapper matches that contract.
    Training fine-tunes cond_emb to adapt the JP feature distribution, so
    it MUST be exported and loaded at inference — using the base cond_emb
    with JP fine-tuned preflow+embedding causes severe phoneme corruption.
    """
    def __init__(self, cond_emb_linear):
        super().__init__()
        self.cond_emb = cond_emb_linear

    def forward(self, cond_code):
        return self.cond_emb(cond_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/lora_jp/stage3/best.pt')
    parser.add_argument('--base_model', type=str, default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'onnx_models', 'fp16', 'JP'))
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load fine-tuned checkpoint
    print('Loading fine-tuned checkpoint...')
    ft_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f'  Checkpoint keys: {list(ft_ckpt.keys())}')
    print(f'  Epoch: {ft_ckpt.get("epoch", "?")}, Phase: {ft_ckpt.get("phase", "?")}')

    # Dequantize any NVFP4 tensors in checkpoint to fp16
    for k, v in ft_ckpt.items():
        if isinstance(v, dict):
            ft_ckpt[k] = dequantize_state_dict(v, torch.float16)
            n_dq = sum(1 for vv in ft_ckpt[k].values()
                       if isinstance(vv, torch.Tensor) and vv.dtype == torch.float16)
            if n_dq > 0:
                print(f'  Dequantized {k} to fp16')

    # Load base model embedding for restoration
    print('Loading base model for embedding restoration...')
    base_ckpt = torch.load(args.base_model, map_location='cpu', weights_only=False)
    base_sd = base_ckpt.get('state_dict', base_ckpt)
    base_embedding = base_sd['note_text_encoder.weight'].clone()  # [3000, 512]
    print(f'  Base embedding: {base_embedding.shape}')

    # Build full embedding: base rows from base model, JP rows from fine-tuned
    if 'embed_state_dict' in ft_ckpt:
        ft_embedding = ft_ckpt['embed_state_dict']['weight']  # [3033, 512]
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
        opset_version=args.opset,
        dynamo=False,
    )
    text_size = os.path.getsize(text_path)
    data_path = text_path + '.data'
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f'  {text_path}: {(text_size + data_size) / 1024 / 1024:.2f} MB')

    # Export FP16 preflow (no LayerNorm — matches SXSEditor inference)
    print('\nExporting FP16 preflow.onnx...')
    pf_sd = ft_ckpt.get('preflow_state_dict', {})

    preflow_model = PreflowONNX({k: v.half() for k, v in pf_sd.items()})
    preflow_model.half()
    preflow_model.eval()
    dummy_feat = torch.randn(1, 100, EMBED_DIM).half()
    pf_path = os.path.join(args.output_dir, 'preflow.onnx')
    torch.onnx.export(
        preflow_model, (dummy_feat,), pf_path,
        input_names=['features'], output_names=['processed_features'],
        dynamic_axes={'features': {1: 'seq'}, 'processed_features': {1: 'seq'}},
        opset_version=args.opset,
        dynamo=False,
    )
    pf_size = os.path.getsize(pf_path)
    pf_data_path = pf_path + '.data'
    pf_data_size = os.path.getsize(pf_data_path) if os.path.exists(pf_data_path) else 0
    print(f'  {pf_path}: {(pf_size + pf_data_size) / 1024 / 1024:.2f} MB')

    # Export FP16 cond_emb (Linear 512->1024, fine-tuned for JP).
    # This is critical: training adapts cond_emb to the JP feature
    # distribution, so inference MUST use the fine-tuned cond_emb. Using the
    # base cond_emb with JP preflow+embedding causes severe phoneme
    # corruption (e.g. "watashino" -> "nin to shi so").
    print('\nExporting FP16 cond_emb.onnx...')
    if 'cond_emb_state_dict' not in ft_ckpt:
        raise RuntimeError(
            "Checkpoint missing 'cond_emb_state_dict'. The fine-tuned "
            "cond_emb is required for correct JP inference. Re-train with "
            "the updated train_staged.py (which saves cond_emb) before export."
        )
    from soulxsinger.models.modules.flow_matching import CFMDecoder
    # Build a Linear(512, 1024) and load the fine-tuned weights.
    # We don't need the full CFMDecoder — just the cond_emb submodule.
    cond_emb_linear = nn.Linear(EMBED_DIM, 1024)
    cond_emb_linear.load_state_dict(ft_ckpt['cond_emb_state_dict'])
    cond_model = CondEmbONNX(cond_emb_linear)
    cond_model.half().eval()
    dummy_cond = torch.randn(1, 20, EMBED_DIM).half()
    cond_path = os.path.join(args.output_dir, 'cond_emb.onnx')
    torch.onnx.export(
        cond_model, (dummy_cond,), cond_path,
        input_names=['cond_code'], output_names=['cond_embedding'],
        dynamic_axes={'cond_code': {1: 'seq'}, 'cond_embedding': {1: 'seq'}},
        opset_version=args.opset,
        dynamo=False,
    )
    cond_size = os.path.getsize(cond_path)
    cond_data_path = cond_path + '.data'
    cond_data_size = os.path.getsize(cond_data_path) if os.path.exists(cond_data_path) else 0
    print(f'  {cond_path}: {(cond_size + cond_data_size) / 1024 / 1024:.2f} MB')

    # NOTE: note_pitch_encoder is intentionally NOT exported.
    # JP LoRA shares the base model's pitch encoder because:
    # 1. note_pitch is a MIDI index (0-127) with no language-specific semantic
    # 2. Exporting a JP-specific pitch_encoder would cause train/inference
    #    mismatch when switching languages (pitch semantics are identical).
    # If a stale note_pitch_encoder.onnx exists in the output dir from a
    # previous (buggy) export, remove it so SXSEditor uses the base model's.
    stale_pitch = os.path.join(args.output_dir, 'note_pitch_encoder.onnx')
    if os.path.exists(stale_pitch):
        os.remove(stale_pitch)
        print(f'\nRemoved stale note_pitch_encoder.onnx (was incorrectly exported before).')
    print('Skipping note_pitch_encoder export (shared with base model).')

    # Verify ONNX models
    print('\nVerifying ONNX models...')
    import onnxruntime as ort
    for fname in ['note_text_encoder.onnx', 'preflow.onnx', 'cond_emb.onnx']:
        fpath = os.path.join(args.output_dir, fname)
        sess = ort.InferenceSession(fpath, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        if 'input_ids' in inp.name:
            max_id = full_embedding.shape[0]
            test = torch.randint(0, max_id, (1, 5), dtype=torch.long).numpy()
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

    # Verify <SP> embedding is the base model's value (since pau now maps to <SP>)
    sp_id = 1
    sp_out = sess.run(None, {'input_ids': torch.tensor([[sp_id]], dtype=torch.long).numpy()})[0]
    print(f'\n  <SP> (ID={sp_id}) embedding: mean={sp_out.mean():.6f}, std={sp_out.std():.6f}')

    print(f'\nExport complete. Files in: {args.output_dir}')
    print('Exported: note_text_encoder, preflow, cond_emb.')
    print('Skipped: note_pitch_encoder (shared with base model).')
    print('Other models (diff_step, vocoder, etc.) are used as-is from base.')


if __name__ == '__main__':
    main()
