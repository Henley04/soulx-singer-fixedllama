"""
Pre-quantize the base SoulX-Singer model to NVFP4 (true quantization).

This is a ONE-TIME operation that produces a truly quantized NVFP4 base
model in a separate directory. The fine-tuning script (train_staged.py)
loads this pre-quantized model instead of performing in-place pseudo-
quantization at training time.

NVFP4 = NVIDIA FP4 (E2M1 + microscaling, float8_e4m3fn block scales).
Requires Blackwell (sm100+) and torchao.

Only nn.Linear layers with both dims divisible by 16 are quantized.
nn.Embedding, Conv, and LayerNorm stay in their original precision.

Usage:
    python train/lora_jp/quantize_base_to_nvfp4.py
    # Or with explicit paths:
    python train/lora_jp/quantize_base_to_nvfp4.py \
        --model_path pretrained_models/SoulX-Singer/model.pt \
        --output_dir pretrained_models/SoulX-Singer-nvfp4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from soulxsinger.models.soulxsinger import SoulXSinger

try:
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    TORCHAO_NVFP4_AVAILABLE = True
except ImportError:
    TORCHAO_NVFP4_AVAILABLE = False


def _nvfp4_filter(mod, fqn):
    """Filter: quantize nn.Linear with both dims divisible by 16."""
    if not isinstance(mod, nn.Linear):
        return False
    N, K = mod.weight.shape
    return K % 16 == 0 and N % 16 == 0


def main():
    parser = argparse.ArgumentParser(description="Pre-quantize base model to NVFP4")
    parser.add_argument('--model_path', default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--config', default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--output_dir', default='pretrained_models/SoulX-Singer-nvfp4')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if not TORCHAO_NVFP4_AVAILABLE:
        print("ERROR: torchao is not installed. Install with: pip install torchao>=0.17.0")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for NVFP4 quantization.")
        sys.exit(1)
    cap = torch.cuda.get_device_capability()[0]
    if cap < 10:
        print(f"ERROR: NVFP4 requires Blackwell (sm100+), got sm{cap}.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'model.pt')

    print(f"Loading base model: {args.model_path}")
    config = OmegaConf.load(args.config)
    model = SoulXSinger(config)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)

    # Count Linear layers before quantization
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_eligible = sum(1 for m in model.modules()
                     if isinstance(m, nn.Linear)
                     and m.weight.shape[0] % 16 == 0
                     and m.weight.shape[1] % 16 == 0)
    print(f"  Linear layers: {n_linear} total, {n_eligible} eligible for NVFP4")

    print(f"Moving to {args.device}...")
    model = model.to(args.device)

    print("Quantizing to NVFP4 (true quantization, not pseudo-quant)...")
    quantize_(model, NVFP4WeightOnlyConfig(), _nvfp4_filter)

    # Verify quantization
    n_q = sum(1 for m in model.modules()
              if isinstance(m, nn.Linear)
              and "NVFP4" in type(m.weight).__name__)
    print(f"  Quantized {n_q} Linear layer(s) to NVFP4")

    # Measure size reduction
    sd = model.state_dict()
    orig_size = sum(v.numel() * 4 for v in sd.values()
                    if isinstance(v, torch.Tensor) and not hasattr(v, 'dequantize'))
    quant_size = sum(v.qdata.numel() + v.scale.numel()
                     for v in sd.values() if hasattr(v, 'qdata'))
    print(f"  Estimated weight storage: {orig_size/1024/1024:.1f} MB (fp32) -> "
          f"~{quant_size/1024/1024:.1f} MB (NVFP4)")

    print(f"Saving NVFP4 base model to: {output_path}")
    torch.save({
        'state_dict': sd,
        'nvfp4_quantized': True,
        'torchao_required': True,
        'orig_model_path': args.model_path,
    }, output_path)
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"  Location: {args.output_dir} (separate from base model)")
    print("Done. Use this path as --nvfp4_base in train_staged.py.")


if __name__ == '__main__':
    main()
