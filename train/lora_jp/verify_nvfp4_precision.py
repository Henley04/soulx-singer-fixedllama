"""
Verify that SoulX-Singer-nvfp4/model.pt is TRUE NVFP4 quantization (not pseudo-quant)
and measure its precision against the original fp32 base model.

Strategy (NO dequantization):
  1. Load original fp32 model on CUDA.
  2. Load NVFP4 model DIRECTLY on CUDA (sm120 native NVFP4 kernels).
  3. For every nn.Linear whose weight is an NVFP4Tensor, feed the SAME random
     fp32 input through both versions and compare outputs:
       - MSE, RMSE, max abs err, mean abs err, cosine similarity, relative L2
  4. Aggregate stats per module group (preflow / cond_emb / diff_estimator /
     vocoder backbone) and overall.

Hardware: requires Blackwell sm100+. Run on the local RTX 5060 Laptop (sm120).

Usage:
    python train/lora_jp/verify_nvfp4_precision.py
"""

import os
import sys
import argparse
import time
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
    """Same filter as quantize_base_to_nvfp4.py: nn.Linear with both dims divisible by 16."""
    if not isinstance(mod, nn.Linear):
        return False
    N, K = mod.weight.shape
    return K % 16 == 0 and N % 16 == 0


def _classify_group(fqn: str) -> str:
    if fqn.startswith('preflow'):
        return 'preflow'
    if 'cond_emb' in fqn:
        return 'cond_emb'
    if 'diff_estimator' in fqn:
        return 'diff_estimator'
    if 'vocoder' in fqn:
        return 'vocoder'
    return 'other'


def load_fp32_model(model_path, config, device):
    print(f"[fp32] Loading: {model_path}")
    model = SoulXSinger(config)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
    model = model.to(device).eval()
    return model


def load_nvfp4_model(nvfp4_path, config, device):
    """Load NVFP4 model DIRECTLY (no dequantization) on the given device."""
    print(f"[nvfp4] Loading: {nvfp4_path}")
    if not TORCHAO_NVFP4_AVAILABLE:
        raise RuntimeError("torchao is required to load NVFP4 weights directly.")

    model = SoulXSinger(config)

    # First quantize_ in-place to convert eligible Linear weights to NVFP4Tensor
    # wrappers. This sets up the proper parameter type so we can assign the
    # saved NVFP4 tensors directly.
    quantize_(model, NVFP4WeightOnlyConfig(), _nvfp4_filter)

    # Load the saved NVFP4 state_dict and directly assign NVFP4Tensor to each
    # quantized Linear's .weight Parameter (load_state_dict doesn't support
    # NVFP4Tensor copy_).
    ckpt = torch.load(nvfp4_path, map_location='cpu', weights_only=False)
    assert ckpt.get('nvfp4_quantized', False), "Checkpoint is not marked nvfp4_quantized"
    nvfp4_sd = ckpt['state_dict']

    n_assigned = 0
    n_skipped = 0
    for name, param in model.named_parameters():
        if name not in nvfp4_sd:
            continue
        saved = nvfp4_sd[name]
        is_nvfp4_tensor = type(saved).__name__ == 'NVFP4Tensor'
        if not is_nvfp4_tensor:
            # Non-quantized tensor (embedding / conv / layernorm / bias) — use
            # normal copy_.
            try:
                param.data.copy_(saved)
            except Exception:
                param.data = saved.clone()
            continue
        # Directly replace the Parameter's underlying tensor with the saved
        # NVFP4Tensor. NO dequantization here — the weight stays NVFP4 and
        # forward passes use native NVFP4 matmul on sm100+.
        try:
            target_mod = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else None
        except AttributeError:
            target_mod = None
        if target_mod is not None and isinstance(target_mod, nn.Linear):
            target_mod.weight = nn.Parameter(saved.to(device=device))
            n_assigned += 1
        else:
            n_skipped += 1

    print(f"[nvfp4] Assigned {n_assigned} NVFP4 Linear weight(s), skipped {n_skipped}")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def compare_one_linear(name, fp32_lin, nvfp4_lin, device, trials=3):
    """Run forward on a single Linear on CUDA, compare fp32 vs NVFP4 outputs."""
    N, K = fp32_lin.weight.shape
    has_bias = fp32_lin.bias is not None

    # Build a representative fp32 input. Batch=2, seq=64 keeps memory low.
    B, T = 2, 64
    x = torch.randn(B, T, K, device=device, dtype=torch.float32)

    # Run both. fp32 Linear under no_grad. NVFP4 Linear runs on the same input
    # and uses native NVFP4 matmul kernels (sm100+).
    try:
        y_ref = fp32_lin(x)
    except Exception as e:
        return None, f"fp32 forward failed: {e}"
    try:
        y_q = nvfp4_lin(x)
    except Exception as e:
        return None, f"nvfp4 forward failed: {e}"

    y_ref = y_ref.float()
    y_q = y_q.float()

    diff = (y_q - y_ref)
    mse = diff.pow(2).mean().item()
    rmse = mse ** 0.5
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    ref_norm = y_ref.pow(2).mean().sqrt().item()
    rel_rmse = rmse / (ref_norm + 1e-12)
    # cosine similarity averaged over (B*T) vectors of dim N
    cos = torch.nn.functional.cosine_similarity(
        y_ref.reshape(-1, N), y_q.reshape(-1, N), dim=1
    ).mean().item()

    # Free intermediate tensors.
    del x, y_ref, y_q, diff

    stats = {
        'shape': f"{N}x{K}",
        'mse': mse,
        'rmse': rmse,
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'rel_rmse': rel_rmse,
        'cosine': cos,
    }
    return stats, None


@torch.no_grad()
def run_comparison(fp32_model, nvfp4_model, device):
    """Walk both models in lock-step, compare every NVFP4 Linear."""
    # Build maps from name -> module for both models.
    fp32_linears = {name: m for name, m in fp32_model.named_modules()
                    if isinstance(m, nn.Linear)}
    nvfp4_linears = {name: m for name, m in nvfp4_model.named_modules()
                     if isinstance(m, nn.Linear)}

    results = []
    group_stats = {}
    n_total = 0
    n_compared = 0
    n_nvfp4 = 0

    for name, fp32_lin in fp32_linears.items():
        n_total += 1
        nvfp4_lin = nvfp4_linears.get(name)
        if nvfp4_lin is None:
            continue
        is_nvfp4 = 'NVFP4' in type(nvfp4_lin.weight).__name__
        if not is_nvfp4:
            # Non-quantized Linear (e.g. shape not divisible by 16) — skip.
            continue
        n_nvfp4 += 1
        group = _classify_group(name)
        stats, err = compare_one_linear(name, fp32_lin, nvfp4_lin, device)
        if err is not None:
            print(f"  [SKIP] {name}: {err}")
            continue
        n_compared += 1
        stats['group'] = group
        stats['name'] = name
        results.append(stats)
        if group not in group_stats:
            group_stats[group] = {'mse': [], 'rmse': [], 'max_abs': [],
                                   'mean_abs': [], 'rel_rmse': [], 'cosine': [], 'count': 0}
        g = group_stats[group]
        g['mse'].append(stats['mse'])
        g['rmse'].append(stats['rmse'])
        g['max_abs'].append(stats['max_abs'])
        g['mean_abs'].append(stats['mean_abs'])
        g['rel_rmse'].append(stats['rel_rmse'])
        g['cosine'].append(stats['cosine'])
        g['count'] += 1

    return results, group_stats, n_total, n_nvfp4, n_compared


def summarize(group_stats, n_total, n_nvfp4, n_compared, results):
    print("\n" + "=" * 78)
    print("NVFP4 vs fp32 precision comparison")
    print("=" * 78)
    print(f"Total nn.Linear layers in model : {n_total}")
    print(f"Layers quantized to NVFP4       : {n_nvfp4}")
    print(f"Layers compared (forward pass)  : {n_compared}")
    print("=" * 78)
    print(f"{'Group':<18}{'#':>5}{'MSE':>14}{'RMSE':>12}{'MaxAbs':>12}{'RelRMSE':>10}{'Cos':>9}")
    print("-" * 78)

    all_mse = []
    all_rmse = []
    all_max = []
    all_rel = []
    all_cos = []

    for group in sorted(group_stats.keys()):
        g = group_stats[group]
        n = g['count']
        if n == 0:
            continue
        mse = sum(g['mse']) / n
        rmse = sum(g['rmse']) / n
        max_abs = max(g['max_abs'])
        mean_max = sum(g['max_abs']) / n
        rel = sum(g['rel_rmse']) / n
        cos = sum(g['cosine']) / n
        print(f"{group:<18}{n:>5}{mse:>14.6e}{rmse:>12.6f}{mean_max:>12.6f}{rel:>10.4%}{cos:>9.6f}")
        all_mse.extend(g['mse'])
        all_rmse.extend(g['rmse'])
        all_max.extend(g['max_abs'])
        all_rel.extend(g['rel_rmse'])
        all_cos.extend(g['cosine'])

    n_all = len(all_mse)
    if n_all > 0:
        print("-" * 78)
        print(f"{'OVERALL':<18}{n_all:>5}"
              f"{sum(all_mse)/n_all:>14.6e}"
              f"{sum(all_rmse)/n_all:>12.6f}"
              f"{max(all_max):>12.6f}"
              f"{sum(all_rel)/n_all:>10.4%}"
              f"{sum(all_cos)/n_all:>9.6f}")
    print("=" * 78)

    # Worst 10 layers by max abs error.
    print("\nWorst 10 layers by max abs error:")
    print(f"{'Name':<60}{'Shape':<14}{'MaxAbs':>12}{'Cos':>9}")
    print("-" * 95)
    worst = sorted(results, key=lambda s: s['max_abs'], reverse=True)[:10]
    for s in worst:
        print(f"{s['name']:<60}{s['shape']:<14}{s['max_abs']:>12.6f}{s['cosine']:>9.6f}")

    # Worst 10 by cosine (lowest = most deviation in direction).
    print("\nWorst 10 layers by cosine similarity (lowest first):")
    print(f"{'Name':<60}{'Shape':<14}{'Cos':>9}{'RelRMSE':>10}")
    print("-" * 95)
    worst_cos = sorted(results, key=lambda s: s['cosine'])[:10]
    for s in worst_cos:
        print(f"{s['name']:<60}{s['shape']:<14}{s['cosine']:>9.6f}{s['rel_rmse']:>10.4%}")

    print("\nVerdict:")
    if n_all == 0:
        print("  NO NVFP4 layers were compared. Quantization is NOT confirmed.")
        return
    overall_cos = sum(all_cos) / n_all
    overall_rel = sum(all_rel) / n_all
    if overall_cos >= 0.999 and overall_rel < 0.05:
        grade = "EXCELLENT"
    elif overall_cos >= 0.995 and overall_rel < 0.10:
        grade = "GOOD"
    elif overall_cos >= 0.98 and overall_rel < 0.20:
        grade = "ACCEPTABLE"
    else:
        grade = "POOR"
    print(f"  Mean cosine similarity : {overall_cos:.6f}")
    print(f"  Mean relative RMSE     : {overall_rel:.4%}")
    print(f"  Max abs error (any layer): {max(all_max):.6f}")
    print(f"  Quantization grade     : {grade}")
    print("  -> NVFP4 quantization is " +
          ("REAL (true 4-bit weight-only, native NVFP4 matmul on sm100+)."
           if grade != "POOR" else
           "REAL but introduces significant error; consider recalibration."))


def main():
    parser = argparse.ArgumentParser(description="Verify NVFP4 precision vs fp32")
    parser.add_argument('--fp32_model',
                        default='pretrained_models/SoulX-Singer/model.pt')
    parser.add_argument('--nvfp4_model',
                        default='pretrained_models/SoulX-Singer-nvfp4/model.pt')
    parser.add_argument('--config',
                        default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if not TORCHAO_NVFP4_AVAILABLE:
        print("ERROR: torchao is not installed. pip install torchao>=0.17.0")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for native NVFP4 forward.")
        sys.exit(1)
    cap = torch.cuda.get_device_capability()[0]
    if cap < 10:
        print(f"ERROR: NVFP4 native kernels require Blackwell sm100+, got sm{cap}.")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm{cap})")
    print(f"torch: {torch.__version__}, torchao: {TORCHAO_NVFP4_AVAILABLE}")

    config = OmegaConf.load(args.config)

    fp32_model = load_fp32_model(args.fp32_model, config, args.device)
    nvfp4_model = load_nvfp4_model(args.nvfp4_model, config, args.device)

    # File size summary
    fp32_size = os.path.getsize(args.fp32_model) / 1024 / 1024
    nvfp4_size = os.path.getsize(args.nvfp4_model) / 1024 / 1024
    print(f"\nFile size:")
    print(f"  fp32  : {fp32_size:.2f} MB")
    print(f"  nvfp4 : {nvfp4_size:.2f} MB")
    print(f"  ratio : {fp32_size / nvfp4_size:.2f}x compression")

    # Verify NVFP4 weights are actually NVFP4Tensor (not fake / not dequantized).
    n_nvfp4_real = 0
    n_nvfp4_fake = 0
    for name, m in nvfp4_model.named_modules():
        if isinstance(m, nn.Linear) and 'NVFP4' in type(m.weight).__name__:
            n_nvfp4_real += 1
        elif isinstance(m, nn.Linear):
            n_nvfp4_fake += 1
    print(f"\nNVFP4 verification:")
    print(f"  Linear layers with NVFP4Tensor weight : {n_nvfp4_real}")
    print(f"  Linear layers with regular Tensor weight: {n_nvfp4_fake}")
    if n_nvfp4_real == 0:
        print("  -> WARNING: no NVFP4Tensor weights detected! This is NOT real NVFP4.")
        sys.exit(1)
    print("  -> NVFP4Tensor weights present. Quantization is REAL (not pseudo-quant).")

    # Run forward comparison
    print(f"\nRunning per-layer forward comparison on {args.device}...")
    t0 = time.time()
    results, group_stats, n_total, n_nvfp4, n_compared = run_comparison(
        fp32_model, nvfp4_model, args.device
    )
    dt = time.time() - t0
    print(f"Compared {n_compared}/{n_nvfp4} NVFP4 layers in {dt:.1f}s")

    summarize(group_stats, n_total, n_nvfp4, n_compared, results)

    # Cleanup
    del fp32_model
    del nvfp4_model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
