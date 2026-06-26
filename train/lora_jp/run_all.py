"""
End-to-end JP LoRA fine-tuning orchestrator.

Runs the full pipeline in one shot:
  Step 0:  init_embeddings.py  -> outputs/lora_jp/init_embed.pt
  Step 1:  train_staged phase1 (warmup, freeze embed, bf16 autocast)
  Step 2:  train_staged phase2 (embed FT, lower LR, bf16 autocast)
  Step 3:  train_staged phase3 (joint, full FT, bf16 autocast)
  Step 4:  export_onnx.py -> onnx_models/fp16/JP/ (fp16)

All parameters are baked in below — just run:
    python train/lora_jp/run_all.py            # resume (skip done stages)
    python train/lora_jp/run_all.py --force    # re-run ALL stages from scratch
    python train/lora_jp/run_all.py --stage 1  # run only a specific stage (0/1/2/3/4)

Stage chaining convention (matches train_staged.py defaults):
  stage1 -> outputs/lora_jp/stage1/best.pt
  stage2 -> outputs/lora_jp/stage2/best.pt
  stage3 -> outputs/lora_jp/stage3/best.pt

Precision: bf16 autocast + gradient checkpointing (no NVFP4, no precision loss).
"""
import os
import sys
import argparse
import subprocess

# ─── All-in-one configuration (edit here, no CLI args) ───────────────────
CONFIG = {
    # Paths (relative to repo root, which is two levels up from this file)
    "repo_root":      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "model_path":     "pretrained_models/SoulX-Singer/model.pt",
    "config":         "soulxsinger/config/soulxsinger.yaml",
    "phoneset":       "train/lora_jp/jp_phone_set.json",
    "mapping":        "train/lora_jp/jp_phoneme_mapping.json",
    "dataset_meta":   "train/lora_jp/dataset/metadata.json",
    "dataset_wav":    "train/lora_jp/dataset/wavs",
    "output_dir":     "outputs/lora_jp",
    "app_root":       os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),

    # Init embeddings
    "target_std":     0.9,

    # Training
    "device":         "cuda",
    "batch_size":     4,
    "lr_phase1":      5e-5,
    "lr_phase2":      5e-5,
    "lr_phase3":      2e-5,
    "num_workers":    0,
    "val_ratio":      0.1,
    "seed":           42,

    # Gradient checkpointing (enabled by default)
    "grad_checkpoint": True,

    # Skip already-completed stages (idempotent re-runs)
    "skip_if_done":   True,
}


def run(cmd, cwd):
    """Run a subprocess, raise on failure."""
    print("\n" + "=" * 70)
    print("$ " + " ".join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def exists(path):
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser(description="End-to-end JP LoRA training")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing checkpoints and re-run all stages")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4], default=None,
                        help="Run only a specific stage (0=init, 1/2/3=train, 4=export). Default: all.")
    args = parser.parse_args()

    cfg = CONFIG
    repo = cfg["repo_root"]
    os.chdir(repo)
    print(f"Working dir: {repo}")
    print(f"Force re-run: {args.force}")
    if args.stage is not None:
        print(f"Single-stage mode: {args.stage}")

    skip = cfg["skip_if_done"] and not args.force

    out = cfg["output_dir"]
    init_embed = os.path.join(out, "init_embed.pt")
    stage1_ckpt = os.path.join(out, "stage1", "best.pt")
    stage2_ckpt = os.path.join(out, "stage2", "best.pt")
    stage3_ckpt = os.path.join(out, "stage3", "best.pt")

    py = sys.executable

    # ── Step 0: init embeddings ─────────────────────────────────────
    if args.stage is not None and args.stage != 0:
        print("[SKIP] stage 0 (init) not selected")
    elif skip and exists(init_embed):
        print(f"[SKIP] init_embed already exists: {init_embed}")
    else:
        run([py, "train/lora_jp/init_embeddings.py",
             "--model_path", cfg["model_path"],
             "--mapping",    cfg["mapping"],
             "--phoneset",   cfg["phoneset"],
             "--output",     init_embed,
             "--target_std", str(cfg["target_std"])], cwd=repo)

    # ── Step 1: Phase 1 (warmup) ────────────────────────────────────
    if args.stage is not None and args.stage != 1:
        print("[SKIP] stage 1 not selected")
    elif skip and exists(stage1_ckpt):
        print(f"[SKIP] stage1 already done: {stage1_ckpt}")
    else:
        run([py, "train/lora_jp/train_staged.py",
             "--phase",           "1",
             "--model_path",      cfg["model_path"],
             "--config",          cfg["config"],
             "--phoneset_path",   cfg["phoneset"],
             "--mapping_path",    cfg["mapping"],
             "--dataset_metadata", cfg["dataset_meta"],
             "--dataset_wav_dir", cfg["dataset_wav"],
             "--output_dir",      out,
             "--init_embed",      init_embed,
             "--device",          cfg["device"],
             "--batch_size",      str(cfg["batch_size"]),
             "--lr",              str(cfg["lr_phase1"]),
             "--num_workers",     str(cfg["num_workers"]),
             "--val_ratio",       str(cfg["val_ratio"]),
             "--seed",            str(cfg["seed"])], cwd=repo)

    # ── Step 2: Phase 2 (embed FT) ──────────────────────────────────
    if args.stage is not None and args.stage != 2:
        print("[SKIP] stage 2 not selected")
    elif skip and exists(stage2_ckpt):
        print(f"[SKIP] stage2 already done: {stage2_ckpt}")
    else:
        run([py, "train/lora_jp/train_staged.py",
             "--phase",           "2",
             "--model_path",      cfg["model_path"],
             "--config",          cfg["config"],
             "--phoneset_path",   cfg["phoneset"],
             "--mapping_path",    cfg["mapping"],
             "--dataset_metadata", cfg["dataset_meta"],
             "--dataset_wav_dir", cfg["dataset_wav"],
             "--output_dir",      out,
             "--resume",          stage1_ckpt,
             "--device",          cfg["device"],
             "--batch_size",      str(cfg["batch_size"]),
             "--lr",              str(cfg["lr_phase2"]),
             "--num_workers",     str(cfg["num_workers"]),
             "--val_ratio",       str(cfg["val_ratio"]),
             "--seed",            str(cfg["seed"])], cwd=repo)

    # ── Step 3: Phase 3 (joint) ─────────────────────────────────────
    if args.stage is not None and args.stage != 3:
        print("[SKIP] stage 3 not selected")
    elif skip and exists(stage3_ckpt):
        print(f"[SKIP] stage3 already done: {stage3_ckpt}")
    else:
        run([py, "train/lora_jp/train_staged.py",
             "--phase",           "3",
             "--model_path",      cfg["model_path"],
             "--config",          cfg["config"],
             "--phoneset_path",   cfg["phoneset"],
             "--mapping_path",    cfg["mapping"],
             "--dataset_metadata", cfg["dataset_meta"],
             "--dataset_wav_dir", cfg["dataset_wav"],
             "--output_dir",      out,
             "--resume",          stage2_ckpt,
             "--device",          cfg["device"],
             "--batch_size",      str(cfg["batch_size"]),
             "--lr",              str(cfg["lr_phase3"]),
             "--num_workers",     str(cfg["num_workers"]),
             "--val_ratio",       str(cfg["val_ratio"]),
             "--seed",            str(cfg["seed"])], cwd=repo)

    # ── Step 4: export ONNX (fp16, place in JP dir) ─────────────────
    jp_dir = os.path.join(cfg["app_root"], "onnx_models", "fp16", "JP")
    if args.stage is not None and args.stage != 4:
        print("[SKIP] export stage not selected")
    elif skip and exists(os.path.join(jp_dir, "preflow.onnx")):
        print(f"[SKIP] JP ONNX already exported: {jp_dir}")
    else:
        run([py, "train/lora_jp/export_onnx.py",
             "--checkpoint", stage3_ckpt,
             "--base_model", cfg["model_path"],
             "--output_dir", jp_dir], cwd=repo)

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE")
    print(f"  Final checkpoint: {stage3_ckpt}")
    print(f"  JP ONNX dir:     {jp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
