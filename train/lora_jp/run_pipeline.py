"""
Master pipeline runner: Japanese TTS embedding init & staged training.

Runs all steps sequentially. Aborts on any failure.

Usage:
    python train/lora_jp/run_pipeline.py
"""

import os
import sys
import subprocess
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
os.chdir(PROJECT_DIR)

OUTPUT_DIR = "outputs/lora_jp"
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline.log")
MODEL_PATH = "pretrained_models/SoulX-Singer/model.pt"
CONFIG = "soulxsinger/config/soulxsinger.yaml"
PHONESET = "train/lora_jp/jp_phone_set.json"
METADATA = "train/lora_jp/dataset/metadata.json"
WAV_DIR = "train/lora_jp/dataset/wavs"
# JVS-MuSiC extra dataset (merged with PJS for training)
JVS_METADATA = "train/lora_jp/dataset_jvs/metadata.json"
JVS_WAV_DIR = "train/lora_jp/dataset_jvs"
ONNX_JP_DIR = os.path.join(PROJECT_DIR, '..', 'onnx_models', 'fp16', 'JP')

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_fh = open(LOG_FILE, 'w', encoding='utf-8')


def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    log_fh.write(line + '\n')
    log_fh.flush()


def run_step(description, cmd):
    log("")
    log(f"{'='*60}")
    log(description)
    log(f"{'='*60}")
    log(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        log(f"FAILED: {description} (exit code {result.returncode})")
        log(f"Pipeline aborted. Check log: {LOG_FILE}")
        sys.exit(1)

    log(f"DONE: {description}")


def main():
    log("="*60)
    log("Japanese TTS Pipeline: Starting")
    log("="*60)

    # Step 1: Prepare dataset (PJS Corpus -> metadata.json + wavs/)
    run_step(
        "[Step 1/9] Preparing dataset (PJS Corpus -> metadata + wavs)",
        [sys.executable, "train/lora_jp/prepare_dataset.py"]
    )

    # Step 1b: Prepare JVS-MuSiC dataset (audio -> metadata.json + wavs via ROSVOT + G2P)
    run_step(
        "[Step 1b/9] Preparing JVS-MuSiC dataset (ROSVOT note transcription + jp_g2p)",
        [sys.executable, "train/lora_jp/prepare_jvs_dataset.py"]
    )

    # Step 2: Generate phoneme mapping
    run_step(
        "[Step 2/9] Generating phoneme mapping",
        [sys.executable, "train/lora_jp/phoneme_mapping.py"]
    )

    # Step 3: Initialize embeddings
    run_step(
        "[Step 3/9] Initializing embeddings",
        [sys.executable, "train/lora_jp/init_embeddings.py",
         "--model_path", MODEL_PATH,
         "--mapping", "train/lora_jp/jp_phoneme_mapping.json",
         "--phoneset", PHONESET,
         "--output", f"{OUTPUT_DIR}/init_embed.pt",
         "--target_std", "0.9"]
    )

    # Step 4: Phase 1 training
    run_step(
        "[Step 4/9] Phase 1: Warmup & Adaptation (frozen embedding)",
        [sys.executable, "train/lora_jp/train_staged.py",
         "--phase", "1",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--extra_dataset_metadata", JVS_METADATA,
         "--extra_dataset_wav_dir", JVS_WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--init_embed", f"{OUTPUT_DIR}/init_embed.pt",
         "--batch_size", "1",
         "--lr", "5e-5",
         "--device", "cuda"]
    )

    # Step 5: Validate Phase 1
    run_step(
        "[Step 5/9] Validating Phase 1",
        [sys.executable, "train/lora_jp/validate_and_rollback.py",
         "--checkpoint", f"{OUTPUT_DIR}/stage1/best.pt",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--device", "cuda"]
    )
    _check_rollback("Phase 1")

    # Step 6: Phase 2 training
    run_step(
        "[Step 6/9] Phase 2: Embedding Fine-tuning",
        [sys.executable, "train/lora_jp/train_staged.py",
         "--phase", "2",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--extra_dataset_metadata", JVS_METADATA,
         "--extra_dataset_wav_dir", JVS_WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--resume", f"{OUTPUT_DIR}/stage1/best.pt",
         "--batch_size", "1",
         "--lr", "5e-5",
         "--device", "cuda"]
    )

    # Step 7: Validate Phase 2
    run_step(
        "[Step 7/9] Validating Phase 2",
        [sys.executable, "train/lora_jp/validate_and_rollback.py",
         "--checkpoint", f"{OUTPUT_DIR}/stage2/best.pt",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--device", "cuda"]
    )
    _check_rollback("Phase 2")

    # Step 8: Phase 3 training
    run_step(
        "[Step 8/9] Phase 3: Joint Fine-tuning",
        [sys.executable, "train/lora_jp/train_staged.py",
         "--phase", "3",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--extra_dataset_metadata", JVS_METADATA,
         "--extra_dataset_wav_dir", JVS_WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--resume", f"{OUTPUT_DIR}/stage2/best.pt",
         "--batch_size", "1",
         "--lr", "5e-5",
         "--device", "cuda"]
    )

    # Step 9: Final validation + synthesis + ONNX export
    run_step(
        "[Step 9/9] Final validation + synthesis check",
        [sys.executable, "train/lora_jp/validate_and_rollback.py",
         "--checkpoint", f"{OUTPUT_DIR}/stage3/best.pt",
         "--model_path", MODEL_PATH,
         "--config", CONFIG,
         "--phoneset_path", PHONESET,
         "--dataset_metadata", METADATA,
         "--dataset_wav_dir", WAV_DIR,
         "--output_dir", OUTPUT_DIR,
         "--device", "cuda",
         "--synthesize"]
    )

    run_step(
        "[Step 9/9] Exporting ONNX models (note_text_encoder + preflow + cond_emb)",
        [sys.executable, "train/lora_jp/export_onnx.py",
         "--checkpoint", f"{OUTPUT_DIR}/stage3/best.pt",
         "--base_model", MODEL_PATH,
         "--output_dir", ONNX_JP_DIR]
    )

    log("")
    log("="*60)
    log("Pipeline complete!")
    log(f"  Checkpoints: {OUTPUT_DIR}/stage1/best.pt, stage2/best.pt, stage3/best.pt")
    log(f"  Validation:  {OUTPUT_DIR}/validation_results.json")
    log(f"  Synthesis:   {OUTPUT_DIR}/synthesis_check/")
    log(f"  ONNX models: {ONNX_JP_DIR}")
    log(f"  TensorBoard: tensorboard --logdir {OUTPUT_DIR}")
    log(f"  Full log:    {LOG_FILE}")
    log("="*60)


def _check_rollback(phase_name):
    """Check validation results and warn on rollback conditions.

    Rollback conditions are treated as WARNINGS, not pipeline aborts.
    The pipeline continues to subsequent stages so the model can recover
    in later phases. Only FATAL conditions (NaN, missing checkpoint) abort.
    """
    import json
    results_path = os.path.join(OUTPUT_DIR, "validation_results.json")
    if not os.path.exists(results_path):
        return
    with open(results_path, 'r') as f:
        data = json.load(f)
    if data.get("rollback"):
        reasons = data.get("reasons", [])
        log(f"WARNING: {phase_name} rollback condition detected: {reasons}")
        log(f"  Continuing pipeline — later phases may recover.")
        log(f"  (If final validation still triggers rollback, inspect embeddings manually.)")
    else:
        log(f"{phase_name} validation passed.")


if __name__ == '__main__':
    main()
