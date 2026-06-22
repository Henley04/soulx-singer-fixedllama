#!/bin/bash
# Master pipeline: Japanese TTS embedding initialization & staged training
#
# Runs all 5 tasks sequentially:
#   1. Generate phoneme mapping config
#   2. Initialize embeddings with scale calibration
#   3. Phase 1 training (warmup)
#   4. Validate Phase 1 + rollback check
#   5. Phase 2 training (embedding fine-tuning)
#   6. Validate Phase 2 + rollback check
#   7. Phase 3 training (joint fine-tuning)
#   8. Final validation + synthesis check
#
# Usage:
#   bash train/lora_jp/run_pipeline.sh
#
# Any step failure aborts the pipeline with error message.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_DIR="outputs/lora_jp"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"
MODEL_PATH="pretrained_models/SoulX-Singer/model.pt"
CONFIG="soulxsinger/config/soulxsinger.yaml"
PHONESET="train/lora_jp/jp_phone_set.json"
METADATA="train/lora_jp/dataset/metadata.json"
WAV_DIR="train/lora_jp/dataset/wavs"

mkdir -p "$OUTPUT_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

fail() {
    log "FAILED: $1"
    echo ""
    echo "Pipeline aborted. Check log: $LOG_FILE"
    exit 1
}

echo "=============================================" | tee "$LOG_FILE"
log "Japanese TTS Pipeline: Starting"
echo "=============================================" | tee -a "$LOG_FILE"

# ── Step 1: Generate phoneme mapping ─────────────────────────────────
log ""
log "[Step 1/8] Generating phoneme mapping..."
python train/lora_jp/phoneme_mapping.py 2>&1 | tee -a "$LOG_FILE" || fail "phoneme_mapping.py"
log "Mapping saved to: train/lora_jp/jp_phoneme_mapping.json"

# ── Step 2: Initialize embeddings ────────────────────────────────────
log ""
log "[Step 2/8] Initializing embeddings..."
python train/lora_jp/init_embeddings.py \
    --model_path "$MODEL_PATH" \
    --mapping train/lora_jp/jp_phoneme_mapping.json \
    --phoneset "$PHONESET" \
    --output "${OUTPUT_DIR}/init_embed.pt" \
    --target_std 0.9 \
    2>&1 | tee -a "$LOG_FILE" || fail "init_embeddings.py"
log "Init embeddings saved to: ${OUTPUT_DIR}/init_embed.pt"

# ── Step 3: Phase 1 training ─────────────────────────────────────────
log ""
log "[Step 3/8] Phase 1: Warmup & Adaptation (frozen embedding)..."
python train/lora_jp/train_staged.py \
    --phase 1 \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --init_embed "${OUTPUT_DIR}/init_embed.pt" \
    --batch_size 2 \
    --lr 5e-5 \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE" || fail "train_staged.py phase 1"
log "Phase 1 complete. Checkpoint: ${OUTPUT_DIR}/stage1/best.pt"

# ── Step 4: Validate Phase 1 ────────────────────────────────────────
log ""
log "[Step 4/8] Validating Phase 1..."
python train/lora_jp/validate_and_rollback.py \
    --checkpoint "${OUTPUT_DIR}/stage1/best.pt" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE" || fail "validate phase 1"

# Check for rollback
if grep -q "ROLLBACK TRIGGERED" "${OUTPUT_DIR}/validation_results.json" 2>/dev/null; then
    ROLLBACK=$(python -c "import json; d=json.load(open('${OUTPUT_DIR}/validation_results.json')); print(d.get('rollback', False))")
    if [ "$ROLLBACK" = "True" ]; then
        log "Phase 1 validation triggered rollback. See ${OUTPUT_DIR}/validation_results.json"
        fail "Phase 1 rollback triggered"
    fi
fi

# ── Step 5: Phase 2 training ─────────────────────────────────────────
log ""
log "[Step 5/8] Phase 2: Embedding Fine-tuning..."
python train/lora_jp/train_staged.py \
    --phase 2 \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resume "${OUTPUT_DIR}/stage1/best.pt" \
    --batch_size 2 \
    --lr 5e-5 \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE" || fail "train_staged.py phase 2"
log "Phase 2 complete. Checkpoint: ${OUTPUT_DIR}/stage2/best.pt"

# ── Step 6: Validate Phase 2 ────────────────────────────────────────
log ""
log "[Step 6/8] Validating Phase 2..."
python train/lora_jp/validate_and_rollback.py \
    --checkpoint "${OUTPUT_DIR}/stage2/best.pt" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE" || fail "validate phase 2"

if grep -q "ROLLBACK TRIGGERED" "${OUTPUT_DIR}/validation_results.json" 2>/dev/null; then
    ROLLBACK=$(python -c "import json; d=json.load(open('${OUTPUT_DIR}/validation_results.json')); print(d.get('rollback', False))")
    if [ "$ROLLBACK" = "True" ]; then
        log "Phase 2 validation triggered rollback."
        fail "Phase 2 rollback triggered"
    fi
fi

# ── Step 7: Phase 3 training ─────────────────────────────────────────
log ""
log "[Step 7/8] Phase 3: Joint Fine-tuning..."
python train/lora_jp/train_staged.py \
    --phase 3 \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resume "${OUTPUT_DIR}/stage2/best.pt" \
    --batch_size 2 \
    --lr 5e-5 \
    --device cuda \
    2>&1 | tee -a "$LOG_FILE" || fail "train_staged.py phase 3"
log "Phase 3 complete. Checkpoint: ${OUTPUT_DIR}/stage3/best.pt"

# ── Step 8: Final validation + synthesis ─────────────────────────────
log ""
log "[Step 8/8] Final validation + synthesis check..."
python train/lora_jp/validate_and_rollback.py \
    --checkpoint "${OUTPUT_DIR}/stage3/best.pt" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --phoneset_path "$PHONESET" \
    --dataset_metadata "$METADATA" \
    --dataset_wav_dir "$WAV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --synthesize \
    2>&1 | tee -a "$LOG_FILE" || fail "final validation"

echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
log "Pipeline complete!"
log "  Checkpoints: ${OUTPUT_DIR}/stage{1,2,3}/best.pt"
log "  Validation:  ${OUTPUT_DIR}/validation_results.json"
log "  Synthesis:   ${OUTPUT_DIR}/synthesis_check/"
log "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
log "  Full log:    ${LOG_FILE}"
echo "=============================================" | tee -a "$LOG_FILE"
