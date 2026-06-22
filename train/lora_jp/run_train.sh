#!/bin/bash
# LoRA Japanese phoneme fine-tuning for SoulX-Singer
#
# Usage:
#   bash train/lora_jp/run_train.sh
#
# Steps:
#   1. Prepare dataset from PJS Corpus
#   2. Train LoRA adapters
#   3. Export to ONNX
#   4. Validate with inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " SoulX-Singer LoRA Japanese Fine-tuning"
echo "=========================================="

# Step 1: Prepare dataset
echo ""
echo "[Step 1/4] Preparing dataset from PJS Corpus..."
python train/lora_jp/prepare_dataset.py \
    --corpus_dir pretrained_models/SoulX-Singer/assets/LoRA-JP/PJS_corpus_ver1.1 \
    --output_dir train/lora_jp/dataset \
    --sample_rate 24000

# Step 2: Train LoRA
echo ""
echo "[Step 2/4] Training LoRA adapters..."
python train/lora_jp/train_lora.py \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --phoneset_path train/lora_jp/jp_phone_set.json \
    --dataset_metadata train/lora_jp/dataset/metadata.json \
    --dataset_wav_dir train/lora_jp/dataset/wavs \
    --output_dir outputs/lora_jp \
    --batch_size 4 \
    --lr 5e-5 \
    --embedding_lr 1e-4 \
    --epochs 200 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --save_every 20 \
    --eval_every 10 \
    --device cuda \
    --use_amp

# Step 3: Export to ONNX
echo ""
echo "[Step 3/4] Exporting LoRA to ONNX..."
python train/lora_jp/export_onnx.py \
    --lora_checkpoint outputs/lora_jp/lora_jp_latest.pt \
    --output_dir outputs/lora_jp/onnx \
    --verify

# Step 4: Validate with inference
echo ""
echo "[Step 4/4] Validating with inference..."
python train/lora_jp/infer_lora.py \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --lora_checkpoint outputs/lora_jp/lora_jp_latest.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --phoneset_path train/lora_jp/jp_phone_set.json \
    --dataset_dir train/lora_jp/dataset \
    --output_dir outputs/lora_jp/inference \
    --num_samples 5 \
    --device cuda

echo ""
echo "=========================================="
echo " Training complete!"
echo " Checkpoints: outputs/lora_jp/"
echo " ONNX models: outputs/lora_jp/onnx/"
echo " Inference results: outputs/lora_jp/inference/"
echo " TensorBoard: tensorboard --logdir outputs/lora_jp/runs"
echo "=========================================="
