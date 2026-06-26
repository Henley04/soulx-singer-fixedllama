# SoulX-Singer 日语音素微调

通过三阶段微调 preflow 模块 + 扩展音素嵌入层，使 SoulX-Singer 支持日语歌声合成。

## 训练结果

| 指标 | 值 |
|------|-----|
| 数据集 | PJS Corpus (100首歌，~15分钟) |
| 可训练参数 | ~4.25M (preflow + JP embedding + cond_emb) |
| 训练阶段 | 三阶段 (warmup → embed FT → joint) |
| GPU | RTX 5060 8GB (bf16 autocast + gradient checkpointing) |
| 精度 | bfloat16 autocast (无量化，无精度损失) |
| batch_size | 4 (gradient checkpointing 省显存) |

## 文件结构

```
train/lora_jp/
├── jp_phone_set.json       # 扩展音素表 (3000 base + 33 JP = 3033)
├── jp_phoneme_mapping.json # JP 音素到 EN 源音素的映射 (配置驱动)
├── prepare_dataset.py      # PJS Corpus → 训练数据 (含 SP/slur 样本)
├── dataset.py              # PyTorch Dataset (确定性 wav 匹配)
├── phoneme_mapping.py      # 生成 jp_phoneme_mapping.json
├── init_embeddings.py      # 初始化 JP 嵌入 (只 L2 归一化 JP 行)
├── train_staged.py         # 三阶段训练脚本 (主入口)
├── validate_and_rollback.py # 验证 + 回滚决策 (使用独立 val 集)
├── verify_checkpoint.py    # Checkpoint 质量检查
├── export_onnx.py          # 导出 ONNX (text_encoder + preflow)
├── run_pipeline.py         # 一键训练管线 (8 步)
└── run_pipeline.sh         # Shell 包装
```

## 音素集设计

### 停顿处理（关键决策）

**`jp_pau` 已删除**。SXSEditor 推理时所有停顿都使用跨语言共享的 `<SP>` (ID=1)，
而非语言特定的 `jp_pau`。为对齐训练-推理语义，训练数据中 PJS lab 的 `pau` 标签
直接映射到 `<SP>` (ID=1)，复用 base 模型已学好的停顿嵌入，无需新增训练。

### 日语音素（索引 3000-3032，共 33 个）

在原有 3000 个音素（含特殊 token 和占位符）基础上追加 33 个日语音素：

| 索引范围 | 音素 | 说明 |
|---------|------|------|
| 3000-3004 | jp_a, jp_i, jp_u, jp_e, jp_o | 五元音 |
| 3005-3024 | jp_k, jp_s, ..., jp_ts | 清辅音、浊辅音、摩擦音、塞擦音 |
| 3025-3032 | jp_ky, jp_gy, ..., jp_cl | 拗音 + 促音 (jp_cl) |

**注意**：`jp_pau` 已从音素表中移除，停顿统一使用 `<SP>` (ID=1)。

## 快速开始

```bash
cd SoulX-Singer
pip install -r train/lora_jp/requirements.txt

# 一键训练 (数据预处理 → 初始化 → 三阶段训练 → 验证)
python train/lora_jp/run_pipeline.py
```

或分步执行：

```bash
# 1. 数据预处理 (生成 metadata.json + wavs/)
python train/lora_jp/prepare_dataset.py

# 2. 生成音素映射配置
python train/lora_jp/phoneme_mapping.py

# 3. 初始化 JP 嵌入
python train/lora_jp/init_embeddings.py \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --mapping train/lora_jp/jp_phoneme_mapping.json \
    --phoneset train/lora_jp/jp_phone_set.json \
    --output outputs/lora_jp/init_embed.pt

# 4-6. 三阶段训练
python train/lora_jp/train_staged.py --phase 1 --init_embed outputs/lora_jp/init_embed.pt
python train/lora_jp/train_staged.py --phase 2 --resume outputs/lora_jp/stage1/best.pt
python train/lora_jp/train_staged.py --phase 3 --resume outputs/lora_jp/stage2/best.pt

# 7. 验证
python train/lora_jp/validate_and_rollback.py --checkpoint outputs/lora_jp/stage3/best.pt

# 8. 导出 ONNX
python train/lora_jp/export_onnx.py \
    --checkpoint outputs/lora_jp/stage3/best.pt \
    --base_model pretrained_models/SoulX-Singer/model.pt \
    --output_dir onnx_models/fp16/JP
```

## ONNX 兼容性

导出的 ONNX 文件与 SXSEditor 推理管线完全兼容：

| 文件 | 用途 | 是否导出 |
|------|------|---------|
| note_text_encoder.onnx | 扩展嵌入 (3033×512, FP16) | ✅ |
| preflow.onnx | 微调后 preflow (无 LayerNorm) | ✅ |
| note_pitch_encoder.onnx | 音高嵌入 | ❌ (共享 base 模型) |

**部署方式**：将 `note_text_encoder.onnx` 和 `preflow.onnx` 放到
`onnx_models/fp16/JP/` 目录。SXSEditor 会自动检测 JP 模型并在日语歌词时切换。

`phone_set.json` 需同步更新到 `src/inference/phone_set.json`（已包含 3033 个音素）。

## 训练-推理一致性

| 环节 | 一致性 |
|------|--------|
| BOW/EOW/SEP token | ✅ 日语不加 SEP（与英文不同） |
| **BOW/EOW 包裹结构** | ✅ 一个 note 的所有音素共享一个 BOW/EOW（`jp_t-a` 格式，匹配推理） |
| note_pitch (MIDI 0-127) | ✅ 直接索引 Embedding(256)，pitch_encoder 冻结复用 base |
| note_type (1=SP, 2=normal, 3=slur) | ✅ 训练数据含 SP 和 slur 样本 |
| f0 量化 (361 bins, 20 cents) | ✅ |
| preflow 输入 (无 LayerNorm) | ✅ |
| 停顿音素 (`<SP>` ID=1) | ✅ 训练-推理统一 |

### 音素包裹格式（关键修复）

**修复前**（导致音素混淆）：prepare_dataset.py 把一个 MIDI note 内的辅音+元音拆成独立 entry，
每个音素独立 BOW/EOW。训练 token: `[PAD BOW jp_t EOW BOW jp_a EOW ...]`

**修复后**（与推理一致）：一个 MIDI note 的所有音素合并成一个 entry `jp_t-a`，
共享一个 BOW/EOW。训练 token: `[PAD BOW jp_t jp_a EOW ...]`

此格式与 SXSEditor 推理侧 [preprocessing.js](../../../src/inference/pipeline/preprocessing.js) 的处理完全一致：
一个 note 调用 `_japaneseG2p` 得到多音素，全部放在同一个 BOW/EOW 块内（不加 SEP）。

## pitch_encoder 冻结说明

**pitch_encoder 在 JP LoRA 中完全冻结，不微调。** 原因：

1. **pitch 是 MIDI 索引，无语种语义** — MIDI 60 在日语/英语/中文里都是 C4
2. **base 模型已全局覆盖** — 实测 base pitch_encoder 256 行 norm 均匀（20-24），覆盖 MIDI 0-255
3. **PJS 数据 pitch 范围窄**（36-72），微调会扭曲 base 对其他音高的建模
4. **ONNX 导出时不导出 pitch_encoder** — 复用 base 模型，微调结果会被丢弃

因此日语推理时即使用户输入 MIDI 84+（训练集外），base 模型的 pitch_encoder 也能提供合理表征。

## 训练优化

### 移除 NVFP4，改用 bf16 autocast

早期版本使用 NVFP4 weight-only 量化加速训练，但实际测试发现：

1. **加速有限**：NVFP4 只加速 forward matmul，微调时主要瓶颈在 backward dgrad（对 input 的梯度），NVFP4 dgrad 内核效率低
2. **精度损失**：276 层平均 cosine 0.9954，相对 RMSE 9.5%，影响梯度信号质量

改用 bf16 autocast 后：
- forward 和 backward 都有原生 bf16 内核（无精度损失）
- bf16 指数范围与 fp32 相同，无需 GradScaler
- 实际训练速度与 NVFP4 相当或更快（backward 加速弥补 forward 差距）

### Gradient Checkpointing

22 层 DiffLlama (diff_estimator) 启用 `gradient_checkpointing`：
- 用 `use_reentrant=False` 模式支持 `cond_embedding` kwargs（旧版 reentrant 不支持）
- 以 ~20% 额外 forward 计算换取大幅激活显存节省
- 使 batch_size 从 1 提升到 4，提高 GPU 利用率

### Hidden States 按需保存

原 DiffLlama 每层都 `clone()` 保存 hidden_states（供 repa/ctc 使用），但 JP LoRA 不用 repa/ctc。改为只在 `use_repa` 或 `use_ctc` 为 True 时才保存，省去 22 次 clone 的内存和时间开销。

## 自定义训练

使用自己的日语数据集：

1. 准备 WAV (24kHz) + Lab 音素标签 + MIDI 文件
2. Lab 格式：`开始时间(100ns) 结束时间(100ns) 音素名`
3. 音素名使用 `jp_` 前缀或裸名（自动映射）
4. 修改 `prepare_dataset.py` 中的路径后运行训练

## 依赖

- PyTorch >= 2.0
- torchaudio
- omegaconf
- tensorboard
- mido (MIDI解析)
- praat-parselmouth (F0提取)
- soundfile
- onnxruntime (ONNX 导出验证)
