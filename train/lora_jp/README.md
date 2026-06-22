# SoulX-Singer 日语音素微调

通过微调 preflow 模块 + 扩展音素嵌入层，使 SoulX-Singer 支持日语歌声合成。

## 训练结果

| 指标 | 值 |
|------|-----|
| 数据集 | PJS Corpus (100首歌，~15分钟) |
| 可训练参数 | 4.25M (0.6%) |
| 训练 Loss | 4.02 → 1.42 (50 epochs) |
| GPU | RTX 5060 8GB (AMP) |
| 训练耗时 | ~60秒 |

## 文件结构

```
train/lora_jp/
├── jp_phone_set.json      # 扩展音素表 (2820原有 + 34日语 = 2854)
├── prepare_dataset.py      # PJS Corpus → 训练数据
├── dataset.py              # PyTorch Dataset
├── train_lora.py           # 训练脚本 (preflow全量微调 + JP embedding)
├── export_onnx.py          # 导出 ONNX (FP32 + FP16)
├── infer_lora.py           # PyTorch 推理验证
├── run_train.sh            # 一键训练
└── requirements.txt        # 依赖
```

## 快速开始

```bash
cd SoulX-Singer
pip install -r train/lora_jp/requirements.txt

# 一键训练 (数据预处理 → 训练 → 导出ONNX)
bash train/lora_jp/run_train.sh
```

或分步执行：

```bash
# 1. 数据预处理
python train/lora_jp/prepare_dataset.py

# 2. 训练
python train/lora_jp/train_lora.py --epochs 50 --batch_size 4 --use_amp

# 3. 导出 ONNX
PYTHONIOENCODING=utf-8 python train/lora_jp/export_onnx.py
```

## 日语音素集

在原有 2820 个音素基础上追加 34 个日语音素（索引 2820-2853）：

| 音素 | 索引 | 说明 |
|------|------|------|
| jp_pau | 2820 | 静音/停顿 |
| jp_a ~ jp_o | 2821-2825 | 五元音 |
| jp_k ~ jp_cl | 2826-2853 | 辅音及辅音组合 |

## ONNX 兼容性

导出的 ONNX 文件与现有 FP16 推理管线完全兼容：

| 文件 | 大小 | 用途 |
|------|------|------|
| note_text_encoder.onnx | 5.9 MB | 扩展嵌入 (2854×512) |
| note_text_encoder_fp16.onnx | 3.0 MB | FP16 版本 |
| preflow.onnx | 16.2 MB | 微调后 preflow |
| preflow_fp16.onnx | 16.2 MB | FP16 版本 |

**部署方式**：将 ONNX 文件替换到模型目录，同时替换 `phone_set.json` 为 `jp_phone_set.json`。

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
