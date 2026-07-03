# SoulX-Singer 日语音素微调

通过三阶段微调 preflow 模块 + 扩展音素嵌入层，使 SoulX-Singer 支持日语歌声合成。

## 训练结果

| 指标 | 值 |
|------|-----|
| 数据集 | PJS Corpus (100首,~15分钟) + JVS-MuSiC (100说话人×2首,~120分钟) + GTSinger Japanese (可选,IPA 标注) |
| 训练样本数 | 300 (PJS 100 + JVS 200,可追加 GTSinger) |
| 可训练参数 | ~4.25M (preflow + JP embedding + cond_emb) |
| 训练阶段 | 三阶段 (warmup 15ep → embed FT 40ep → joint 80ep) |
| GPU | RTX 5060 8GB (bf16 autocast + gradient checkpointing) |
| 精度 | bfloat16 autocast (无量化，无精度损失) |
| batch_size | 1 (ConcatDataset 合并 PJS+JVS) |

> **云端训练**：`SoulX-Singer/train_jp_lora_cloud.py` 是 c2net 云端一键训练脚本，
> 在本地 `train/lora_jp/` 流水线基础上整合了 PJS+JSUT+JVS+GTSinger 数据准备与
> 三阶段训练，支持 `--gtsinger_only` / `--no_gtsinger` 等开关。详见脚本顶部文档字符串。
>
> **v3.1 修复（Stage 2/3 resume 路径）**：原代码在 `load_checkpoint` 替换
> `nn.Embedding` 之前注册 `zero_base_grad` hook 和构建 optimizer，导致 hook 失效、
> optimizer 引用孤儿张量，JP 嵌入永不更新（jp_std 恒定 0.7984）。修复后顺序为
> `apply_lora_structure → load_checkpoint → setup_trainable_and_hooks → build_optimizer
> → restore_optimizer_state_by_name → verify_optimizer_embed_alignment`，并新增
> embedding 梯度范数日志（`train/embed_grad_norm`）用于验证更新有效性。

## 文件结构

```
train/lora_jp/
├── jp_phone_set.json       # 扩展音素表 (3000 base + 33 JP = 3033)
├── jp_phoneme_mapping.json # JP 音素到 EN 源音素的映射 (配置驱动)
├── prepare_dataset.py      # PJS Corpus → 训练数据 (含 SP/slur 样本)
├── prepare_jvs_dataset.py  # JVS-MuSiC → 训练数据 (ROSVOT + jp_g2p)
├── jp_g2p.py               # 日语 G2P (JS 移植，训练-推理一致)
├── jvs_lyrics.json         # JVS-MuSiC 公有领域歌词 (26 首童谣)
├── dataset.py              # PyTorch Dataset (确定性 wav 匹配, 支持 pjs*/jvs* 前缀)
├── phoneme_mapping.py      # 生成 jp_phoneme_mapping.json
├── init_embeddings.py      # 初始化 JP 嵌入 (只 L2 归一化 JP 行)
├── train_staged.py         # 三阶段训练脚本 (主入口, --extra_dataset_* 合并数据)
├── validate_and_rollback.py # 验证 + 回滚决策 (使用独立 val 集)
├── verify_checkpoint.py    # Checkpoint 质量检查
├── export_onnx.py          # 导出 ONNX (text_encoder + preflow + cond_emb)
├── run_pipeline.py         # 一键训练管线 (9 步, 含 JVS 预处理)
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
| cond_emb.onnx | 微调后条件嵌入投影 (512→1024, FP16) | ✅ |
| note_pitch_encoder.onnx | 音高嵌入 | ❌ (共享 base 模型) |

**部署方式**：将 `note_text_encoder.onnx`、`preflow.onnx` 和 `cond_emb.onnx` 放到
`onnx_models/fp16/JP/` 目录。SXSEditor 会自动检测 JP 模型（三者必须同时存在）
并在日语歌词时切换。

**重要**：`cond_emb.onnx` 必须与 JP `preflow.onnx`/`note_text_encoder.onnx` 一起部署。
训练时 `cond_emb` 被微调以适配日语音素分布，若推理时使用 base 的 `cond_emb` 会导致
严重音素错乱（例如 "わたしの" → "nin to shi so"）。

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

**修复前 v1**（导致音素混淆）：prepare_dataset.py 把一个 MIDI note 内的辅音+元音拆成独立 entry，
每个音素独立 BOW/EOW。训练 token: `[PAD BOW jp_t EOW BOW jp_a EOW ...]`

**修复前 v2**（导致 3+ 音素损坏条目）：把一个 MIDI note 时间范围内**所有**重叠的 lab 音素
合并成一个 entry，长 note 会跨多个音节，产生 `jp_t-a-a`、`jp_sh-y-u-o` 等损坏条目
（2379 条损坏数据）。

**修复后**（按音节切分）：先将 lab 音素流切分为音节单元（CV / V / cl），再为每个 MIDI note
分配其中心时间所在的单个音节，确保每个 note 最多 2 个音素（辅音+元音）。
训练 token: `[PAD BOW jp_t jp_a EOW ...]`

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

## JVS-MuSiC 数据集支持

JVS-MuSiC 提供纯音频（无 .lab/.mid/歌词），需要额外预处理：

### 预处理流程 (`prepare_jvs_dataset.py`)

1. **ROSVOT NoteTranscriber** — 从音频转录音符（音高 + 时序）
2. **RMVPE F0Extractor** — 提取帧级 F0
3. **jp_g2p.py** — 公有领域歌词转音素（JS `_japaneseG2p` 的 Python 移植）
4. **音节-音符对齐** — 按音节数量匹配（1:1 / 合并 / 拆分），输出 `jp_t-a` 合并格式

### 数据集布局

```
datasets/jvs_music_ver1/
├── jvs001/ ... jvs100/
│   ├── song_common/wav/{modified_grouped,modified,raw}.wav  # 共同曲 (かたつむり)
│   └── song_unique/wav/raw.wav                               # 独唱曲 (童谣)
├── singer_info.txt
└── oneness/, similarity/  # 说话人相似度评估 (训练未用)
```

### 歌词来源

`jvs_lyrics.json` 包含 26 首日本公有领域童谣歌词（纯假名，无汉字）。
共同曲为 `かたつむり`，独唱曲从歌词库按 `singer_info.txt` 匹配。

### 合并训练

`train_staged.py` 通过 `--extra_dataset_metadata` / `--extra_dataset_wav_dir` 参数
使用 `ConcatDataset` 合并 PJS + JVS 数据。`run_pipeline.py` 已自动传递这些参数。

### 修复记录

- **F0Extractor 构造**：使用 `model_path=`/`is_half=` 而非 `f0_extractor="rmvpe"`
- **F0Extractor 方法**：调用 `process(wav_path)` 而非 `extract(wav, sr)`
- **ROSVOT word_bd_pred**：`infer_sample` 中 `outputs['word_bd_pred']` 改为 `word_bd[0]`（变量来自 wbd_predictor）

## GTSinger Japanese 数据集支持（云端脚本）

GTSinger 提供带 IPA 音素标注和 MIDI 音高的多技巧演唱数据，可作为 PJS/JVS 的补充数据源。
此支持仅在云端脚本 `SoulX-Singer/train_jp_lora_cloud.py` 中实现（本地 `train/lora_jp/` 流水线不涉及）。

### 目录结构

```
GTSinger/Japanese/{singer}/{technique}/{song}/{group}/{sample_id}.json+.wav
```

- `singer`：如 `JA-Soprano-1`、`JA-Tenor-2`
- `technique`：如 `Breathy`、`Vibrato`、`Glissando`、`Control_Group`、`Falsetto` 等
- `song`：歌曲名（可能为日文/中文，非 ASCII 字符自动转 md5 哈希）
- `group`：如 `Breathy_Group`、`Control_Group`
- `sample_id`：`0000`、`0001`...（每个样本对应一对 .json + .wav）

### JSON 格式

每个 `.json` 是一个 word 数组，每个 word 含：

| 字段 | 说明 |
|------|------|
| `word` | 歌词（含 `<AP>` 气口标记） |
| `ph` | IPA 音素数组（如 `["t","a"]`、`["ɕ","i"]`、`["<AP>"]`） |
| `ph_start` / `ph_end` | 每个音素的起止时间（秒） |
| `note` | MIDI 音高数组（**支持 melisma**：一个 word 可对应多个 note） |
| `note_dur` / `note_start` / `note_end` | 音符时序 |

### IPA → PJS base 音素映射

GTSinger 使用 IPA 标注，需要映射到 PJS base 音素集以复用现有 `jp_*` token 体系：

| IPA | PJS base | 说明 |
|-----|----------|------|
| `a` `i` `u` `e` `o` | a i u e o | 五元音（直接映射） |
| `ɯ` | u | 日语闭后不圆唇元音 → u |
| `ɨ` `ɨ̥` | u | 元音清化版本 → u |
| `oː` | o | 长音 → 去长音标记 |
| `k` `s` `t` `n` `h` `m` `r` `w` `j` `g` `d` `b` `p` | 同名 | 直接映射 |
| `dz` | z | 浊塞擦音 → z |
| `ɡ` | g | 浊软腭塞音 → g |
| `c` | k | 清硬腭塞音 → k |
| `ɲ` | ny | 硬腭鼻音 → ny |
| `ɾ` `ɾʲ` | r | 闪音 → r（`ɾʲ` 拗音形式） |
| `ɕ` | sh | 清龈硬腭擦音 → sh |
| `ts` | ts | 清齿龈塞擦音 → ts |
| `bʲ` | b | 硬腭化双唇塞音 → b |
| `ʔ` | cl | 声门塞音 → cl（促音） |
| `ɴ` | n | 拨音 → n |
| `<AP>` | `<AP>` | 气口 → note_type=1 |

未知 IPA 音素会通过 `_gtsinger_map_phoneme()` 中的安全回退逻辑处理：
剥离 `ː`（长音）、`̥`（清化）、`ʲ`（硬腭化）等变音符号后重新查表；
仍无法识别则回退到 `pau`（静音）。

### Melisma（一字多音）处理

GTSinger 中一个 word 可能对应多个 note（melisma 转音）。
`_gtsinger_word_to_tokens()` 使用**音素中点对齐**策略将 word 内的音素分配到各 note：

1. 计算每个音素的中点时间 `(ph_start + ph_end) / 2`
2. 计算每个 note 的中点时间 `(note_start + note_end) / 2`
3. 把每个音素分配给距其时间中点最近的 note
4. 同一 note 范围内的连续音素合并为一个 `jp_辅音-元音` token（如 `jp_t-a`）

这保证了 melisma 音节与 PJS lab 数据的 `jp_*` 合并格式一致，复用现有训练-推理对齐逻辑。

### 用法

```bash
# 仅准备 GTSinger 数据
python train_jp_lora_cloud.py --prepare_only --gtsinger_only

# 全流水线（含 GTSinger）
python train_jp_lora_cloud.py

# 排除 GTSinger 数据
python train_jp_lora_cloud.py --no_gtsinger
```

数据集挂载路径（c2net）：
- `{dataset_path}/GTSinger/Japanese/...`（推荐）
- `{dataset_path}/Japanese/...`（兼容布局）

### 输出格式

每个 GTSinger 样本生成：
- `wavs/gts_{singer}_{technique}_{md5(song)}_{sample_id}.wav` — 重采样到 24kHz 的音频
- `metadata.json` 中追加条目，字段与 PJS/JVS 一致：
  - `id`、`phoneme`（`jp_*` token 数组）、`duration`、`note_pitch`、`note_type`、`f0`

## 依赖

- PyTorch >= 2.0
- torchaudio
- omegaconf
- tensorboard
- mido (MIDI解析)
- praat-parselmouth (F0提取)
- soundfile
- onnxruntime (ONNX 导出验证)
- librosa (JVS 音频加载)
- pyworld (ROSVOT 依赖，Python 3.14 需 stub)
