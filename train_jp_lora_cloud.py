"""
单文件云端日语微调训练脚本（v3 架构，云端 A100 / Ascend 910B 一键执行）。
这个脚本是本地编写的，需要你针对云端环境二次修改适配！！
将 SoulX-Singer/train/lora_jp_v3 下的多文件流水线（jp_g2p.py / prepare_dataset.py /
lora.py / init_embeddings.py / dataset.py / train_lora.py / validate.py / export_onnx.py）
整合为单个自包含脚本，便于在云端 Jupyter / SSH 环境一次性执行。

支持设备：
  - NVIDIA CUDA（A100 40G）
  - 华为 NPU（Ascend 910B 64G，需 torch_npu）
  - CPU（仅用于 dry_run / verify_protection）

执行模式：
  python train_jp_lora_cloud.py --dry_run             # 部署前验证（不需要真实模型/数据）
  python train_jp_lora_cloud.py --prepare_only        # 仅准备数据集（PJS+JSUT+JVS+GTSinger）
  python train_jp_lora_cloud.py --train_only         # 仅训练（数据已准备）
  python train_jp_lora_cloud.py --export_only        # 仅导出 ONNX
  python train_jp_lora_cloud.py --verify_protection  # 验证源语言权重未变化
  python train_jp_lora_cloud.py                       # 全流水线（prepare→train→export）
  python train_jp_lora_cloud.py --prepare_only --gtsinger_only  # 仅准备 GTSinger 数据
  python train_jp_lora_cloud.py --no_gtsinger        # 排除 GTSinger 数据

训练目标（按组件）：
  - note_text_encoder (embedding): 全量训练
      仅 JP phoneme 行（3000-3032）更新，base 行梯度置零保护源语言。
  - preflow (encoder): LoRA 深度微调（rank=32, alpha=64）
      4 层 ConvNeXtV2Block 的 pwconv1/pwconv2（8 个适配器）。
  - cond_emb (跨模态对齐层): 全量微调
      cfm_decoder.model.cond_emb Linear（EMBED_DIM→COND_DIM）直接更新权重。
  - diff_estimator (diffstep / DiffLlama): 冻结，不微调
      22 层 attention 完全冻结，仅参与前向计算（梯度不更新其参数）。

源语言保护策略：
  - embedding 的 base 行（0-2999）通过 zero_base_grad 钩子置零梯度，源语言音素不变。
  - diff_estimator 完全冻结，粤语/中文/英文的 diffusion 路径不变。
  - preflow 通过 LoRA 合并回基础权重，源语言相关通道由低秩增量主导（可控）。
  - cond_emb 为全量微调（共享层），训练后权重直接变化；如需严格源语言保护，
    可在导出时用基础模型 cond_emb 替换（但会损失日语对齐增益）。
  - vocoder / f0_encoder / note_pitch_encoder / note_type_encoder 始终冻结。
  verify_source_language_protection() 验证非预期权重是否保持不变。

三阶段训练：
  Stage 1（JP 嵌入 + cond_emb 预热，10 ep）：preflow LoRA 冻结（B=0 no-op）
  Stage 2（全量联合训练，50 ep）：JP embed + preflow LoRA + cond_emb 可训练
  Stage 3（prompt 条件化微调，20 ep）：100% prompt 切分训练

数据集：
  - PJS Corpus（日语歌唱，主要数据源，100 样本已预处理）
https://www.modelscope.cn/datasets/aihobbyist/StarRail_Dataset/resolve/master/StarRail4.2_JP.7z  #Genshin Impact JP Voice,modelscope highspeed download.
  - GTSinger Japanese（日语歌唱，IPA 音素标注，含 Breathy/Glissando/Vibrato 技巧）
    目录结构: GTSinger/Japanese/{singer}/{technique}/{song}/{group}/{sample_id}.json+.wav
    每个 JSON 含 word 数组，每个 word 含 ph(音素)/note(MIDI)/note_dur/时序信息。
    支持 melisma（一个 word 对应多个 note），IPA 音素自动映射到 PJS base 音素集。
"""

# LoRA 超参数（A100 40G 最佳实践）
# 仅 preflow (encoder) 采用 LoRA 深度微调；cond_emb 全量微调；diff_estimator 不微调
LORA_RANK = 32           # rank=32 提供更高容量用于 preflow 深度微调
LORA_ALPHA = 64          # alpha = 2 * rank
BATCH_SIZE = 8           # A100 40G（无 diff_estimator LoRA，显存充裕）
GRADIENT_ACCUMULATION = 2  # 有效 batch = 16
GRAD_CHECKPOINT = True
VAL_RATIO = 0.1
SEED = 42

# 三阶段训练配置
# 训练目标说明：
#   - note_text_encoder (embedding): 全量训练（仅 JP 行，base 行梯度置零保护源语言）
#   - preflow (encoder): LoRA 深度微调（rank=32）
#   - cond_emb (跨模态对齐层): 全量微调
#   - diff_estimator (diffstep): 冻结，不微调
STAGE_CONFIGS = {
    1: {
        'epochs': 10,
        'lora_lr': 0.0,        # preflow LoRA 冻结
        'embed_lr': 1e-3,      # JP embedding 预热
        'cond_emb_lr': 1e-4,   # cond_emb 全量微调（预热阶段）
        'train_lora': False,
        'train_cond_emb': True,
        'warmup_steps': 0,
        'use_prompt_split': False,
        'prompt_split_prob': 0.0,
        'prompt_split_start_ep': 999,
    },
    2: {
        'epochs': 50,
        'lora_lr': 1e-4,       # preflow LoRA 标准学习率
        'embed_lr': 3e-4,
        'cond_emb_lr': 5e-5,   # cond_emb 全量微调（联合训练，低于 LoRA）
        'train_lora': True,
        'train_cond_emb': True,
        'warmup_steps': 200,
        'use_prompt_split': True,
        'prompt_split_prob': 0.5,
        'prompt_split_start_ep': 21,
    },
    3: {
        'epochs': 20,
        'lora_lr': 3e-5,       # preflow LoRA 精调
        'embed_lr': 1e-4,
        'cond_emb_lr': 2e-5,   # cond_emb 全量微调（精调阶段）
        'train_lora': True,
        'train_cond_emb': True,
        'warmup_steps': 100,
        'use_prompt_split': True,
        'prompt_split_prob': 1.0,
        'prompt_split_start_ep': 1,
    },
}

# ===========================================================================
# 常量（与 v3 流水线一致，不要修改）
# ===========================================================================
JP_PHONEME_START = 3000
JP_PHONEME_COUNT = 33
EMBED_DIM = 512
COND_DIM = 1024
MEL_DIM = 128
SAMPLE_RATE = 24000
HOP_SIZE = 480
NOISE_SCALE = 0.3  # v3: cos ~ 0.96，保留语音学相似性
NUM_PREFLOW_BLOCKS = 4

# ===========================================================================
# 导入（标准库 + 可选依赖）
# ===========================================================================
import os
import sys
import math
import json
import time
import argparse
import hashlib
import shutil
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# c2net 云平台接口（必须在所有其他逻辑之前初始化）
from c2net.context import prepare, upload_output
c2net_context = prepare()

# 持久化 checkpoint 目录（在 _output_path 之外，避免 upload_output 清理）
_PERSISTENT_CKPT_DIR = "/tmp/_stage_checkpoints"

# ===========================================================================
# 云端配置变量（根据 c2net 环境自动设置）
# 数据集 / 模型 / 预处理文件分别从 c2net 挂载路径获取
# ===========================================================================

# 基础路径
_dataset_path = c2net_context.dataset_path
_pretrain_path = c2net_context.pretrain_model_path
_preprocess_path = _pretrain_path + "/SoulX-Singer-Preprocess"
_output_path = c2net_context.output_path

# 将 SoulX-Singer 代码目录加入 Python 路径（soulxsinger 包位于此）
_code_dir = os.path.dirname(os.path.abspath(__file__))
_soulx_path = os.path.join(_code_dir, "SoulX-Singer")
if os.path.isdir(_soulx_path):
   sys.path.insert(0, _soulx_path)

DEVICE = 'auto'  # 'cuda' / 'npu' / 'auto'

# 基础模型与配置
BASE_MODEL_PATH = _pretrain_path + "/SoulX-Singer/model.pt"
CONFIG_PATH = _pretrain_path + "/SoulX-Singer/config.yaml"

# ---------- 数据准备（解压 + 生成配置文件） ----------
# c2net 只下载原始 zip，需要自行解压
_PJS_RAW_DIR = _dataset_path + "/PJS"
_PJS_EXTRACT_DIR = _output_path + "/_pjs_extracted"
_PJS_CORPUS_EXTRACT = _PJS_EXTRACT_DIR + "/PJS_corpus_ver1.1"
_PJS_LAB_EXTRACT = _PJS_EXTRACT_DIR + "/lab"

# 预处理的音素／映射文件（若不存在则自动生成）
_LOCAL_PREPROCESS_DIR = _output_path + "/_preprocess"

def _ensure_pjs_extracted():
    """解压 PJS 数据集（zip → 目录）。"""
    if os.path.isdir(_PJS_CORPUS_EXTRACT) and os.path.isdir(_PJS_LAB_EXTRACT):
        return  # 已解压
    print('\n[数据准备] 解压 PJS 数据集...')
    os.makedirs(_PJS_EXTRACT_DIR, exist_ok=True)
    # 主语料
    corpus_zip = _PJS_RAW_DIR + "/PJS_corpus_ver1.1.zip"
    if os.path.exists(corpus_zip):
        import zipfile
        with zipfile.ZipFile(corpus_zip, 'r') as zf:
            zf.extractall(_PJS_EXTRACT_DIR)
        print(f'  解压完成: {corpus_zip} → {_PJS_CORPUS_EXTRACT}')
    # 修正 lab
    lab_zip = _PJS_RAW_DIR + "/pjs-manual-labels-main.zip"
    if os.path.exists(lab_zip):
        import zipfile
        _lab_tmp = _PJS_EXTRACT_DIR + "/_lab_tmp"
        os.makedirs(_lab_tmp, exist_ok=True)
        with zipfile.ZipFile(lab_zip, 'r') as zf:
            zf.extractall(_lab_tmp)
        # 拷贝修正后的 lab 到目标 lab 目录
        src_lab = _lab_tmp + "/pjs-manual-labels-main/lab"
        if os.path.isdir(src_lab):
            os.makedirs(_PJS_LAB_EXTRACT, exist_ok=True)
            for fname in os.listdir(src_lab):
                if fname.endswith('.lab'):
                    shutil.copy2(os.path.join(src_lab, fname),
                                  os.path.join(_PJS_LAB_EXTRACT, fname))
            print(f'  修正 lab 解压完成: {len(os.listdir(_PJS_LAB_EXTRACT))} files')

def _ensure_preprocess_files():
    """若 phone_set / mapping 不存在则从 lab 数据自动生成。"""
    if os.path.exists(_LOCAL_PREPROCESS_DIR + "/jp_phone_set.json") and \
       os.path.exists(_LOCAL_PREPROCESS_DIR + "/jp_phoneme_mapping.json"):
        return
    print('\n[数据准备] 生成音素集和映射文件...')
    os.makedirs(_LOCAL_PREPROCESS_DIR, exist_ok=True)

    # 提取 lab 中的所有基础音素
    base_phones = set()
    lab_dir = _PJS_LAB_EXTRACT if os.path.isdir(_PJS_LAB_EXTRACT) else _PJS_CORPUS_EXTRACT
    lab_files = []
    if os.path.isdir(lab_dir):
        lab_files = [os.path.join(lab_dir, f) for f in os.listdir(lab_dir) if f.endswith('.lab')]
    if not lab_files and os.path.isdir(_PJS_CORPUS_EXTRACT):
        # 回退到原版 lab
        for d in sorted(os.listdir(_PJS_CORPUS_EXTRACT)):
            dd = os.path.join(_PJS_CORPUS_EXTRACT, d)
            if os.path.isdir(dd):
                for f in os.listdir(dd):
                    if f.endswith('.lab'):
                        lab_files.append(os.path.join(dd, f))
    for lf in lab_files:
        try:
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        base_phones.add(parts[2])
        except Exception:
            pass

    base_phones = sorted(base_phones)
    if not base_phones:
        # 兜底：使用已知 Japanese 音素作为 base
        print('  [WARNING] 无法从 lab 提取音素，使用默认日语音素集')
        jp_vowels_def = {'a', 'i', 'u', 'e', 'o'}
        jp_cons_def = {'k','s','t','n','h','m','r','w','y','g','z','d','b','p',
                       'f','j','ch','sh','ts','ky','gy','ny','hy','my','ry','py','by','cl'}
        base_phones = sorted(['pau'] + list(jp_vowels_def | jp_cons_def))

    # JP phoneme list (33 个)
    jp_vowels = {'a', 'i', 'u', 'e', 'o'}
    jp_consonants = {'k','s','t','n','h','m','r','w','y','g','z','d','b','p',
                     'f','j','ch','sh','ts','ky','gy','ny','hy','my','ry','py','by','cl'}
    jp_phones = sorted(jp_vowels | jp_consonants)

    # 特殊 token（须与原版 SoulX-Singer phone_set.json 一致，
    # DataProcessor.preprocess() 会插入 <PAD>/<BOW>/<EOW>/<SEP> 等 token）
    _special_tokens = [
        '<PAD>', '<SP>', '<AP>', '<UNK>', '<BOW>', '<EOW>',
        '<BOS>', '<EOS>', '<MASK>', '<SEP>',
        '<SPECIAL_TOKEN_1>', '<SPECIAL_TOKEN_2>', '<SPECIAL_TOKEN_3>',
        '<SPECIAL_TOKEN_4>', '<SPECIAL_TOKEN_5>', '<SPECIAL_TOKEN_6>',
        '<SPECIAL_TOKEN_7>', '<SPECIAL_TOKEN_8>', '<SPECIAL_TOKEN_9>',
        '<SPECIAL_TOKEN_10>',
    ]

    # 生成所有可能的 JP 组合 token（lab_phonemes_to_notes 会生成 jp_辅音-元音 形式）
    jp_cons_list = ['k','s','t','n','h','m','r','w','y','g','z','d','b','p',
                    'f','j','ch','sh','ts','ky','gy','ny','hy','my','ry','py','by','cl']
    jp_vowel_list = ['a', 'i', 'u', 'e', 'o']
    jp_combined = []
    # 辅音+元音组合: jp_k-a, jp_k-i, ..., jp_by-a, ...
    for c in jp_cons_list:
        for v in jp_vowel_list:
            jp_combined.append(f'jp_{c}-{v}')
    # 单独元音: jp_a, jp_i, jp_u, jp_e, jp_o
    for v in jp_vowel_list:
        jp_combined.append(f'jp_{v}')
    # 单独辅音: jp_k, jp_s, ..., jp_cl (lab 数据中可能出现无声调独立辅音)
    for c in jp_cons_list:
        jp_combined.append(f'jp_{c}')
    # 特殊 JP 音素: jp_n (拨音, 如果上面没被 cons_list 包含)
    if 'n' not in jp_cons_list or 'jp_n' not in jp_combined:
        jp_combined.append('jp_n')

    # phone_set.json: 特殊 token + base_phones + jp 组合 token
    phone_set = _special_tokens + base_phones + jp_combined

    # jp_phoneme_mapping.json: jp_ 音素 → base 音素
    mapping = {}
    for jp in jp_phones:
        if jp in base_phones:
            mapping['jp_' + jp] = {
                'sources': [{'phone': jp, 'weight': 1.0}],
                'init_weight': 1.0
            }
        else:
            mapping['jp_' + jp] = {
                'sources': [{'phone': 'pau' if 'pau' in base_phones else jp, 'weight': 1.0}],
                'init_weight': 0.5,
                'strategy': 'fallback'
            }

    with open(_LOCAL_PREPROCESS_DIR + "/jp_phone_set.json", 'w') as f:
        json.dump(phone_set, f, ensure_ascii=False, indent=2)
    with open(_LOCAL_PREPROCESS_DIR + "/jp_phoneme_mapping.json", 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'  生成完成: {len(phone_set)} 音素, {len(mapping)} 映射条目')

# 执行数据准备
_ensure_pjs_extracted()
_ensure_preprocess_files()

# 音素集与映射（自动生成后的路径）
PHONE_SET_PATH = _LOCAL_PREPROCESS_DIR + "/jp_phone_set.json"
JP_PHONEME_MAPPING_PATH = _LOCAL_PREPROCESS_DIR + "/jp_phoneme_mapping.json"

# 数据集路径（解压后的实际目录）
PJS_CORPUS_DIR = _PJS_CORPUS_EXTRACT if os.path.isdir(_PJS_CORPUS_EXTRACT) else _dataset_path + "/PJS/PJS_corpus_ver1.1"
PJS_LAB_DIR = _PJS_LAB_EXTRACT if os.path.isdir(_PJS_LAB_EXTRACT) else _dataset_path + "/PJS/lab"
JSUT_DIR = _dataset_path + "/JSUT" if os.path.isdir(_dataset_path + "/JSUT") else ""
JVS_PREPARED_DIR = _preprocess_path + "/dataset"  # JVS 预处理数据打包在预处理文件中（若不存在则跳过）

# GTSinger 数据集路径（c2net 挂载路径下）
# 支持两种目录结构：GTSinger/Japanese/... 或直接 Japanese/...
GTSINGER_DIR = ""
GTSINGER_JA_DIR = ""
for _gts_root in [_dataset_path + "/GTSinger", _dataset_path]:
    _gts_ja = os.path.join(_gts_root, "Japanese")
    if os.path.isdir(_gts_ja):
        GTSINGER_DIR = _gts_root
        GTSINGER_JA_DIR = _gts_ja
        break

# 输出路径（所有输出必须保存在 c2net_context.output_path 下）
PREPARED_DATA_DIR = _output_path + "/dataset"
CHECKPOINT_DIR = _output_path + "/checkpoints"
ONNX_OUTPUT_DIR = _output_path + "/onnx_jp"
TENSORBOARD_DIR = _output_path + "/tensorboard"
OUTPUT_DIR = _output_path

# Windows 控制台 Unicode 修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 可选依赖（延迟导入）
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# 音频/MIDI 处理（可选）
try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import mido
except ImportError:
    mido = None
try:
    import parselmouth
except ImportError:
    parselmouth = None
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


# ===========================================================================
# 设备检测（CUDA / NPU / CPU）
# ===========================================================================
def detect_device():
    """自动检测可用设备：CUDA > NPU > CPU。"""
    global _TORCH_AVAILABLE
    if not _TORCH_AVAILABLE:
        return 'cpu'

    # 优先 CUDA
    if torch.cuda.is_available():
        return 'cuda'

    # 其次 NPU（华为 Ascend）
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    return 'cpu'


def move_to_device(obj, device):
    """将 tensor / module 移到指定设备（兼容 NPU）。"""
    if device == 'npu':
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            raise RuntimeError("NPU 设备需要安装 torch_npu")
    return obj.to(device)


def amp_context(device, enabled=True, dtype=None):
    """autocast 上下文（兼容 CUDA/NPU/CPU）。"""
    if not enabled:
        class _NullCtx:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _NullCtx()
    if dtype is None:
        dtype = torch.bfloat16
    dev_type = 'cuda' if device.startswith('cuda') else ('npu' if device == 'npu' else 'cpu')
    return torch.amp.autocast(dev_type, dtype=dtype)


def empty_cache(device):
    """清空设备缓存。"""
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == 'npu':
        try:
            import torch_npu  # noqa: F401
            torch.npu.empty_cache()
        except Exception:
            pass


# ===========================================================================
# JP G2P（来自 jp_g2p.py，与 JS _japaneseG2p 字节一致）
# ===========================================================================
import re

JP_HIRAGANA_MAP = {
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    'か': 'k a', 'き': 'k i', 'く': 'k u', 'け': 'k e', 'こ': 'k o',
    'さ': 's a', 'し': 'sh i', 'す': 's u', 'せ': 's e', 'そ': 's o',
    'た': 't a', 'ち': 'ch i', 'つ': 'ts u', 'て': 't e', 'と': 't o',
    'な': 'n a', 'に': 'ny i', 'ぬ': 'n u', 'ね': 'n e', 'の': 'n o',
    'は': 'h a', 'ひ': 'hy i', 'ふ': 'f u', 'へ': 'h e', 'ほ': 'h o',
    'ま': 'm a', 'み': 'my i', 'む': 'm u', 'め': 'm e', 'も': 'm o',
    'や': 'y a', 'ゆ': 'y u', 'よ': 'y o',
    'ら': 'r a', 'り': 'ry i', 'る': 'r u', 'れ': 'r e', 'ろ': 'r o',
    'わ': 'w a', 'を': 'o', 'ん': 'n',
    'が': 'g a', 'ぎ': 'gy i', 'ぐ': 'g u', 'げ': 'g e', 'ご': 'g o',
    'ざ': 'z a', 'じ': 'j i', 'ず': 'z u', 'ぜ': 'z e', 'ぞ': 'z o',
    'だ': 'd a', 'ぢ': 'j i', 'づ': 'z u', 'で': 'd e', 'ど': 'd o',
    'ば': 'b a', 'び': 'by i', 'ぶ': 'b u', 'べ': 'b e', 'ぼ': 'b o',
    'ぱ': 'p a', 'ぴ': 'py i', 'ぷ': 'p u', 'ぺ': 'p e', 'ぽ': 'p o',
    'きゃ': 'ky a', 'きゅ': 'ky u', 'きょ': 'ky o',
    'しゃ': 'sh a', 'しゅ': 'sh u', 'しょ': 'sh o',
    'ちゃ': 'ch a', 'ちゅ': 'ch u', 'ちょ': 'ch o',
    'にゃ': 'ny a', 'にゅ': 'ny u', 'にょ': 'ny o',
    'ひゃ': 'hy a', 'ひゅ': 'hy u', 'ひょ': 'hy o',
    'みゃ': 'my a', 'みゅ': 'my u', 'みょ': 'my o',
    'りゃ': 'ry a', 'りゅ': 'ry u', 'りょ': 'ry o',
    'ぎゃ': 'gy a', 'ぎゅ': 'gy u', 'ぎょ': 'gy o',
    'じゃ': 'j a', 'じゅ': 'j u', 'じょ': 'j o',
    'びゃ': 'by a', 'びゅ': 'by u', 'びょ': 'by o',
    'ぴゃ': 'py a', 'ぴゅ': 'py u', 'ぴょ': 'py o',
    'てゃ': 't a', 'てゅ': 't u', 'てょ': 't o',
    'でゃ': 'd a', 'でゅ': 'd u', 'でょ': 'd o',
    'っ': 'cl',
}

# Katakana map: shift hiragana char code by +0x60
JP_KATAKANA_MAP = {}
for _hira, _ph in JP_HIRAGANA_MAP.items():
    _kata = chr(ord(_hira[0]) + 0x60) + _hira[1:]
    JP_KATAKANA_MAP[_kata] = _ph

JP_KANJI_DICT = {
    '愛': 'a i', '雨': 'a m e', '空': 's o r a', '花': 'h a n a',
    '風': 'k a z e', '月': 'ts u k i', '星': 'h o sh i', '雪': 'y u k i',
    '海': 'u m i', '山': 'y a m a', '川': 'k a w a', '森': 'm o r i',
    '光': 'h i k a r i', '音': 'o t o', '声': 'k o e', '梦': 'y u m e',
    '心': 'k o k o r o', '恋': 'k o i', '涙': 'n a m i d a',
    '歌': 'u t a', '飛': 't o b u', '歩': 'a r u k u',
    '走': 'h a sh i r u', '泳': 'o y o g u', '読': 'y o m u',
    '食': 't a b e r u', '飲': 'n o m u', '見': 'm i r u', '聞': 'k i k u',
    '帰': 'k a e r u', '行': 'i k u', '来': 'k u r u', '立': 't a ts u',
    '入': 'h a i r u', '出': 'd e r u', '上': 'u e', '下': 's h i t a',
    '大': 'o o', '小': 'ch i i s a', '長': 'n a g a i', '強': 'ts u y o i',
    '春': 'h a r u', '夏': 'n a ts u', '秋': 'a k i', '冬': 'f u y u',
    '朝': 'a s a', '昼': 'h i r u', '夜': 'y o r u',
    '今': 'i m a', '私': 'w a t a sh i', '君': 'k i m i',
    '一': 'i ch i', '二': 'n i', '三': 's a n', '四': 'y o n',
    '五': 'g o', '六': 'r o k u', '七': 'n a n a', '八': 'h a ch i',
    '九': 'ky u', '十': 'j u',
}

_JP_KANA_RE = re.compile(r'[ぁ-ゟァ-ヿ]')
_JP_KANJI_RE = re.compile(r'[一-鿿]')
_LATIN_RE = re.compile(r'[a-zA-Z]')

JP_CONSONANTS = {
    'k', 's', 't', 'n', 'h', 'm', 'r', 'w', 'y', 'g', 'z', 'd', 'b', 'p',
    'f', 'j', 'ch', 'sh', 'ts', 'ky', 'gy', 'ny', 'hy', 'my', 'ry', 'py', 'by', 'cl',
}
JP_VOWELS = {'a', 'i', 'u', 'e', 'o'}


def japanese_g2p(text: str) -> List[str]:
    """日语 → 音素列表（与 JS _japaneseG2p 字节一致）。"""
    result: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == 'ー' or ch == '〜':
            i += 1
            continue
        if i + 1 < n:
            combo = ch + text[i + 1]
            ph = JP_HIRAGANA_MAP.get(combo) or JP_KATAKANA_MAP.get(combo)
            if ph is not None:
                result.extend(ph.split(' '))
                i += 2
                continue
        ph = JP_HIRAGANA_MAP.get(ch) or JP_KATAKANA_MAP.get(ch)
        if ph is not None:
            result.extend(ph.split(' '))
            i += 1
            continue
        if _JP_KANJI_RE.match(ch):
            found = False
            max_len = min(4, n - i)
            for length in range(max_len, 1, -1):
                compound = text[i:i + length]
                if compound in JP_KANJI_DICT:
                    result.extend(JP_KANJI_DICT[compound].split(' '))
                    i += length
                    found = True
                    break
            if not found:
                kanji_ph = JP_KANJI_DICT.get(ch)
                if kanji_ph is not None:
                    result.extend(kanji_ph.split(' '))
                else:
                    result.append('pau')
                i += 1
            continue
        if _LATIN_RE.match(ch):
            result.append(ch.lower())
            i += 1
            continue
        i += 1
    return result


def syllabify_jp(phonemes: List[str]) -> List[List[str]]:
    """将平铺音素列表分组为日语音节单元。"""
    syllables: List[List[str]] = []
    i = 0
    n = len(phonemes)
    while i < n:
        ph = phonemes[i]
        if ph == 'cl':
            syllables.append([ph])
            i += 1
            continue
        if ph == 'n':
            if i + 1 < n and phonemes[i + 1] in JP_VOWELS:
                pass
            else:
                syllables.append([ph])
                i += 1
                continue
        if ph == 'pau':
            syllables.append([ph])
            i += 1
            continue
        if ph in JP_VOWELS:
            syllables.append([ph])
            i += 1
            continue
        if ph in JP_CONSONANTS:
            consonant = ph
            if i + 1 < n and phonemes[i + 1] in JP_VOWELS:
                vowel = phonemes[i + 1]
                syllables.append([consonant, vowel])
                i += 2
            else:
                syllables.append([consonant])
                i += 1
            continue
        syllables.append([ph])
        i += 1
    return syllables


def lyrics_to_jp_tokens(lyrics: str) -> List[str]:
    """完整管线：歌词 → jp_* 合并 token（如 'jp_t-a'）。"""
    phonemes = japanese_g2p(lyrics)
    syllables = syllabify_jp(phonemes)
    tokens = []
    for syl in syllables:
        if syl == ['pau']:
            tokens.append('<SP>')
            continue
        if syl == ['cl']:
            tokens.append('jp_cl')
            continue
        parts = ['jp_' + p for p in syl]
        merged = 'jp_' + '-'.join(p[3:] for p in parts)
        tokens.append(merged)
    return tokens


# ===========================================================================
# PJS 数据集处理（来自 prepare_dataset.py）
# ===========================================================================
PJS_PHONEME_MAP = {
    'pau': '<SP>', 'xx': '<SP>',
    'a': 'jp_a', 'i': 'jp_i', 'u': 'jp_u', 'e': 'jp_e', 'o': 'jp_o',
    'k': 'jp_k', 's': 'jp_s', 't': 'jp_t', 'n': 'jp_n', 'h': 'jp_h',
    'm': 'jp_m', 'r': 'jp_r', 'w': 'jp_w', 'y': 'jp_y',
    'g': 'jp_g', 'z': 'jp_z', 'd': 'jp_d', 'b': 'jp_b', 'p': 'jp_p',
    'f': 'jp_f', 'j': 'jp_j', 'ch': 'jp_ch', 'sh': 'jp_sh', 'ts': 'jp_ts',
    'ky': 'jp_ky', 'gy': 'jp_gy', 'ny': 'jp_ny', 'hy': 'jp_hy',
    'my': 'jp_my', 'ry': 'jp_ry', 'py': 'jp_py', 'by': 'jp_by',
    'cl': 'jp_cl',
    'I': 'jp_a', 'N': 'jp_n', 'O': 'jp_o', 'U': 'jp_u',
}

PJS_CONSONANTS = {
    'k', 's', 't', 'n', 'h', 'm', 'r', 'w', 'y', 'g', 'z', 'd', 'b', 'p',
    'f', 'j', 'ch', 'sh', 'ts', 'ky', 'gy', 'ny', 'hy', 'my', 'ry', 'py', 'by', 'cl',
}
PJS_VOWELS = {'a', 'i', 'u', 'e', 'o', 'I', 'N', 'O', 'U'}


# ===========================================================================
# GTSinger 数据集处理（IPA 音素 → PJS base 音素映射）
# ===========================================================================
# GTSinger 使用 IPA 标注，需映射到 PJS base 音素集以复用 jp_* token 体系
GTSINGER_PHONEME_MAP = {
    # 元音
    'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
    'ɯ': 'u',      # 日语不圆唇后元音 → u
    'ɨ': 'u',      # 中央元音（す/つ）→ u
    'ɨ̥': 'u',     # 清化版本 → u
    'oː': 'o',     # 长元音去长度标记
    'aː': 'a', 'iː': 'i', 'uː': 'u', 'eː': 'e', 'ɯː': 'u',
    # 塞音
    'k': 'k', 'ɡ': 'g', 'g': 'g',
    't': 't', 'd': 'd',
    'p': 'p', 'b': 'b',
    'c': 'k',     # [c] 是 /k/ 在高元音前的硬腭变体 → k
    # 擦音
    's': 's', 'z': 'z', 'dz': 'z',
    'h': 'h', 'f': 'f', 'ɸ': 'f',   # ɸ 双唇擦音 → f
    'ɕ': 'sh',    # 齿龈硬腭擦音 → sh
    'sh': 'sh',   # 安全回退（部分文件可能用 sh 而非 ɕ）
    'ç': 'hy',    # 硬腭擦音 → hy（ひ）
    # 塞擦音
    'ts': 'ts', 'ch': 'ch',
    # 鼻音
    'n': 'n', 'ɲ': 'ny',   # 硬腭鼻音 → ny（に）
    'm': 'm',
    'ɴ': 'n',     # 拨音（音节末鼻音）→ n
    # 近音/闪音
    'j': 'j', 'w': 'w', 'y': 'y',
    'ɾ': 'r',     # 齿龈闪音 → r
    'ɾʲ': 'ry',   # 硬腭化闪音 → ry（り）
    'r': 'r',
    # 硬腭化辅音
    'bʲ': 'by',
    'kʲ': 'ky', 'gʲ': 'gy', 'pʲ': 'py', 'mʲ': 'my',
    'hʲ': 'hy', 'sʲ': 'sy',
    # 促音/喉塞音
    'ʔ': 'cl', 'cl': 'cl', 'Q': 'cl',
    # 停顿
    '<AP>': '<SP>',
}


def split_lab_into_syllables(lab_segments: List[Dict]) -> List[Dict]:
    """将 lab 音素按音节分组（CV 或 V 或 cl 或停顿）。"""
    syllables = []
    i = 0
    n = len(lab_segments)
    while i < n:
        seg = lab_segments[i]
        ph = seg['phoneme']
        if ph in ('pau', 'xx'):
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': True})
            i += 1
            continue
        if ph == 'cl':
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': False})
            i += 1
            continue
        if ph in PJS_VOWELS:
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': False})
            i += 1
            continue
        cons_start = seg['start']
        j = i + 1
        if j < n and lab_segments[j]['phoneme'] in PJS_VOWELS:
            vow_seg = lab_segments[j]
            syllables.append({'start': cons_start, 'end': vow_seg['end'],
                              'phonemes': [ph, vow_seg['phoneme']], 'is_pause': False})
            i = j + 1
            continue
        syllables.append({'start': seg['start'], 'end': seg['end'],
                          'phonemes': [ph], 'is_pause': False})
        i += 1
    return syllables


def parse_lab_file(lab_path: str) -> List[Dict]:
    """解析 .lab 文件（HTK 100ns 时间戳）。"""
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            segments.append({
                'start': int(parts[0]) / 1e7,
                'end': int(parts[1]) / 1e7,
                'phoneme': parts[2],
            })
    return segments


def parse_midi_file(midi_path: str) -> List[Dict]:
    """解析 MIDI 文件，提取音符事件。"""
    if mido is None:
        raise ImportError("mido is required. Install: pip install mido")
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    events = []
    for track in mid.tracks:
        absolute_time = 0
        for msg in track:
            absolute_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({'type': 'on', 'time': absolute_time,
                               'note': msg.note, 'velocity': msg.velocity})
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({'type': 'off', 'time': absolute_time, 'note': msg.note})
    notes = []
    active_notes = {}
    for event in sorted(events, key=lambda x: x['time']):
        if event['type'] == 'on':
            active_notes[event['note']] = event
        elif event['type'] == 'off' and event['note'] in active_notes:
            on_event = active_notes.pop(event['note'])
            start_sec = mido.tick2second(on_event['time'], ticks_per_beat, tempo)
            end_sec = mido.tick2second(event['time'], ticks_per_beat, tempo)
            if end_sec > start_sec:
                notes.append({'start': start_sec, 'end': end_sec,
                              'pitch': event['note'], 'velocity': on_event['velocity']})
    return sorted(notes, key=lambda x: x['start'])


def extract_f0_parselmouth(wav_path: str, sr: int = 24000) -> np.ndarray:
    """使用 parselmouth (Praat) 提取 F0。"""
    if parselmouth is None:
        raise ImportError("parselmouth is required. Install: pip install praat-parselmouth")
    if sf is None:
        raise ImportError("soundfile is required")
    y, orig_sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    if orig_sr != sr:
        import scipy.signal
        y = scipy.signal.resample(y, int(len(y) * sr / orig_sr))
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=480 / sr, pitch_floor=32.7031956625, pitch_ceiling=1000.0)
    return pitch.selected_array['frequency'].astype(np.float32)


def lab_phonemes_to_notes(lab_segments, midi_notes, sample_rate=24000, hop_size=480):
    """将 lab 音素按音节对齐到 MIDI 音符。"""
    phonemes = []
    durations = []
    note_pitches = []
    note_types = []
    min_dur = 6 * hop_size / sample_rate
    max_dur = 2.0
    syllables = split_lab_into_syllables(lab_segments)
    consumed = [False] * len(syllables)
    prev_note_end = None
    for note in midi_notes:
        note_start = note['start']
        note_end = note['end']
        note_pitch = note['pitch']
        note_duration = max(min_dur, min(note_end - note_start, max_dur))
        note_center = (note_start + note_end) / 2
        if prev_note_end is not None and note_start > prev_note_end + 0.05:
            gap_dur = note_start - prev_note_end
            phonemes.append('<SP>')
            durations.append(max(gap_dur, min_dur))
            note_pitches.append(0)
            note_types.append(1)
        prev_note_end = note_end
        best_idx = -1
        best_dist = float('inf')
        for si, syl in enumerate(syllables):
            if syl['end'] <= note_start or syl['start'] >= note_end:
                continue
            syl_center = (syl['start'] + syl['end']) / 2
            dist = abs(note_center - syl_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = si
        if best_idx == -1:
            phonemes.append('jp_a')
            durations.append(max(note_duration, min_dur))
            note_pitches.append(note_pitch)
            note_types.append(2)
            continue
        syl = syllables[best_idx]
        is_slur = consumed[best_idx]
        consumed[best_idx] = True
        if syl.get('is_pause', False):
            phonemes.append('<SP>')
            durations.append(max(note_duration, min_dur))
            note_pitches.append(0)
            note_types.append(1)
            continue
        jp_parts = []
        for raw_ph in syl['phonemes']:
            jp_ph = PJS_PHONEME_MAP.get(raw_ph, f'jp_{raw_ph}')
            if jp_ph == '<SP>':
                continue
            jp_parts.append(jp_ph)
        if not jp_parts:
            phonemes.append('<SP>')
            durations.append(max(note_duration, min_dur))
            note_pitches.append(0)
            note_types.append(1)
            continue
        merged_phoneme = 'jp_' + '-'.join(p[3:] for p in jp_parts)
        phonemes.append(merged_phoneme)
        durations.append(max(note_duration, min_dur))
        note_pitches.append(note_pitch)
        note_types.append(3 if is_slur else 2)
    return phonemes, durations, note_pitches, note_types


def process_one_pjs_sample(sample_dir, sample_id, output_dir, sample_rate=24000, lab_dir=None):
    """处理一个 PJS 样本（wav + lab + midi）。"""
    if sf is None:
        raise ImportError("soundfile is required")
    wav_path = os.path.join(sample_dir, f'{sample_id}_song.wav')
    if lab_dir:
        lab_path = os.path.join(lab_dir, f'{sample_id}.lab')
        if not os.path.exists(lab_path):
            lab_path = os.path.join(sample_dir, f'{sample_id}.lab')
    else:
        lab_path = os.path.join(sample_dir, f'{sample_id}.lab')
    midi_path = os.path.join(sample_dir, f'{sample_id}.mid')
    if not os.path.exists(wav_path) or not os.path.exists(lab_path):
        return None
    lab_segments = parse_lab_file(lab_path)
    midi_notes = []
    if os.path.exists(midi_path) and mido is not None:
        try:
            midi_notes = parse_midi_file(midi_path)
        except Exception as e:
            print(f'  Warning: MIDI parse failed for {sample_id}: {e}')
    if not midi_notes:
        for seg in lab_segments:
            if seg['phoneme'] == 'pau':
                continue
            if seg['end'] - seg['start'] <= 0:
                continue
            midi_notes.append({'start': seg['start'], 'end': seg['end'],
                               'pitch': 69, 'velocity': 80})
    if not midi_notes:
        return None
    phonemes, durations, note_pitches, note_types = lab_phonemes_to_notes(
        lab_segments, midi_notes, sample_rate)
    if not phonemes:
        return None
    f0 = None
    if parselmouth is not None:
        try:
            f0 = extract_f0_parselmouth(wav_path, sample_rate)
        except Exception as e:
            print(f'  Warning: F0 extraction failed for {sample_id}: {e}')
    y, orig_sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    if orig_sr != sample_rate:
        import scipy.signal
        y = scipy.signal.resample(y, int(len(y) * sample_rate / orig_sr))
    resampled_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(resampled_dir, exist_ok=True)
    sf.write(os.path.join(resampled_dir, f'{sample_id}_song.wav'), y, sample_rate)
    metadata = {
        'id': sample_id,
        'phoneme': ' '.join(phonemes),
        'duration': ' '.join(f'{d:.6f}' for d in durations),
        'note_pitch': ' '.join(str(p) for p in note_pitches),
        'note_type': ' '.join(str(t) for t in note_types),
    }
    if f0 is not None and len(f0) > 0:
        metadata['f0'] = ' '.join(f'{v:.2f}' for v in f0.tolist())
    return metadata


def process_pjs_corpus(corpus_dir, lab_dir, output_dir, sample_rate=24000):
    """处理 PJS 语料库。"""
    print('=' * 60)
    print('处理 PJS Corpus')
    print('=' * 60)
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f'  [WARNING] PJS 语料库不存在: {corpus_dir}')
        return []
    sample_dirs = sorted([d for d in corpus_path.iterdir()
                          if d.is_dir() and d.name.startswith('pjs')])
    print(f'Found {len(sample_dirs)} samples in {corpus_dir}')
    metadata_list = []
    for i, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name
        print(f'[{i+1}/{len(sample_dirs)}] Processing {sample_id}...')
        metadata = process_one_pjs_sample(
            str(sample_dir), sample_id, output_dir, sample_rate, lab_dir=lab_dir)
        if metadata is not None:
            metadata_list.append(metadata)
    print(f'PJS: processed {len(metadata_list)}/{len(sample_dirs)} samples')
    return metadata_list


# ===========================================================================
# JSUT 数据集处理（新增加，语音 → 歌唱格式）
# ===========================================================================
def process_jsut_dataset(jsut_dir, output_dir, sample_rate=24000, max_samples=0):
    """处理 JSUT 语音语料库，转换为歌唱训练格式。

    JSUT 是日语语音语料库（非歌唱），用于补充音素覆盖。
    将转写文本通过 G2P 转换为 jp_* 音素，F0 使用中位数作为固定音高。

    期望目录结构：
      jsut_dir/
        basic5000/
          transcript/
            transcript.txt  # 格式: "filename_0001\tsentence"
          wav/
            filename_0001.wav
            filename_0002.wav
            ...
    """
    if not jsut_dir or not os.path.isdir(jsut_dir):
        print(f'  [INFO] JSUT 未配置或路径不存在: {jsut_dir}')
        return []

    if sf is None:
        print('  [WARNING] soundfile 未安装，跳过 JSUT')
        return []

    print('=' * 60)
    print('处理 JSUT Corpus')
    print('=' * 60)

    transcript_path = os.path.join(jsut_dir, 'basic5000', 'transcript', 'transcript.txt')
    wav_dir = os.path.join(jsut_dir, 'basic5000', 'wav')
    if not os.path.exists(transcript_path) or not os.path.isdir(wav_dir):
        print(f'  [WARNING] JSUT 目录结构不符合预期: {jsut_dir}')
        return []

    # 解析 transcript
    transcripts = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                transcripts.append((parts[0], parts[1]))
            else:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    transcripts.append((parts[0], parts[1]))

    if max_samples > 0:
        transcripts = transcripts[:max_samples]

    print(f'  转写条目: {len(transcripts)}')

    wavs_out_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_out_dir, exist_ok=True)
    metadata_list = []

    for i, (wav_id, text) in enumerate(transcripts):
        wav_path = os.path.join(wav_dir, f'{wav_id}.wav')
        if not os.path.exists(wav_path):
            continue
        try:
            # G2P 转换
            tokens = lyrics_to_jp_tokens(text)
            if not tokens:
                continue

            # 重采样并保存
            y, orig_sr = sf.read(wav_path, dtype='float32')
            if y.ndim > 1:
                y = y[:, 0]
            if orig_sr != sample_rate:
                import scipy.signal
                y = scipy.signal.resample(y, int(len(y) * sample_rate / orig_sr))
            out_wav = os.path.join(wavs_out_dir, f'jsut_{wav_id}.wav')
            sf.write(out_wav, y, sample_rate)

            # F0 提取（中位数作为固定音高）
            f0 = None
            if parselmouth is not None:
                try:
                    f0_arr = extract_f0_parselmouth(wav_path, sample_rate)
                    voiced = f0_arr[f0_arr > 0]
                    median_f0 = float(np.median(voiced)) if len(voiced) > 0 else 150.0
                    f0_arr = np.full(max(1, len(y) // HOP_SIZE), median_f0, dtype=np.float32)
                    f0 = f0_arr
                except Exception:
                    pass

            # 构造 metadata（JSUT 无 MIDI，使用固定音高 60=C4）
            note_pitches = [60] * len(tokens)
            note_types = [2] * len(tokens)
            durations = [0.3] * len(tokens)  # 默认 0.3s/音节

            meta = {
                'id': f'jsut_{wav_id}',
                'phoneme': ' '.join(tokens),
                'duration': ' '.join(f'{d:.6f}' for d in durations),
                'note_pitch': ' '.join(str(p) for p in note_pitches),
                'note_type': ' '.join(str(t) for t in note_types),
            }
            if f0 is not None and len(f0) > 0:
                meta['f0'] = ' '.join(f'{v:.2f}' for v in f0.tolist())
            metadata_list.append(meta)

            if (i + 1) % 100 == 0:
                print(f'  [JSUT {i+1}/{len(transcripts)}] processed')

        except Exception as e:
            print(f'  [JSUT] 跳过 {wav_id}: {e}')
            continue

    print(f'JSUT: processed {len(metadata_list)} samples')
    return metadata_list


# ===========================================================================
# GTSinger 数据集处理（IPA 标注 → jp_* token）
# ===========================================================================
def _gtsinger_map_phoneme(ipa_ph):
    """GTSinger IPA 音素 → PJS base 音素，未知音素回退到 pau。"""
    base = GTSINGER_PHONEME_MAP.get(ipa_ph)
    if base is not None:
        return base
    # 去除长度标记 ː 后重试
    stripped = ipa_ph.replace('ː', '').replace('̥', '').replace('̩', '')
    if stripped != ipa_ph:
        base = GTSINGER_PHONEME_MAP.get(stripped)
        if base is not None:
            return base
    print(f'  [GTSinger] WARNING: 未知音素 "{ipa_ph}" → pau')
    return 'pau'


def _gtsinger_word_to_tokens(word_entry, min_dur=0.12):
    """将一个 GTSinger word 条目转换为 (phonemes, durations, note_pitches, note_types)。

    处理逻辑：
      - <AP> → <SP> token (note_pitch=0, type=1)
      - 单 note: 合并所有音素为 jp_C-V 形式，一个 token
      - 多 note (melisma): 按时序对齐音素到 note，每个 note 一个 token
        无法对齐时回退为重复合并 token（首 note type=2，后续 type=3）
    """
    ph_list = word_entry.get('ph', [])
    notes = word_entry.get('note', [0])
    note_dur = word_entry.get('note_dur', [0.3])
    word = word_entry.get('word', '')

    # 停顿
    if word == '<AP>' or ph_list == ['<AP>']:
        dur = float(note_dur[0]) if note_dur else 0.3
        return ['<SP>'], [max(dur, min_dur)], [0], [1]

    if not ph_list:
        return [], [], [], []

    # 映射 IPA → PJS base
    mapped_ph = []
    for ipa in ph_list:
        base = _gtsinger_map_phoneme(ipa)
        if base == '<SP>':
            continue
        mapped_ph.append(base)

    if not mapped_ph:
        return ['<SP>'], [max(float(note_dur[0]) if note_dur else 0.3, min_dur)], [0], [1]

    # 构建 jp_ 前缀音素
    jp_parts = []
    for base_ph in mapped_ph:
        jp_ph = PJS_PHONEME_MAP.get(base_ph, f'jp_{base_ph}')
        if jp_ph == '<SP>':
            continue
        jp_parts.append(jp_ph)

    if not jp_parts:
        return ['<SP>'], [max(float(note_dur[0]) if note_dur else 0.3, min_dur)], [0], [1]

    # 合并 token: jp_{p1}-{p2}-...
    def _merge(parts):
        return 'jp_' + '-'.join(p[3:] for p in parts)

    merged_token = _merge(jp_parts)

    # 单 note
    if len(notes) <= 1:
        dur = float(note_dur[0]) if note_dur else 0.3
        pitch = int(notes[0]) if notes and notes[0] else 60
        return [merged_token], [max(dur, min_dur)], [pitch], [2]

    # 多 note (melisma): 尝试按时序对齐音素到 note
    ph_starts = word_entry.get('ph_start', [])
    ph_ends = word_entry.get('ph_end', [])
    note_starts = word_entry.get('note_start', [])
    note_ends = word_entry.get('note_end', [])

    phonemes_out = []
    durations_out = []
    note_pitches_out = []
    note_types_out = []

    can_align = (len(ph_starts) == len(ph_list) and
                 len(note_starts) == len(notes) and
                 len(ph_list) >= len(notes))

    if can_align:
        # 按音素中点距离最近的 note 分组
        note_groups = [[] for _ in range(len(notes))]
        for pi, ipa in enumerate(ph_list):
            if ipa == '<AP>':
                continue
            ph_mid = (float(ph_starts[pi]) + float(ph_ends[pi])) / 2
            best_ni, best_dist = 0, float('inf')
            for ni in range(len(notes)):
                nm = (float(note_starts[ni]) + float(note_ends[ni])) / 2
                dist = abs(ph_mid - nm)
                if dist < best_dist:
                    best_dist = dist
                    best_ni = ni
            note_groups[best_ni].append(ipa)

        for ni, group_ipas in enumerate(note_groups):
            if not group_ipas:
                # 该 note 无对应音素 → 用合并 token 作为 slur
                dur = float(note_dur[ni]) if ni < len(note_dur) else 0.3
                phonemes_out.append(merged_token)
                durations_out.append(max(dur, min_dur))
                note_pitches_out.append(int(notes[ni]) if notes[ni] else 60)
                note_types_out.append(3 if ni > 0 else 2)
                continue
            group_mapped = []
            for ipa in group_ipas:
                base = _gtsinger_map_phoneme(ipa)
                if base == '<SP>':
                    continue
                group_mapped.append(base)
            if not group_mapped:
                group_jp = jp_parts
            else:
                group_jp = [PJS_PHONEME_MAP.get(b, f'jp_{b}') for b in group_mapped
                            if PJS_PHONEME_MAP.get(b, f'jp_{b}') != '<SP>']
            token = _merge(group_jp) if group_jp else merged_token
            dur = float(note_dur[ni]) if ni < len(note_dur) else 0.3
            phonemes_out.append(token)
            durations_out.append(max(dur, min_dur))
            note_pitches_out.append(int(notes[ni]) if notes[ni] else 60)
            note_types_out.append(3 if ni > 0 else 2)
    else:
        # 无法对齐 → 重复合并 token
        for ni, note in enumerate(notes):
            dur = float(note_dur[ni]) if ni < len(note_dur) else 0.3
            phonemes_out.append(merged_token)
            durations_out.append(max(dur, min_dur))
            note_pitches_out.append(int(note) if note else 60)
            note_types_out.append(3 if ni > 0 else 2)

    return phonemes_out, durations_out, note_pitches_out, note_types_out


def process_one_gtsinger_sample(json_path, wav_path, output_dir, sample_id,
                                 sample_rate=24000, hop_size=480):
    """处理一个 GTSinger 样本（json + wav）。"""
    if sf is None:
        raise ImportError("soundfile is required")
    with open(json_path, 'r', encoding='utf-8') as f:
        words = json.load(f)
    if not words or not isinstance(words, list):
        return None

    min_dur = 6 * hop_size / sample_rate
    phonemes = []
    durations = []
    note_pitches = []
    note_types = []

    for word_entry in words:
        if not isinstance(word_entry, dict):
            continue
        phs, durs, pitches, types = _gtsinger_word_to_tokens(word_entry, min_dur)
        phonemes.extend(phs)
        durations.extend(durs)
        note_pitches.extend(pitches)
        note_types.extend(types)

    if not phonemes:
        return None

    # 处理音频
    y, orig_sr = sf.read(wav_path, dtype='float32')
    if y.ndim > 1:
        y = y[:, 0]
    if orig_sr != sample_rate:
        import scipy.signal
        y = scipy.signal.resample(y, int(len(y) * sample_rate / orig_sr))

    wavs_out_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_out_dir, exist_ok=True)
    out_wav = os.path.join(wavs_out_dir, f'{sample_id}.wav')
    sf.write(out_wav, y, sample_rate)

    # F0 提取
    f0 = None
    if parselmouth is not None:
        try:
            f0 = extract_f0_parselmouth(wav_path, sample_rate)
        except Exception as e:
            print(f'  Warning: F0 extraction failed for {sample_id}: {e}')

    metadata = {
        'id': sample_id,
        'phoneme': ' '.join(phonemes),
        'duration': ' '.join(f'{d:.6f}' for d in durations),
        'note_pitch': ' '.join(str(p) for p in note_pitches),
        'note_type': ' '.join(str(t) for t in note_types),
    }
    if f0 is not None and len(f0) > 0:
        metadata['f0'] = ' '.join(f'{v:.2f}' for v in f0.tolist())
    return metadata


def process_gtsinger_dataset(gtsinger_ja_dir, output_dir, sample_rate=24000,
                              hop_size=480, max_samples=0):
    """遍历 GTSinger Japanese 目录树，处理所有 json+wav 样本。

    目录结构: {ja_dir}/{singer}/{technique}/{song}/{group}/{sample_id}.json+.wav
    """
    if not gtsinger_ja_dir or not os.path.isdir(gtsinger_ja_dir):
        print(f'  [INFO] GTSinger 未配置或路径不存在: {gtsinger_ja_dir}')
        return []

    if sf is None:
        print('  [WARNING] soundfile 未安装，跳过 GTSinger')
        return []

    print('=' * 60)
    print('处理 GTSinger Japanese Corpus')
    print('=' * 60)

    # 收集所有 json+wav 对
    samples = []
    for singer in sorted(os.listdir(gtsinger_ja_dir)):
        singer_dir = os.path.join(gtsinger_ja_dir, singer)
        if not os.path.isdir(singer_dir):
            continue
        for technique in sorted(os.listdir(singer_dir)):
            tech_dir = os.path.join(singer_dir, technique)
            if not os.path.isdir(tech_dir):
                continue
            for song in sorted(os.listdir(tech_dir)):
                song_dir = os.path.join(tech_dir, song)
                if not os.path.isdir(song_dir):
                    continue
                for group in sorted(os.listdir(song_dir)):
                    group_dir = os.path.join(song_dir, group)
                    if not os.path.isdir(group_dir):
                        continue
                    for fname in sorted(os.listdir(group_dir)):
                        if not fname.endswith('.json'):
                            continue
                        sample_id_num = os.path.splitext(fname)[0]
                        json_path = os.path.join(group_dir, fname)
                        wav_path = os.path.join(group_dir, sample_id_num + '.wav')
                        if not os.path.exists(wav_path):
                            continue
                        # 构建 ASCII 安全的 sample_id
                        sample_id = f'gts_{singer}_{technique}_{group}_{sample_id_num}'
                        # 替换非 ASCII 字符为短哈希
                        if not sample_id.isascii():
                            import hashlib as _hl
                            _h = _hl.md5(sample_id.encode('utf-8')).hexdigest()[:10]
                            sample_id = f'gts_{singer}_{group}_{sample_id_num}_{_h}'
                        samples.append((json_path, wav_path, sample_id))

    if max_samples > 0:
        samples = samples[:max_samples]

    print(f'  发现 {len(samples)} 个 GTSinger 样本')
    metadata_list = []
    for i, (json_path, wav_path, sample_id) in enumerate(samples):
        try:
            meta = process_one_gtsinger_sample(
                json_path, wav_path, output_dir, sample_id, sample_rate, hop_size)
            if meta is not None:
                metadata_list.append(meta)
            if (i + 1) % 50 == 0:
                print(f'  [GTSinger {i+1}/{len(samples)}] processed')
        except Exception as e:
            print(f'  [GTSinger] 跳过 {sample_id}: {e}')
            continue

    print(f'GTSinger: processed {len(metadata_list)}/{len(samples)} samples')
    return metadata_list


# ===========================================================================
# 数据集准备主函数
# ===========================================================================
def prepare_datasets(include_pjs=True, include_jsut=True, include_jvs=True,
                     include_gtsinger=True):
    """合并 PJS + JSUT + JVS + GTSinger 数据集，输出统一的 metadata.json。"""
    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    all_metadata = []

    if include_pjs:
        pjs_meta = process_pjs_corpus(PJS_CORPUS_DIR, PJS_LAB_DIR, PREPARED_DATA_DIR, SAMPLE_RATE)
        all_metadata.extend(pjs_meta)
        print(f'  PJS: {len(pjs_meta)} samples')

    if include_jsut:
        jsut_meta = process_jsut_dataset(JSUT_DIR, PREPARED_DATA_DIR, SAMPLE_RATE)
        all_metadata.extend(jsut_meta)
        print(f'  JSUT: {len(jsut_meta)} samples')

    if include_gtsinger and GTSINGER_JA_DIR:
        gts_meta = process_gtsinger_dataset(GTSINGER_JA_DIR, PREPARED_DATA_DIR, SAMPLE_RATE, HOP_SIZE)
        all_metadata.extend(gts_meta)
        print(f'  GTSinger: {len(gts_meta)} samples')

    if include_jvs and os.path.isdir(JVS_PREPARED_DIR):
        # 复制已预处理的 JVS 数据
        jvs_metadata_path = os.path.join(JVS_PREPARED_DIR, 'metadata.json')
        jvs_wavs_dir = os.path.join(JVS_PREPARED_DIR, 'wavs')
        if os.path.exists(jvs_metadata_path):
            with open(jvs_metadata_path, 'r', encoding='utf-8') as f:
                jvs_meta = json.load(f)
            # 复制 wav 文件
            dst_wavs = os.path.join(PREPARED_DATA_DIR, 'wavs')
            os.makedirs(dst_wavs, exist_ok=True)
            for meta in jvs_meta:
                sid = meta.get('id', '')
                for prefix in ('jvs', 'pjs'):
                    src_wav = os.path.join(jvs_wavs_dir, f'{sid}.wav')
                    if not os.path.exists(src_wav) and sid.startswith(prefix):
                        src_wav = os.path.join(jvs_wavs_dir, f'{sid}_song.wav')
                    if os.path.exists(src_wav):
                        dst_wav = os.path.join(dst_wavs, os.path.basename(src_wav))
                        if not os.path.exists(dst_wav):
                            shutil.copy2(src_wav, dst_wav)
                        break
            all_metadata.extend(jvs_meta)
            print(f'  JVS: {len(jvs_meta)} samples (from {JVS_PREPARED_DIR})')

    # 保存合并 metadata
    metadata_path = os.path.join(PREPARED_DATA_DIR, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f'\n合并数据集: {len(all_metadata)} samples')
    print(f'Metadata: {metadata_path}')
    print(f'Wavs: {os.path.join(PREPARED_DATA_DIR, "wavs")}')

    # 统计音素
    jp_phonemes = set()
    for meta in all_metadata:
        for ph in meta.get('phoneme', '').split():
            if ph.startswith('jp_'):
                jp_phonemes.add(ph)
    print(f'使用的 jp_* 音素数: {len(jp_phonemes)}')

    return all_metadata


# ===========================================================================
# LoRA 层实现（来自 lora.py）
# ===========================================================================
def _define_lora_linear():
    """延迟定义 LoRALinear（需要 torch 已导入）。"""
    class LoRALinear(nn.Module):
        """用低秩加性适配器包装一个 nn.Linear。"""

        def __init__(self, original: nn.Linear, rank: int = 16, alpha: int = 32):
            super().__init__()
            self.original = original
            self.rank = rank
            self.alpha = alpha
            self.scale = alpha / rank
            in_f = original.in_features
            out_f = original.out_features
            self.lora_A = nn.Parameter(torch.empty(rank, in_f))
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            for p in self.original.parameters():
                p.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            delta = (x @ self.lora_A.t()) @ self.lora_B.t()
            return self.original(x) + self.scale * delta

        def merged_weight(self) -> torch.Tensor:
            with torch.no_grad():
                return self.original.weight + self.scale * (self.lora_B @ self.lora_A)

        def merged_bias(self):
            return self.original.bias

    return LoRALinear


def apply_lora_to_model(model, rank=32, alpha=64):
    """将 preflow 的目标 Linear 替换为 LoRALinear。

    训练目标分配（按用户要求）：
      - preflow (encoder): LoRA 深度微调 ← 仅此部分注入 LoRA
      - cond_emb (跨模态对齐层): 全量微调，不注入 LoRA（直接训练原始权重）
      - diff_estimator (diffstep): 冻结，不注入 LoRA
      - note_text_encoder (embedding): 全量训练，不注入 LoRA
    """
    LoRALinear = _define_lora_linear()
    replaced = []

    # preflow: 4 个 ConvNeXtV2Block（仅此处注入 LoRA）
    for i, block in enumerate(model.preflow):
        for attr in ("pwconv1", "pwconv2"):
            original = getattr(block, attr)
            if not isinstance(original, nn.Linear):
                continue
            lora = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(block, attr, lora)
            replaced.append((block, attr, lora))

    # cond_emb: 全量微调，不注入 LoRA（在 apply_lora_and_setup_trainable 中设 requires_grad=True）
    # diff_estimator: 冻结，不注入 LoRA
    return replaced


def merge_lora_into_base(model):
    """将 preflow 的 LoRA 增量合并回基础 Linear 权重（in-place）。

    注意：cond_emb 为全量微调，无需合并；diff_estimator 未微调，无需合并。
    """
    LoRALinear = _define_lora_linear()
    for block in model.preflow:
        for attr in ("pwconv1", "pwconv2"):
            layer = getattr(block, attr)
            if isinstance(layer, LoRALinear):
                merged_lin = nn.Linear(
                    layer.original.in_features, layer.original.out_features,
                    bias=layer.original.bias is not None)
                with torch.no_grad():
                    merged_lin.weight.copy_(layer.merged_weight())
                    if layer.original.bias is not None:
                        merged_lin.bias.copy_(layer.original.bias)
                merged_lin.to(layer.original.weight.device)
                setattr(block, attr, merged_lin)


def count_trainable_params(model) -> int:
    """统计 requires_grad=True 的参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 嵌入初始化（来自 init_embeddings.py）
# ===========================================================================
def add_orthogonal_noise(vec, noise_scale=NOISE_SCALE):
    """添加正交扰动（保持范数，改变方向）。"""
    rand = torch.randn_like(vec)
    dot = torch.dot(rand, vec)
    norm_sq = torch.dot(vec, vec)
    rand_orth = rand - (dot / (norm_sq + 1e-8)) * vec
    rand_orth_norm = rand_orth.norm()
    if rand_orth_norm < 1e-8:
        return vec
    perturbation = rand_orth / rand_orth_norm * vec.norm() * noise_scale
    return vec + perturbation


def initialize_jp_embeddings(embed_weight, mapping, phone2idx):
    """从 EN 源映射初始化 JP 行。"""
    log = {}
    pending = {}
    for jp_phone, entry in mapping.items():
        if jp_phone not in phone2idx:
            log[jp_phone] = {"status": "failed", "reason": "not in phone2idx"}
            continue
        jp_idx = phone2idx[jp_phone]
        if entry.get("strategy") == "pause_mean":
            pause_indices = [phone2idx[s] for s in entry.get("pause_sources", []) if s in phone2idx]
            if pause_indices:
                mean_vec = embed_weight[pause_indices].mean(dim=0)
                embed_weight[jp_idx] = mean_vec + torch.randn_like(mean_vec) * 0.01
                log[jp_phone] = {"status": "pause_mean"}
            else:
                log[jp_phone] = {"status": "failed", "reason": "no pause sources"}
            continue
        sources = entry.get("sources", [])
        weighted_sum = torch.zeros(EMBED_DIM)
        total_weight = 0.0
        for src in sources:
            sp = src["phone"]
            if sp in phone2idx:
                weighted_sum += embed_weight[phone2idx[sp]] * src["weight"]
                total_weight += src["weight"]
        if total_weight > 0:
            if abs(total_weight - 1.0) > 1e-6:
                weighted_sum = weighted_sum / total_weight
            pending[jp_phone] = {
                "jp_idx": jp_idx, "weighted_sum": weighted_sum,
                "init_weight": entry.get("init_weight", 1.0)}
        else:
            log[jp_phone] = {"status": "failed", "reason": "no valid sources"}
    full_weight_indices = [p["jp_idx"] for p in pending.values() if p["init_weight"] >= 1.0]
    fallback_mean = embed_weight[full_weight_indices].mean(dim=0) if full_weight_indices else None
    for jp_phone, pdata in pending.items():
        jp_idx = pdata["jp_idx"]
        weighted_sum = pdata["weighted_sum"]
        iw = pdata["init_weight"]
        if iw >= 1.0:
            embed_weight[jp_idx] = add_orthogonal_noise(weighted_sum, NOISE_SCALE)
            log[jp_phone] = {"status": "ok", "norm": embed_weight[jp_idx].norm().item()}
        elif fallback_mean is not None:
            blended = iw * weighted_sum + (1 - iw) * fallback_mean
            embed_weight[jp_idx] = add_orthogonal_noise(blended, NOISE_SCALE)
            log[jp_phone] = {"status": "reduced_init", "init_weight": iw,
                             "norm": embed_weight[jp_idx].norm().item()}
        else:
            embed_weight[jp_idx] = weighted_sum * iw
            log[jp_phone] = {"status": "reduced_init_no_fallback", "norm": embed_weight[jp_idx].norm().item()}
    if fallback_mean is not None:
        for jp_phone, info in log.items():
            if info.get("status") == "failed" and jp_phone in phone2idx:
                jp_idx = phone2idx[jp_phone]
                embed_weight[jp_idx] = fallback_mean + torch.randn_like(fallback_mean) * 0.05
                log[jp_phone] = {"status": "fallback_mean"}
    return embed_weight, log


def init_jp_embeddings(model, mapping_path, phoneset_path):
    """初始化 JP 嵌入到 model.note_text_encoder。"""
    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}
    print(f"  Phone set: {len(phone2idx)} entries")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"  Mapping: {len(mapping)} JP phonemes")

    embed_weight = model.note_text_encoder.weight.data.clone()
    full_size = JP_PHONEME_START + JP_PHONEME_COUNT
    if embed_weight.shape[0] < full_size:
        new_embed = torch.zeros(full_size, EMBED_DIM)
        new_embed[:embed_weight.shape[0]] = embed_weight
        embed_weight = new_embed
    print(f"  Extended to: {embed_weight.shape}")

    embed_weight, init_log = initialize_jp_embeddings(embed_weight, mapping, phone2idx)

    # 范数缩放
    base_mean_norm = embed_weight[:JP_PHONEME_START].norm(dim=1).mean().item()
    jp_rows = embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT]
    jp_mean_norm = jp_rows.norm(dim=1).mean().item()
    if jp_mean_norm > 1e-8 and abs(base_mean_norm / jp_mean_norm - 1.0) > 0.10:
        scale = base_mean_norm / jp_mean_norm
        embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT] *= scale
        print(f"  Applied norm scale: {scale:.4f}")

    # 更新模型 embedding
    if embed_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
        new_emb = nn.Embedding(embed_weight.shape[0], EMBED_DIM)
        new_emb.weight.data = embed_weight
        model.note_text_encoder = new_emb
    else:
        model.note_text_encoder.weight.data[:embed_weight.shape[0]] = embed_weight

    # 验证初始 cos 相似度
    cos_sims = []
    for jp_name, entry in mapping.items():
        if jp_name not in phone2idx or entry.get("strategy") == "pause_mean":
            continue
        jp_idx = phone2idx[jp_name]
        for src in entry.get("sources", []):
            sp = src["phone"]
            if sp in phone2idx:
                en_idx = phone2idx[sp]
                cos = F.cosine_similarity(
                    embed_weight[jp_idx:jp_idx+1],
                    embed_weight[en_idx:en_idx+1]).item()
                cos_sims.append(cos)
    if cos_sims:
        print(f"  Initial JP-EN cos: avg={sum(cos_sims)/len(cos_sims):.4f} "
              f"(target ~ 0.96, noise_scale={NOISE_SCALE})")


# ===========================================================================
# 数据集类（来自 dataset.py）
# ===========================================================================
def _define_dataset_class():
    """延迟定义 JpLoRADataset（需要 torch 和 DataProcessor）。"""
    # 延迟导入 DataProcessor（需要 SoulX-Singer 包）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from soulxsinger.utils.data_processor import DataProcessor

    class JpLoRADataset(Dataset):
        """日语 LoRA 微调数据集。"""

        def __init__(self, metadata_path, wav_dir, phoneset_path,
                     sample_rate=24000, hop_size=480, device='cpu', max_frames=2000):
            self.wav_dir = wav_dir
            self.sample_rate = sample_rate
            self.hop_size = hop_size
            self.device = device
            self.max_frames = max_frames
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.data_processor = DataProcessor(
                hop_size=hop_size, sample_rate=sample_rate,
                phoneset_path=phoneset_path, device=device)
            self.wav_path_map = {}
            if os.path.isdir(wav_dir):
                wav_files = sorted(f for f in os.listdir(wav_dir) if f.endswith('.wav'))
                id_to_wav = {os.path.splitext(wf)[0]: wf for wf in wav_files}
                for i, meta in enumerate(self.metadata):
                    sid = meta.get("id") or meta.get("sample_id") or meta.get("index")
                    if sid and sid in id_to_wav:
                        self.wav_path_map[i] = os.path.join(wav_dir, id_to_wav[sid])
                    elif i < len(wav_files):
                        self.wav_path_map[i] = os.path.join(wav_dir, wav_files[i])
                print(f'[JpLoRADataset] Matched {len(self.wav_path_map)}/{len(self.metadata)} wav files')
            else:
                print(f'[JpLoRADataset] WARNING: wav_dir does not exist: {wav_dir}')
            print(f'[JpLoRADataset] Loaded {len(self.metadata)} samples')

        def __len__(self):
            return len(self.metadata)

        def _ensure_min_duration(self, meta):
            durations = [float(x) for x in meta["duration"].split()]
            min_dur = 3 * self.hop_size / self.sample_rate
            if all(d >= min_dur for d in durations):
                return
            adjusted = [max(d, min_dur) for d in durations]
            meta["duration"] = ' '.join(f'{d:.6f}' for d in adjusted)

        def __getitem__(self, idx):
            meta = __import__('copy').deepcopy(self.metadata[idx])
            try:
                wav_path = self.wav_path_map.get(idx)
                if wav_path is None or not os.path.exists(wav_path):
                    return None
                self._ensure_min_duration(meta)
                item = self.data_processor.process(meta, wav_path)
                if item.get('f0') is not None and item['mel2note'] is not None:
                    min_len = min(item['mel2note'].shape[1], item['f0'].shape[1])
                    item['mel2note'] = item['mel2note'][:, :min_len]
                    item['f0'] = item['f0'][:, :min_len]
                if item['mel2note'].shape[1] > self.max_frames:
                    item['mel2note'] = item['mel2note'][:, :self.max_frames]
                    if item.get('f0') is not None:
                        item['f0'] = item['f0'][:, :self.max_frames]
                    if 'waveform' in item:
                        item['waveform'] = item['waveform'][:, :self.max_frames * self.hop_size]
                item['mel_len'] = torch.tensor([item['mel2note'].shape[1]], dtype=torch.long)
                return item
            except Exception as e:
                print(f'[JpLoRADataset] Error loading sample {idx}: {e}')
                return None

    return JpLoRADataset


def collate_fn(batch):
    """自定义 collate 函数，过滤 None 样本。"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_ph_len = max(item['phoneme'].shape[1] for item in batch)
    max_mel_len = max(item['mel2note'].shape[1] for item in batch)
    padded_batch = []
    for item in batch:
        padded = {}
        for key, val in item.items():
            if val is None:
                padded[key] = None
                continue
            if key in ('phoneme', 'note_pitch', 'note_type'):
                pad_len = max_ph_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_ph_len]
            elif key in ('mel2note', 'f0'):
                pad_len = max_mel_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_mel_len]
            elif key == 'waveform':
                max_audio_len = max_mel_len * 480
                pad_len = max_audio_len - val.shape[1]
                if pad_len > 0:
                    padded[key] = torch.nn.functional.pad(val, (0, pad_len), value=0)
                else:
                    padded[key] = val[:, :max_audio_len]
            else:
                padded[key] = val
        padded_batch.append(padded)
    result = {}
    for key in padded_batch[0]:
        vals = [item[key] for item in padded_batch if item[key] is not None]
        if vals and all(v is not None for v in vals):
            result[key] = torch.cat(vals, dim=0)
        else:
            result[key] = None
    return result


# ===========================================================================
# 训练函数（来自 train_lora.py）
# ===========================================================================
def apply_lora_and_setup_trainable(model, stage):
    """注入 LoRA 并按阶段设置 requires_grad。

    训练目标分配：
      - preflow (encoder): LoRA 深度微调（按 stage 配置开关）
      - cond_emb (跨模态对齐层): 全量微调（按 stage 配置开关）
      - note_text_encoder (embedding): 全量训练（仅 JP 行，base 行梯度置零）
      - diff_estimator (diffstep): 始终冻结
      - vocoder / f0_encoder / note_pitch_encoder / note_type_encoder: 始终冻结
    """
    for param in model.parameters():
        param.requires_grad = False
    cfg = STAGE_CONFIGS[stage]
    replaced = apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    print(f"  Applied LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA}) to {len(replaced)} preflow layers")

    # preflow LoRA
    if cfg['train_lora']:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
        print(f"  Stage {stage}: preflow LoRA 适配器可训练")
    else:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False
        print(f"  Stage {stage}: preflow LoRA 适配器冻结（B=0 no-op）")

    # cond_emb 全量微调（跨模态对齐层）
    cond_parent = model.cfm_decoder.model
    if cfg.get('train_cond_emb', True):
        for param in cond_parent.cond_emb.parameters():
            param.requires_grad = True
        print(f"  Stage {stage}: cond_emb (跨模态对齐层) 全量微调")

    # embedding 全量训练（仅 JP 行，base 行梯度置零保护源语言）
    embed_weight = model.note_text_encoder.weight
    embed_weight.requires_grad = True

    def zero_base_grad(grad):
        grad = grad.clone()
        grad[:JP_PHONEME_START] = 0
        return grad
    embed_weight.register_hook(zero_base_grad)

    # diff_estimator (diffstep) 始终冻结 — 不在此处设置，已通过全局 requires_grad=False 冻结
    # vocoder 始终冻结 — 同上
    return replaced


def count_trainable_params_v3(model, stage):
    """统计可训练参数数量。

    返回 (total, lora_n, embed_n, cond_emb_n)。
    """
    lora_n = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and ('lora_A' in n or 'lora_B' in n))
    embed_n = 0
    if model.note_text_encoder.weight.requires_grad:
        embed_n = JP_PHONEME_COUNT * EMBED_DIM
    cond_emb_n = 0
    cond_parent = model.cfm_decoder.model
    for p in cond_parent.cond_emb.parameters():
        if p.requires_grad:
            cond_emb_n += p.numel()
    total = lora_n + embed_n + cond_emb_n
    return total, lora_n, embed_n, cond_emb_n


def build_optimizer(model, stage, lora_lr, embed_lr):
    """构建优化器（LoRA / embed / cond_emb 分组）。

    cond_emb 为全量微调，使用独立学习率（低于 LoRA，符合全量微调最佳实践）。
    """
    cfg = STAGE_CONFIGS[stage]
    cond_emb_lr = cfg.get('cond_emb_lr', 5e-5)
    lora_params = []
    embed_params = []
    cond_emb_params = []
    cond_parent = model.cfm_decoder.model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif 'note_text_encoder' in name:
            embed_params.append(param)
        elif param is cond_parent.cond_emb.weight or (
                cond_parent.cond_emb.bias is not None and param is cond_parent.cond_emb.bias):
            cond_emb_params.append(param)
        else:
            # 兜底：未分类的可训练参数归入 lora 组
            lora_params.append(param)
    param_groups = []
    if cfg['train_lora'] and lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr, 'name': 'lora', 'weight_decay': 0.01})
    if embed_params:
        param_groups.append({'params': embed_params, 'lr': embed_lr, 'name': 'embed', 'weight_decay': 0.0})
    if cfg.get('train_cond_emb', True) and cond_emb_params:
        param_groups.append({'params': cond_emb_params, 'lr': cond_emb_lr, 'name': 'cond_emb', 'weight_decay': 0.01})
    print(f"  Optimizer: LoRA params={sum(p.numel() for p in lora_params)} "
          f"(lr={lora_lr if cfg['train_lora'] else 'frozen'}), "
          f"Embed params={sum(p.numel() for p in embed_params)} (lr={embed_lr}), "
          f"cond_emb params={sum(p.numel() for p in cond_emb_params)} (lr={cond_emb_lr})")
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def build_scheduler(optimizer, total_steps, warmup_steps):
    """cosine 调度 + warmup。"""
    if total_steps <= 0:
        return None
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.01, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_prompt_mask(B, T, device):
    """构造 prompt 切分掩码（20%-40% 切分点）。"""
    low = max(int(T * 0.2), 1)
    high = max(int(T * 0.4), low + 1)
    prompt_len = torch.randint(low, high, (B,), device=device).long()
    col = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    is_prompt = (col < prompt_len.unsqueeze(1)).float()
    return is_prompt


def extract_features(model, batch, device):
    """提取 mel_feat / target_mel / x_mask。"""
    note_text = batch['phoneme'].to(device)
    note_pitch = batch['note_pitch'].to(device)
    note_type = batch['note_type'].to(device)
    mel2note = batch['mel2note'].to(device)
    mel_lens = batch['mel_len'].to(device)
    f0 = batch.get('f0')
    if f0 is not None:
        f0 = f0.to(device)
    waveform = batch['waveform'].to(device)
    features = (model.note_text_encoder(note_text) +
                model.note_pitch_encoder(note_pitch) +
                model.note_type_encoder(note_type))
    features = model.preflow(features)
    mel_feat = model.expand_states(features, mel2note)
    if f0 is not None and f0.shape[1] > 0:
        f0_coarse = model.f0_to_coarse(f0)
        f0_enc = model.f0_encoder(f0_coarse)
        mel_feat = mel_feat + f0_enc[:, :mel_feat.shape[1], :]
    target_mel = model.mel(waveform.float())
    T = min(target_mel.shape[1], mel_feat.shape[1])
    target_mel = target_mel[:, :T, :]
    mel_feat = mel_feat[:, :T, :]
    x_mask = (torch.arange(T, device=device).unsqueeze(0)
              < mel_lens.unsqueeze(1).clamp(max=T)).float()
    return target_mel, x_mask, mel_feat


def compute_flow_loss(model, target_mel, x_mask, mel_feat,
                      use_prompt_split, prompt_split_prob, device):
    """计算 flow-matching 损失。"""
    B, T, _ = target_mel.shape
    do_split = use_prompt_split and (torch.rand(1).item() < prompt_split_prob)
    if do_split:
        is_prompt = make_prompt_mask(B, T, device)
        noise, x, flow_pred, final_mask, prompt_len = model.cfm_decoder(
            target_mel, x_mask, mel_feat, is_prompt=is_prompt)
    else:
        noise, x, flow_pred, final_mask, prompt_len = model.cfm_decoder(
            target_mel, x_mask, mel_feat, is_prompt=None)
    sigma = model.cfm_decoder.model.sigma
    flow_target = x - (1 - sigma) * noise
    loss = ((flow_pred - flow_target) ** 2 * final_mask).sum() / final_mask.sum().clamp(min=1)
    return loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer,
                    stage, use_prompt_split, prompt_split_prob,
                    gradient_accumulation, amp_dtype, global_step):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = device.startswith('cuda') or device == 'npu'
    optimizer.zero_grad()
    for bi, batch in enumerate(dataloader):
        if batch is None:
            continue
        waveform = batch.get('waveform')
        if waveform is None:
            continue
        try:
            with amp_context(device, enabled=use_amp, dtype=amp_dtype):
                target_mel, x_mask, mel_feat = extract_features(model, batch, device)
                loss = compute_flow_loss(
                    model, target_mel, x_mask, mel_feat,
                    use_prompt_split, prompt_split_prob, device)
                loss_scaled = loss / gradient_accumulation
            loss_scaled.backward()
            if (bi + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                lr_now = optimizer.param_groups[0]['lr']
                if writer is not None:
                    writer.add_scalar('train/learning_rate', lr_now, global_step)
            total_loss += loss.item()
            n_batches += 1
            if writer is not None:
                writer.add_scalar('train/loss_step', loss.item(), epoch * 1000 + bi)
            if bi % 10 == 0:
                print(f'  Epoch {epoch} [{bi}] loss={loss.item():.4f} '
                      f'split={use_prompt_split and (prompt_split_prob > 0)}')
        except RuntimeError as e:
            if 'device' in str(e).lower() or 'cuda' in str(e).lower() or 'npu' in str(e).lower():
                print(f'  FATAL: Device error at batch {bi}: {e}')
                raise
            print(f'  Error batch {bi}: {e}')
            import traceback
            traceback.print_exc()
    avg = total_loss / max(n_batches, 1)
    if writer is not None:
        writer.add_scalar('train/loss_epoch', avg, epoch)
    return avg, global_step


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer, stage,
             use_prompt_split, prompt_split_prob, amp_dtype):
    """验证（flow-matching 损失）。"""
    model.eval()
    total_loss = 0.0
    n = 0
    use_amp = device.startswith('cuda') or device == 'npu'
    for batch in dataloader:
        if batch is None:
            continue
        waveform = batch.get('waveform')
        if waveform is None:
            continue
        try:
            with amp_context(device, enabled=use_amp, dtype=amp_dtype):
                target_mel, x_mask, mel_feat = extract_features(model, batch, device)
                loss = compute_flow_loss(
                    model, target_mel, x_mask, mel_feat,
                    use_prompt_split, prompt_split_prob, device)
            total_loss += loss.item()
            n += 1
        except RuntimeError as e:
            if 'device' in str(e).lower() or 'cuda' in str(e).lower():
                raise
            print(f'  Val error: {e}')
    avg_loss = total_loss / max(n, 1)
    if writer is not None:
        writer.add_scalar('val/loss_epoch', avg_loss, epoch)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, stage, loss, best_val_loss,
                    output_dir, is_best=False):
    """保存 checkpoint。"""
    LoRALinear = _define_lora_linear()
    os.makedirs(output_dir, exist_ok=True)
    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_state[name] = {
                'lora_A': module.lora_A.data.clone(),
                'lora_B': module.lora_B.data.clone(),
                'rank': module.rank,
                'alpha': module.alpha,
            }
    embed_weight = model.note_text_encoder.weight.data.clone()
    jp_embed = embed_weight[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT].clone()
    # cond_emb 全量微调权重（跨模态对齐层）
    cond_parent = model.cfm_decoder.model
    cond_emb_state = {
        'weight': cond_parent.cond_emb.weight.data.clone(),
    }
    if cond_parent.cond_emb.bias is not None:
        cond_emb_state['bias'] = cond_parent.cond_emb.bias.data.clone()
    ckpt = {
        'epoch': epoch, 'stage': stage, 'loss': loss, 'best_val_loss': best_val_loss,
        'lora_state': lora_state, 'embed_weight': embed_weight, 'jp_embed': jp_embed,
        'cond_emb_state': cond_emb_state,
        'jp_phoneme_start': JP_PHONEME_START, 'jp_phoneme_count': JP_PHONEME_COUNT,
        'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
    }
    hist_path = os.path.join(output_dir, f'stage{stage}_epoch{epoch:03d}.pt')
    torch.save(ckpt, hist_path)
    last_path = os.path.join(output_dir, f'stage{stage}_last.pt')
    torch.save(ckpt, last_path)
    if is_best:
        best_path = os.path.join(output_dir, f'stage{stage}_best.pt')
        torch.save(ckpt, best_path)
    # 额外保存到持久化目录（避免 upload_output 清理后丢失）
    persist_dir = os.path.join(_PERSISTENT_CKPT_DIR, f'stage{stage}')
    os.makedirs(persist_dir, exist_ok=True)
    persist_last = os.path.join(persist_dir, f'stage{stage}_last.pt')
    torch.save(ckpt, persist_last)
    if is_best:
        torch.save(ckpt, os.path.join(persist_dir, f'stage{stage}_best.pt'))
    jp_std = jp_embed.std().item()
    tag = 'BEST' if is_best else ''
    print(f'  Saved stage{stage} epoch {epoch} (jp_std={jp_std:.4f}, loss={loss:.4f}) {tag}')
    return last_path


def load_checkpoint(model, optimizer, resume_path, stage):
    """从 checkpoint 恢复。"""
    if not os.path.exists(resume_path):
        # 主路径不存在 → 尝试从持久化目录恢复（避免 upload_output 清理后丢失）
        persist_path = os.path.join(_PERSISTENT_CKPT_DIR, f'stage{stage-1}',
                                     f'stage{stage-1}_best.pt')
        if os.path.exists(persist_path):
            print(f'  [WARNING] {resume_path} 不存在，使用持久化副本: {persist_path}')
            resume_path = persist_path
        else:
            # 尝试在 stage{stage-1} 目录下找任意 .pt 文件
            fallback_dir = os.path.join(_PERSISTENT_CKPT_DIR, f'stage{stage-1}')
            if os.path.isdir(fallback_dir):
                pt_files = sorted([f for f in os.listdir(fallback_dir) if f.endswith('.pt')])
                if pt_files:
                    resume_path = os.path.join(fallback_dir, pt_files[-1])
                    print(f'  [WARNING] 使用持久化目录下最新的 checkpoint: {resume_path}')
                else:
                    raise FileNotFoundError(
                        f'Checkpoint {resume_path} 不存在，持久化目录 {fallback_dir} 也为空。\n'
                        f'请先完成 Stage {stage-1} 训练。')
            else:
                raise FileNotFoundError(
                    f'Checkpoint {resume_path} 不存在，且持久化目录 {fallback_dir} 也不存在。\n'
                    f'请先完成 Stage {stage-1} 训练。')
    print(f"  Resuming from: {resume_path}")
    ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
    print(f"  Checkpoint stage={ckpt.get('stage', '?')}, epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?')}")
    if 'embed_weight' in ckpt:
        ew = ckpt['embed_weight']
        if ew.shape[0] > model.note_text_encoder.weight.shape[0]:
            device = model.note_text_encoder.weight.device
            new_emb = nn.Embedding(ew.shape[0], EMBED_DIM).to(device)
            new_emb.weight.data = ew.to(device)
            model.note_text_encoder = new_emb
        else:
            model.note_text_encoder.weight.data[:ew.shape[0]] = ew
    lora_state = ckpt.get('lora_state', {})
    loaded = 0
    if lora_state:
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and name in lora_state:
                ls = lora_state[name]
                module.lora_A.data.copy_(ls['lora_A'])
                module.lora_B.data.copy_(ls['lora_B'])
                loaded += 1
        print(f"  Restored LoRA state for {loaded} layers")
    # 恢复 cond_emb 全量微调权重（跨模态对齐层）
    cond_emb_state = ckpt.get('cond_emb_state', {})
    if cond_emb_state:
        cond_parent = model.cfm_decoder.model
        target = cond_parent.cond_emb
        # cond_emb 可能被 LoRA 包装（旧 checkpoint 兼容）
        if hasattr(target, 'original'):
            target = target.original
        target.weight.data.copy_(cond_emb_state['weight'])
        if 'bias' in cond_emb_state and target.bias is not None:
            target.bias.data.copy_(cond_emb_state['bias'])
        print(f"  Restored cond_emb weights (full fine-tune)")
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        if ckpt.get('stage') == stage:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
                print(f"  Restored optimizer state (same stage {stage})")
            except Exception as e:
                print(f"  Skip optimizer state: {e}")
        else:
            print(f"  Skip optimizer state (cross-stage {ckpt.get('stage')}->{stage})")
    epoch_start = ckpt.get('epoch', 0) + 1
    best_val_loss = ckpt.get('best_val_loss', ckpt.get('loss', float('inf')))
    return epoch_start, best_val_loss


def split_train_val(dataset, val_ratio=0.1, seed=42):
    """划分 train / val 子集。"""
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(n, generator=g).tolist()
    train_subset = Subset(dataset, indices[:n_train])
    val_subset = Subset(dataset, indices[n_train:])
    print(f'  Split: {n_train} train / {n_val} val (seed={seed})')
    return train_subset, val_subset


def train_one_stage(stage, init_embed_path=None, resume_path=None, device='cuda'):
    """运行单个训练阶段。"""
    cfg = STAGE_CONFIGS[stage]
    print(f'\n=== Stage {stage} 训练 ===')
    print(f'  epochs={cfg["epochs"]}, lora_lr={cfg["lora_lr"]}, '
          f'embed_lr={cfg["embed_lr"]}, cond_emb_lr={cfg["cond_emb_lr"]}, '
          f'train_lora={cfg["train_lora"]}, train_cond_emb={cfg.get("train_cond_emb", True)}')
    print(f'  use_prompt_split={cfg["use_prompt_split"]}, '
          f'prompt_split_prob={cfg["prompt_split_prob"]}, '
          f'prompt_split_start_ep={cfg["prompt_split_start_ep"]}')

    output_dir = os.path.join(CHECKPOINT_DIR, f'stage{stage}')
    os.makedirs(output_dir, exist_ok=True)

    use_amp = device.startswith('cuda') or device == 'npu'
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f'  [Precision] bf16 autocast' if use_amp else '  [Precision] fp32')

    # 加载基础模型
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from soulxsinger.models.soulxsinger import SoulXSinger
    config = OmegaConf.load(CONFIG_PATH)
    print('Loading base model...')
    model = SoulXSinger(config)
    ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
    model = move_to_device(model, device)

    # gradient checkpointing
    if GRAD_CHECKPOINT and hasattr(model.cfm_decoder.model.diff_estimator, 'gradient_checkpointing'):
        model.cfm_decoder.model.diff_estimator.gradient_checkpointing = True
        print('[GradCheckpoint] Enabled on diff_estimator (22 layers)')

    # 加载 init embedding 或 resume
    if resume_path:
        print('Applying LoRA structure before resume...')
        apply_lora_and_setup_trainable(model, stage)
        model = move_to_device(model, device)
        optimizer = build_optimizer(model, stage, cfg['lora_lr'], cfg['embed_lr'])
        epoch_start, best_val_loss = load_checkpoint(model, optimizer, resume_path, stage)
        model = move_to_device(model, device)
    else:
        if stage == 1:
            if init_embed_path and os.path.exists(init_embed_path):
                print(f'  Loading init embeddings from: {init_embed_path}')
                init_data = torch.load(init_embed_path, map_location='cpu', weights_only=False)
                embed_weight = init_data['embed_weight']
                target_device = model.note_text_encoder.weight.device
                embed_weight = embed_weight.to(target_device)
                if embed_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
                    new_emb = nn.Embedding(embed_weight.shape[0], EMBED_DIM)
                    new_emb.weight.data = embed_weight
                    model.note_text_encoder = new_emb
                else:
                    model.note_text_encoder.weight.data[:embed_weight.shape[0]] = embed_weight
            else:
                # 直接从 mapping 初始化
                print('  Initializing JP embeddings from mapping...')
                init_jp_embeddings(model, JP_PHONEME_MAPPING_PATH, PHONE_SET_PATH)
        elif stage in (2, 3):
            raise ValueError(f'Stage {stage} 需通过 resume_path 继承上一阶段权重')
        print('Setting up LoRA + trainable params...')
        apply_lora_and_setup_trainable(model, stage)
        model = move_to_device(model, device)
        optimizer = build_optimizer(model, stage, cfg['lora_lr'], cfg['embed_lr'])
        epoch_start = 1
        best_val_loss = float('inf')

    total, lora_n, embed_n, cond_emb_n = count_trainable_params_v3(model, stage)
    print(f'  可训练参数: {total:,} ({total / 1e6:.3f}M) '
          f'[preflow LoRA={lora_n:,}, JP embed={embed_n:,}, cond_emb={cond_emb_n:,}]')

    # 数据集
    JpLoRADataset = _define_dataset_class()
    metadata_path = os.path.join(PREPARED_DATA_DIR, 'metadata.json')
    wav_dir = os.path.join(PREPARED_DATA_DIR, 'wavs')
    print('Loading dataset...')
    full_dataset = JpLoRADataset(
        metadata_path=metadata_path, wav_dir=wav_dir,
        phoneset_path=PHONE_SET_PATH,
        sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE)
    train_dataset, val_dataset = split_train_val(full_dataset, VAL_RATIO, SEED)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, f'stage{stage}'))
    except ImportError:
        writer = None
        print('  [WARNING] tensorboard 未安装，跳过日志记录')

    # 调度器
    steps_per_epoch = max(1, len(train_dataloader) // GRADIENT_ACCUMULATION)
    total_steps = cfg['epochs'] * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, cfg['warmup_steps'])
    print(f'  Scheduler: total_steps={total_steps}, warmup={cfg["warmup_steps"]}')

    epoch_end = epoch_start + cfg['epochs'] - 1
    print(f'\n=== Stage {stage} 训练开始: epochs {epoch_start}-{epoch_end} ({cfg["epochs"]} epochs) ===\n')

    global_step = 0
    avg_loss = 0.0
    for epoch in range(epoch_start, epoch_end + 1):
        t0 = time.time()
        if cfg['use_prompt_split'] and epoch >= cfg['prompt_split_start_ep']:
            use_ps = True
            ps_prob = cfg['prompt_split_prob']
        else:
            use_ps = False
            ps_prob = 0.0
        avg_loss, global_step = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device,
            epoch, writer, stage, use_ps, ps_prob,
            GRADIENT_ACCUMULATION, amp_dtype, global_step)
        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{epoch_end} loss={avg_loss:.4f} '
              f'time={elapsed:.1f}s split={use_ps}(p={ps_prob})')

        if epoch % 5 == 0:
            jp_embed = model.note_text_encoder.weight.data[
                JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT]
            jp_std = jp_embed.std().item()
            jp_norm = jp_embed.norm(dim=1).mean().item()
            print(f'  [Embed] JP std={jp_std:.4f}, norm={jp_norm:.3f}')
            if writer is not None:
                writer.add_scalar('embed/jp_std', jp_std, epoch)
                writer.add_scalar('embed/jp_mean_norm', jp_norm, epoch)

        if epoch % 2 == 0:
            val_loss = validate(
                model, val_dataloader, device, epoch, writer, stage,
                use_ps, ps_prob, amp_dtype)
            print(f'  val_loss={val_loss:.4f}')
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, stage, val_loss,
                            best_val_loss, output_dir, is_best=is_best)
            # 回传 checkpoint 到云平台，防止超时丢失
            print('  [c2net] 上传 checkpoint 到云存储...')
            upload_output()
        elif epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, stage, avg_loss,
                            best_val_loss, output_dir, is_best=False)
            print('  [c2net] 上传 checkpoint 到云存储...')
            upload_output()

    save_checkpoint(model, optimizer, epoch_end, stage, avg_loss,
                    best_val_loss, output_dir, is_best=False)
    print('  [c2net] 上传最终 stage checkpoint 到云存储...')
    upload_output()
    if writer is not None:
        writer.close()
    print(f'\nStage {stage} 训练完成. Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints in: {output_dir}/')
    empty_cache(device)
    # 返回持久化路径（而非 output_dir 下的路径），确保后续 stage 能找到
    persist_best = os.path.join(
        _PERSISTENT_CKPT_DIR, f'stage{stage}', f'stage{stage}_best.pt')
    if os.path.exists(persist_best):
        return persist_best
    return os.path.join(output_dir, f'stage{stage}_best.pt')


def train_all_stages(device='cuda'):
    """运行三阶段训练。"""
    print('\n' + '=' * 60)
    print('三阶段 LoRA 训练（v3）')
    print('=' * 60)

    # Stage 1
    stage1_best = train_one_stage(stage=1, init_embed_path=None, resume_path=None, device=device)
    empty_cache(device)

    # Stage 2
    stage2_best = train_one_stage(stage=2, resume_path=stage1_best, device=device)
    empty_cache(device)

    # Stage 3
    stage3_best = train_one_stage(stage=3, resume_path=stage2_best, device=device)
    empty_cache(device)

    print('\n' + '=' * 60)
    print('全部三阶段训练完成')
    print('=' * 60)
    print(f'  Stage 1 best: {stage1_best}')
    print(f'  Stage 2 best: {stage2_best}')
    print(f'  Stage 3 best: {stage3_best}')
    return stage3_best


# ===========================================================================
# ONNX 导出（来自 export_onnx.py）
# ===========================================================================
def _define_onnx_classes():
    """延迟定义 ONNX 导出包装类。"""
    NUM_PREFLOW_BLOCKS = 4

    class PreflowONNX(nn.Module):
        def __init__(self, preflow_state_dict):
            super().__init__()
            from soulxsinger.models.modules.convnext import ConvNeXtV2Block
            self.blocks = nn.ModuleList()
            for i in range(NUM_PREFLOW_BLOCKS):
                block = ConvNeXtV2Block(EMBED_DIM, EMBED_DIM * 2)
                prefix = f'{i}.'
                block_state = {}
                for k, v in preflow_state_dict.items():
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
            for block in self.blocks:
                x = block(x)
            return x

    class TextEncoderONNX(nn.Module):
        def __init__(self, full_embedding_weight):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(full_embedding_weight, freeze=True)

        def forward(self, input_ids):
            return self.embedding(input_ids)

    class CondEmbONNX(nn.Module):
        def __init__(self, cond_emb_linear):
            super().__init__()
            self.cond_emb = cond_emb_linear

        def forward(self, cond_code):
            return self.cond_emb(cond_code)

    class DiffStepONNX(nn.Module):
        def __init__(self, diff_estimator):
            super().__init__()
            self.diff_estimator = diff_estimator

        def forward(self, xt_input, t, cond, xt_mask):
            return self.diff_estimator(xt_input, t, cond, xt_mask)

    return PreflowONNX, TextEncoderONNX, CondEmbONNX, DiffStepONNX


def _ensure_rotary_emb(model):
    """确保 DiffLlama 有 rotary_emb（transformers 5.x 兼容）。"""
    diff_estimator = model.cfm_decoder.model.diff_estimator
    if not hasattr(diff_estimator, 'rotary_emb') or diff_estimator.rotary_emb is None:
        try:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            layer_cfg = diff_estimator.layers[0].self_attn.config
            diff_estimator.rotary_emb = LlamaRotaryEmbedding(config=layer_cfg)
            print('  已添加 rotary_emb（transformers 兼容）')
        except Exception as e:
            print(f'  警告: 无法添加 rotary_emb: {e}')


def build_full_embedding(ft_ckpt, base_ckpt):
    """构建完整嵌入：base 行来自 base 模型，JP 行来自微调 checkpoint。"""
    base_sd = base_ckpt.get('state_dict', base_ckpt)
    base_embed = base_sd['note_text_encoder.weight'].clone()
    if 'embed_weight' in ft_ckpt:
        ft_embed = ft_ckpt['embed_weight']
        full_embed = ft_embed.clone()
        full_embed[:base_embed.shape[0]] = base_embed
    else:
        full_embed = base_embed.clone()
    jp_std = full_embed[JP_PHONEME_START:JP_PHONEME_START + JP_PHONEME_COUNT].std().item()
    print(f'  完整嵌入: {full_embed.shape}, JP std={jp_std:.4f}')
    return full_embed


def export_onnx_files(checkpoint_path, device='cpu'):
    """导出 4 个 ONNX 文件 + merged_model.pt。"""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for ONNX export")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    PreflowONNX, TextEncoderONNX, CondEmbONNX, DiffStepONNX = _define_onnx_classes()

    print('=' * 60)
    print('导出 ONNX 模型')
    print('=' * 60)
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

    # 加载基础模型
    from soulxsinger.models.soulxsinger import SoulXSinger
    config = OmegaConf.load(CONFIG_PATH)
    print('Loading base model...')
    model = SoulXSinger(config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=True)

    # 注入 LoRA 结构（仅 preflow）
    apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    # 加载 checkpoint
    print(f'Loading checkpoint: {checkpoint_path}')
    ft_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    lora_state = ft_ckpt.get('lora_state', {})
    loaded = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and name in lora_state:
            ls = lora_state[name]
            module.lora_A.data.copy_(ls['lora_A'])
            module.lora_B.data.copy_(ls['lora_B'])
            loaded += 1
    print(f'  Loaded LoRA state for {loaded} preflow layers')

    # 加载 cond_emb 全量微调权重（跨模态对齐层）
    cond_emb_state = ft_ckpt.get('cond_emb_state', {})
    if cond_emb_state:
        cond_parent = model.cfm_decoder.model
        target = cond_parent.cond_emb
        if hasattr(target, 'original'):
            target = target.original
        target.weight.data.copy_(cond_emb_state['weight'])
        if 'bias' in cond_emb_state and target.bias is not None:
            target.bias.data.copy_(cond_emb_state['bias'])
        print(f'  Loaded cond_emb weights (full fine-tune)')
    else:
        print(f'  [WARNING] checkpoint 无 cond_emb_state，使用基础模型 cond_emb')

    # 构建完整 embedding
    full_embed = build_full_embedding(ft_ckpt, base_ckpt)

    # 合并 LoRA（仅 preflow；cond_emb 已是全量权重，无需合并；diff_estimator 未微调）
    print('Merging preflow LoRA into base weights...')
    merge_lora_into_base(model)

    opset = 17

    # 1. note_text_encoder.onnx
    print('\n导出 FP16 note_text_encoder.onnx...')
    text_model = TextEncoderONNX(full_embed.half())
    text_model.eval()
    dummy_ids = torch.randint(0, full_embed.shape[0], (1, 10), dtype=torch.long)
    text_path = os.path.join(ONNX_OUTPUT_DIR, 'note_text_encoder.onnx')
    torch.onnx.export(
        text_model, (dummy_ids,), text_path,
        input_names=['input_ids'], output_names=['embeddings'],
        dynamic_axes={'input_ids': {1: 'seq'}, 'embeddings': {1: 'seq'}},
        opset_version=opset)
    print(f'  {text_path}: {os.path.getsize(text_path) / 1024 / 1024:.2f} MB')

    # 2. preflow.onnx
    print('\n导出 FP16 preflow.onnx...')
    pf_sd = model.preflow.state_dict()
    preflow_model = PreflowONNX({k: v.half() for k, v in pf_sd.items()})
    preflow_model.half()
    preflow_model.eval()
    dummy_feat = torch.randn(1, 100, EMBED_DIM).half()
    pf_path = os.path.join(ONNX_OUTPUT_DIR, 'preflow.onnx')
    torch.onnx.export(
        preflow_model, (dummy_feat,), pf_path,
        input_names=['features'], output_names=['processed_features'],
        dynamic_axes={'features': {1: 'seq'}, 'processed_features': {1: 'seq'}},
        opset_version=opset)
    print(f'  {pf_path}: {os.path.getsize(pf_path) / 1024 / 1024:.2f} MB')

    # 3. cond_emb.onnx
    print('\n导出 FP16 cond_emb.onnx...')
    cond_emb_linear = nn.Linear(EMBED_DIM, COND_DIM)
    cond_emb_linear.load_state_dict(model.cfm_decoder.model.cond_emb.state_dict())
    cond_model = CondEmbONNX(cond_emb_linear)
    cond_model.half().eval()
    dummy_cond = torch.randn(1, 20, EMBED_DIM).half()
    cond_path = os.path.join(ONNX_OUTPUT_DIR, 'cond_emb.onnx')
    torch.onnx.export(
        cond_model, (dummy_cond,), cond_path,
        input_names=['cond_code'], output_names=['cond_embedding'],
        dynamic_axes={'cond_code': {1: 'seq'}, 'cond_embedding': {1: 'seq'}},
        opset_version=opset)
    print(f'  {cond_path}: {os.path.getsize(cond_path) / 1024 / 1024:.2f} MB')

    # 4. diff_step_dml.onnx
    print('\n导出 FP16 diff_step_dml.onnx...')
    _ensure_rotary_emb(model)
    diff_estimator = model.cfm_decoder.model.diff_estimator
    diff_step_model = DiffStepONNX(diff_estimator)
    diff_step_model.half().eval()
    seq_len = 100
    xt_input = torch.randn(1, seq_len, MEL_DIM).half()
    t = torch.tensor([0.5]).half()
    cond = torch.randn(1, seq_len, COND_DIM).half()
    xt_mask = torch.ones(1, seq_len).half()
    diff_path = os.path.join(ONNX_OUTPUT_DIR, 'diff_step_dml.onnx')
    with torch.no_grad():
        torch.onnx.export(
            diff_step_model, (xt_input, t, cond, xt_mask), diff_path,
            input_names=['xt_input', 't', 'cond', 'xt_mask'],
            output_names=['flow_pred'],
            dynamic_axes={
                'xt_input': {0: 'batch_size', 1: 'seq_len'},
                't': {0: 'batch_size'},
                'cond': {0: 'batch_size', 1: 'seq_len'},
                'xt_mask': {0: 'batch_size', 1: 'seq_len'},
                'flow_pred': {0: 'batch_size', 1: 'seq_len'},
            },
            opset_version=opset, do_constant_folding=True)
    print(f'  {diff_path}: {os.path.getsize(diff_path) / 1024 / 1024:.2f} MB')

    # 5. merged_model.pt
    print('\n保存 merged_model.pt...')
    merged_path = os.path.join(ONNX_OUTPUT_DIR, 'merged_model.pt')
    save_obj = {
        'state_dict': model.state_dict(),
        'description': 'v3 JP fine-tuned model (preflow LoRA merged + cond_emb full + JP embed; diff_estimator frozen)',
        'epoch': ft_ckpt.get('epoch'),
        'stage': ft_ckpt.get('stage'),
        'loss': ft_ckpt.get('loss'),
    }
    torch.save(save_obj, merged_path)
    print(f'  {merged_path}: {os.path.getsize(merged_path) / 1024 / 1024:.2f} MB')

    # 清理过时的 note_pitch_encoder.onnx
    stale_pitch = os.path.join(ONNX_OUTPUT_DIR, 'note_pitch_encoder.onnx')
    if os.path.exists(stale_pitch):
        os.remove(stale_pitch)
        print('\n已移除过时的 note_pitch_encoder.onnx')

    print('\n' + '=' * 60)
    print('ONNX 导出完成')
    print('=' * 60)
    print(f'  输出目录: {ONNX_OUTPUT_DIR}')
    print(f'  文件:')
    for f in sorted(os.listdir(ONNX_OUTPUT_DIR)):
        fpath = os.path.join(ONNX_OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            print(f'    {f}: {os.path.getsize(fpath) / 1024 / 1024:.2f} MB')

    # 验证 ONNX
    try:
        import onnxruntime as ort
        print('\n验证 ONNX 模型...')
        for fname in ['note_text_encoder.onnx', 'preflow.onnx', 'cond_emb.onnx', 'diff_step_dml.onnx']:
            fpath = os.path.join(ONNX_OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                sess = ort.InferenceSession(fpath, providers=['CPUExecutionProvider'])
                inputs = [i.name for i in sess.get_inputs()]
                outputs = [o.name for o in sess.get_outputs()]
                print(f'  [OK] {fname}: inputs={inputs}, outputs={outputs}')
    except ImportError:
        print('  onnxruntime 未安装，跳过验证')

    return ONNX_OUTPUT_DIR


# ===========================================================================
# 源语言保护验证
# ===========================================================================
def compute_weight_hash(model, exclude_patterns=None):
    """计算模型权重的哈希值（用于验证源语言保护）。"""
    if exclude_patterns is None:
        exclude_patterns = ['lora_A', 'lora_B', 'note_text_encoder']
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for name in sorted(state_dict.keys()):
        if any(p in name for p in exclude_patterns):
            continue
        tensor = state_dict[name].cpu().float()
        hasher.update(name.encode('utf-8'))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def verify_source_language_protection(checkpoint_path=None):
    """验证源语言（粤语/中文/英文）权重在训练前后未变化。

    训练配置：
      - preflow: LoRA 深度微调（合并后权重变化，预期行为）
      - cond_emb: 全量微调（权重直接变化，预期行为）
      - diff_estimator: 冻结（权重应完全不变）
      - note_text_encoder: JP 行变化，base 行不变（已排除）
      - 其他（vocoder/f0_encoder/note_pitch_encoder/note_type_encoder 等）: 应保持不变
    """
    if not _TORCH_AVAILABLE:
        print('[verify_protection] torch 未安装，跳过验证')
        return False

    print('\n' + '=' * 60)
    print('源语言保护验证')
    print('=' * 60)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from soulxsinger.models.soulxsinger import SoulXSinger

    config = OmegaConf.load(CONFIG_PATH)
    print('Loading base model...')
    base_model = SoulXSinger(config)
    base_ckpt = torch.load(BASE_MODEL_PATH, map_location='cpu', weights_only=False)
    base_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=True)

    # 计算原始基础模型的权重哈希（排除 note_text_encoder，因为 JP 扩展了它）
    base_hash = compute_weight_hash(base_model)
    print(f'  Base model hash (excl embed/lora): {base_hash[:16]}...')

    # 如果提供了 checkpoint，加载并验证
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        ft_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 构造合并后的模型
        merged_model = SoulXSinger(config)
        merged_model.load_state_dict(base_ckpt.get('state_dict', base_ckpt), strict=True)
        apply_lora_to_model(merged_model, rank=LORA_RANK, alpha=LORA_ALPHA)

        # 加载 LoRA 权重（仅 preflow）
        lora_state = ft_ckpt.get('lora_state', {})
        loaded = 0
        for name, module in merged_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and name in lora_state:
                ls = lora_state[name]
                module.lora_A.data.copy_(ls['lora_A'])
                module.lora_B.data.copy_(ls['lora_B'])
                loaded += 1
        print(f'  Loaded LoRA state for {loaded} preflow layers')

        # 加载 cond_emb 全量微调权重
        cond_emb_state = ft_ckpt.get('cond_emb_state', {})
        if cond_emb_state:
            cond_parent = merged_model.cfm_decoder.model
            target = cond_parent.cond_emb
            if hasattr(target, 'original'):
                target = target.original
            target.weight.data.copy_(cond_emb_state['weight'])
            if 'bias' in cond_emb_state and target.bias is not None:
                target.bias.data.copy_(cond_emb_state['bias'])
            print(f'  Loaded cond_emb weights (full fine-tune)')

        # 合并 LoRA（仅 preflow）
        merge_lora_into_base(merged_model)

        # 计算合并后模型的权重哈希
        merged_hash = compute_weight_hash(merged_model)
        print(f'  Merged model hash (excl embed/lora): {merged_hash[:16]}...')

        # 对比：preflow（LoRA 合并）和 cond_emb（全量微调）权重应有变化，
        # diff_estimator 应完全不变（冻结），其他权重也应不变。
        base_sd = base_model.state_dict()
        merged_sd = merged_model.state_dict()

        unchanged_keys = []
        changed_keys = []
        # 微调影响的层：preflow（LoRA 合并）和 cond_emb（全量微调）
        # diff_estimator 不在此列表中 — 它应保持不变
        tuned_patterns = [
            'preflow.', 'cfm_decoder.model.cond_emb'
        ]

        for key in sorted(base_sd.keys()):
            if 'note_text_encoder' in key:
                continue  # 跳过 embedding（JP 扩展了它）
            if key not in merged_sd:
                continue
            base_tensor = base_sd[key].cpu().float()
            merged_tensor = merged_sd[key].cpu().float()
            if base_tensor.shape != merged_tensor.shape:
                continue
            diff = (base_tensor - merged_tensor).abs().max().item()
            is_tuned = any(p in key for p in tuned_patterns)
            if diff < 1e-8:
                unchanged_keys.append(key)
            elif is_tuned:
                changed_keys.append((key, diff))
            else:
                changed_keys.append((key, diff))

        print(f'\n  非微调权重（应保持不变）: {len(unchanged_keys)} keys')
        print(f'  微调权重（preflow+cond_emb，应有变化）: {len([k for k, _ in changed_keys if any(p in k for p in tuned_patterns)])} keys')
        # 特别检查 diff_estimator 是否保持不变
        diff_estimator_changed = [(k, d) for k, d in changed_keys
                                   if 'cfm_decoder.model.diff_estimator' in k]
        if diff_estimator_changed:
            print(f'  [WARNING] diff_estimator (应冻结) 有 {len(diff_estimator_changed)} 个权重变化:')
            for k, d in diff_estimator_changed[:5]:
                print(f'    {k}: max_diff={d:.6f}')
        unexpected_changed = [(k, d) for k, d in changed_keys
                               if not any(p in k for p in tuned_patterns)]
        if unexpected_changed:
            print(f'  [WARNING] {len(unexpected_changed)} 个非微调权重发生了意外变化:')
            for k, d in unexpected_changed[:5]:
                print(f'    {k}: max_diff={d:.6f}')
            print('  结果: FAIL（源语言保护可能受影响）')
            return False
        else:
            print('  结果: PASS（仅 preflow+cond_emb 被修改，diff_estimator 及其他权重保持不变）')
            return True
    else:
        # 仅验证基础模型哈希（部署后对比用）
        print('  [INFO] 未提供 checkpoint，仅计算基础模型哈希')
        print(f'  部署后可用此哈希对比合并后模型的非 LoRA 权重')
        return True


# ===========================================================================
# dry-run 验证
# ===========================================================================
def dry_run():
    """dry-run：验证 G2P、LoRA 合并一致性、嵌入初始化（无需真实模型/数据）。"""
    print('=' * 60)
    print('DRY-RUN: 验证单文件训练脚本')
    print('=' * 60)

    # 1. G2P 验证
    print('\n[Test 1] JP G2P 验证')
    test_cases = [
        ("かたつむり", ['k', 'a', 't', 'a', 'ts', 'u', 'm', 'u', 'ry', 'i']),
        ("さくら", ['s', 'a', 'k', 'u', 'r', 'a']),
        ("っ", ['cl']),
        ("ん", ['n']),
    ]
    all_passed = True
    for text, expected in test_cases:
        result = japanese_g2p(text)
        ok = result == expected
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_passed = False
        print(f"  [{status}] '{text}' -> {result}" + ("" if ok else f"  (expected {expected})"))

    # 2. Token 格式验证
    print('\n[Test 2] Token 格式验证')
    token_cases = [
        ("かたつむり", ['jp_k-a', 'jp_t-a', 'jp_ts-u', 'jp_m-u', 'jp_ry-i']),
    ]
    for text, expected in token_cases:
        result = lyrics_to_jp_tokens(text)
        ok = result == expected
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_passed = False
        print(f"  [{status}] '{text}' -> {result}" + ("" if ok else f"  (expected {expected})"))

    # 3. LoRA 合并一致性验证（需要 torch）
    if _TORCH_AVAILABLE:
        print('\n[Test 3] LoRA 合并一致性验证')
        try:
            LoRALinear = _define_lora_linear()

            class _DummyBlock(nn.Module):
                def __init__(self, dim=64, intermediate=128):
                    super().__init__()
                    self.pwconv1 = nn.Linear(dim, intermediate)
                    self.pwconv2 = nn.Linear(intermediate, dim)

                def forward(self, x):
                    return x + self.pwconv2(self.pwconv1(x))

            class _DummySelfAttn(nn.Module):
                def __init__(self, hidden=64):
                    super().__init__()
                    self.q_proj = nn.Linear(hidden, hidden)
                    self.k_proj = nn.Linear(hidden, hidden)
                    self.v_proj = nn.Linear(hidden, hidden)
                    self.o_proj = nn.Linear(hidden, hidden)

            class _DummyLayer(nn.Module):
                def __init__(self, hidden=64):
                    super().__init__()
                    self.self_attn = _DummySelfAttn(hidden)

            class _DummyDiffEstimator(nn.Module):
                def __init__(self, num_layers=2, hidden=64):
                    super().__init__()
                    self.layers = nn.ModuleList([_DummyLayer(hidden) for _ in range(num_layers)])

            class _DummyCFMModel(nn.Module):
                def __init__(self, embed_dim=64, hidden=64):
                    super().__init__()
                    self.cond_emb = nn.Linear(embed_dim, hidden)
                    self.diff_estimator = _DummyDiffEstimator(2, hidden)
                    self.sigma = 1e-5

            class _DummyCFMDecoder(nn.Module):
                def __init__(self, embed_dim=64, hidden=64):
                    super().__init__()
                    self.model = _DummyCFMModel(embed_dim, hidden)

            class _DummyModel(nn.Module):
                def __init__(self, embed_dim=64, hidden=64):
                    super().__init__()
                    self.preflow = nn.Sequential(
                        *[_DummyBlock(embed_dim, embed_dim * 2) for _ in range(NUM_PREFLOW_BLOCKS)])
                    self.cfm_decoder = _DummyCFMDecoder(embed_dim, hidden)

            model = _DummyModel(64, 64)
            replaced = apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA)
            print(f'  注入 LoRA: {len(replaced)} 个适配器（仅 preflow）')
            preflow_n = sum(1 for _, attr, _ in replaced if attr in ('pwconv1', 'pwconv2'))
            cond_n = sum(1 for _, attr, _ in replaced if attr == 'cond_emb')
            attn_n = sum(1 for _, attr, _ in replaced if attr in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            print(f'    preflow: {preflow_n}, cond_emb: {cond_n} (应为0，全量微调), DiffLlama attn: {attn_n} (应为0，冻结)')
            # 验证：仅 preflow 有 LoRA，cond_emb 和 diff_estimator 无 LoRA
            if cond_n != 0 or attn_n != 0:
                print(f'  [FAIL] cond_emb 或 diff_estimator 被错误注入 LoRA')
                all_passed = False

            # 设置非零 LoRA_B
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    nn.init.normal_(module.lora_B, std=0.02)

            # 合并前前向
            sample_input = torch.randn(1, 10, 64)
            with torch.no_grad():
                out_before = model.preflow(sample_input)

            # 合并
            merge_lora_into_base(model)

            # 合并后前向
            with torch.no_grad():
                out_after = model.preflow(sample_input)

            max_error = (out_before - out_after).abs().max().item()
            print(f'  合并一致性: max abs error = {max_error:.2e} (tol=1e-5)')
            if max_error < 1e-5:
                print('  [PASS] LoRA 合并一致性验证通过')
            else:
                print('  [FAIL] LoRA 合并一致性验证失败')
                all_passed = False

            # 验证合并后无 LoRALinear 残留
            residue = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
            print(f'  LoRALinear 残留: {residue} 个')
            if residue == 0:
                print('  [PASS] 合并后无 LoRALinear 残留')
            else:
                print('  [FAIL] 合并后仍有 LoRALinear 残留')
                all_passed = False

        except Exception as e:
            print(f'  [FAIL] LoRA 验证异常: {e}')
            import traceback
            traceback.print_exc()
            all_passed = False
    else:
        print('\n[Test 3] LoRA 合并一致性验证（跳过：torch 未安装）')

    # 4. 设备检测
    print('\n[Test 4] 设备检测')
    device = detect_device()
    print(f'  检测到设备: {device}')
    if device.startswith('cuda'):
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    elif device == 'npu':
        try:
            import torch_npu
            print(f'  NPU: {torch.npu.get_device_name(0)}')
        except Exception:
            pass

    print('\n' + '=' * 60)
    if all_passed:
        print('DRY-RUN 全部通过：脚本可安全部署到云端')
    else:
        print('DRY-RUN 有失败项，请检查后再部署')
    print('=' * 60)
    return all_passed


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='单文件云端日语 LoRA 微调训练脚本（v3）')
    parser.add_argument('--dry_run', action='store_true',
                        help='dry-run：验证脚本逻辑（不需要真实模型/数据）')
    parser.add_argument('--prepare_only', action='store_true',
                        help='仅准备数据集（PJS+JSUT+JVS）')
    parser.add_argument('--train_only', action='store_true',
                        help='仅训练（数据已准备）')
    parser.add_argument('--export_only', action='store_true',
                        help='仅导出 ONNX（需先训练完成）')
    parser.add_argument('--verify_protection', action='store_true',
                        help='验证源语言权重未变化')
    parser.add_argument('--checkpoint', default=None,
                        help='ONNX 导出/验证用的 checkpoint 路径')
    parser.add_argument('--device', default=None,
                        help='覆盖设备选择（cuda/npu/cpu）')
    parser.add_argument('--no_jsut', action='store_true',
                        help='不包含 JSUT 数据集')
    parser.add_argument('--no_jvs', action='store_true',
                        help='不包含 JVS 数据集')
    parser.add_argument('--no_gtsinger', action='store_true',
                        help='不包含 GTSinger 数据集')
    parser.add_argument('--gtsinger_only', action='store_true',
                        help='仅使用 GTSinger 数据集（跳过 PJS/JSUT/JVS）')
    args = parser.parse_args()

    # 设备检测
    device = args.device or DEVICE
    if device == 'auto':
        device = detect_device()
    print(f'使用设备: {device}')

    # dry-run
    if args.dry_run:
        dry_run()
        return

    # 验证源语言保护
    if args.verify_protection:
        verify_source_language_protection(args.checkpoint)
        return

    # 仅准备数据集
    if args.prepare_only:
        if args.gtsinger_only:
            prepare_datasets(
                include_pjs=False,
                include_jsut=False,
                include_jvs=False,
                include_gtsinger=True)
        else:
            prepare_datasets(
                include_pjs=True,
                include_jsut=not args.no_jsut,
                include_jvs=not args.no_jvs,
                include_gtsinger=not args.no_gtsinger)
        return

    # 仅训练
    if args.train_only:
        if not _TORCH_AVAILABLE:
            print('错误: torch 未安装，无法训练')
            sys.exit(1)
        if device == 'cpu':
            print('警告: 使用 CPU 训练将非常缓慢，建议使用 CUDA 或 NPU')
        train_all_stages(device=device)
        return

    # 仅导出 ONNX
    if args.export_only:
        if not _TORCH_AVAILABLE:
            print('错误: torch 未安装，无法导出 ONNX')
            sys.exit(1)
        if not args.checkpoint:
            # 自动查找 stage3_best.pt
            stage3_best = os.path.join(CHECKPOINT_DIR, 'stage3', 'stage3_best.pt')
            if os.path.exists(stage3_best):
                args.checkpoint = stage3_best
            else:
                print('错误: 未指定 --checkpoint，且未找到 stage3_best.pt')
                sys.exit(1)
        export_onnx_files(args.checkpoint, device='cpu')
        return

    # 全流水线
    print('\n' + '=' * 60)
    print('全流水线执行：prepare → train → export')
    print('=' * 60)

    # 1. 准备数据集
    if not os.path.exists(os.path.join(PREPARED_DATA_DIR, 'metadata.json')):
        print('\n[1/3] 准备数据集...')
        if args.gtsinger_only:
            prepare_datasets(
                include_pjs=False,
                include_jsut=False,
                include_jvs=False,
                include_gtsinger=True)
        else:
            prepare_datasets(
                include_pjs=True,
                include_jsut=not args.no_jsut,
                include_jvs=not args.no_jvs,
                include_gtsinger=not args.no_gtsinger)
    else:
        print(f'\n[1/3] 数据集已存在: {os.path.join(PREPARED_DATA_DIR, "metadata.json")}')

    # 2. 训练
    if not _TORCH_AVAILABLE:
        print('错误: torch 未安装，无法训练')
        sys.exit(1)
    print('\n[2/3] 开始训练...')
    stage3_best = train_all_stages(device=device)

    # 3. 导出 ONNX
    print('\n[3/3] 导出 ONNX...')
    export_onnx_files(stage3_best, device='cpu')

    # 4. 验证源语言保护
    print('\n[验证] 源语言保护验证...')
    verify_source_language_protection(stage3_best)

    print('\n' + '=' * 60)
    print('全流水线完成')
    print('=' * 60)
    print(f'  ONNX 输出: {ONNX_OUTPUT_DIR}')
    print(f'  Checkpoints: {CHECKPOINT_DIR}')
    print(f'  数据集: {PREPARED_DATA_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
