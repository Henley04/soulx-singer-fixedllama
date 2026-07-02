"""
train_jp_lora_cloud_v2.py
=========================
v2 改动版：只训练促音(jp_cl)和拨音(jp_n)，其他日语音素近似映射到英语音素复用；
新增强制对齐（lab 时间戳精确对齐 + 无 lab 时的音素时长先验对齐）。

与 train_jp_lora_cloud.py (v1) 的差异：
  1. 音素映射：31 个日语音素 → 单个最接近的英语音素（1:1，上下文解耦）。
     不再新建 jp_k / jp_a / ... 等 31 个 token，直接复用 base en_* embedding。
  2. 新训练 token：只保留 jp_cl（促音，3032）和 jp_n（拨音，3008）两个新行。
     embedding 表从 3000 扩展到 3002（仅 +2 行，而非 v1 的 +33 行）。
  3. 强制对齐：用 lab 的音素级时间戳生成精确 mel2note（替代 v1 的 note 内均匀分配）；
     无 lab 时用辅音:元音时长先验分配。
  4. 训练目标：embedding 只训练 jp_cl/jp_n 两行；preflow LoRA + cond_emb 保持 v1 方案
     （因音素组合变化，跨模态对齐层仍需微调）。

接入方式：本文件提供改动函数，可在 train_jp_lora_cloud.py 中 import 后替换对应函数。
  from train_jp_lora_cloud_v2 import (
      JP_TO_BASE_PHONEME_MAP, JP_NEW_TOKENS,
      init_jp_embeddings_v2, apply_lora_and_setup_trainable_v2,
      lab_phonemes_to_notes_v2, gtsinger_word_to_tokens_v2,
      forced_align_preprocess_grouped,
  )
"""

import os
import json
import math
from typing import List, Dict, Tuple, Optional

# ===========================================================================
# 常量（与 v1 一致，仅 JP token 数量变化）
# ===========================================================================
EMBED_DIM = 512
SAMPLE_RATE = 24000
HOP_SIZE = 480
NOISE_SCALE = 0.3

# v2：只保留 2 个新 token
# 复用 v1 的索引位（jp_n=3008, jp_cl=3032）以兼容已有 checkpoint；
# 若从头训练，可改为 3000/3001。这里沿用 v1 索引，便于在 v1 模型上继续微调。
JP_NEW_TOKENS = ['jp_cl', 'jp_n']
JP_NEW_TOKEN_INDICES = {'jp_cl': 3032, 'jp_n': 3008}
JP_PHONEME_START_V2 = 3000
JP_PHONEME_COUNT_V2 = 33  # embedding 表大小沿用 v1（3033），但只训练 2 行

LORA_RANK = 32
LORA_ALPHA = 64

# ===========================================================================
# 1. 音素映射表：31 个日语音素 → 英语音素（1:1，上下文解耦）
# ===========================================================================
# 原则：每个日语音素独立映射到单个最接近的英语音素，不考虑上下文。
# ARPABET 无完全对应的（如 ts 塞擦音、拗音硬腭化）取主辅音近似。
# 重音标记：日语无重音，元音统一用 1（主重音）作为英语代表音素。
# jp_cl（促音）和 jp_n（拨音）不在此表中——它们是新训练 token。
JP_TO_BASE_PHONEME_MAP = {
    # --- 五元音 ---
    'jp_a':  'en_AA1',   # 开后不圆唇 [a]
    'jp_i':  'en_IY1',   # 闭前不圆唇 [i]
    'jp_u':  'en_UW1',   # 闭后（日语不圆唇 [ɯ]，英语圆唇 [u]，近似）
    'jp_e':  'en_EH1',   # 半开前 [e]
    'jp_o':  'en_OW1',   # 半开后圆唇 [o]
    # --- 清辅音 ---
    'jp_k':  'en_K',     # [k]
    'jp_s':  'en_S',     # [s]
    'jp_t':  'en_T',     # [t]
    'jp_h':  'en_HH',    # [h]（日语 [h] / [ç] / [ɸ] 变体，取 base [h]）
    'jp_p':  'en_P',     # [p]
    'jp_f':  'en_F',     # 日语双唇擦音 [ɸ] → 英语唇齿 [f]，近似
    'jp_ch': 'en_CH',    # [tʃ]
    'jp_sh': 'en_SH',    # [ʃ]
    'jp_ts': 'en_T',     # 塞擦音 [ts]，ARPABET 无对应，取塞音起始 [t]
    # --- 浊辅音 ---
    'jp_g':  'en_G',     # [g]
    'jp_z':  'en_Z',     # [z]
    'jp_d':  'en_D',     # [d]
    'jp_b':  'en_B',     # [b]
    'jp_j':  'en_JH',    # 浊塞擦音 [dʒ]
    # --- 鼻音 / 近音 / 闪音 ---
    'jp_m':  'en_M',     # [m]
    'jp_r':  'en_L',     # 日语齿龈闪音 [ɾ] → 英语边音 [l]，近似
    'jp_w':  'en_W',     # [w]
    'jp_y':  'en_Y',     # [j]
    # --- 拗音（硬腭化辅音）→ 取主辅音，丢失腭化色彩 ---
    'jp_ky': 'en_K',     # [kʲ] → [k]
    'jp_gy': 'en_G',     # [gʲ] → [g]
    'jp_ny': 'en_N',     # 硬腭鼻音 [ɲ] → 齿龈鼻音 [n]，近似
    'jp_hy': 'en_HH',    # 清硬腭擦音 [ç] → 声门擦音 [h]，近似
    'jp_my': 'en_M',     # [mʲ] → [m]
    'jp_ry': 'en_L',     # [ɾʲ] → [l]
    'jp_py': 'en_P',     # [pʲ] → [p]
    'jp_by': 'en_B',     # [bʲ] → [b]
}

# 反向映射：v1 的 PJS 裸音素 → v2 的 base 英语音素（用于 lab 路径）
# 键是 PJS lab 里的裸音素名（a/i/u/.../cl/n），值是 base token 或 jp_ 新 token
PJS_RAW_TO_V2_TOKEN = {
    'pau': '<SP>', 'xx': '<SP>',
    # 元音 → en_*
    'a': 'en_AA1', 'i': 'en_IY1', 'u': 'en_UW1', 'e': 'en_EH1', 'o': 'en_OW1',
    'I': 'en_AA1', 'O': 'en_OW1', 'U': 'en_UW1',
    # 辅音 → en_*
    'k': 'en_K', 's': 'en_S', 't': 'en_T', 'h': 'en_HH',
    'm': 'en_M', 'r': 'en_L', 'w': 'en_W', 'y': 'en_Y',
    'g': 'en_G', 'z': 'en_Z', 'd': 'en_D', 'b': 'en_B', 'p': 'en_P',
    'f': 'en_F', 'j': 'en_JH', 'ch': 'en_CH', 'sh': 'en_SH', 'ts': 'en_T',
    'ky': 'en_K', 'gy': 'en_G', 'ny': 'en_N', 'hy': 'en_HH',
    'my': 'en_M', 'ry': 'en_L', 'py': 'en_P', 'by': 'en_B',
    # 促音 / 拨音 → 新训练 token
    'cl': 'jp_cl',
    'n':  'jp_n',    # 拨音（moraic nasal）
    'N':  'jp_n',
}

# GTSinger IPA → v2 token（先映射到 PJS 裸音素，再用上面的表）
GTSINGER_IPA_TO_PJS_RAW = {
    'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
    'ɯ': 'u', 'ɨ': 'u', 'ɨ̥': 'u', 'i̥': 'i', 'ɯ̥': 'u',
    'ɰ̃': 'n',
    'oː': 'o', 'aː': 'a', 'iː': 'i', 'uː': 'u', 'eː': 'e', 'ɯː': 'u',
    'k': 'k', 'ɡ': 'g', 'g': 'g', 't': 't', 'd': 'd', 'p': 'p', 'b': 'b', 'c': 'k',
    's': 's', 'z': 'z', 'dz': 'z', 'h': 'h', 'f': 'f', 'ɸ': 'f',
    'ɕ': 'sh', 'sh': 'sh', 'ç': 'hy',
    'ts': 'ts', 'ch': 'ch', 'tɕ': 'ch', 'dʑ': 'j', 'ʑ': 'j', 'ɟ': 'g',
    'n': 'n', 'ɲ': 'ny', 'm': 'm', 'ɴ': 'n',
    'j': 'j', 'w': 'w', 'y': 'y', 'ɾ': 'r', 'ɾʲ': 'ry', 'r': 'r',
    'bʲ': 'by', 'kʲ': 'ky', 'gʲ': 'gy', 'pʲ': 'py', 'mʲ': 'my',
    'hʲ': 'hy', 'sʲ': 'sy',
    'ʔ': 'cl', 'cl': 'cl', 'Q': 'cl',
    '<AP>': '<SP>', '<SP>': '<SP>',
}

# 音素时长先验（秒），用于无 lab 数据的强制对齐
# 来源：日语语音学统计（辅音短、元音长、促音约 0.1s、拨音约 0.1s）
PHONEME_DURATION_PRIOR = {
    'vowel': 0.12,      # 元音
    'consonant': 0.06,  # 普通辅音
    'cl': 0.10,         # 促音（阻塞段）
    'n': 0.10,          # 拨音
    'sp': 0.15,         # 停顿
}


# ===========================================================================
# 2. G2P 转换：日语歌词 → base 英语音素 + jp_cl + jp_n 序列
# ===========================================================================
def jp_lyrics_to_base_tokens(lyrics: str) -> List[str]:
    """日语歌词 → token 列表（CV 合并为 en_C-V，促音/拨音独立）。

    与 v1 的 jp_g2p.lyrics_to_jp_tokens 输出格式平行，但 CV 音节用 en_ 前缀。
    例如 'かつ' → ['en_K-AA1', 'jp_cl', 'en_T-UW1']

    本函数是 v1 jp_g2p 的薄包装：先用 v1 G2P 得到 jp_C-V token，
    再按 JP_TO_BASE_PHONEME_MAP 把 jp_ 音素替换为 en_ 音素（促音/拨音保留）。
    需要 train/lora_jp/jp_g2p.py 可导入。
    """
    import sys as _sys
    _code_dir = os.path.dirname(os.path.abspath(__file__))
    _soulx_path = os.path.join(_code_dir, "SoulX-Singer")
    if os.path.isdir(_soulx_path):
        _sys.path.insert(0, _soulx_path)
    _jp_dir = os.path.join(_code_dir, "train", "lora_jp")
    if os.path.isdir(_jp_dir):
        _sys.path.insert(0, _jp_dir)
    from jp_g2p import lyrics_to_jp_tokens

    jp_tokens = lyrics_to_jp_tokens(lyrics)  # e.g. ['jp_k-a', 'jp_cl', 'jp_ts-u']
    out = []
    for tok in jp_tokens:
        if tok in ('jp_cl', 'jp_n'):
            out.append(tok)
            continue
        # 拆分 jp_C-V → 用映射表逐个替换
        parts = tok[3:].split('-')  # ['k', 'a']
        mapped = []
        for p in parts:
            jp_ph = 'jp_' + p
            base = JP_TO_BASE_PHONEME_MAP.get(jp_ph)
            if base is None:
                # 未知音素，回退到 <UNK>
                mapped.append('<UNK>')
            else:
                mapped.append(base)
        out.append('-'.join(mapped))  # 'en_K-en_AA1'
    return out


def gtsinger_ipa_to_v2_token(ipa: str) -> str:
    """GTSinger 单个 IPA 音素 → v2 token。"""
    raw = GTSINGER_IPA_TO_PJS_RAW.get(ipa)
    if raw is None:
        # 剥离变音符号重试
        stripped = ipa
        for diac in ('ː', '̥', '̩', '̃', 'ʲ'):
            stripped = stripped.replace(diac, '')
        raw = GTSINGER_IPA_TO_PJS_RAW.get(stripped, 'pau')
    return PJS_RAW_TO_V2_TOKEN.get(raw, '<SP>')


# ===========================================================================
# 3. 强制对齐
# ===========================================================================
def forced_align_from_lab(
    lab_segments: List[Dict],
    hop_size: int = HOP_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[str, int, int]]:
    """用 lab 的音素级时间戳做帧级强制对齐。

    输入：lab_segments = [{'phoneme': 'k', 'start': 0.0, 'end': 0.05}, ...]
    输出：[(token, start_frame, end_frame), ...]，token 已映射为 v2 格式。

    这是精确对齐——每个音素按其实际时间边界占据 mel 帧，
    替代 v1 DataProcessor.preprocess 的 note 内均匀分配。
    """
    aligned = []
    for seg in lab_segments:
        raw_ph = seg['phoneme']
        token = PJS_RAW_TO_V2_TOKEN.get(raw_ph, '<SP>')
        start_frame = int(round(seg['start'] * sample_rate / hop_size))
        end_frame = int(round(seg['end'] * sample_rate / hop_size))
        if end_frame <= start_frame:
            end_frame = start_frame + 1  # 至少 1 帧，防吞字
        aligned.append((token, start_frame, end_frame))
    return aligned


def forced_align_from_durations(
    phonemes_per_note: List[List[str]],
    note_durations: List[float],
    hop_size: int = HOP_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[str, int, int]]:
    """无 lab 时用音素时长先验做强制对齐。

    在每个 note 内按音素类型（元音/辅音/促音/拨音）的统计时长比例分配帧。
    比 v1 的均匀分配更符合语音学：辅音短、元音长。
    """
    VOWELS = {'en_AA1', 'en_IY1', 'en_UW1', 'en_EH1', 'en_OW1',
              'en_AA0', 'en_IY0', 'en_UW0', 'en_EH0', 'en_OW0'}
    aligned = []
    frame_cursor = 0
    for phs, note_dur in zip(phonemes_per_note, note_durations):
        note_total_frames = max(1, int(round(note_dur * sample_rate / hop_size)))
        # 计算每个音素的先验权重
        weights = []
        for ph in phs:
            if ph == 'jp_cl':
                weights.append(PHONEME_DURATION_PRIOR['cl'])
            elif ph == 'jp_n':
                weights.append(PHONEME_DURATION_PRIOR['n'])
            elif ph == '<SP>':
                weights.append(PHONEME_DURATION_PRIOR['sp'])
            elif ph in VOWELS:
                weights.append(PHONEME_DURATION_PRIOR['vowel'])
            else:
                weights.append(PHONEME_DURATION_PRIOR['consonant'])
        total_w = sum(weights)
        if total_w <= 0:
            total_w = 1.0
        # 按权重分配帧（至少 1 帧/音素）
        allocated = [max(1, int(note_total_frames * w / total_w)) for w in weights]
        # 修正总数偏差
        diff = note_total_frames - sum(allocated)
        if diff != 0 and phs:
            idx = max(range(len(phs)), key=lambda i: weights[i])
            allocated[idx] = max(1, allocated[idx] + diff)
        for ph, n_frames in zip(phs, allocated):
            start = frame_cursor
            end = frame_cursor + n_frames
            aligned.append((ph, start, end))
            frame_cursor = end
    return aligned


def forced_align_preprocess_grouped(
    aligned_per_note: List[List[Tuple[str, int, int]]],
    note_pitches: List[int],
    note_types: List[int],
    phone2idx: Dict[str, int],
    total_mel_frames: Optional[int] = None,
    device: str = 'cpu',
):
    """强制对齐预处理（按 note 分组）。

    Args:
        aligned_per_note: 每个 note 内的音素对齐列表，元素为 (token, start_frame, end_frame)
        note_pitches: 每个 note 的 MIDI pitch
        note_types: 每个 note 的类型 (1=SP, 2=normal, 3=slur)
        phone2idx: 音素名 → 索引
        total_mel_frames: mel 总帧数（None 则取最大 end_frame）
        device: torch 设备

    Returns: 与 DataProcessor.preprocess 相同格式的 dict
    """
    import torch

    if total_mel_frames is None:
        total_mel_frames = max(
            (end for group in aligned_per_note for _, _, end in group), default=1
        )
    mel2note = torch.zeros(total_mel_frames, dtype=torch.long)

    new_phonemes = ['<PAD>']
    new_note_pitch = [0]
    new_note_type = [1]
    ph_idx = 1  # 全局 phoneme ID 计数器（含 BOW/EOW）

    for note_i, group in enumerate(aligned_per_note):
        # 插入 <BOW> 在 note 起始帧
        if group:
            bow_frame = group[0][1]
        else:
            continue
        if bow_frame < total_mel_frames:
            mel2note[bow_frame] = ph_idx
        new_phonemes.append('<BOW>')
        new_note_pitch.append(note_pitches[note_i])
        new_note_type.append(note_types[note_i])

        # 音素帧：直接用对齐的 [start, end) 区间
        for p_i, (token, s_frame, e_frame) in enumerate(group):
            ph_id = ph_idx + 1 + p_i
            s = min(s_frame, total_mel_frames)
            e = min(e_frame, total_mel_frames)
            if e > s:
                mel2note[s:e] = ph_id
            elif s < total_mel_frames:
                mel2note[s] = ph_id  # 至少 1 帧
            new_phonemes.append(token)
            new_note_pitch.append(note_pitches[note_i])
            new_note_type.append(note_types[note_i])

        # 插入 <EOW> 在 note 结束帧
        eow_frame = min(group[-1][2] - 1, total_mel_frames - 1) if group else bow_frame
        if eow_frame >= 0:
            mel2note[eow_frame] = ph_idx + len(group) + 1
        new_phonemes.append('<EOW>')
        new_note_pitch.append(note_pitches[note_i])
        new_note_type.append(note_types[note_i])

        ph_idx += len(group) + 2  # <BOW> + 音素 + <EOW>

    # 索引化 phoneme
    unk_idx = phone2idx.get('<UNK>', phone2idx.get('<PAD>', 0))
    phoneme_ids = [phone2idx.get(p, unk_idx) for p in new_phonemes]

    return {
        'phoneme': torch.tensor(phoneme_ids, device=device).unsqueeze(0),
        'note_pitch': torch.tensor(new_note_pitch, device=device).unsqueeze(0),
        'note_type': torch.tensor(new_note_type, device=device).unsqueeze(0),
        'mel2note': mel2note.clone().detach().to(device).unsqueeze(0),
    }


# ===========================================================================
# 4. embedding 初始化（只新增 jp_cl, jp_n 两行）
# ===========================================================================
def init_jp_embeddings_v2(model, phoneset_path: str):
    """初始化 v2 的 2 个新 JP embedding 行。

    - jp_cl（促音）：从 <SP> 均值初始化（促音是无声阻塞，类似停顿）。
    - jp_n（拨音）：从 en_N 初始化（拨音是 moraic nasal，最接近英语 N）。
    其余日语音素复用 en_ embedding，不新增行。

    embedding 表大小沿用 v1（3033），保证与已有 checkpoint 兼容；
    但只初始化并训练 jp_cl / jp_n 两行。
    """
    import torch
    import torch.nn.functional as F

    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    embed_weight = model.note_text_encoder.weight.data.clone()
    full_size = JP_PHONEME_START_V2 + JP_PHONEME_COUNT_V2  # 3033
    if embed_weight.shape[0] < full_size:
        new_embed = torch.zeros(full_size, EMBED_DIM)
        new_embed[:embed_weight.shape[0]] = embed_weight
        embed_weight = new_embed
    print(f"  [v2] Embedding table: {embed_weight.shape}")

    # jp_cl ← <SP> 均值 + 小噪声
    sp_idx = phone2idx.get('<SP>')
    if sp_idx is not None:
        cl_vec = embed_weight[sp_idx].clone()
        cl_vec = cl_vec + torch.randn_like(cl_vec) * 0.01
        embed_weight[JP_NEW_TOKEN_INDICES['jp_cl']] = cl_vec
        print(f"  [v2] jp_cl (idx={JP_NEW_TOKEN_INDICES['jp_cl']}) <- <SP> mean")
    else:
        print(f"  [v2] WARNING: <SP> not in phoneset, jp_cl init skipped")

    # jp_n ← en_N + 正交扰动
    en_n_idx = phone2idx.get('en_N')
    if en_n_idx is not None:
        n_vec = embed_weight[en_n_idx].clone()
        # 添加正交扰动，避免与 en_N 完全相同
        rand = torch.randn_like(n_vec)
        dot = torch.dot(rand, n_vec)
        norm_sq = torch.dot(n_vec, n_vec)
        rand_orth = rand - (dot / (norm_sq + 1e-8)) * n_vec
        rand_orth_norm = rand_orth.norm()
        if rand_orth_norm > 1e-8:
            n_vec = n_vec + rand_orth / rand_orth_norm * n_vec.norm() * NOISE_SCALE
        embed_weight[JP_NEW_TOKEN_INDICES['jp_n']] = n_vec
        cos = F.cosine_similarity(
            embed_weight[JP_NEW_TOKEN_INDICES['jp_n']:JP_NEW_TOKEN_INDICES['jp_n']+1],
            embed_weight[en_n_idx:en_n_idx+1]).item()
        print(f"  [v2] jp_n (idx={JP_NEW_TOKEN_INDICES['jp_n']}) <- en_N "
              f"(cos={cos:.4f})")
    else:
        print(f"  [v2] WARNING: en_N not in phoneset, jp_n init skipped")

    # 写回模型
    if embed_weight.shape[0] > model.note_text_encoder.weight.shape[0]:
        import torch.nn as nn
        new_emb = nn.Embedding(embed_weight.shape[0], EMBED_DIM)
        new_emb.weight.data = embed_weight
        model.note_text_encoder = new_emb
    else:
        model.note_text_encoder.weight.data[:embed_weight.shape[0]] = embed_weight


# ===========================================================================
# 5. 训练配置 + trainable 设置（只训练 2 个 JP embedding 行）
# ===========================================================================
# v2 三阶段：与 v1 结构一致，但 embedding 只训练 jp_cl/jp_n 两行
STAGE_CONFIGS_V2 = {
    1: {
        'epochs': 10,
        'lora_lr': 0.0,        # preflow LoRA 冻结（预热）
        'embed_lr': 1e-3,      # jp_cl/jp_n embedding 预热
        'cond_emb_lr': 1e-4,
        'train_lora': False,
        'train_cond_emb': True,
        'warmup_steps': 0,
        'use_prompt_split': False,
        'prompt_split_prob': 0.0,
        'prompt_split_start_ep': 999,
    },
    2: {
        'epochs': 50,
        'lora_lr': 1e-4,
        'embed_lr': 3e-4,
        'cond_emb_lr': 5e-5,
        'train_lora': True,
        'train_cond_emb': True,
        'warmup_steps': 200,
        'use_prompt_split': True,
        'prompt_split_prob': 0.5,
        'prompt_split_start_ep': 21,
    },
    3: {
        'epochs': 20,
        'lora_lr': 3e-5,
        'embed_lr': 1e-4,
        'cond_emb_lr': 2e-5,
        'train_lora': True,
        'train_cond_emb': True,
        'warmup_steps': 100,
        'use_prompt_split': True,
        'prompt_split_prob': 1.0,
        'prompt_split_start_ep': 1,
    },
}


def apply_lora_and_setup_trainable_v2(model, stage: int, phoneset_path: str):
    """v2 trainable 设置：embedding 只训练 jp_cl/jp_n 两行，其余冻结。

    与 v1 的区别：
      - v1 训练 33 个 JP embedding 行（JP_PHONEME_START..3032）
      - v2 只训练 2 行（jp_cl=3032, jp_n=3008），其余 JP 行（虽然存在但未使用）也冻结
      - base 行（0-2999）梯度置零保护源语言（与 v1 一致）
    """
    import torch

    for param in model.parameters():
        param.requires_grad = False

    cfg = STAGE_CONFIGS_V2[stage]

    # preflow LoRA（需要外部先 apply_lora_to_model，此处只设 requires_grad）
    if cfg['train_lora']:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
        print(f"  [v2] Stage {stage}: preflow LoRA 可训练")
    else:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False
        print(f"  [v2] Stage {stage}: preflow LoRA 冻结")

    # cond_emb 全量微调
    cond_parent = model.cfm_decoder.model
    if cfg.get('train_cond_emb', True):
        for param in cond_parent.cond_emb.parameters():
            param.requires_grad = True
        print(f"  [v2] Stage {stage}: cond_emb 全量微调")

    # embedding：只训练 jp_cl / jp_n 两行
    embed_weight = model.note_text_encoder.weight
    embed_weight.requires_grad = True

    trainable_indices = set(JP_NEW_TOKEN_INDICES.values())
    print(f"  [v2] Stage {stage}: embedding 只训练 {len(trainable_indices)} 行: "
          f"{JP_NEW_TOKEN_INDICES}")

    def zero_non_target_grad(grad):
        grad = grad.clone()
        # 只保留 jp_cl/jp_n 行的梯度，其余（含 base 和未使用的 JP 行）置零
        mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        for idx in trainable_indices:
            if idx < grad.shape[0]:
                mask[idx] = True
        return grad * mask.unsqueeze(1).float()

    embed_weight.register_hook(zero_non_target_grad)


def build_optimizer_v2(model, stage: int, lora_lr: float, embed_lr: float):
    """v2 优化器：LoRA / embed(2行) / cond_emb 分组。"""
    import torch

    cfg = STAGE_CONFIGS_V2[stage]
    cond_emb_lr = cfg.get('cond_emb_lr', 5e-5)
    lora_params, embed_params, cond_emb_params = [], [], []
    cond_parent = model.cfm_decoder.model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif 'note_text_encoder' in name:
            embed_params.append(param)
        elif (param is cond_parent.cond_emb.weight or
              (cond_parent.cond_emb.bias is not None and
               param is cond_parent.cond_emb.bias)):
            cond_emb_params.append(param)
        else:
            lora_params.append(param)

    param_groups = []
    if cfg['train_lora'] and lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr,
                             'name': 'lora', 'weight_decay': 0.01})
    if embed_params:
        # v2: embedding 实际只有 2 行有梯度，但优化器收整个 weight tensor
        param_groups.append({'params': embed_params, 'lr': embed_lr,
                             'name': 'embed', 'weight_decay': 0.0})
    if cfg.get('train_cond_emb', True) and cond_emb_params:
        param_groups.append({'params': cond_emb_params, 'lr': cond_emb_lr,
                             'name': 'cond_emb', 'weight_decay': 0.01})
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ===========================================================================
# 6. 数据预处理：lab_phonemes_to_notes_v2（新映射 + 强制对齐）
# ===========================================================================
def lab_phonemes_to_notes_v2(
    lab_segments: List[Dict],
    midi_notes: List[Dict],
    sample_rate: int = SAMPLE_RATE,
    hop_size: int = HOP_SIZE,
) -> Tuple[List[str], List[float], List[int], List[int]]:
    """v2 的 lab → notes 对齐（用强制对齐替代 v1 的中心距离 + 均匀分配）。

    输出与 v1 lab_phonemes_to_notes 相同的 4 元组，可直接写入 metadata：
      (phonemes, durations, note_pitches, note_types)
    其中 phonemes 用 v2 格式（en_C-V / jp_cl / jp_n / <SP>）。

    对齐策略：
      1. 先用 lab 时间戳做音素级强制对齐（forced_align_from_lab）
      2. 按 MIDI note 边界把音素分组到 note（时间重叠最大原则）
      3. note 内音素时长来自 lab 实际时间戳（精确），而非均匀分配
    """
    min_dur = 6 * hop_size / sample_rate   # 0.12s
    max_dur = 2.0

    # 1. 音素级强制对齐
    aligned = forced_align_from_lab(lab_segments, hop_size, sample_rate)
    # aligned: [(token, start_frame, end_frame), ...]

    # 2. 音节切分（与 v1 一致：CV / V / cl / n / pause）
    # 用 lab 的音素序列切分，但 token 已映射为 v2
    syllables = _split_lab_into_syllables_v2(lab_segments)

    # 3. 把音节分配到 MIDI note（中心距离最近 + 时间重叠）
    phonemes, durations, note_pitches, note_types = [], [], [], []
    consumed = [False] * len(syllables)
    prev_note_end = None

    for note in midi_notes:
        note_start, note_end, note_pitch = note['start'], note['end'], note['pitch']
        note_duration = max(min_dur, min(note_end - note_start, max_dur))
        note_center = (note_start + note_end) / 2

        # gap → <SP>
        if prev_note_end is not None and note_start > prev_note_end + 0.05:
            gap_dur = note_start - prev_note_end
            phonemes.append('<SP>')
            durations.append(max(gap_dur, min_dur))
            note_pitches.append(0)
            note_types.append(1)
        prev_note_end = note_end

        # 找最近音节
        best_idx, best_dist = -1, float('inf')
        for si, syl in enumerate(syllables):
            if syl['end'] <= note_start or syl['start'] >= note_end:
                continue
            syl_center = (syl['start'] + syl['end']) / 2
            dist = abs(note_center - syl_center)
            if dist < best_dist:
                best_dist, best_idx = dist, si

        if best_idx == -1:
            # 无重叠音节 → 兜底 en_AA1（v1 用 jp_a，v2 映射为 en_AA1）
            phonemes.append('en_AA1')
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

        # 用 lab 实际时长（精确），而非 note_duration
        syl_dur = max(min_dur, min(syl['end'] - syl['start'], max_dur))
        # 合并音节内音素为 token（en_C-V 或单 token）
        jp_parts = []
        for raw_ph in syl['phonemes']:
            tok = PJS_RAW_TO_V2_TOKEN.get(raw_ph, '<SP>')
            if tok == '<SP>':
                continue
            jp_parts.append(tok)

        if not jp_parts:
            phonemes.append('<SP>')
            note_pitches.append(0)
            note_types.append(1)
        elif len(jp_parts) == 1:
            # 单音素音节（促音 jp_cl / 拨音 jp_n / 元音 en_*）
            phonemes.append(jp_parts[0])
            note_pitches.append(note_pitch)
            note_types.append(3 if is_slur else 2)
        else:
            # CV 合并：en_C-en_V（DataProcessor 会按 '-' 拆分 + 加 <SEP>）
            merged = '-'.join(jp_parts)
            phonemes.append(merged)
            note_pitches.append(note_pitch)
            note_types.append(3 if is_slur else 2)
        durations.append(syl_dur)

    return phonemes, durations, note_pitches, note_types


def _split_lab_into_syllables_v2(lab_segments: List[Dict]) -> List[Dict]:
    """v2 音节切分（与 v1 split_lab_into_syllables 逻辑一致，token 映射不同）。

    促音(cl)、拨音(n)独立成节；辅音+元音合并为 CV。
    """
    VOWELS = {'a', 'i', 'u', 'e', 'o', 'I', 'O', 'U'}  # 注意 N 是拨音，不在此
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
        if ph == 'cl':  # 促音独立
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': False})
            i += 1
            continue
        if ph in ('n', 'N'):  # 拨音独立
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': False})
            i += 1
            continue
        if ph in VOWELS:  # 元音独立
            syllables.append({'start': seg['start'], 'end': seg['end'],
                              'phonemes': [ph], 'is_pause': False})
            i += 1
            continue
        # 辅音 + 后跟元音 → CV
        cons_start = seg['start']
        j = i + 1
        if j < n and lab_segments[j]['phoneme'] in VOWELS:
            vow_seg = lab_segments[j]
            syllables.append({'start': cons_start, 'end': vow_seg['end'],
                              'phonemes': [ph, vow_seg['phoneme']],
                              'is_pause': False})
            i = j + 1
            continue
        # 辅音无配对元音 → 独立
        syllables.append({'start': seg['start'], 'end': seg['end'],
                          'phonemes': [ph], 'is_pause': False})
        i += 1
    return syllables


def gtsinger_word_to_tokens_v2(word_entry: dict) -> Tuple[List[str], List[float], List[int], List[int]]:
    """v2 的 GTSinger word → tokens（用 IPA → v2 映射）。

    与 v1 _gtsinger_word_to_tokens 平行，但输出 en_/jp_cl/jp_n token。
    """
    min_dur = 6 * HOP_SIZE / SAMPLE_RATE
    tokens, durs, pitches, types_ = [], [], [], []
    notes = word_entry.get('notes') or []
    pitch = int(notes[0]) if notes and notes[0] else 60

    for ph_entry in word_entry.get('ph', []):
        ipa = ph_entry.get('ph', '')
        tok = gtsinger_ipa_to_v2_token(ipa)
        dur = float(ph_entry.get('ph_dur', 0.12))
        dur = max(dur, min_dur)
        if tok == '<SP>':
            tokens.append('<SP>')
            pitches.append(0)
            types_.append(1)
        else:
            tokens.append(tok)
            pitches.append(pitch)
            types_.append(2)
        durs.append(dur)
    return tokens, durs, pitches, types_


# ===========================================================================
# 7. 接入文档与 main
# ===========================================================================
USAGE = """
=========================================================================
train_jp_lora_cloud_v2.py — 接入说明
=========================================================================

本文件提供 v2 改动函数，不包含完整训练循环（沿用 train_jp_lora_cloud.py）。
接入步骤：

1. 在 train_jp_lora_cloud.py 顶部 import：
   from train_jp_lora_cloud_v2 import (
       init_jp_embeddings_v2,
       apply_lora_and_setup_trainable_v2,
       build_optimizer_v2,
       lab_phonemes_to_notes_v2,
       gtsinger_word_to_tokens_v2,
       forced_align_from_lab,
       forced_align_preprocess_grouped,
       STAGE_CONFIGS_V2,
   )

2. 替换数据预处理函数：
   - lab_phonemes_to_notes  → lab_phonemes_to_notes_v2
   - _gtsinger_word_to_tokens → gtsinger_word_to_tokens_v2

3. 替换 embedding 初始化：
   - init_jp_embeddings(model, JP_PHONEME_MAPPING_PATH, PHONE_SET_PATH)
   + init_jp_embeddings_v2(model, PHONE_SET_PATH)

4. 替换 trainable 设置：
   - apply_lora_and_setup_trainable(model, stage)
   + apply_lora_and_setup_trainable_v2(model, stage, PHONE_SET_PATH)

5. 替换优化器：
   - build_optimizer(model, stage, lora_lr, embed_lr)
   + build_optimizer_v2(model, stage, lora_lr, embed_lr)

6. 替换 STAGE_CONFIGS：
   - STAGE_CONFIGS  → STAGE_CONFIGS_V2

7. （可选）启用强制对齐预处理：
   在 JpLoRADataset 中用 forced_align_preprocess_grouped 替代 DataProcessor.preprocess，
   需在 metadata 生成时保存 per-note 的音素对齐结果。

设计要点：
  - embedding 表只新增 2 行（jp_cl=3032, jp_n=3008），其余 31 个日语音素复用 en_*
  - 训练时只有这 2 行 embedding 有梯度（zero_non_target_grad 钩子）
  - 强制对齐用 lab 时间戳精确分配帧，替代 note 内均匀分配
  - preflow LoRA + cond_emb 仍微调（音素组合变化，需重新对齐跨模态特征）
=========================================================================
"""


def _demo():
    """演示音素映射和强制对齐的核心逻辑（不需要模型/数据）。"""
    print('=' * 60)
    print('v2 音素映射表示例（31 个日语音素 → 英语音素）')
    print('=' * 60)
    for jp, en in JP_TO_BASE_PHONEME_MAP.items():
        print(f'  {jp:8s} → {en}')
    print(f'\n  新训练 token: {JP_NEW_TOKENS}')
    print(f'  索引: {JP_NEW_TOKEN_INDICES}')

    print('\n' + '=' * 60)
    print('强制对齐演示（lab 时间戳 → 帧级对齐）')
    print('=' * 60)
    demo_lab = [
        {'phoneme': 'k', 'start': 0.00, 'end': 0.04},
        {'phoneme': 'a', 'start': 0.04, 'end': 0.14},
        {'phoneme': 'cl', 'start': 0.14, 'end': 0.24},
        {'phoneme': 't', 'start': 0.24, 'end': 0.28},
        {'phoneme': 'u', 'start': 0.28, 'end': 0.40},
    ]
    aligned = forced_align_from_lab(demo_lab)
    print(f'  输入 lab: {demo_lab}')
    print(f'  对齐结果 (token, start_frame, end_frame):')
    for tok, s, e in aligned:
        print(f'    {tok:10s} [{s:3d}, {e:3d})')
    print(f'\n  → 促音 jp_cl 和拨音 jp_n 保留为新 token，其余映射为 en_*')

    print('\n' + '=' * 60)
    print('G2P 转换演示（歌词 → v2 token）')
    print('=' * 60)
    try:
        tokens = jp_lyrics_to_base_tokens('かつん')
        print(f"  'かつん' → {tokens}")
    except Exception as e:
        print(f'  (需要 train/lora_jp/jp_g2p.py 可导入) 跳过: {e}')

    print('\n' + USAGE)


if __name__ == '__main__':
    _demo()
