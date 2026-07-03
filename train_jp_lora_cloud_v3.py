"""
train_jp_lora_cloud_v3.py
=========================
v3 改动版：在 v2 基础上，放开 22 个复用英语音素的梯度（复用微调），
用极小学习率缓慢适应日语，同时尽量不破坏源语言。

与 v2 的差异：
  - v2：复用音素（en_K 等）完全冻结，只训练 jp_cl/jp_n 两行
  - v3：复用音素也可微调，但用极小 lr（1e-5），jp_cl/jp_n 用正常 lr（3e-4）
  - 优化器分 4 组：jp_cl/jp_n（新 token）/ 复用 en_*（小 lr）/ LoRA / cond_emb
  - GRAD_CHECKPOINT 同 v2 建议关闭（显存充裕时重计算是纯速度开销）

设计权衡：
  - 收益：复用音素能缓慢适应日语发音特征（如让 en_L 稍微偏向闪音 [ɾ]）
  - 风险：复用音素是英语/中文/粤语的共享权重，微调会改变源语言合成
  - 缓解：极小 lr（1e-5）+ 梯度裁剪 + 短训练周期，控制偏移幅度
  - 保护：非复用的 base 行（zh_*, yue_*, 未复用的 en_*）仍梯度置零

直接运行：python train_jp_lora_cloud_v3.py
  - 无参数：全流水线（prepare → train → export）
  - --prepare_only / --train_only / --export_only：分阶段执行
  - --demo：仅查看复用音素列表和 lr 配置（不需模型/数据）
  - 其他参数同 train_jp_lora_cloud.py（--no_jsut / --device 等）
实现方式：运行时 monkey-patch v1 模块的关键函数为 v3 版本，复用 v1 全部基础设施。
"""

import os
import sys
import json
from typing import List, Dict, Tuple, Optional, Set

# 复用 v2 的映射表、G2P、对齐、数据预处理（这些不变）
from train_jp_lora_cloud_v2 import (
    JP_TO_BASE_PHONEME_MAP,
    PJS_RAW_TO_V2_TOKEN,
    GTSINGER_IPA_TO_PJS_RAW,
    PHONEME_DURATION_PRIOR,
    JP_NEW_TOKENS,
    JP_NEW_TOKEN_INDICES,
    JP_PHONEME_START_V2,
    JP_PHONEME_COUNT_V2,
    EMBED_DIM,
    SAMPLE_RATE,
    HOP_SIZE,
    NOISE_SCALE,
    LORA_RANK,
    LORA_ALPHA,
    jp_lyrics_to_base_tokens,
    gtsinger_ipa_to_v2_token,
    forced_align_from_lab,
    forced_align_from_durations,
    forced_align_preprocess_grouped,
    lab_phonemes_to_notes_v2,
    gtsinger_word_to_tokens_v2,
    init_jp_embeddings_v2,
)

# ===========================================================================
# v3 新增：复用音素集合
# ===========================================================================
# 日语复用的 22 个英语音素（去重）
REUSED_BASE_PHONEMES: List[str] = sorted(set(JP_TO_BASE_PHONEME_MAP.values()))

# v3 可训练的 embedding 行 = jp_cl/jp_n（2 个新 token）+ 22 个复用 en_* 音素
# 其余 base 行（zh_*, yue_*, 未复用的 en_*, 特殊 token）仍梯度置零保护源语言


def _get_reused_indices(phone2idx: Dict[str, int]) -> Set[int]:
    """获取复用音素的 embedding 索引集合。"""
    return {phone2idx[p] for p in REUSED_BASE_PHONEMES if p in phone2idx}


# ===========================================================================
# v3 训练配置：复用音素用极小 lr
# ===========================================================================
# 学习率策略：
#   - jp_cl/jp_n（新 token）：正常 lr，需要从初始化快速学习
#   - 复用 en_*（共享权重）：极小 lr（1e-5），缓慢适应日语不破坏源语言
#   - LoRA / cond_emb：同 v2
REUSED_EMBED_LR = 1e-5  # 复用音素学习率（比新 token 小 30 倍）

STAGE_CONFIGS_V3 = {
    1: {
        'epochs': 10,
        'lora_lr': 0.0,            # preflow LoRA 冻结（预热）
        'embed_lr': 1e-3,          # jp_cl/jp_n 新 token 预热
        'reused_embed_lr': 0.0,    # 复用音素冻结（预热阶段不动）
        'cond_emb_lr': 1e-4,
        'train_lora': False,
        'train_cond_emb': True,
        'train_reused': False,     # Stage 1 复用音素冻结
        'warmup_steps': 0,
        'use_prompt_split': False,
        'prompt_split_prob': 0.0,
        'prompt_split_start_ep': 999,
    },
    2: {
        'epochs': 50,
        'lora_lr': 1e-4,
        'embed_lr': 3e-4,          # jp_cl/jp_n
        'reused_embed_lr': REUSED_EMBED_LR,  # 1e-5，缓慢微调
        'cond_emb_lr': 5e-5,
        'train_lora': True,
        'train_cond_emb': True,
        'train_reused': True,      # Stage 2 开始微调复用音素
        'warmup_steps': 200,
        'use_prompt_split': True,
        'prompt_split_prob': 0.5,
        'prompt_split_start_ep': 21,
    },
    3: {
        'epochs': 20,
        'lora_lr': 3e-5,
        'embed_lr': 1e-4,          # jp_cl/jp_n 精调
        'reused_embed_lr': 5e-6,   # 更小 lr，收敛阶段
        'cond_emb_lr': 2e-5,
        'train_lora': True,
        'train_cond_emb': True,
        'train_reused': True,
        'warmup_steps': 100,
        'use_prompt_split': True,
        'prompt_split_prob': 1.0,
        'prompt_split_start_ep': 1,
    },
}


# ===========================================================================
# v3 embedding 初始化（复用音素保持原值，不重新初始化）
# ===========================================================================
def init_jp_embeddings_v3(model, phoneset_path: str):
    """v3 embedding 初始化。

    与 v2 相同：jp_cl 从 <SP> 初始化，jp_n 从 en_N 初始化。
    复用音素（en_K 等）保持 base 模型原值不动——它们将在训练中微调，
    但初始化时不改变，确保起点是源语言的原始分布。
    """
    # 直接复用 v2 的初始化逻辑
    init_jp_embeddings_v2(model, phoneset_path)
    print(f"  [v3] 复用音素（{len(REUSED_BASE_PHONEMES)} 个）保持原值，"
          f"将在 Stage 2/3 以 lr<={REUSED_EMBED_LR} 微调")


# ===========================================================================
# v3 trainable 设置：jp_cl/jp_n + 复用音素可训练，其余冻结
# ===========================================================================
def apply_lora_and_setup_trainable_v3(model, stage: int, phoneset_path: str):
    """v3 trainable 设置。

    与 v2 的区别：
      - v2：只有 jp_cl/jp_n 两行可训练
      - v3：jp_cl/jp_n + 22 个复用 en_* 音素都可训练（按 stage 开关）
      - 非复用的 base 行（zh_*, yue_*, 未复用 en_*）仍梯度置零
    """
    import torch

    for param in model.parameters():
        param.requires_grad = False

    cfg = STAGE_CONFIGS_V3[stage]

    # preflow LoRA
    if cfg['train_lora']:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
        print(f"  [v3] Stage {stage}: preflow LoRA 可训练")
    else:
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False
        print(f"  [v3] Stage {stage}: preflow LoRA 冻结")

    # cond_emb 全量微调
    cond_parent = model.cfm_decoder.model
    if cfg.get('train_cond_emb', True):
        for param in cond_parent.cond_emb.parameters():
            param.requires_grad = True
        print(f"  [v3] Stage {stage}: cond_emb 全量微调")

    # embedding：jp_cl/jp_n 始终可训练；复用音素按 stage 开关
    embed_weight = model.note_text_encoder.weight
    embed_weight.requires_grad = True

    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    # 可训练索引集合
    trainable_indices = set(JP_NEW_TOKEN_INDICES.values())  # jp_cl, jp_n
    if cfg.get('train_reused', False):
        reused_indices = _get_reused_indices(phone2idx)
        trainable_indices |= reused_indices
        print(f"  [v3] Stage {stage}: embedding 可训练 "
              f"{len(trainable_indices)} 行 "
              f"(jp_cl/jp_n + {len(reused_indices)} 复用 en_*)")
    else:
        print(f"  [v3] Stage {stage}: embedding 只训练 "
              f"{len(trainable_indices)} 行 (jp_cl/jp_n，复用冻结)")

    def zero_non_target_grad(grad):
        grad = grad.clone()
        mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        for idx in trainable_indices:
            if idx < grad.shape[0]:
                mask[idx] = True
        return grad * mask.unsqueeze(1).float()

    embed_weight.register_hook(zero_non_target_grad)


# ===========================================================================
# v3 优化器：4 组参数（新 token / 复用音素 / LoRA / cond_emb）
# ===========================================================================
def build_optimizer_v3(model, stage: int, lora_lr: float, embed_lr: float,
                       phoneset_path: str):
    """v3 优化器：LoRA / 新 token embed / 复用 embed / cond_emb 分组。

    复用音素用极小 lr（reused_embed_lr），避免破坏源语言。
    """
    import torch

    cfg = STAGE_CONFIGS_V3[stage]
    cond_emb_lr = cfg.get('cond_emb_lr', 5e-5)
    reused_lr = cfg.get('reused_embed_lr', REUSED_EMBED_LR)

    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}
    reused_indices = _get_reused_indices(phone2idx)
    new_token_indices = set(JP_NEW_TOKEN_INDICES.values())

    lora_params, new_embed_params, reused_embed_params, cond_emb_params = \
        [], [], [], []
    cond_parent = model.cfm_decoder.model

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        elif 'note_text_encoder' in name:
            # embedding 是单个 weight tensor，无法按行分组
            # 全部放入 new_embed_params，用钩子控制实际更新行
            # 但 lr 只能设一个——这里用新 token 的 lr
            # 复用音素的"小 lr"效果通过梯度缩放实现（见下方 hook）
            new_embed_params.append(param)
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
    if new_embed_params:
        param_groups.append({'params': new_embed_params, 'lr': embed_lr,
                             'name': 'embed_new', 'weight_decay': 0.0})
    if cfg.get('train_cond_emb', True) and cond_emb_params:
        param_groups.append({'params': cond_emb_params, 'lr': cond_emb_lr,
                             'name': 'cond_emb', 'weight_decay': 0.01})

    # 复用音素的 lr 缩放通过梯度 hook 实现（因为 embedding 是单 tensor）
    # 注册第二个 hook：对复用音素行的梯度乘以 reused_lr / embed_lr
    if cfg.get('train_reused', False) and reused_lr < embed_lr:
        scale = reused_lr / embed_lr
        embed_weight = model.note_text_encoder.weight

        def scale_reused_grad(grad):
            grad = grad.clone()
            for idx in reused_indices:
                if idx < grad.shape[0]:
                    grad[idx] = grad[idx] * scale
            return grad

        # 注意：hook 执行顺序与注册顺序相反（后注册的先执行）
        # zero_non_target_grad 已在 apply_lora_and_setup_trainable_v3 注册
        # 这里再注册一个，会在 zero_non_target_grad 之前执行
        # 但 PyTorch 的 hook 是按注册顺序执行的（非反向）
        # 实际上多个 hook 会按注册顺序依次执行，前一个的输出是后一个的输入
        # 所以顺序：scale_reused_grad → zero_non_target_grad
        # scale 先放大/缩小复用行，再由 zero 钩子保留/置零
        embed_weight.register_hook(scale_reused_grad)
        print(f"  [v3] 复用音素梯度缩放: {scale:.4f} "
              f"(reused_lr={reused_lr} / embed_lr={embed_lr})")

    print(f"  [v3] Optimizer: LoRA(lr={lora_lr if cfg['train_lora'] else 'frozen'}), "
          f"new_embed(lr={embed_lr}), "
          f"reused_embed(effective_lr={reused_lr}), "
          f"cond_emb(lr={cond_emb_lr})")
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ===========================================================================
# v3 验证工具：检查复用音素偏移量
# ===========================================================================
def check_reused_phoneme_drift(model, base_model, phoneset_path: str):
    """对比 v3 训练前后复用音素 embedding 的偏移量。

    用途：训练后调用，检查复用音素偏离源语言原始分布的程度。
    若偏移过大（如 cos < 0.95），说明可能破坏源语言，应降低 reused_lr。

    Args:
        model: 训练后的模型
        base_model: 训练前的基础模型
        phoneset_path: 音素表路径
    """
    import torch
    import torch.nn.functional as F

    with open(phoneset_path, 'r', encoding='utf-8') as f:
        phone_list = json.load(f)
    phone2idx = {ph: i for i, ph in enumerate(phone_list)}

    ft_emb = model.note_text_encoder.weight.data
    base_emb = base_model.note_text_encoder.weight.data

    print(f"  [v3] 复用音素偏移检查（cos 相似度，>0.95 为安全）：")
    max_drift = 0.0
    for ph in REUSED_BASE_PHONEMES:
        idx = phone2idx.get(ph)
        if idx is None or idx >= base_emb.shape[0] or idx >= ft_emb.shape[0]:
            continue
        cos = F.cosine_similarity(
            ft_emb[idx:idx+1], base_emb[idx:idx+1]).item()
        l2 = (ft_emb[idx] - base_emb[idx]).norm().item()
        flag = '⚠' if cos < 0.95 else '✓'
        print(f"    {flag} {ph:8s} cos={cos:.4f}  L2={l2:.4f}")
        max_drift = max(max_drift, 1.0 - cos)
    print(f"  [v3] 最大偏移: {max_drift:.4f} "
          f"({'安全' if max_drift < 0.05 else '建议降低 reused_lr'})")


# ===========================================================================
# 接入文档
# ===========================================================================
USAGE_V3 = """
=========================================================================
train_jp_lora_cloud_v3.py — 接入说明（复用微调版）
=========================================================================

与 v2 的区别：v3 放开 22 个复用英语音素的梯度，用极小 lr（1e-5）微调，
让复用音素缓慢适应日语发音，同时尽量不破坏源语言。

接入步骤（在 train_jp_lora_cloud.py 基础上）：

1. import v3：
   from train_jp_lora_cloud_v3 import (
       init_jp_embeddings_v3,
       apply_lora_and_setup_trainable_v3,
       build_optimizer_v3,
       check_reused_phoneme_drift,
       STAGE_CONFIGS_V3,
       REUSED_BASE_PHONEMES,
   )
   # 数据预处理、对齐、G2P 等沿用 v2

2. 替换函数：
   - init_jp_embeddings → init_jp_embeddings_v3
   - apply_lora_and_setup_trainable → apply_lora_and_setup_trainable_v3
   - build_optimizer → build_optimizer_v3
   - STAGE_CONFIGS → STAGE_CONFIGS_V3

3. 训练后验证：
   check_reused_phoneme_drift(model, base_model, PHONE_SET_PATH)
   若有音素 cos < 0.95，降低 REUSED_EMBED_LR 重训。

复用音素列表（{n_reused} 个）：
{reused_list}

学习率分组：
  - jp_cl/jp_n（新 token）：3e-4（Stage 2）/ 1e-4（Stage 3）
  - 复用 en_*（共享权重）：1e-5（Stage 2）/ 5e-6（Stage 3）
  - LoRA：1e-4 / 3e-5
  - cond_emb：5e-5 / 2e-5

风险提示：
  复用音素是英语/中文/粤语的共享 embedding 行。微调会改变源语言合成。
  极小 lr + 梯度缩放 + 训练后 cos 检查是三层防护。若源语言听感下降，
  回退到 v2（完全冻结复用音素）。
=========================================================================
""".format(
    n_reused=len(REUSED_BASE_PHONEMES),
    reused_list=', '.join(REUSED_BASE_PHONEMES),
)


def _demo():
    print('=' * 60)
    print('v3 复用微调版')
    print('=' * 60)
    print(f'复用音素（{len(REUSED_BASE_PHONEMES)} 个，可微调）：')
    for p in REUSED_BASE_PHONEMES:
        print(f'  {p}')
    print(f'\n新训练 token: {JP_NEW_TOKENS}')
    print(f'复用音素 lr: Stage2={REUSED_EMBED_LR}, Stage3={STAGE_CONFIGS_V3[3]["reused_embed_lr"]}')
    print(f'\n学习率对比（Stage 2）：')
    print(f'  jp_cl/jp_n (新 token):  {STAGE_CONFIGS_V3[2]["embed_lr"]}')
    print(f'  复用 en_*  (共享权重):  {STAGE_CONFIGS_V3[2]["reused_embed_lr"]}')
    print(f'  比值: {STAGE_CONFIGS_V3[2]["reused_embed_lr"] / STAGE_CONFIGS_V3[2]["embed_lr"]:.4f}'
          f' (复用音素梯度缩放至此比例)')
    print('\n' + USAGE_V3)


# ===========================================================================
# 自包含入口：直接 python train_jp_lora_cloud_v3.py 即可启动训练
# ===========================================================================
# 策略：import v1 模块，用 monkey-patch 把 v1 的关键函数/常量替换为 v3 版本，
# 然后调用 v1 的 main()。数据准备、训练循环、导出逻辑复用 v1。
# v3 相对 v2 的差异：复用音素也微调（极小 lr），需替换 apply_lora/build_optimizer
# 和 STAGE_CONFIGS。init_jp_embeddings/lab 对齐等复用 v2 实现。

def _patch_v1_with_v3():
    """把 v1 模块的关键符号替换为 v3 版本。返回 v1 模块对象。"""
    import train_jp_lora_cloud as v1

    # --- 替换 STAGE_CONFIGS ---
    v1.STAGE_CONFIGS = STAGE_CONFIGS_V3

    # --- 替换 init_jp_embeddings ---
    # v1 签名: init_jp_embeddings(model, mapping_path, phoneset_path)
    # v3 签名: init_jp_embeddings_v3(model, phoneset_path)  -- 忽略 mapping_path
    def init_jp_embeddings_wrapper(model, mapping_path=None, phoneset_path=None):
        return init_jp_embeddings_v3(model, phoneset_path or v1.PHONE_SET_PATH)
    v1.init_jp_embeddings = init_jp_embeddings_wrapper

    # --- 替换 apply_lora_and_setup_trainable ---
    # v1 签名: apply_lora_and_setup_trainable(model, stage)
    # v3 签名: apply_lora_and_setup_trainable_v3(model, stage, phoneset_path)
    def apply_lora_wrapper(model, stage):
        return apply_lora_and_setup_trainable_v3(model, stage, v1.PHONE_SET_PATH)
    v1.apply_lora_and_setup_trainable = apply_lora_wrapper

    # --- 替换 build_optimizer ---
    # 签名一致：(model, stage, lora_lr, embed_lr)
    v1.build_optimizer = build_optimizer_v3

    # --- 替换 lab_phonemes_to_notes / _gtsinger_word_to_tokens ---
    # v3 沿用 v2 的对齐与映射逻辑
    v1.lab_phonemes_to_notes = lab_phonemes_to_notes_v2
    def gtsinger_wrapper(word_entry, min_dur=0.12):
        return gtsinger_word_to_tokens_v2(word_entry)
    v1._gtsinger_word_to_tokens = gtsinger_wrapper

    return v1


def main():
    """v3 主入口：--demo 跑逻辑演示，否则 patch v1 后启动完整训练流水线。"""
    if '--demo' in sys.argv:
        _demo()
        return

    print('=' * 60)
    print('train_jp_lora_cloud_v3.py — 促音/拨音 + 复用音素微调版')
    print('=' * 60)
    print('策略：jp_cl/jp_n 用正常 lr（3e-4）训练，22 个复用英语音素')
    print(f'      用极小 lr（{REUSED_EMBED_LR}）微调，缓慢适应日语不破坏源语言。')
    print()

    v1 = _patch_v1_with_v3()
    v1.main()


if __name__ == '__main__':
    main()
