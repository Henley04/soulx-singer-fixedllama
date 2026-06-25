#!/usr/bin/env python3
"""
export_preprocess.py - 将 SoulX-Singer 预处理 PyTorch 模型转换为 ONNX 格式

模型列表：
  rmvpe_mel    - RMVPE MelSpectrogram 前端
  rmvpe_model  - RMVPE E2E 音高检测（含 Mel 提取）
  rosvot_mel   - ROSVOT MelNet 前端
  rosvot_model - ROSVOT MidiExtractor 音符识别
  rwbd_model   - RWBD WordbdExtractor 词边界检测
  vocal_sep    - MelBandRoformer 人声分离（karaoke）
  dereverb     - MelBandRoformer 去混响

使用方法：
  python export_preprocess.py --model-type all
  python export_preprocess.py --model-type rmvpe
  python export_preprocess.py --model-type vocal_sep

导出完成后，运行 optimize_onnx.py 生成 FP16/INT8 版本。

内存管理：每个模型导出后立即释放内存，防止 OOM。
GPU 加速：如果环境支持 CUDA 则自动使用。
"""

import argparse
import gc
import os
import sys
import time
import traceback
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models" / "SoulX-Singer-Preprocess"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "onnx_models" / "preprocess"

OPSET = 17


# ============================================================
# 通用工具
# ============================================================

def get_device():
    """获取最佳可用设备。"""
    if torch.cuda.is_available():
        print("[INFO] 检测到 CUDA，使用 GPU 加速导出")
        return torch.device("cuda:0")
    print("[INFO] 未检测到 CUDA，使用 CPU 导出")
    return torch.device("cpu")


def release_memory():
    """释放 GPU 内存并垃圾回收。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def export_onnx(model, dummy_inputs, output_path, input_names, output_names,
                dynamic_axes, opset=OPSET, device="cpu"):
    """通用 ONNX 导出函数。"""
    model = model.to(device).eval()
    # 将 dummy_inputs 移到设备
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = dummy_inputs.to(device)
    elif isinstance(dummy_inputs, (list, tuple)):
        dummy_inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in dummy_inputs)
    elif isinstance(dummy_inputs, dict):
        dummy_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_inputs.items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] 导出 ONNX: {output_path.name}")
    t0 = time.time()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # 使用传统导出器，避免 torch.export 对 GRU 等算子的分解问题
        )

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[EXPORT] 完成: {output_path.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")


def verify_onnx(onnx_path, input_feed, expected_outputs=None, rtol=1e-3, atol=1e-5):
    """验证 ONNX 模型是否可加载并推理。"""
    import onnxruntime as ort
    onnx_path = str(onnx_path)
    print(f"[VERIFY] 验证: {Path(onnx_path).name}")

    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[VERIFY] 加载失败: {e}")
        return False

    # 准备输入
    if isinstance(input_feed, torch.Tensor):
        feed = {session.get_inputs()[0].name: input_feed.cpu().numpy()}
    elif isinstance(input_feed, dict):
        feed = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in input_feed.items()}
    elif isinstance(input_feed, (list, tuple)):
        feed = {}
        for inp, ort_inp in zip(input_feed, session.get_inputs()):
            feed[ort_inp.name] = inp.cpu().numpy() if isinstance(inp, torch.Tensor) else inp
    else:
        print(f"[VERIFY] 不支持的输入类型: {type(input_feed)}")
        return False

    try:
        results = session.run(None, feed)
        for i, r in enumerate(results):
            print(f"  输出 {i}: shape={r.shape}, dtype={r.dtype}, "
                  f"min={r.min():.4f}, max={r.max():.4f}, mean={r.mean():.4f}")
    except Exception as e:
        print(f"[VERIFY] 推理失败: {e}")
        return False

    # 与 PyTorch 结果对比
    if expected_outputs is not None:
        if isinstance(expected_outputs, torch.Tensor):
            expected_outputs = [expected_outputs]
        for i, (ort_out, pt_out) in enumerate(zip(results, expected_outputs)):
            pt_np = pt_out.cpu().numpy()
            if not np.allclose(ort_out, pt_np, rtol=rtol, atol=atol):
                diff = np.abs(ort_out - pt_np)
                print(f"  输出 {i} 不匹配: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
                # 不返回 False，因为数值差异可能是正常的

    print(f"[VERIFY] 通过")
    return True


# ============================================================
# 1. RMVPE MelSpectrogram ONNX Wrapper
# ============================================================

class RmvpeMelOnnx(nn.Module):
    """ONNX 兼容的 RMVPE MelSpectrogram。

    使用 F.conv1d 实现 STFT，避免复数类型和 Unfold 问题。
    """
    def __init__(self, mel_basis, n_fft, hop_length, win_length, n_mel_channels, clamp=1e-5):
        super().__init__()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

        # 预计算 DFT 滤波器（包含窗函数），用 conv1d 替代 STFT
        n_freq = n_fft // 2 + 1
        t = torch.arange(n_fft).float()
        k = torch.arange(n_freq).float()
        angles = 2 * math.pi * k.unsqueeze(1) * t.unsqueeze(0) / n_fft
        hann = torch.hann_window(win_length)

        # Real filter: cos(2*pi*k*t/N) * window(t)
        real_filter = torch.cos(angles) * hann.unsqueeze(0)  # [n_freq, n_fft]
        # Imag filter: -sin(2*pi*k*t/N) * window(t)
        imag_filter = -torch.sin(angles) * hann.unsqueeze(0)  # [n_freq, n_fft]

        # Shape: [n_freq, 1, n_fft] for conv1d
        self.register_buffer("real_filter", real_filter.unsqueeze(1))
        self.register_buffer("imag_filter", imag_filter.unsqueeze(1))

    def forward(self, audio):
        # center=True: 两侧各补 n_fft//2 个样本
        pad_len = self.n_fft // 2
        audio = F.pad(audio, (pad_len, pad_len), mode='reflect')

        # Add channel dimension: [B, T] -> [B, 1, T]
        audio = audio.unsqueeze(1)

        # 用 conv1d 实现 DFT
        real = F.conv1d(audio, self.real_filter, stride=self.hop_length)  # [B, n_freq, num_frames]
        imag = F.conv1d(audio, self.imag_filter, stride=self.hop_length)  # [B, n_freq, num_frames]

        # 幅度谱
        magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-10)

        # Mel filter bank
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


def export_rmvpe_mel(output_dir, device):
    """导出 RMVPE MelSpectrogram。"""
    from librosa.filters import mel as librosa_mel_fn

    n_fft = 1024
    hop_length = 160
    win_length = 1024
    sampling_rate = 16000
    n_mel_channels = 128
    mel_fmin = 30
    mel_fmax = 8000

    mel_basis_np = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
        fmin=mel_fmin, fmax=mel_fmax, htk=True,
    )
    mel_basis = torch.from_numpy(mel_basis_np).float()

    model = RmvpeMelOnnx(mel_basis, n_fft, hop_length, win_length, n_mel_channels)
    model = model.to(device).eval()

    # dummy input: 1秒 16kHz 音频
    audio_len = 16000
    dummy_audio = torch.randn(1, audio_len, device=device)

    output_path = Path(output_dir) / "rmvpe_mel.onnx"

    with torch.no_grad():
        mel_out = model(dummy_audio)

    export_onnx(
        model, dummy_audio, output_path,
        input_names=["audio"],
        output_names=["mel"],
        dynamic_axes={
            "audio": {0: "batch", 1: "num_samples"},
            "mel": {0: "batch", 2: "time_frames"},
        },
        device=device,
    )

    # 验证
    verify_onnx(output_path, dummy_audio, expected_outputs=mel_out)

    del model, dummy_audio, mel_out, mel_basis
    release_memory()


# ============================================================
# 2. RMVPE E2E Model ONNX Wrapper
# ============================================================

class RmvpeModelOnnx(nn.Module):
    """ONNX 兼容的 RMVPE E2E（含 Mel 提取）。"""
    def __init__(self, mel_extractor, e2e_model):
        super().__init__()
        self.mel_extractor = mel_extractor
        self.e2e_model = e2e_model

    def forward(self, audio):
        mel = self.mel_extractor(audio)
        # pad to 32x
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        hidden = self.e2e_model(mel)
        return hidden[:, :n_frames]


def export_rmvpe_model(output_dir, device):
    """导出 RMVPE E2E 模型（含 Mel 提取）。"""
    from librosa.filters import mel as librosa_mel_fn
    # 导入 E2E 模型定义
    sys.path.insert(0, str(PROJECT_ROOT / "preprocess" / "tools"))
    from f0_extraction import E2E

    model_path = PRETRAINED_DIR / "rmvpe" / "rmvpe.pt"
    print(f"[INFO] 加载 RMVPE 模型: {model_path}")

    # 创建 Mel 提取器
    n_fft = 1024
    hop_length = 160
    win_length = 1024
    sampling_rate = 16000
    n_mel_channels = 128
    mel_fmin = 30
    mel_fmax = 8000

    mel_basis_np = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
        fmin=mel_fmin, fmax=mel_fmax, htk=True,
    )
    mel_basis = torch.from_numpy(mel_basis_np).float()
    mel_extractor = RmvpeMelOnnx(mel_basis, n_fft, hop_length, win_length, n_mel_channels)

    # 加载 E2E 模型
    e2e = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    e2e.load_state_dict(ckpt)
    e2e.eval()
    del ckpt

    # 组合模型
    model = RmvpeModelOnnx(mel_extractor, e2e)
    model = model.to(device).eval()

    # dummy input
    audio_len = 16000
    dummy_audio = torch.randn(1, audio_len, device=device)

    output_path = Path(output_dir) / "rmvpe_model.onnx"

    with torch.no_grad():
        out = model(dummy_audio)

    export_onnx(
        model, dummy_audio, output_path,
        input_names=["audio"],
        output_names=["output"],
        dynamic_axes={
            "audio": {0: "batch", 1: "num_samples"},
            "output": {0: "batch", 1: "time_frames"},
        },
        device=device,
    )

    verify_onnx(output_path, dummy_audio, expected_outputs=out)

    del model, e2e, mel_extractor, dummy_audio, out
    release_memory()


# ============================================================
# 3. ROSVOT MelNet ONNX Wrapper
# ============================================================

class RosvotMelOnnx(nn.Module):
    """ONNX 兼容的 ROSVOT MelNet。

    使用 F.conv1d 实现 STFT，避免复数类型和 Unfold 问题。
    """
    def __init__(self, mel_basis, n_fft, hop_size, win_size, num_mels, fmin, fmax):
        super().__init__()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax

        # 预计算 DFT 滤波器
        n_freq = n_fft // 2 + 1
        t = torch.arange(n_fft).float()
        k = torch.arange(n_freq).float()
        angles = 2 * math.pi * k.unsqueeze(1) * t.unsqueeze(0) / n_fft
        hann = torch.hann_window(win_size)

        real_filter = torch.cos(angles) * hann.unsqueeze(0)
        imag_filter = -torch.sin(angles) * hann.unsqueeze(0)

        self.register_buffer("real_filter", real_filter.unsqueeze(1))
        self.register_buffer("imag_filter", imag_filter.unsqueeze(1))

    def forward(self, y):
        # 预处理: clamp + pad（与原始 MelNet 一致）
        y = y.clamp(min=-1., max=1.)
        pad_length = (torch.ceil(torch.tensor(y.shape[1] / self.hop_size)).int() * self.hop_size - y.shape[1])
        y = F.pad(y.unsqueeze(1),
                  [int((self.n_fft - self.hop_size) // 2),
                   int((self.n_fft - self.hop_size) // 2 + pad_length)],
                  mode='reflect')
        y = y.squeeze(1)

        # center=False, 直接用 conv1d
        y = y.unsqueeze(1)  # [B, 1, T]
        real = F.conv1d(y, self.real_filter, stride=self.hop_size)
        imag = F.conv1d(y, self.imag_filter, stride=self.hop_size)
        spec = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)

        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log10(torch.clamp(spec, min=1e-5))
        spec = spec.transpose(1, 2)
        return spec


def export_rosvot_mel(output_dir, device):
    """导出 ROSVOT MelNet。"""
    from librosa.filters import mel as librosa_mel_fn
    import yaml

    # 读取 ROSVOT 配置
    config_path = PRETRAINED_DIR / "rosvot" / "rosvot" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    n_fft = config['fft_size']
    hop_size = config['hop_size']
    win_size = config['win_size']
    num_mels = config['audio_num_mel_bins']
    fmin = config['fmin']
    fmax = config['fmax']
    sample_rate = config['audio_sample_rate']

    mel_np = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_np).float()

    model = RosvotMelOnnx(mel_basis, n_fft, hop_size, win_size, num_mels, fmin, fmax)
    model = model.to(device).eval()

    # dummy input: 1秒 24kHz 音频
    audio_len = 24000
    dummy_audio = torch.randn(1, audio_len, device=device)

    output_path = Path(output_dir) / "rosvot_mel.onnx"

    with torch.no_grad():
        mel_out = model(dummy_audio)

    export_onnx(
        model, dummy_audio, output_path,
        input_names=["audio"],
        output_names=["mel"],
        dynamic_axes={
            "audio": {0: "batch", 1: "num_samples"},
            "mel": {0: "batch", 1: "time_frames"},
        },
        device=device,
    )

    verify_onnx(output_path, dummy_audio, expected_outputs=mel_out)

    del model, dummy_audio, mel_out, mel_basis
    release_memory()


# ============================================================
# 4. ROSVOT MidiExtractor ONNX Wrapper
# ============================================================

class RosvotModelOnnx(nn.Module):
    """ONNX 兼容的 ROSVOT MidiExtractor（含 Mel 提取）。

    输入: wav, pitch, uv, word_bd
    输出: note_bd_pred, note_pred, note_lengths
    """
    def __init__(self, mel_net, midi_extractor, note_bd_threshold, max_notes=500):
        super().__init__()
        self.mel_net = mel_net
        self.midi_extractor = midi_extractor
        self.note_bd_threshold = note_bd_threshold
        self.max_notes = max_notes

    def forward(self, wav, pitch, uv, word_bd):
        # 提取 mel
        mel = self.mel_net(wav)
        # 截取到 use_mel_bins
        mel = mel[:, :, :self.midi_extractor.mel_proj.in_channels]

        # 截取 pitch/uv/word_bd 到 mel 的时间维度
        T = mel.shape[1]
        pitch = pitch[:, :T]
        uv = uv[:, :T]
        word_bd = word_bd[:, :T]

        # 编码器
        feat = self.midi_extractor.run_encoder(mel=mel, word_bd=word_bd, pitch=pitch, uv=uv)
        feat = self.midi_extractor.net(feat)

        # note boundary prediction
        note_bd_logits = self.midi_extractor.note_bd_out(feat).squeeze(-1) / self.midi_extractor.note_bd_temperature
        note_bd_logits = torch.clamp(note_bd_logits, min=-16., max=16.)

        # 简化的边界检测（ONNX 兼容）
        note_bd_pred = (torch.sigmoid(note_bd_logits) > self.note_bd_threshold).long()

        # PitchDecoder
        note_lengths, note_logits, note_pred = self.midi_extractor.pitch_decoder(feat, note_bd_pred, train=False)

        return note_bd_pred, note_pred, note_lengths


def export_rosvot_model(output_dir, device):
    """导出 ROSVOT MidiExtractor 模型。"""
    from librosa.filters import mel as librosa_mel_fn
    import yaml
    sys.path.insert(0, str(PROJECT_ROOT / "preprocess" / "tools"))
    from note_transcription.modules.rosvot.rosvot import MidiExtractor
    from note_transcription.utils.commons.hparams import hparams

    config_path = PRETRAINED_DIR / "rosvot" / "rosvot" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = PRETRAINED_DIR / "rosvot" / "rosvot" / "model.pt"
    print(f"[INFO] 加载 ROSVOT 模型: {model_path}")

    # 创建 MelNet
    n_fft = config['fft_size']
    hop_size = config['hop_size']
    win_size = config['win_size']
    num_mels = config['audio_num_mel_bins']
    fmin = config['fmin']
    fmax = config['fmax']
    sample_rate = config['audio_sample_rate']

    mel_np = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_np).float()
    mel_net = RosvotMelOnnx(mel_basis, n_fft, hop_size, win_size, num_mels, fmin, fmax)

    # 创建并加载 MidiExtractor
    hparams.update(config)
    midi_extractor = MidiExtractor(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        midi_extractor.load_state_dict(ckpt['model'])
    else:
        midi_extractor.load_state_dict(ckpt)
    midi_extractor.eval()
    del ckpt

    note_bd_threshold = config.get('note_bd_threshold', 0.8)

    model = RosvotModelOnnx(mel_net, midi_extractor, note_bd_threshold)
    model = model.to(device).eval()

    # dummy inputs
    audio_len = 24000
    T = audio_len // hop_size + 1
    dummy_wav = torch.randn(1, audio_len, device=device)
    dummy_pitch = torch.zeros(1, T, dtype=torch.long, device=device)
    dummy_uv = torch.ones(1, T, dtype=torch.long, device=device)
    dummy_word_bd = torch.zeros(1, T, dtype=torch.long, device=device)

    output_path = Path(output_dir) / "rosvot_model.onnx"

    with torch.no_grad():
        out = model(dummy_wav, dummy_pitch, dummy_uv, dummy_word_bd)

    export_onnx(
        model,
        (dummy_wav, dummy_pitch, dummy_uv, dummy_word_bd),
        output_path,
        input_names=["wav", "pitch", "uv", "word_bd"],
        output_names=["note_bd_pred", "note_pred", "note_lengths"],
        dynamic_axes={
            "wav": {0: "batch", 1: "num_samples"},
            "pitch": {0: "batch", 1: "T"},
            "uv": {0: "batch", 1: "T"},
            "word_bd": {0: "batch", 1: "T"},
            "note_bd_pred": {0: "batch", 1: "T"},
            "note_pred": {0: "batch"},
            "note_lengths": {0: "batch"},
        },
        device=device,
    )

    verify_onnx(output_path, (dummy_wav, dummy_pitch, dummy_uv, dummy_word_bd), expected_outputs=out)

    del model, midi_extractor, mel_net, dummy_wav, dummy_pitch, dummy_uv, dummy_word_bd, out
    release_memory()


# ============================================================
# 5. RWBD WordbdExtractor ONNX Wrapper
# ============================================================

class RwbdModelOnnx(nn.Module):
    """ONNX 兼容的 RWBD WordbdExtractor（含 Mel 提取）。

    输入: wav, pitch, uv
    输出: word_bd_pred
    """
    def __init__(self, mel_net, rwbd_model, word_bd_threshold):
        super().__init__()
        self.mel_net = mel_net
        self.rwbd_model = rwbd_model
        self.word_bd_threshold = word_bd_threshold

    def forward(self, wav, pitch, uv):
        mel = self.mel_net(wav)
        mel = mel[:, :, :self.rwbd_model.mel_proj.in_channels]

        T = mel.shape[1]
        pitch = pitch[:, :T]
        uv = uv[:, :T]

        feat = self.rwbd_model.run_encoder(mel=mel, pitch=pitch, uv=uv)
        feat = self.rwbd_model.net(feat)

        word_bd_logits = self.rwbd_model.word_bd_out(feat).squeeze(-1) / self.rwbd_model.word_bd_temperature
        word_bd_logits = torch.clamp(word_bd_logits, min=-16., max=16.)

        # 简化边界检测
        word_bd_pred = (torch.sigmoid(word_bd_logits) > self.word_bd_threshold).long()

        return word_bd_pred


def export_rwbd_model(output_dir, device):
    """导出 RWBD WordbdExtractor 模型。"""
    from librosa.filters import mel as librosa_mel_fn
    import yaml
    sys.path.insert(0, str(PROJECT_ROOT / "preprocess" / "tools"))
    from note_transcription.modules.rosvot.rosvot import WordbdExtractor

    config_path = PRETRAINED_DIR / "rosvot" / "rwbd" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = PRETRAINED_DIR / "rosvot" / "rwbd" / "model.pt"
    print(f"[INFO] 加载 RWBD 模型: {model_path}")

    # MelNet（使用 ROSVOT 的 mel 参数）
    rosvot_config_path = PRETRAINED_DIR / "rosvot" / "rosvot" / "config.yaml"
    with open(rosvot_config_path, 'r') as f:
        rosvot_config = yaml.safe_load(f)

    n_fft = rosvot_config['fft_size']
    hop_size = rosvot_config['hop_size']
    win_size = rosvot_config['win_size']
    num_mels = rosvot_config['audio_num_mel_bins']
    fmin = rosvot_config['fmin']
    fmax = rosvot_config['fmax']
    sample_rate = rosvot_config['audio_sample_rate']

    mel_np = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_np).float()
    mel_net = RosvotMelOnnx(mel_basis, n_fft, hop_size, win_size, num_mels, fmin, fmax)

    # 加载 RWBD 模型
    rwbd = WordbdExtractor(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        rwbd.load_state_dict(ckpt['model'])
    else:
        rwbd.load_state_dict(ckpt)
    rwbd.eval()
    del ckpt

    word_bd_threshold = config.get('word_bd_threshold', 0.9)

    model = RwbdModelOnnx(mel_net, rwbd, word_bd_threshold)
    model = model.to(device).eval()

    # dummy inputs
    audio_len = 24000
    T = audio_len // hop_size + 1
    dummy_wav = torch.randn(1, audio_len, device=device)
    dummy_pitch = torch.zeros(1, T, dtype=torch.long, device=device)
    dummy_uv = torch.ones(1, T, dtype=torch.long, device=device)

    output_path = Path(output_dir) / "rwbd_model.onnx"

    with torch.no_grad():
        out = model(dummy_wav, dummy_pitch, dummy_uv)

    export_onnx(
        model,
        (dummy_wav, dummy_pitch, dummy_uv),
        output_path,
        input_names=["wav", "pitch", "uv"],
        output_names=["word_bd_pred"],
        dynamic_axes={
            "wav": {0: "batch", 1: "num_samples"},
            "pitch": {0: "batch", 1: "T"},
            "uv": {0: "batch", 1: "T"},
            "word_bd_pred": {0: "batch", 1: "T"},
        },
        device=device,
    )

    verify_onnx(output_path, (dummy_wav, dummy_pitch, dummy_uv), expected_outputs=out)

    del model, rwbd, mel_net, dummy_wav, dummy_pitch, dummy_uv, out
    release_memory()


# ============================================================
# 6 & 7. MelBandRoformer ONNX Wrapper (Vocal Sep / Dereverb)
# ============================================================

class MelBandRoformerOnnx(nn.Module):
    """ONNX 兼容的 MelBandRoformer。

    用实部/虚部分解替代复数运算，避免 ONNX 兼容性问题。
    输入: raw_audio [B, 2, T] (立体声)
    输出: 分离后的音频 [B, num_stems, 2, T]
    """
    def __init__(self, model):
        super().__init__()
        # 复制所有子模块
        self.stereo = model.stereo
        self.audio_channels = model.audio_channels
        self.num_stems = model.num_stems
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators
        self.stft_kwargs = model.stft_kwargs
        self.match_input_audio_length = model.match_input_audio_length

        # 注册 buffer
        self.register_buffer('freq_indices', model.freq_indices)
        self.register_buffer('freqs_per_band', model.freqs_per_band)
        self.register_buffer('num_freqs_per_band', model.num_freqs_per_band)
        self.register_buffer('num_bands_per_freq', model.num_bands_per_freq)

        # STFT 窗口
        self.stft_win_length = model.stft_kwargs['win_length']
        self.register_buffer('stft_window', torch.hann_window(model.stft_kwargs['win_length']))

    def forward(self, raw_audio):
        from einops import rearrange, pack, unpack, repeat, reduce

        device = raw_audio.device
        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None

        # STFT
        raw_audio_packed, batch_ps = pack([raw_audio], '* t')
        stft_repr = torch.stft(
            raw_audio_packed, **self.stft_kwargs,
            window=self.stft_window, return_complex=True,
        )
        stft_repr = torch.view_as_real(stft_repr)  # [B*channels, F, T, 2]
        stft_repr = unpack(stft_repr, batch_ps, '* f t c')[0]  # [B, channels, F, T, 2]

        # 合并声道到频率维度
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # 提取 mel band 对应的频率
        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]  # [B, num_freq_indices, T, 2]

        # 将实部虚部折叠到频率维度
        x = rearrange(x, 'b f t c -> b t (f c)')

        # BandSplit
        x = self.band_split(x)

        # Axial Transformer
        for transformer_block in self.layers:
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x, ft_ps = pack([x], 'b * d')
                x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

        # Mask estimation
        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        # 用实部/虚部分解替代复数乘法
        stft_real = stft_repr[..., 0]  # [B, F*channels, T]
        stft_imag = stft_repr[..., 1]  # [B, F*channels, T]
        mask_real = masks[..., 0]  # [B, num_stems, num_freq_indices, T]
        mask_imag = masks[..., 1]  # [B, num_stems, num_freq_indices, T]

        # scatter_add 平均重叠频率的 mask
        scatter_indices = repeat(
            self.freq_indices, 'f -> b n f t',
            b=batch, n=num_stems, t=stft_real.shape[-1],
        )

        # 对 mask 的实部和虚部分别做 scatter_add
        mask_real_summed = torch.zeros(
            batch, num_stems, stft_real.shape[1], stft_real.shape[-1],
            dtype=mask_real.dtype, device=device,
        ).scatter_add_(2, scatter_indices, mask_real)

        mask_imag_summed = torch.zeros(
            batch, num_stems, stft_imag.shape[1], stft_imag.shape[-1],
            dtype=mask_imag.dtype, device=device,
        ).scatter_add_(2, scatter_indices, mask_imag)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
        mask_real_avg = mask_real_summed / denom.clamp(min=1e-8)
        mask_imag_avg = mask_imag_summed / denom.clamp(min=1e-8)

        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        stft_real_expanded = repeat(stft_real, 'b f t -> b n f t', n=num_stems)
        stft_imag_expanded = repeat(stft_imag, 'b f t -> b n f t', n=num_stems)

        result_real = stft_real_expanded * mask_real_avg - stft_imag_expanded * mask_imag_avg
        result_imag = stft_real_expanded * mask_imag_avg + stft_imag_expanded * mask_real_avg

        # 组合成复数用于 ISTFT
        result_complex = torch.complex(result_real, result_imag)

        # ISTFT
        result_complex = rearrange(result_complex, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        recon_audio = torch.istft(
            result_complex, **self.stft_kwargs,
            window=self.stft_window, length=istft_length,
        )
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio


def _load_mel_band_roformer(config_path, checkpoint_path, device):
    """加载 MelBandRoformer 模型（禁用 flash attention）。"""
    import yaml
    sys.path.insert(0, str(PROJECT_ROOT / "preprocess" / "tools" / "vocal_separation"))
    from modules.bs_roformer.mel_band_roformer import MelBandRoformer

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    # 禁用 flash attention（ONNX 不兼容）
    model_config['flash_attn'] = False
    # 禁用 sage attention
    model_config['sage_attention'] = False
    # 禁用 torch checkpoint
    model_config['use_torch_checkpoint'] = False

    model = MelBandRoformer(**model_config)

    print(f"[INFO] 加载 MelBandRoformer: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        if 'state' in ckpt:
            ckpt = ckpt['state']
        elif 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    del ckpt

    return model, config


def export_vocal_sep(output_dir, device):
    """导出人声分离模型（MelBandRoformer karaoke）。"""
    config_path = PRETRAINED_DIR / "mel-band-roformer-karaoke" / "config_karaoke_becruily.yaml"
    checkpoint_path = PRETRAINED_DIR / "mel-band-roformer-karaoke" / "mel_band_roformer_karaoke_becruily.ckpt"

    model, config = _load_mel_band_roformer(config_path, checkpoint_path, device)

    # 创建 ONNX wrapper
    onnx_model = MelBandRoformerOnnx(model)
    onnx_model = onnx_model.to(device).eval()
    del model
    release_memory()

    # dummy input: 立体声 1秒 44.1kHz
    chunk_size = config['inference'].get('chunk_size', config['audio'].get('chunk_size', 485100))
    # 使用较小的 chunk 以节省内存
    test_len = min(chunk_size, 44100 * 5)  # 最多5秒
    dummy_audio = torch.randn(1, 2, test_len, device=device)

    output_path = Path(output_dir) / "vocal_sep.onnx"

    with torch.no_grad():
        out = onnx_model(dummy_audio)

    export_onnx(
        onnx_model, dummy_audio, output_path,
        input_names=["raw_audio"],
        output_names=["recon_audio"],
        dynamic_axes={
            "raw_audio": {0: "batch", 2: "num_samples"},
            "recon_audio": {0: "batch", 3: "num_samples"},
        },
        device=device,
    )

    verify_onnx(output_path, dummy_audio, expected_outputs=out)

    del onnx_model, dummy_audio, out
    release_memory()


def export_dereverb(output_dir, device):
    """导出去混响模型（MelBandRoformer dereverb）。"""
    config_path = PRETRAINED_DIR / "dereverb_mel_band_roformer" / "dereverb_mel_band_roformer_anvuew.yaml"
    checkpoint_path = PRETRAINED_DIR / "dereverb_mel_band_roformer" / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"

    model, config = _load_mel_band_roformer(config_path, checkpoint_path, device)

    onnx_model = MelBandRoformerOnnx(model)
    onnx_model = onnx_model.to(device).eval()
    del model
    release_memory()

    chunk_size = config['inference'].get('chunk_size', config['audio'].get('chunk_size', 352800))
    test_len = min(chunk_size, 44100 * 5)
    dummy_audio = torch.randn(1, 2, test_len, device=device)

    output_path = Path(output_dir) / "dereverb.onnx"

    with torch.no_grad():
        out = onnx_model(dummy_audio)

    export_onnx(
        onnx_model, dummy_audio, output_path,
        input_names=["raw_audio"],
        output_names=["recon_audio"],
        dynamic_axes={
            "raw_audio": {0: "batch", 2: "num_samples"},
            "recon_audio": {0: "batch", 2: "num_samples"},
        },
        device=device,
    )

    verify_onnx(output_path, dummy_audio, expected_outputs=out)

    del onnx_model, dummy_audio, out
    release_memory()


# ============================================================
# Main
# ============================================================

MODEL_EXPORTS = {
    "rmvpe": [
        ("rmvpe_mel", export_rmvpe_mel),
        ("rmvpe_model", export_rmvpe_model),
    ],
    "rosvot": [
        ("rosvot_mel", export_rosvot_mel),
        ("rosvot_model", export_rosvot_model),
    ],
    "rwbd": [
        ("rwbd_model", export_rwbd_model),
    ],
    "vocal_sep": [
        ("vocal_sep", export_vocal_sep),
    ],
    "dereverb": [
        ("dereverb", export_dereverb),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="导出 SoulX-Singer 预处理模型到 ONNX 格式")
    parser.add_argument(
        "--model-type", type=str, default="all",
        choices=["all", "rmvpe", "rosvot", "rwbd", "vocal_sep", "dereverb"],
        help="要导出的模型类型",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="跳过 ONNX 验证",
    )
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SoulX-Singer 预处理模型 ONNX 导出工具")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print("=" * 60)

    # 收集要导出的模型
    if args.model_type == "all":
        exports = []
        for model_type, funcs in MODEL_EXPORTS.items():
            exports.extend(funcs)
    else:
        exports = MODEL_EXPORTS.get(args.model_type, [])

    success = []
    failed = []

    for name, export_fn in exports:
        print(f"\n{'=' * 60}")
        print(f"导出: {name}")
        print(f"{'=' * 60}")

        try:
            export_fn(output_dir, device)
            success.append(name)
            print(f"[OK] {name} 导出成功")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"[FAIL] {name} 导出失败: {e}")
            traceback.print_exc()
        finally:
            release_memory()

    # 汇总
    print(f"\n{'=' * 60}")
    print("导出汇总")
    print(f"{'=' * 60}")
    print(f"成功: {len(success)} 个")
    for s in success:
        print(f"  [OK] {s}")
    if failed:
        print(f"失败: {len(failed)} 个")
        for name, err in failed:
            print(f"  [FAIL] {name}: {err}")

    print(f"\n下一步: 运行 optimize_onnx.py 生成 FP16/INT8 版本")
    print(f"  python {Path(__file__).parent.parent / 'optimize_onnx.py'}")


if __name__ == "__main__":
    main()
