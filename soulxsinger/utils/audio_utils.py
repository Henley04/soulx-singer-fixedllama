import torch
import torchaudio


def load_wav(wav_path: str, sample_rate: int):
    """Load wav file and resample to target sample rate.

    Args:
        wav_path (str): Path to wav file.
        sample_rate (int): Target sample rate.

    Returns:
        torch.Tensor: Waveform tensor with shape (1, T).
    """
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception:
        # Fallback to soundfile if torchaudio backend fails
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        waveform = torch.from_numpy(data).unsqueeze(0)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform
