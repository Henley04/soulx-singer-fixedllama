"""Compare synthesized audio against real PJS reference.

Analyzes formant structure, spectral envelope, and F0 contour to
determine if the synthesized Japanese speech is intelligible.
"""
import os
import sys
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def analyze_audio(path, label):
    """Compute key audio features."""
    if not os.path.exists(path):
        print(f"  [{label}] MISSING: {path}")
        return None

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.abs(audio).max()

    # Spectral analysis via FFT on middle segment
    mid = len(audio) // 2
    win = audio[max(0, mid - 2048):mid + 2048]
    if len(win) < 256:
        print(f"  [{label}] Too short: {duration:.2f}s")
        return None

    fft = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1 / sr)

    # Energy bands
    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return np.sqrt(np.sum(fft[mask] ** 2)) if mask.any() else 0

    e_low = band_energy(0, 500)      # F0 region
    e_mid = band_energy(500, 2000)   # F1/F2 (vowel identity)
    e_high = band_energy(2000, 5000) # F2/F3 (consonant clarity)

    # Spectral centroid (brightness)
    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)

    # Zero crossing rate (voicing indicator)
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

    print(f"  [{label}] dur={duration:.2f}s rms={rms:.4f} peak={peak:.4f} "
          f"centroid={centroid:.0f}Hz zcr={zcr:.3f} "
          f"E[0-500]={e_low:.1f} E[500-2k]={e_mid:.1f} E[2k-5k]={e_high:.1f}")

    return {
        'duration': duration, 'rms': rms, 'peak': peak,
        'centroid': centroid, 'zcr': zcr,
        'e_low': e_low, 'e_mid': e_mid, 'e_high': e_high,
    }


def main():
    print("=" * 70)
    print("Audio Quality Analysis: synthesized vs reference")
    print("=" * 70)

    # 1. Real PJS reference (ground truth Japanese)
    print("\n--- Real PJS reference (Japanese ground truth) ---")
    for i in range(3):
        ref_path = f"train/lora_jp/dataset/wavs/pjs00{i+1}_song.wav"
        analyze_audio(ref_path, f"pjs00{i+1}")

    # 2. Synthesized test sentences
    print("\n--- Synthesized test sentences (JP LoRA stage3) ---")
    test_names = [
        "sa-ku-ra",
        "a-i-u-e-o",
        "ka-ki-ku-ke-ko",
        "sa-ku-ra-pause",
        "ha-ru",
    ]
    synth_results = []
    for i, name in enumerate(test_names, 1):
        path = f"outputs/lora_jp/synthesis/test_{i:02d}.wav"
        r = analyze_audio(path, name)
        if r:
            synth_results.append((name, r))

    # 3. Comparison
    print("\n--- Comparison ---")
    if synth_results:
        # Average synthesized vs reference
        ref_path = "train/lora_jp/dataset/wavs/pjs001_song.wav"
        ref = analyze_audio(ref_path, "ref")

        if ref:
            print(f"\n  Reference centroid: {ref['centroid']:.0f}Hz")
            print(f"  Synth centroids:    {[r['centroid'] for _, r in synth_results]}")
            print(f"\n  Reference RMS:      {ref['rms']:.4f}")
            synth_rms = [f"{r['rms']:.4f}" for _, r in synth_results]
            print(f"  Synth RMS:          {synth_rms}")
            print(f"\n  Reference ZCR:      {ref['zcr']:.3f}")
            synth_zcr = [f"{r['zcr']:.3f}" for _, r in synth_results]
            print(f"  Synth ZCR:          {synth_zcr}")

            # Intelligibility heuristics
            print(f"\n  --- Intelligibility heuristics ---")
            synth_centroids = [r['centroid'] for _, r in synth_results]
            ref_centroid = ref['centroid']
            centroid_ratio = np.mean(synth_centroids) / ref_centroid
            print(f"  Centroid ratio (synth/ref): {centroid_ratio:.2f}")
            if centroid_ratio < 0.7:
                print(f"    WARNING: Synth too muffled (centroid << ref)")
            elif centroid_ratio > 1.3:
                print(f"    WARNING: Synth too bright (centroid >> ref)")
            else:
                print(f"    OK: Spectral brightness matches reference")

            # Check if all synth sentences sound the same (collapse)
            rms_std = np.std([r['rms'] for _, r in synth_results])
            rms_mean = np.mean([r['rms'] for _, r in synth_results])
            print(f"  RMS variation across sentences: {rms_std/rms_mean*100:.1f}%")
            if rms_std / rms_mean < 0.05:
                print(f"    WARNING: All sentences have nearly identical energy — possible collapse")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
