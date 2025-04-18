#!/usr/bin/env python3
"""
wav_spectrograms.py
-------------------
Create a frequency‑spectrogram for every .wav file under <INPUT_DIR>
and save each one as a .png inside <OUTPUT_DIR>, preserving sub‑folders.

Example
-------
python wav_spectrograms.py data/Audio_Files  spectrograms
# add --show if you also want to see the figures pop up
"""

from pathlib import Path
import argparse

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


def plot_and_save(audio, sr, save_path, title=None, show=False):
    """Generate and save (optionally show) a spectrogram image."""
    freqs, times, Sxx = spectrogram(audio, fs=sr, nperseg=1024, noverlap=768)
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(times, freqs, Sxx_db, shading="auto")
    plt.colorbar(label="dB")
    plt.ylim(0, sr / 2)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    if title:
        plt.title(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  ↳ saved {save_path.relative_to(save_path.parents[2])}")

    if show:
        plt.show()
    else:
        plt.close()


def process_wav(wav_path: Path, input_root: Path, output_root: Path, show: bool):
    """Read a .wav file and produce a spectrogram PNG under output_root."""
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:                        # stereo → mono
        data = data.mean(axis=1)
    rel = wav_path.relative_to(input_root).with_suffix(".png")
    out_png = output_root / rel
    plot_and_save(
        audio=data.astype(float),
        sr=sr,
        save_path=out_png,
        title=f"{wav_path.name}  (sr={sr} Hz)",
        show=show,
    )


def main():
    ap = argparse.ArgumentParser(
        description=("Generate spectrograms for every .wav in INPUT_DIR "
                     "and save them under OUTPUT_DIR (created if needed)."))
    ap.add_argument("input_dir", help="Root directory containing .wav files")
    ap.add_argument("output_dir", help="Destination directory for PNGs")
    ap.add_argument("--show", action="store_true",
                    help="Also display each spectrogram interactively")
    args = ap.parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not input_root.is_dir():
        ap.error(f"{input_root} is not a directory or does not exist.")

    wav_files = sorted(input_root.rglob("*.wav"))
    if not wav_files:
        ap.error(f"No .wav files found under {input_root}")

    print(f"Found {len(wav_files)} .wav file(s) under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for wav in wav_files:
        print(f"• {wav.relative_to(input_root)}")
        try:
            process_wav(wav, input_root, output_root, show=args.show)
        except Exception as e:
            print(f"  ! Skipped (could not process): {e}")


if __name__ == "__main__":
    main()
