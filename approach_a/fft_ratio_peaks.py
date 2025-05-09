# Updated script: run_fft_to_csv.py

import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks

def compute_peaks_and_ratio(path, min_freq=50, height_frac=0.05):
    """
    Reads a WAV file, computes its FFT, finds the first two peaks,
    and returns (fund_freq, sec_freq, ratio).
    """
    fs, data = wavfile.read(path)
    data = data.astype(float)
    N = len(data)
    # compute half-spectrum magnitude
    spectrum = np.abs(fft(data))[:N//2] * (2.0 / N)
    freqs    = fftfreq(N, 1 / fs)[:N//2]

    # detect peaks above min_freq
    peaks, _ = find_peaks(
        spectrum,
        height=spectrum.max() * height_frac,
        distance=10
    )
    peaks = peaks[freqs[peaks] > min_freq]
    if len(peaks) < 2:
        return None, None, None

    # sort by frequency and take first two
    sorted_peaks = peaks[np.argsort(freqs[peaks])]
    fund_idx = sorted_peaks[0]
    sec_idx  = sorted_peaks[1]

    fund_freq = freqs[fund_idx]
    sec_freq  = freqs[sec_idx]
    ratio     = sec_freq / fund_freq

    return fund_freq, sec_freq, ratio

def build_dataframe(audio_dir):
    """
    Walk through Major/Minor subfolders, compute fundamental,
    second peak, and ratio for each file, and return a DataFrame.
    """
    records = []
    for label_str in ["Major", "Minor"]:
        folder = os.path.join(audio_dir, label_str)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith('.wav'):
                continue
            path = os.path.join(folder, fname)
            fund, sec, ratio = compute_peaks_and_ratio(path)
            if ratio is not None:
                records.append({
                    'filename': fname,
                    'label': label_str,
                    'fundamental_freq': fund,
                    'second_peak_freq': sec,
                    'ratio': ratio
                })
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    audio_dir = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
    df = build_dataframe(audio_dir)

    output_csv = os.path.join(os.path.dirname(audio_dir), 'fft_ratio_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"saved fft ratio data to {output_csv}")

