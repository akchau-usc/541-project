import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

'''
find_harmonics --> reads one .wav file and returns the frequencies of its spectral peaks.
computes the positive magnitude spectrum and corresponding freqs + detects large peaks
and discards any small peaks (to ignore noise)
'''
def find_harmonics(path, min_freq=50, height_frac=0.05):
    fs, data = wavfile.read(path)
    data = data.astype(float)
    N = len(data)
    spectrum = np.abs(fft(data))[:N//2] * (2.0/N)
    freqs    = fftfreq(N, 1/fs)[:N//2]
    peaks, _ = find_peaks(
        spectrum,
        height=spectrum.max()*height_frac,
        distance=10
    )
    peaks = peaks[freqs[peaks] > min_freq]
    return freqs[peaks]

'''
extract_intervals --> builds 7 dimensional feature vector of harmonic interval ratios

example: H2/H1 --> tells us how far apart the 2nd peak is from the 1st 
         H3/H2 H4/H3 H5/H4 --> tells us the step sizes between each pair of peaks

        H5/H1, H6/H1 H7/H1 --> tell us how high  the 5th 6th and 7th peaks sit relative to H1 

essentially, these ratios will tell us the overall "fingerprint" of the signal's structure
and can help the RF distinguish certain patterns in these ratios to classify major vs minor chords
'''
def extract_intervals(path):
    harm = find_harmonics(path)
    if len(harm) < 7:
        harm = np.pad(harm, (0, 7 - len(harm)), mode='edge')

    # consecutive ratios --> H2/H1, H3/H2, H4/H3, H5/H4
    consec = harm[1:5] / harm[:4]

    # root-relative ratios --> H5/H1, H6/H1, H7/H1
    rel    = harm[4:7] / harm[0]
    return np.concatenate([consec, rel], axis=0)

'''
build_intervals_dataframe --> creates df for the harmonic features
'''
def build_intervals_dataframe(data_dir):
    rows = []
    for chord, lbl in [("Minor", 0), ("Major", 1)]:
        folder = os.path.join(data_dir, chord)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                feats = extract_intervals(os.path.join(folder, fname))
                row = {f"interval_{i+1}": feats[i] for i in range(len(feats))}
                row['label']    = lbl
                row['filename'] = fname
                rows.append(row)
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    DATA_DIR = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
    df_intervals = build_intervals_dataframe(DATA_DIR)
    # print(df_intervals.head())
    
    # save to CSV
    TARGET_DIR = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data"
    output_csv = os.path.join(TARGET_DIR, "harmonic_data.csv")
    df_intervals.to_csv(output_csv, index=False)
    print(f"saved intervals df to {output_csv}")