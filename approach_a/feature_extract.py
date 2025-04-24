# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import librosa

def extract_features(path, sr=44100, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)
    # Chroma: 12 dims × frames
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # MFCCs: 13 dims × frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Spectral Contrast (6 bands)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Summarize: mean & std over time for each feature dimension
    features = {}
    for name, arr in [('chroma', chroma),
                  ('mfcc', mfccs),
                  ('contrast', spec_con)]:
        # mean stats
        features.update({
            "{}_mean_{}".format(name, i): np.mean(arr[i])
            for i in range(arr.shape[0])
        })
        # std stats
        features.update({
            "{}_std_{}".format(name, i): np.std(arr[i])
            for i in range(arr.shape[0])
        })
    return features

# Build a pandas DataFrame of all features + labels
rows = []
base = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"   # adjust to your folder
for label in ["Major","Minor"]:
    folder = os.path.join(base, label)
    for fname in os.listdir(folder):
        if fname.lower().endswith('.wav'):
            feats = extract_features(os.path.join(folder, fname))
            feats['label'] = label
            rows.append(feats)
df = pd.DataFrame(rows)
print(df.shape, df.columns)

# … your existing code that builds df …
print(df.shape, df.columns)

# Save to CSV (no index column)
output_path = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/chord_features.csv"
df.to_csv(output_path, index=False)
print(f"Saved feature table to {output_path}")

