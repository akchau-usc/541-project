# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import librosa
'''
this  function is designed to extract features from the raw audio files by utilizing the librosa
library. the audio is resampled to 44.1 kHz, normalized, and the following features are extracted:
1. chroma STFT (energy in each of the 12 pitch classes --> C, C#, B, etc..)

2. MFCCs (mel frequency cepstral coefficients --> essentially captures the shape of the sound spectrum
        and highlighs the important freuqnecy components that us humans can perceive as diff sounds)

3. spectral contrast (difference between peaks and valleys of each frequency section)

this function will take in one of the audio clips, and split it up into many short + overlapping frames.
for EACH frame, it will compute a vector of values:

1. chroma STFT -- 12 dim vector
2. mfccs -- 13 dim vector
3. spectral contrast -- 6/7 dim vector

then it will take the average of the mean and std of the resepctive feature..
aka " what is the average value of this coefficient in the entire clip X?"

it will do this for the entire dataset of 859 audio files. 

'''
def extract_features(path, sr=44100, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)
    # chroma --> 12 dims × frames
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # mfcss --> 13 dims × frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # spectral contrast --> 6 bands
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)

    # summarize the mean & std over time for each feature dim
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

# build the features df 
rows = []
base = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
for label in ["Major", "Minor"]:
     folder = os.path.join(base, label)
     for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".wav"):
            continue
        feats = extract_features(os.path.join(folder, fname))
        # build row dict
        row = {
            "filename": fname,
            "label":    label
        }
        # add all feature entries
        row.update(feats)
        rows.append(row)

df = pd.DataFrame(rows)
# reorder columns: filename, label, then sorted feature columns
feature_cols = sorted([c for c in df.columns if c not in ("filename", "label")])
df = df[["filename", "label"] + feature_cols]

# save csv
output_path = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/librosa_data.csv"
df.to_csv(output_path, index=False)
print(f"saved librosa feature table to {output_path}")

