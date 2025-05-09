import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

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
    freqs = fftfreq(N, 1/fs)[:N//2]
    peaks, _ = find_peaks(spectrum,
                          height=spectrum.max()*height_frac,
                          distance=10)
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
        harm = np.pad(harm, (0,7-len(harm)), mode='edge')
    consec = harm[1:5] / harm[:4]
    rel    = harm[4:7] / harm[0]
    return np.concatenate([consec, rel], axis=0)

class CNNChordClassifier:
    def __init__(self, n_features, lr=1e-3):
        self.n_features = n_features
        self.model = self.build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def build_model(self):
        return Sequential([
            Conv1D(256, 3, activation='relu', padding='same', input_shape=(self.n_features,1)),
            Conv1D(128, 3, activation='relu', padding='same'),
            Conv1D(64, 3, activation='relu', padding='same'),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

    def train(self, X_train, y_train, validation_split=0.1,
              epochs=200, batch_size=32, verbose=2):
        return self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)
        print(f"model saved to {path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="train/evaluate cnn")
    parser.add_argument(
        "--data", type=str, required=True,
        help="folder containing Major/Minor .wav files")
    parser.add_argument(
        "--output_model", type=str, default="cnn2.h5",
        help="filename (with .h5) to save the trained model")
    return parser.parse_args()

def main():
    args = parse_args()
    features, labels = [], []

    for chord, lbl in [("Minor",0), ("Major",1)]:
        folder = os.path.join(args.data, chord)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            feats = extract_intervals(path)
            features.append(feats)
            labels.append(lbl)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # hyper parameters
    test_size    = 0.2
    random_seed  = 4
    batch_size   = 32
    epochs       = 200
    lr           = 1e-3

    # data split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)[...,None]
    X_te = scaler.transform(X_te)[...,None]

    # build + train
    n_feats = X_tr.shape[1]
    classifier = CNNChordClassifier(n_features=n_feats, lr=lr)
    classifier.model.summary()

    history = classifier.train(X_tr, y_tr,
                               validation_split=0.1,
                               epochs=epochs,
                               batch_size=batch_size)

    # save model
    classifier.save(args.output_model)

    # eval
    loss, acc = classifier.evaluate(X_te, y_te)
    print(f"\nTest loss: {loss:.4f}, accuracy: {acc:.4f}\n")

    y_pred = (classifier.predict(X_te) > 0.5).astype(int).flatten()
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred, labels=[0,1]))
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["Minor","Major"]))

    # plot training curves
    epochs = range(1, len(history.history['loss'])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history.history['accuracy'], label='Train Acc')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
