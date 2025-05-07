import os
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# 1) Harmonic extraction
def find_harmonics(path, min_freq=50, height_frac=0.05):
    fs, data = wavfile.read(path)
    data = data.astype(float)
    N = len(data)
    spectrum = np.abs(fft(data))[:N//2] * (2.0/N)
    freqs = fftfreq(N, 1/fs)[:N//2]
    peaks, _ = find_peaks(
        spectrum,
        height=spectrum.max()*height_frac,
        distance=10
    )
    peaks = peaks[freqs[peaks] > min_freq]
    return freqs[peaks]

# 2) Feature: 7 interval ratios
def extract_intervals(path):
    harm = find_harmonics(path)
    if len(harm) < 7:
        harm = np.pad(harm, (0, 7 - len(harm)), mode='edge')
    consec = harm[1:5] / harm[:4]      # H2/H1, H3/H2, H4/H3, H5/H4
    rel   = harm[4:7] / harm[0]        # H5/H1, H6/H1, H7/H1
    return np.concatenate([consec, rel], axis=0)

# 3) Load all WAVs into X,y
data_dir = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
features, labels = [], []
for chord, lbl in [("Minor", 0), ("Major", 1)]:
    folder = os.path.join(data_dir, chord)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".wav"):
            features.append(extract_intervals(os.path.join(folder, fname)))
            labels.append(lbl)

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

# 4) Split, scale, reshape for Conv1D
train_X, val_X, train_y, val_y = train_test_split(
    X, y,
    test_size=0.3,
    random_state=0,
    stratify=y
)
scaler  = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)[..., np.newaxis]
val_X   = scaler.transform(val_X)[...,   np.newaxis]

# 5) Simple 1D-CNN
''' 
model = Sequential([
    Conv1D(256, 4, activation='relu', input_shape=(train_X.shape[1], 1)),
    Conv1D(128, 3, activation='relu'),
    Conv1D( 64, 2, activation='relu'),
    Flatten(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
'''
# testing
model = Sequential([
    Conv1D(256, 4, activation='relu', input_shape=(train_X.shape[1], 1)),
    Conv1D(128, 3, activation='relu'),
    Conv1D( 64, 2, activation='relu'),
    Flatten(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6) Train
history = model.fit(
    train_X, train_y,
    epochs=200,
    batch_size=32,
    validation_data=(val_X, val_y),
    verbose=2
)

# 7) Evaluate
val_loss, val_acc = model.evaluate(val_X, val_y, verbose=0)
print("Validation Loss: {:.4f}, Accuracy: {:.4f}".format(val_loss, val_acc))

# 8) Metrics
preds = (model.predict(val_X) > 0.5).astype(int).flatten()
print("Classification Report:\n{}".format(
    classification_report(val_y, preds)))
print("Confusion Matrix:\n{}".format(
    confusion_matrix(val_y, preds)))

# 9) Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.show()
