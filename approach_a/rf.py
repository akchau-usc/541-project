import os
import numpy as np
from scipy.io import wavfile
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1) Harmonic extraction
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

# 2) Build the 7-dim interval feature
def extract_intervals(path):
    harm = find_harmonics(path)
    if len(harm) < 7:
        harm = np.pad(harm, (0, 7 - len(harm)), mode='edge')
    # consecutive ratios: H2/H1, H3/H2, H4/H3, H5/H4
    consec = harm[1:5] / harm[:4]
    # root-relative ratios: H5/H1, H6/H1, H7/H1
    rel    = harm[4:7] / harm[0]
    return np.concatenate([consec, rel], axis=0)

# 3) Load all WAVs into X, y
data_dir = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
features, labels = [], []
for chord, lbl in [("Minor", 0), ("Major", 1)]:
    folder = os.path.join(data_dir, chord)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".wav"):
            features.append(extract_intervals(os.path.join(folder, fname)))
            labels.append(lbl)

X = np.array(features, dtype=np.float32)  # shape (n_samples, 7)
y = np.array(labels, dtype=np.int32)

# 4) Train/val split + scaling
train_X, val_X, train_y, val_y = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
scaler  = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
val_X   = scaler.transform(val_X)

# 5) Random Forest training
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(train_X, train_y)

# 6) Evaluation
preds = rf.predict(val_X)
print("Classification Report:\n", classification_report(val_y, preds))
print("Confusion Matrix:\n", confusion_matrix(val_y, preds))

# 7) (Optional) Feature importance plot
importances = rf.feature_importances_
plt.bar(range(len(importances)), importances)
plt.xticks(range(7), [
    "H2/H1","H3/H2","H4/H3","H5/H4",
    "H5/H1","H6/H1","H7/H1"
], rotation=45)
plt.ylabel("Feature Importance")
plt.title("Random Forest Importances")
plt.tight_layout()
plt.show()
