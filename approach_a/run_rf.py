import os
import numpy as np
import pandas as pd
import joblib
import argparse

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

FEATURES_DIR  = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data"

INTERVALS_CSV = os.path.join(FEATURES_DIR, "harmonic_data.csv")
LIBROSA_CSV   = os.path.join(FEATURES_DIR, "librosa_data.csv")
FFT_CSV       = os.path.join(FEATURES_DIR, "fft_ratio_data.csv") 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, required=True,
        help='folder with harmonic_data.csv, librosa_data.csv, fft_ratio_data.csv'
    )
    parser.add_argument(
        '--output_model', type=str, default='.',
        help='directory in which to save model .pkl files'
    )
    return parser.parse_args()

def plot_confusion(cm, classes, title=None):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=title or "Confusion Matrix"
    )
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def run_intervals_rf(csv_path, output_model, test_size=0.3, random_state=7):
    df = pd.read_csv(csv_path)
    X = df[[f"interval_{i+1}" for i in range(7)]].values
    y = df["label"].astype(str).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    joblib.dump(rf, os.path.join(output_model, "harmonics_rf.pkl"))

    print("=== Harmonic Intervals RF ===")
    print(classification_report(y_val, preds, target_names=["Minor","Major"]))
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:\n", cm)
    plot_confusion(cm, ["Minor","Major"], title="Harmonics RF")

    importances = rf.feature_importances_
    names = [f"interval_{i+1}" for i in range(7)]
    plt.figure(figsize=(6,4))
    plt.bar(names, importances)
    plt.xticks(rotation=45)
    plt.title("Harmonics RF Importances")
    plt.tight_layout()
    plt.show()

def run_librosa_rf(csv_path, output_model, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    le = LabelEncoder().fit(df['label'])
    y = le.transform(df['label'])
    X = df.drop(columns=['filename','label']).values
    feat_names = df.drop(columns=['filename','label']).columns

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    joblib.dump(rf, os.path.join(output_model, "librosa_rf.pkl"))

    print("\n=== Librosa RF ===")
    print(classification_report(y_val, preds, target_names=le.classes_))
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:\n", cm)
    plot_confusion(cm, le.classes_, title="Librosa RF")

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    top_feats = feat_names[idx]
    plt.figure(figsize=(8,5))
    plt.barh(top_feats, importances[idx])
    plt.title("Librosa RF Top 10 Importances")
    plt.tight_layout()
    plt.show()

def run_fft_ratio_rf(csv_path, output_model, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    le = LabelEncoder().fit(df['label'])
    y = le.transform(df['label'])
    X = df[['ratio']].values  # only the r value not the rest

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    joblib.dump(rf, os.path.join(output_model, 'fft_ratio_rf.pkl'))

    print("\n=== FFT Ratio RF ===")
    print(classification_report(y_val, preds, target_names=le.classes_))
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:\n", cm)
    plot_confusion(cm, le.classes_, title="FFT Ratio RF")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_model, exist_ok=True)

    intervals_csv = os.path.join(args.data, 'harmonic_data.csv')
    librosa_csv = os.path.join(args.data, 'librosa_data.csv')
    fft_csv = os.path.join(args.data, 'fft_ratio_data.csv')

    run_intervals_rf(intervals_csv, args.output_model)
    run_librosa_rf(librosa_csv, args.output_model)
    run_fft_ratio_rf(fft_csv, args.output_model)
