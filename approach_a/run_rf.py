# run_rf.py
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import classification_report, confusion_matrix

import matplotlib.pyplot as plt

DATA_DIR      = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files"
FEATURES_DIR  = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data"
INTERVALS_CSV = os.path.join(FEATURES_DIR, "harmonic_data.csv")
LIBROSA_CSV   = os.path.join(FEATURES_DIR, "librosa_data.csv")


def run_intervals_rf(csv_path, test_size=0.3, random_state=7):
    df = pd.read_csv(csv_path)
    X = df[[f"interval_{i+1}" for i in range(7)]].values
    y = df["label"].astype(int).values

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # train random forest
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    print("=== Harmonics Intervals RF ===")
    print(classification_report(y_val, preds, target_names=["Minor","Major"]))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

    cm = confusion_matrix(y_val, preds)
    plot_confusion(cm, ["Minor", "Major"], title="Harmonics RF")

    # plot importances
    importances = rf.feature_importances_
    names = [f"interval_{i+1}" for i in range(7)]
    plt.figure(figsize=(6,4))
    plt.bar(names, importances)
    plt.title("Harmonics RF Importances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_librosa_rf(csv_path, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    le = LabelEncoder().fit(df['label'])
    y = le.transform(df['label'])
    X = df.drop(columns=['label']).values
    feat_names = df.drop(columns=['label']).columns

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # train random forest
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)

    print("=== Librosa RF ===")
    print(classification_report(y_val, preds, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

    cm = confusion_matrix(y_val, preds)
    plot_confusion(cm, le.classes_, title="Librosa RF")

    # plot top importances
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    top_feats = feat_names[idx]
    plt.figure(figsize=(8,5))
    plt.barh(top_feats, importances[idx])
    plt.title("Librosa RF Top 10 Importances")
    plt.tight_layout()
    plt.show()

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
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_intervals_rf(INTERVALS_CSV)
    run_librosa_rf(LIBROSA_CSV)
