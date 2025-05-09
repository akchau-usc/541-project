import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D,MaxPool1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
        reg = 1e-4
        model = Sequential([
            # Block 1
            Conv1D(256, 3, padding='same', activation='relu',
                kernel_regularizer=l2(reg),
                input_shape=(self.n_features, 1)),
            BatchNormalization(),
            MaxPool1D(2),
            Dropout(0.3),

            # Block 2
            Conv1D(128, 3, padding='same', activation='relu',
                kernel_regularizer=l2(reg)),
            BatchNormalization(),
            MaxPool1D(2),
            Dropout(0.3),

            # Block 3
            Conv1D(64, 3, padding='same', activation='relu',
                kernel_regularizer=l2(reg)),
            BatchNormalization(),
            MaxPool1D(2),
            Dropout(0.3),

            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l2(reg)),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        return model


    def train(self, X_train, y_train, validation_split=0.1, epochs=200, batch_size=32, verbose=2):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True,
        help="path to merged_data.csv (all 3 feature sets)")
    parser.add_argument(
        "--output_model", type=str, default="cnn3.h5",
        help="where to save the trained .h5 model")
    return parser.parse_args()

def main():
    #features_csv = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/merged_data.csv"
    args = parse_args()

    test_size    = 0.2
    random_seed  = 7
    batch_size   = 32
    epochs       = 50
    lr           = 1e-3

    df = pd.read_csv(args.data)
    X = df.drop(columns=['filename', 'label']).values
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    n_feats = X_tr.shape[1]
    X_tr = X_tr.reshape(-1, n_feats, 1)
    X_te = X_te.reshape(-1, n_feats, 1)

    classifier = CNNChordClassifier(n_features=n_feats, lr=lr)
    classifier.model.summary()

    history= classifier.train(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size
    )

    # save model
    classifier.save(args.output_model)

    # eval
    loss, acc = classifier.evaluate(X_te, y_te)
    print(f"\nTest loss: {loss:.4f}, accuracy: {acc:.4f}\n")

    y_pred = (classifier.predict(X_te) > 0.5).astype(int).flatten()
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred, labels=[0,1]))
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # plot accuracy and loss
    epochs = range(1, epochs+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history.history['loss'],    label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, history.history['accuracy'],    label='Train Acc')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
