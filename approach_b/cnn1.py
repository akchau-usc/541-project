import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics        import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, GlobalAveragePooling1D,
    Dense, Concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam

FEATURES_DIR  = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data"  
HARM_CSV      = os.path.join(FEATURES_DIR, "harmonic_data.csv")
LIBROSA_CSV   = os.path.join(FEATURES_DIR, "librosa_data.csv")
FFT_CSV       = os.path.join(FEATURES_DIR, "fft_ratio_data.csv")

df_harm = pd.read_csv(HARM_CSV)   
df_lib  = pd.read_csv(LIBROSA_CSV) 
df_fft  = pd.read_csv(FFT_CSV)    

df = pd.concat([
    df_harm,
    df_lib.drop(columns=["filename", "label"]),
    df_fft.drop(columns=["filename","label","fundamental_freq","second_peak_freq"])
], axis=1)

output_path = "/Users/annachau/Documents/USC/EE541/final_project/541-project/data/merged_data.csv"
df.to_csv(output_path, index=False)

# make sure labels didnt shift
assert (df["label"] == df_lib["label"]).all()
assert (df["label"] == df_fft["label"]).all()

le = LabelEncoder()
y = le.fit_transform(df["label"])  
print(y)

print("overall class distribution:", Counter(y))

interval_cols = [f"interval_{i}" for i in range(1,8)]
librosa_cols  = [c for c in df.columns if c not in ("filename","label") and c.startswith(("chroma","mfcc","contrast"))]
fft_cols      = ["ratio"]

X_int = df[interval_cols].values[..., np.newaxis]   # (n_samples, 7, 1)
X_lib = df[librosa_cols].values[..., np.newaxis]    # (n_samples, n_lib_feats, 1)
X_fft = df[fft_cols].values[..., np.newaxis]        # (n_samples, 1, 1)

# split dataset and make sure both minoor and major classes appear
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
train_idx, test_idx = next(sss.split(df, y))

X_int_train, X_int_test = X_int[train_idx], X_int[test_idx]
X_lib_train, X_lib_test = X_lib[train_idx], X_lib[test_idx]
X_fft_train, X_fft_test = X_fft[train_idx], X_fft[test_idx]
y_train,     y_test     = y[train_idx],     y[test_idx]

print("train class distribution:", Counter(y_train))
print("test class distribution:", Counter(y_test))

# ** MULTI INPUT CNN
inp_int = Input(shape=(7,1), name="interval_input")
x1 = Conv1D(32, 3, padding="same", activation="relu")(inp_int)
x1 = BatchNormalization()(x1)
x1 = Conv1D(32, 3, padding="same", activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.25)(x1)
b1 = GlobalAveragePooling1D()(x1)

inp_lib = Input(shape=(X_lib_train.shape[1],1), name="librosa_input")
x2 = Conv1D(64, 5, padding="same", activation="relu")(inp_lib)
x2 = BatchNormalization()(x2)
x2 = Conv1D(64, 5, padding="same", activation="relu")(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.3)(x2)
b2 = GlobalAveragePooling1D()(x2)

inp_fft = Input(shape=(1,1), name="fft_input")
x3 = Conv1D(16, 1, activation="relu")(inp_fft)
x3 = BatchNormalization()(x3)
x3 = Dropout(0.2)(x3)
b3 = GlobalAveragePooling1D()(x3)

merged = Concatenate()([b1, b2, b3])
x = Dense(64, activation="relu")(merged)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(32, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=[inp_int, inp_lib, inp_fft], outputs=out)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# train
model.fit(
    {"interval_input": X_int_train, "librosa_input": X_lib_train, "fft_input": X_fft_train},
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

# eval
y_pred_prob = model.predict(
    {"interval_input": X_int_test, "librosa_input": X_lib_test, "fft_input": X_fft_test}
).flatten()

print("Min prob:", y_pred_prob.min(), "Max prob: ", y_pred_prob.max())
y_pred = (y_pred_prob > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("\nConfusion Matrix (Test set):")
print(pd.DataFrame(cm,
                   index=[f"true_{cls}" for cls in le.inverse_transform([0,1])],
                   columns=[f"pred_{cls}" for cls in le.inverse_transform([0,1])]))
print("\nClassification Report (Test set):")
print(classification_report(y_test, y_pred, target_names=le.inverse_transform([0,1])))
