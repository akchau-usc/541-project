# Automated Major/Minor Chord Classification

**EE 541 — Intro to Deep Learning (Spring 2025)**
**Authors:** Anna Chau & Mousumi Das
**Date:** May 9, 2025

---

## Description

Automatically classify piano and guitar audio clips as Major or Minor chords using either Random Forest classifiers on engineered features or 1D Convolutional Neural Networks on multiple feature inputs.

---

## Usage

### 1. Generate Features

```bash
# FFT ratio features
python3 approach_a/fft_ratio_peaks.py --data data/Audio_Files

# Harmonic interval features
python3 approach_a/harmonic_feat.py --data data/Audio_Files

# Librosa features
python3 approach_a/librosa_feat.py --data data/Audio_Files
```

### 2. Train Random Forests

```bash
python3 approach_a/run_rf.py --data data
```

### 3. Train CNNs

```bash
# Multi-input CNN
python3 approach_b/cnn1.py --data data

# Harmonic-only CNN
python3 approach_b/cnn2.py --data data/Audio_Files

# Pre-merged features CNN
python3 approach_b/cnn3.py --data data/merged_data.csv
```

---

## Project Structure

```
541-project/
├── approach_a/
│   ├── models/
│   ├── fft_ratio_peaks.py
│   ├── harmonic_feat.py
│   ├── librosa_feat.py
│   └── run_rf.py
├── approach_b/
│   ├── models/
│   ├── cnn1.py
│   ├── cnn2.py
│   └── cnn3.py
├── data/
│   ├── Audio_Files/
│   ├── fft_ratio_data.csv
│   ├── harmonic_data.csv
│   ├── librosa_data.csv
│   └── merged_data.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Approach A: Random Forest

1. **fft_ratio_peaks.py**

   * Detects fundamental and second harmonic peaks via FFT
   * Computes the ratio H2/H1 for each audio sample
   * Outputs `fft_ratio_data.csv`

2. **harmonic_feat.py**

   * Extracts seven consecutive harmonic interval ratios (H2/H1, H3/H2, …, H8/H7)
   * Outputs `harmonic_data.csv`

3. **librosa_feat.py**

   * Uses Librosa to compute STFT, MFCCs, and spectral contrast
   * Summarizes each feature dimension by mean and standard deviation
   * Outputs `librosa_data.csv`
   
4. **run_rf.py**

   * Loads each feature CSV
   * Splits into training and validation sets
   * Standardizes features and trains separate Random Forest classifiers
   * Prints classification reports and confusion matrices

---

## Approach B: Convolutional Neural Networks

* **cnn1.py**

  * **Multi-input**: Three Conv1D branches for FFT ratio, harmonic intervals, and Librosa features
  * Concatenates feature maps and applies dense layers with batch normalization and dropout
  * Trains for 50 epochs
* **cnn2.py**

  * **Harmonic-only**: Single Conv1D branch on seven harmonic interval features
  * Trains for 200 epochs
* **cnn3.py**

  * **Pre-merged**: Single Conv1D branch on concatenated feature CSV
  * Trains for 200 epochs

---

## Saved Models

| Model                           | Size   |
| ------------------------------- | ------ |
| RF — Librosa features           | 2.7 MB |
| RF — FFT ratio (H2/H1)          | 3.5 MB |
| RF — Harmonic interval features | 2.1 MB |
| CNN1 — Multi-input              | 522 KB |
| CNN2 — Harmonic-only            | 1.9 MB |
| CNN3 — Pre-merged               | 2.0 MB |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or feedback, please contact Anna Chau at [anna.chau@example.com](mailto:anna.chau@example.com) or Mousumi Das at [mousumi.das@example.com](mailto:mousumi.das@example.com).
