
# Automated Major/Minor Chord Classification

**EE 541 — Intro to Deep Learning (Spring 2025)**
**Authors:** Anna Chau & Mousumi Das
**Date:** May 9, 2025

---

## DESCRIPTION

Automatically classify piano and guitar audio clips as Major or Minor chords using either Random Forest classifiers on engineered features or 1D Convolutional Neural Networks on multiple feature inputs.

---

To run this project successfully, the following packages will be required:
- python 3.7 or newer 

Note: no special technical requirements are necessary for this project. 


# GENERAL STRUCTURE OF PROJECT FILE SYSTEM

541-project/
├── approach_a
│   ├── models
│   ├── fft_ratio_peaks.py
│   ├── harmonic_feat.py
│   ├── librosa_feat.py
│   └── run_rf.py
├── approach_b
│   ├── models
│   ├── cnn1.py
│   ├── cnn2.py
│   └── cnn3.py
├── data
│   ├── Audio_Files
│   ├── fft_ratio_data.csv
│   ├── harmonic_data.csv
│   ├── librosa_data.csv
│   └── merged_data.csv
├── LICENSE
└── README.md


## USAGE

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
Note: --data data should be a folder consisting of the (3) generated CSV's from part (1).


### 3. Train CNNs

```bash
# Multi-input CNN

python3 approach_b/cnn1.py --data data

```
Note: --data data should be path to folder consisting of the (3) generated CSV's from part (1).

```bash 
# Harmonic-only CNN

python3 approach_b/cnn2.py --data data/Audio_Files

```
Note: --data data should be path to DATASET, aka Audio_Files (will convert to Harmonic CSV in code)

```bash
# Fully merged features (all three ) CNN

python3 approach_b/cnn3.py --data data/merged_data.csv
```
Note: --data data should be path to csv that contains ALL (3) feature sets, merged

---

# APPROACH A: RANDOM FOREST CLASSIFIER 

541-project/
├── approach_a
│   ├── fft_ratio_peaks.py
│   ├── harmonic_feat.py
│   ├── librosa_feat.py
│   └── run_rf.py

Focusing in on a Random Forest Classifier, we ran (3) different feature extraction scripts to test which features would lead to the best classification of Major vs Minor audio chords. 

# 3 METHODS OF FEATURE EXTRACTION + CSV CREATION

1. **fft_ratio_peaks.py**

   * scans the Major/Minor data folders
   * performs the FFT peak detection 
   * computes ONLY the harmonic ratio between the first (also known as fundamental peak in the music world) and the second harmonic peak
   * saves this ratio value for each data sample to 'fft_ratio_data.csv' with columns setup as: filename,    label,  fundamental_freq,   second_peak_freq,   ratio

2. **harmonic_feat.py**

  * scans the Major/Minor data folders
  * performs FFT peak detection
  * computes the first seven harmonic‐interval ratios (e.g. H2/H1, H3/H2, etc)
    (based on the outputs of the FFT peak detection)
  * saves ratio values of each data sample to 'harmonic_data.csv' with columns setup as:    filename,   label,  interval_1,   interval_2, .... ,  interval_7

3. **librosa_feat.py**

  * load each clip in Major/Minor data folders
  * normalizes audio

for each data sample:
  * extracts STFT (Short Time Fourier Transform)
  * extracts MFCCs (Mel Frequency Cepstral Coefficients)
  * extracts spectral contrast
  * summarize all this information by its mean and std (for each category)

  * saves the output table as 'librosa_data.csv' with columns setup as:
filename,   label,  chroma_mean_0,  ...,    chroma_mean11, ....,  contrast_std_6


# AFTER GENERATING THE 3 CSV'S:
# RANDOM FOREST IMPLEMENTATION --> run_rf.py

4. **run_rf.py**

To experiment with these different feature methods, we loaded them all into their own Random Forest Classifier. In run_rf.py, the code essentially:

   * loads each feature table
   * splits it into train/validation sets
   * standardizes the features
   * trains SEPARATE Random Forests based on the three feature extraction methods above
   * prints analysis report (classification, confusion matrix, etc. ) 


## APPROACH B: CONVOLUTIONAL NEURAL NETWORKS 
# data: utilizes the (3) csvs created earlier. 

541-project/
├── approach_b
│   ├── cnn1.py
│   └── cnn2.py

Focusing now on utilizing Deep Learning, we experimented with (3) different Convolutional Neural Networks to see which would lead to the best classification of Major vs Minor audio chords.

* **cnn1.py**

   * multi-input 1D CNN model that merges ALL three feature sets into one large feature set
   * concatenates three CONV1D branches (one for the harmonic intervals, one for the Librosa features, and one for the singualar FFT ratio)
   * followed by dense head with batch normalization and also dropout
   * trains for a total of 50 epochs and then prints out the respective classification report



* **cnn2.py**

   * a stand alone 1D CNN model that ONLY utilizes the seven harmonic intervals features (H2/H1, etc..) and 
   * stacks three CONV1D layers, flattens, and then outputs a single sigmoid neuron
   * model trains for 200 epochs and then prints out the respective classification report of the model. 

* **cnn3.py**

   * a stand alone 1D CNN model that utilizes a PRE-CONCATENATED dataset of the 3 individual feature datasets discussed earlier 
   * this CNN will be receiving the Librosa features, the Harmonic Interval Features, and the FFT Ratio data
   * the difference between CNN #1 and CNN #3 is that CNN #1 takes in three input files and concatenates as part of its process to perform CONV1D on each individually first
   * CNN #3 takes in an already concatenated csv file and performs its process on this directly
   * model trains on 200 epochs and then prints out the respective classification report of the model


## SAVED MODELS

| Model                           | Size   |
| ------------------------------- | ------ |
| RF — Librosa features           | 2.7 MB |
| RF — FFT ratio (H2/H1)          | 3.5 MB |
| RF — Harmonic interval features | 2.1 MB |
| CNN1 — Multi-input              | 522 KB |
| CNN2 — Harmonic-only            | 1.9 MB |
| CNN3 — Pre-merged               | 2.0 MB |

Note: all models are saved in respective approach_X/model folder
