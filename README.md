# EE 541 -- INTRO TO DEEP LEARNING
# FINAL PROJECT SPRING 2025
# ANNA CHAU & MOUSUMI DAS

# PROJECT TITLE:
# AUTOMATED MAJOR/MINOR CHORD CLASSIFICATION IN PIANO AND GUITAR AUDIO

# 9 MAY 2025

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

# APPROACH A: RANDOM FOREST CLASSIFIER 

541-project/
├── approach_a
│   ├── fft_ratio_peaks.py
│   ├── harmonic_feat.py
│   ├── librosa_feat.py
│   └── run_rf.py

Focusing in on a Random Forest Classifier, we ran (3) different feature extraction scripts to test which features would lead to the best classification of Major vs Minor audio chords. 

# 3 METHODS OF FEATURE EXTRACTION + CSV CREATION
1. fft_ratio_peaks.py

run with: python3 fft_ratio_peaks.py

--> scans the Major/Minor data folders
--> performs the FFT peak detection 
--> computes ONLY the harmonic ratio between the first (also known as fundamental peak in the music world) and the second harmonic peak
--> saves this ratio value for each data sample to 'fft_ratio_data.csv' with columns setup as: filename,    label,  fundamental_freq,   second_peak_freq,   ratio

2. harmonic_feat.py

run with: python3 harmonic_feat.py

--> scans the Major/Minor data folders
--> performs FFT peak detection
--> computes the first seven harmonic‐interval ratios (e.g. H2/H1, H3/H2, etc)
    (based on the outputs of the FFT peak detection)
--> saves ratio values of each data sample to 'harmonic_data.csv' with columns setup as:    filename,   label,  interval_1,   interval_2, .... ,  interval_7

3. librosa_feat.py

run with: python3 librosa_feat.py

--> load each clip in Major/Minor data folders
--> normalizes audio

for each data sample:
--> extracts STFT (Short Time Fourier Transform)
--> extracts MFCCs (Mel Frequency Cepstral Coefficients)
--> extracts spectral contrast
--> summarize all this information by its mean and std (for each category)

--> saves the output table as 'librosa_data.csv' with columns setup as:
filename,   label,  chroma_mean_0,  ...,    chroma_mean11, ....,  contrast_std_6

# AFTER GENERATING THE 3 CSV'S:
# TESTING EACH FEATURE EXTRACTION METHOD 
# RANDOM FOREST IMPLEMENTATION --> run_rf.py

run with: python3 run_rf.py --data /path/to/csvs
example: python3 run_rf.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data

To experiment with these different feature methods, we loaded them all into their own Random Forest Classifier. In run_rf.py, the code essentially
-- loads each feature table
-- splits it into train/validation sets
-- standardizes the features
-- trains SEPARATE Random Forests based on the three feature extraction methods above
-- prints analysis report (classification, confusion matrix, etc. ) 


## APPROACH B: CONVOLUTIONAL NEURAL NETWORKS 
# data: utilizes the (3) csvs created earlier. 

541-project/
├── approach_b
│   ├── cnn1.py
│   └── cnn2.py

Focusing now on utilizing Deep Learning, we experimented with (3) different Convolutional Neural Networks to see which would lead to the best classification of Major vs Minor audio chords.

# CNN #1 (cnn1.py)

run with: python3 cnn1.py --data /path/to/FOLDER_WITH_ALL_3_CSVS
example: python3 cnn1.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data
note: this /data folder should contain ALL of the following:
1. harmonic.csv
2. librosa.csv
3. fft_ratio.csv

In this file, you will find a multi-input 1D CNN model that merges ALL three feature sets into one large feature set. The model concatenates three CONV1D branches (one for the harmonic intervals, one for the Librosa features, and one for the singualar FFT ratio), and follows with a dense head with batch normalization and also droout. It trains for a total of 50 epochs and then prints out the respective classification report. 

# CNN #2 (cnn2.py)

run with: python3 cnn2.py --data /path/to/MAJORMINOR_AUDIO_FILES
example: python3 cnn2.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files

In this file, you will find a stand alone 1D CNN model that ONLY utilizes the seven harmonic intervals features (H2/H1, etc..) and stacks three CONV1D layers, flattens, and then outputs a single sigmoid neuron. This model trains for 200 epochs and then prints out the respective classification report of the model. 

# CNN #3 (cnn3.py)

run with: python3 cnn3.py --data /path/to/MERGED_CSV
example: python3 cnn3.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data/merged_data.csv

In this file, you will find a stand alone 1D CNN model that utilizes a PRE-CONCATENATED dataset of the 3 individual feature datasets discussed earlier. In other words, this CNN will be receiving the Librosa features, the Harmonic Interval Features, and the FFT Ratio data. The difference between CNN #1 and CNN #3 is that CNN #1 takes in three input files and concatenates as part of its process to perform CONV1D on each individually first. CNN #3 takes in an already concatenated csv file and performs its process on this directly. This model trainas for 200 epochs and then prints out the respective classification report of the model.


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