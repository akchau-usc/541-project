# EE 541 -- INTRO TO DEEP LEARNING
# FINAL PROJECT SPRING 2025
# ANNA CHAU & MOUSUMI DAS

# PROJECT TITLE:
# AUTOMATED MAJOR/MINOR CHORD CLASSIFICATION IN PIANO AND GUITAR AUDIO

# 9 MAY 2025

tree -I "__pycache__" --dirsfirst -L 2
To run this project successfully, the following packages will be required:
- python 3.7 or newer 

Note: no special technical requirements are necessary for this project. 

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

# 3 METHODS OF FEATURE EXTRACTION 

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

541-project/
├── approach_b
│   ├── cnn1.py
│   └── cnn2.py

Focusing now on utilizing Deep Learning, we experimented with (3) different Convolutional Neural Networks to see which would lead to the best classification of Major vs Minor audio chords.

# CNN #1 (cnn1.py)

run with: python3 cnn1.py --data /path/to/csvs
example: python3 cnn1.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data

In this file, you will find a multi-input 1D CNN model that merges ALL three feature sets into one large feature set called 'merged_data.csv'. The model concatenates three CONV1D branches (one for the harmonic intervals, one for the Librosa features, and one for the singualr FFT ratio), and follows with a dense head with batch normalization and also droout. It trains for a total of 50 epochs and then prints out the respective classification report. 

# CNN #2 (cnn2.py)

run with: python3 cnn2.py --data /path/to/csvs
example: python3 cnn2.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data/Audio_Files

In this file, you will find a stand alone 1D CNN model that ONLY utilizes the seven harmonic intervals features (H2/H1, etc..) and stacks three CONV1D layers, flattens, and then outputs a single sigmoid neuron. This model trains for 200 epochs and then prints out the respective classification report of the model. 

# CNN #3 (cnn3.py)

run with: python3 cnn3.py --data /path/to/csvs
example: python3 cnn3.py --data /Users/annachau/Documents/USC/EE541/final_project/541-project/data/merged_data.csv

In this file, you will find a stand alone 1D CNN model that utilizes a CONCATENATED dataset of the 3 individual feature datasets discussed earlier. In other words, this CNN will be receiving the Librosa features and the Harmonic Interval Features. This model trainas for 200 epochs and then prints out the respective classification report of the model.


## MODEL LIST
1. Random Forest Classifier Model #1 -- Librosa Features (SIZE = 2.7MB)
2. Random Forest Classifier Model #2 -- Singular FFT Ratio Feature (H2/H1) (SIZE = 3.5MB)
3. Random Forest Classifier Model #3 -- Full Harmonic Interval Features (H2/H1, .. H7/H1) (SIZE = 2.1 MB)

4. Convolutional Neural Network #1 -- Single Input: Full Harmonic Interval Features (SIZE = 1.9 MB)
5. Convolutional Neural Network #2 -- Multi Input: Concatenation of the 3 Feature Sets (SIZE = 522 KB)
5. Convolutional Neural Network #3 -- Single Input: Concatenation of 3 Feature Sets (SIZE = 2 MB)

Note: all models are saved in respective approach_X/model folder