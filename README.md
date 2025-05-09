# EE 541 -- INTRO TO DEEP LEARNING
# FINAL PROJECT SPRING 2025
# ANNA CHAU & MOUSUMI DAS

tree -I "__pycache__" --dirsfirst -L 2

541-project/
├── approach_a
│   ├── fft_ratio_peaks.py
│   ├── harmonic_feat.py
│   ├── librosa_feat.py
│   └── run_rf.py
├── approach_b
│   ├── cnn1.py
│   └── cnn2.py
├── data
│   ├── Audio_Files
│   ├── fft_ratio_data.csv
│   ├── harmonic_data.csv
│   ├── librosa_data.csv
│   └── merged_data.csv
├── process_data
│   ├── major_display
│   ├── display_data.py
│   ├── major_data.csv
│   ├── minor_data.csv
│   ├── opendata.py
│   ├── preprocess.py
│   └── test.ipynb
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
--> scans the Major/Minor data folders
--> performs the FFT peak detection 
--> computes ONLY the harmonic ratio between the first (also known as fundamental peak in the music world) and the second harmonic peak
--> saves this ratio value for each data sample to 'fft_ratio_data.csv' with columns setup as: filename,    label,  fundamental_freq,   second_peak_freq,   ratio

2. harmonic_feat.py
--> scans the Major/Minor data folders
--> performs FFT peak detection
--> computes the first seven harmonic‐interval ratios (e.g. H2/H1, H3/H2, etc)
    (based on the outputs of the FFT peak detection)
--> saves ratio values of each data sample to 'harmonic_data.csv' with columns setup as:    filename,   label,  interval_1,   interval_2, .... ,  interval_7

3. librosa_feat.py
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

Focusing now 
