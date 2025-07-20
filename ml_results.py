import os
import numpy as np
import librosa
from glob import glob
import torchaudio
import math
import torchaudio.functional as F
#import disvoice
SAMPLE_RATE= 16000
from scipy import stats
from disvoice.prosody import Prosody

import parselmouth


F0_FORMANT_COUNT = 4  # F1~F4

def get_formants(filename):
    try:
        snd = parselmouth.Sound(filename)
        formant = snd.to_formant_burg()
        duration = snd.duration

        times = np.linspace(0, duration, 100)
        formants = []

        for t in times:
            values = [formant.get_value_at_time(i, t) for i in range(1, F0_FORMANT_COUNT + 1)]
            values = [v if v is not None and not np.isnan(v) else 0.0 for v in values]
            formants.append(values)

        formants = np.array(formants)
        return formants

    except Exception as e:
        print(f"Formant extraction failed for {filename}: {e}")
        return np.zeros((100, F0_FORMANT_COUNT))  # fallback

def get_log_mel(y, sr):
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=400, hop_length=400)
        log_mel = librosa.power_to_db(mel + 1e-6, ref=np.max)
        log_mel = np.nan_to_num(log_mel, nan=0.0)
        return log_mel
    except Exception as e:
        print(f"log-mel extraction failed: {e}")
        return np.zeros((80, 1))  # fallback


def zero_pad(x, target_length):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) >= target_length:
        return x[:target_length]
    else:
        return np.pad(x, (0, target_length - len(x)))

def get_results(exp_name, X_train, Y_train, X_test, y_test):
    print('exp_name {} X_train {} Y_train {} X_test {} y_test {}'.format(exp_name, len(X_train), len(Y_train), len(X_test), len(y_test)))
    # Initialize the SVM classifier
    classifier = svm.SVC(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.
    
    # Train the classifier on the training data
    classifier.fit(X_train, Y_train)
    
    # Predict the probabilities of the positive class for the test set
    y_prob = classifier.decision_function(X_test)
    
    # Evaluate accuracy
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # AUC
    auc = roc_auc_score(y_test, y_prob)
    
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    
    # Precision
    precision = precision_score(y_test, y_pred)
    
    # Recall
    recall = recall_score(y_test, y_pred)
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    
    print('SVM Exp: {} Acc {} Auc {} Precision {} Recall {} F1 {}'.format(exp_name, accuracy, auc, precision, recall, f1))
    
    
    # Initialize the Logistic Regression classifier
    classifier = LogisticRegression()
    # Train the classifier on the training data
    classifier.fit(X_train, Y_train)
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Predict the probabilities of the positive class for the test set
    y_prob = classifier.predict_proba(X_test)[:, 1]
    # Compute AUC
    auc = roc_auc_score(y_test, y_prob)
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    # Precision
    precision = precision_score(y_test, y_pred)
    # Recall
    recall = recall_score(y_test, y_pred)
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print('LR Exp: {} Acc {} Auc {} Precision {} Recall {} F1 {}'.format(exp_name, accuracy, auc, precision, recall, f1))
    
    # Initialize the Random Forest classifier
    classifier = RandomForestClassifier()
    # Train the classifier on the training data
    classifier.fit(X_train, Y_train)
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Predict the probabilities of the positive class for the test set
    y_prob = classifier.predict_proba(X_test)[:, 1]
    # Compute AUC
    auc = roc_auc_score(y_test, y_prob)
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    # Precision
    precision = precision_score(y_test, y_pred)
    # Recall
    recall = recall_score(y_test, y_pred)
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print('RF Exp: {} Acc {} Auc {} Precision {} Recall {} F1 {}'.format(exp_name, accuracy, auc, precision, recall, f1))

def get_features(dataset):
    stacked_list = []
    
    for data in dataset:
        y, sr = librosa.load(data, sr=SAMPLE_RATE)
        
        derivative = np.diff(y)

        # Find zero crossings of the derivative
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
        
        # --- 1. GOP/GCP/GCL (long raw shape)
        gop = np.zeros_like(y)
        gcp = np.zeros_like(y)
        gcl = np.zeros_like(y)
        
        for i in range(len(zero_crossings)-1):
            start = zero_crossings[i]
            end = zero_crossings[i+1]
            midpoint = (start + end) // 2
            gop[start:midpoint] = y[start:midpoint]
            gcp[midpoint:end] = y[midpoint:end]
            gcl[start:end] = y[start:end]
        
        gop = zero_pad(gop, 80000)
        gcp = zero_pad(gcp, 80000)
        gcl = zero_pad(gcl, 80000)
        
        # --- 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        log_spectral_centroids = np.log(np.nan_to_num(spectral_centroids, nan=1e-6))
        log_spectral_centroids = zero_pad(log_spectral_centroids.flatten(), 157)
        
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        log_bandwidth = np.log(np.nan_to_num(bandwidth, nan=1e-6))
        log_bandwidth = zero_pad(log_bandwidth.flatten(), 157)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        log_rolloff = np.log(np.nan_to_num(rolloff, nan=1e-6))
        log_rolloff = zero_pad(log_rolloff.flatten(), 157)
        
        # --- 3. RMS
        rms = librosa.feature.rms(y=y)
        log_rms = np.log(np.nan_to_num(rms, nan=1e-6))
        log_rms = zero_pad(log_rms.flatten(), 157)
        
        # --- 4. Tempo (scalar)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        log_tempo = np.log(np.nan_to_num(tempo, nan=1e-6))
        log_tempo = zero_pad(log_tempo, 1)
        
        # --- 5. F0
        f0, _, _ = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]
        f0 = zero_pad(f0, 150)
        
        # --- 6. Magnitude
        D = librosa.stft(y)
        magnitude, _ = librosa.magphase(D)
        magnitude = magnitude.flatten()
        magnitude = zero_pad(magnitude, 160925)
        
        # --- 7. ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr = zero_pad(zcr.flatten(), 157)
        
        # --- 8. Mel-spectrogram
        log_mel_spectrogram = get_log_mel(y, SAMPLE_RATE)
        log_mel_spectrogram = log_mel_spectrogram.flatten()
        log_mel_spectrogram = zero_pad(log_mel_spectrogram, 16080)
        
        # --- 9. FBank (Kaldi)
        y_torch, sr = torchaudio.load(data)
        fbank = torchaudio.compliance.kaldi.fbank(y_torch)
        log_fbank = np.log(fbank.numpy() + 1e-6)
        log_fbank = np.nan_to_num(log_fbank, nan=0.0, posinf=0.0, neginf=0.0)
        log_fbank = zero_pad(log_fbank.flatten(), 11454)
        
        # --- 10. Stack all features
        feature_list = [
            gop, gcp, gcl,
            log_spectral_centroids, log_bandwidth, log_rolloff,
            log_rms, log_tempo, f0,
            magnitude, zcr,
            log_mel_spectrogram, log_fbank
        ]
        
        stacked = np.concatenate(feature_list)
        stacked_list.append(stacked)
        
        
    x_stack = np.stack(stacked_list)
    
    
    return x_stack


train_audios, train_labels = sorted(glob('...'))
test_audios, test_labels = sorted(glob('...'))

x_train = get_features(train_audios)
x_test = get_features(test_audios)
get_results('ml', x_train, train_labels, x_test, test_labels)





