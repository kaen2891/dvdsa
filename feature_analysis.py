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

def get_features(dataset):
    log_spectrals = []
    log_bandwidths = []
    log_rolloffs = []
    log_spectral_rmss = []
    log_spectral_tempo = []
    f0s = []
    pitches = []
    magnitudes = []
    zcrs = []
    formants = []
    mels = []
    fbanks = []
    prosodies = []
    gops = []
    gcps = []
    gcls = []
    
    
    for data in dataset:
        y, sr = librosa.load(data, sr=SAMPLE_RATE)
        
        derivative = np.diff(y)

        # Find zero crossings of the derivative
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
        
        # Compute glottal opening phase, closing phase, and closed phase
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
        
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        log_spectral_centroids = np.log(np.mean(spectral_centroids))
        
        
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        log_bandwidth = np.log(np.mean(bandwidth))
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        log_rolloff = np.log(np.mean(rolloff))
        
        rms = librosa.feature.rms(y=y)
        log_rms = np.log(np.mean(rms))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        log_tempo = np.log(np.mean(tempo))
        
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        check_nan = [not math.isnan(number) for number in f0]
        f0 = np.mean(f0[check_nan])
        
        torchaudio_y, torchaudio_sr = torchaudio.load(data)
        pitch = F.detect_pitch_frequency(torchaudio_y, torchaudio_sr)
        pitch = np.mean(pitch.numpy())
        
        
        prosody_function = Prosody()
        features = prosody_function.prosody_static(data, plots=False)
        check_nan = [not math.isnan(number) for number in features]
        prosody_features = np.mean(features[check_nan])
        
        
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)
        magnitude = np.mean(magnitude)
        
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        formant = get_formants(data)
        
        log_mel_spectrogram = get_log_mel(y, SAMPLE_RATE)
        
        
        
        y, sr = torchaudio.load(data)
        fbank = torchaudio.compliance.kaldi.fbank(y)
        log_fbank = np.log(fbank.numpy() + 1e-6)  # small epsilon for numerical stability
        # Replace any nan or inf with 0
        log_fbank = np.nan_to_num(log_fbank, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        gops.append(gop)
        gcps.append(gcp)
        gcls.append(gcl)
        log_spectrals.append(log_spectral_centroids)
        log_bandwidths.append(log_bandwidth)
        log_rolloffs.append(log_rolloff)
        log_spectral_rmss.append(log_rms)
        log_spectral_tempo.append(log_tempo)
        f0s.append(f0)
        pitches.append(pitch)
        prosodies.append(prosody_features)
        magnitudes.append(magnitude)
        zcrs.append(zcr)
        formants.append(formant)
        mels.append(log_mel_spectrogram)
        fbanks.append(log_fbank)
        
    
    print('gops', np.mean(gops), np.std(gops))
    print('gcps', np.mean(gcps), np.std(gcps))
    print('gcls', np.mean(gcls), np.std(gcls))
    print('log_spectrals', np.mean(log_spectrals), np.std(log_spectrals))
    print('log_bandwidths', np.mean(log_bandwidths), np.std(log_bandwidths))
    print('log_rolloffs', np.mean(log_rolloffs), np.std(log_rolloffs))
    print('log_spectral_rmss', np.mean(log_spectral_rmss), np.std(log_spectral_rmss))
    print('log_spectral_tempo', np.mean(log_spectral_tempo), np.std(log_spectral_tempo))
    print('f0s', np.mean(f0s), np.std(f0s))
    print('pitches', np.mean(pitches), np.std(pitches))
    print('prosodies', np.mean(prosodies), np.std(prosodies))
    print('magnitudes', np.mean(magnitudes), np.std(magnitudes))
    print('zcrs', np.mean(zcrs), np.std(zcrs))
    print('formants', np.mean(formants), np.std(formants))
    print('log mel', np.mean(mels), np.std(mels))
    print('log fbanks', np.mean(fbanks), np.std(fbanks))
    
    return gops, gcps, gcls, log_spectrals, log_bandwidths, log_rolloffs, log_spectral_rmss, log_spectral_tempo, f0s, pitches, prosodies, magnitudes, zcrs, formants, mels, fbanks

pre_data = sorted(glob('./pre/*.wav'))
post_data = sorted(glob('./post/*.wav'))

pre_gops, pre_gcps, pre_gcls, pre_log_spectrals, pre_log_bandwidths, pre_log_rolloffs, pre_log_spectral_rmss, pre_log_spectral_tempo, pre_f0s, pre_pitches, pre_prosodies, pre_magnitudes, pre_zcrs, pre_formants, pre_mels, pre_fbanks = get_features(pre_data)
post_gops, post_gcps, post_gcls, post_log_spectrals, post_log_bandwidths, post_log_rolloffs, post_log_spectral_rmss, post_log_spectral_tempo, post_f0s, post_pitches, post_prosodies, post_magnitudes, post_zcrs, post_formants, post_mels, post_fbanks = get_features(post_data) 


t, gop_pvalue = stats.ttest_ind(pre_gops, post_gops)
t, gcp_pvalue = stats.ttest_ind(pre_gcps, post_gcps)
t, gcl_pvalue = stats.ttest_ind(pre_gcls, post_gcls)
t, log_spectral_pvalue = stats.ttest_ind(pre_log_spectrals, post_log_spectrals)
t, log_bandwidth_pvalue = stats.ttest_ind(pre_log_bandwidths, post_log_bandwidths)
t, log_rolloff_pvalue = stats.ttest_ind(pre_log_rolloffs, post_log_rolloffs)
t, log_spectral_rms_pvalue = stats.ttest_ind(pre_log_spectral_rmss, post_log_spectral_rmss)
t, log_spectral_tempo_pvalue = stats.ttest_ind(pre_log_spectral_tempo, post_log_spectral_tempo)
t, f0_pvalue = stats.ttest_ind(pre_f0s, post_f0s)
t, pitch_pvalue = stats.ttest_ind(pre_pitches, post_pitches)
t, prosody_pvalue = stats.ttest_ind(pre_prosodies, post_prosodies)
t, magnitude_pvalue = stats.ttest_ind(pre_magnitudes, post_magnitudes)
t, zcr_pvalue = stats.ttest_ind(pre_zcrs, post_zcrs)

t, formant_pvalue = stats.ttest_ind(pre_formants, post_formants)
t, mel_pvalue = stats.ttest_ind(pre_mels, post_mels)
t, fbank_pvalue = stats.ttest_ind(pre_fbanks, post_fbanks)


print('gop_pvalue', gop_pvalue)
print('gcp_pvalue', gcp_pvalue)
print('gcl_pvalue', gcl_pvalue)

print('log_spectral_pvalue', log_spectral_pvalue)
print('log_bandwidth_pvalue', log_bandwidth_pvalue)
print('log_rolloff_pvalue', log_rolloff_pvalue)
print('log_spectral_rms_pvalue', log_spectral_rms_pvalue)
print('log_spectral_tempo_pvalue', log_spectral_tempo_pvalue)
print('f0_pvalue', f0_pvalue)
print('pitch_pvalue', pitch_pvalue)
print('magnitude_pvalue', magnitude_pvalue)
print('zcr_pvalue', zcr_pvalue)
print('formant_pvalue', formant_pvalue)
print('mel_pvalue', mel_pvalue)
print('fbank_pvalue', fbank_pvalue)








