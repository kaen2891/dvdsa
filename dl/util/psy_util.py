from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

from .augmentation import augment_raw_audio

__all__ = ['save_image', 'get_mean_and_std', 'get_individual_samples_torchaudio', 'generate_fbank', 'get_score']


# ==========================================================================

def save_image(image, fpath):
    save_dir = os.path.join(fpath, 'image.jpg')
    cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """

def cut_pad_sample_torchaudio(data, train_flag, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    if train_flag:
        target_duration = args.desired_length * args.sample_rate
    else:
        target_duration = 30 * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data


def get_individual_samples_torchaudio(args, folder_name, data_folder, sample_rate, n_cls, train_flag, label):
    sample_data = []
    
    subject_files = sorted(glob(os.path.join(data_folder, folder_name) + '/*.wav'))
    for data in subject_files:
        if '_3_' in data: # because we use only set 1 and set 2
            break
        data, sr = torchaudio.load(data)
        
        if data.size(0) == 2: # if stereo 
            data = torch.mean(data, dim=0).unsqueeze(0)
        
        if sr != sample_rate:
            resample = T.Resample(sr, sample_rate)
            data = resample(data)
    
        fade_samples_ratio = 16
        fade_samples = int(sample_rate / fade_samples_ratio)
        fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
        data = fade(data)
        
        if train_flag:
            divide_count = 30 // args.divide_length
            #print('train')
            for i in range(divide_count):
                start = i * args.divide_length * sample_rate
                end = (i+1) * args.divide_length * sample_rate
                divided_wav = data[:, i*args.divide_length*sample_rate:(i+1)*args.divide_length*sample_rate]
                sample_data.append((divided_wav, label))
        else:
            sample_data.append((data, label))
    
    padded_sample_data = []
    for data, label in sample_data:
        data = cut_pad_sample_torchaudio(data, train_flag, args) # --> resample to [1, 80000] --> 5 seconds
        padded_sample_data.append((data, label))
        #print('data', data.size())
    return padded_sample_data


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std =  -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    print(hits)
    print(counts)
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
