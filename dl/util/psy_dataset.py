from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .psy_util import generate_fbank, get_individual_samples_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio



class PsychiatryDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        data_folder = os.path.join(args.data_folder, 'psychiatry_speech/wav')
        print('train_flag', train_flag)        
        if args.cut_test:
            if train_flag:
                annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/training_seed{}.csv'.format(args.dataset_seed))
                
            else:
                annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/test_seed{}.csv'.format(args.dataset_seed))
                train_flag = True
        else:
            annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/training_seed{}.csv'.format(args.dataset_seed)) if train_flag else os.path.join(args.data_folder, 'psychiatry_speech/wav/test_seed{}.csv'.format(args.dataset_seed))
        print('train_flag', train_flag)
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels

        # ==========================================================================
        """ get dataset meta information """
        
        label = pd.read_csv(annotation_file, encoding='utf8')
        
        patients = label['patient'].values.tolist()
        before_folders = label['before_folder'].values.tolist()
        before_subjects = label['before_subject'].values.tolist()
        before_labels = label['before_label'].values.tolist()
        after_folders = label['after_subject'].values.tolist()
        after_subjects = label['after_subject'].values.tolist()
        after_labels = label['after_label'].values.tolist()
        '''
        print('before_folders--', before_folders)
        print('before_subjects--', before_subjects)
        print('before_labels', before_labels)
        print('after_folders--', after_folders)
        print('after_subjects--', after_subjects)
        print('after_labels--', after_labels)
        '''
        self.filenames = []
        self.audio_data = []  # each sample is a tuple with (audio_data, label, metadata)

        if print_flag:
            print('*' * 20)  
            print("Extracting individual psychiatry samples..")
        

        self.psy_list = []
        for before_folder, after_folder, before_label, after_label in zip(before_folders, after_folders, before_labels, after_labels):
            before_folder = self.make_num(before_folder) # only for the before_folders
            after_folder = self.make_num(after_folder)
            #print('before_folder', before_folder, type(before_folder))
            #print('after_folder', after_folder, type(after_folder))
            sample_data_before = get_individual_samples_torchaudio(args, before_folder, self.data_folder, self.sample_rate, args.n_cls, self.train_flag, before_label)
            sample_data_after = get_individual_samples_torchaudio(args, after_folder, self.data_folder, self.sample_rate, args.n_cls, self.train_flag, after_label)
            samples_data_before_with_labels = [(data[0], data[1]) for data in sample_data_before]
            samples_data_after_with_labels = [(data[0], data[1]) for data in sample_data_after]
            
            self.psy_list.extend(samples_data_before_with_labels)
            self.psy_list.extend(samples_data_after_with_labels)
        print(len(self.psy_list))
        for sample in self.psy_list:
            self.audio_data.append(sample)
        
        self.class_nums = np.zeros(args.n_cls)
            
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
            
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('total number of audio data: {}'.format(len(self.audio_data)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
        
        if args.framework == 'transformers':
            from transformers import AutoFeatureExtractor
            self.speech_extractor = AutoFeatureExtractor.from_pretrained(args.model)
        # ==========================================================================
        #print('in dataset, extractor', self.speech_extractor)
        """ convert fbank """
        self.audio_images = []
        #self.targets = []
        
        for index in range(len(self.audio_data)): #for the training set, 4142
            audio, label = self.audio_data[index][0], self.audio_data[index][1] # wav, label
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)                
                    
                    image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
                else:
                    if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_large',
            'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
            'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                        image = audio
                        if args.framework == 'transformers':
                            inputs = self.speech_extractor(audio, sampling_rate=self.sample_rate)
                            image = torch.from_numpy(inputs['input_values'][0])
                    else:
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
            self.audio_images.append((audio_image, label))
            
    
    def make_num(self, item):
        item = str(item)
        if len(item) == 1:
            item = '00' + item
        elif len(item) == 2:
            item = '0' + item
        return item
        # ==========================================================================

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
    
        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        return audio_image, label

    def __len__(self):
        return len(self.audio_images)