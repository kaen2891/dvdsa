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
import torchaudio
from .psy_util import generate_fbank, get_individual_samples_torchaudio, cut_pad_sample_torchaudio, get_samples_torchaudio
from .augmentation import augment_raw_audio



class PsychiatryDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, valid=False):
        if train_flag:
            annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/permutation_{}s_seed{}/training_pairs_{}s_seed{}.csv'.format(args.divide_length, args.dataset_seed, args.divide_length, args.dataset_seed))
            cache_path = './data/training_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed)
            print('Training cache_path', cache_path)
        else:
            if valid:
                annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/permutation_{}s_seed{}/valid_pairs_{}s_seed{}.csv'.format(args.divide_length, args.dataset_seed, args.divide_length, args.dataset_seed))
                cache_path = './data/valid_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed)
                print('Validation cache_path', cache_path)
            else:
                annotation_file = os.path.join(args.data_folder, 'psychiatry_speech/wav/permutation_{}s_seed{}/test_pairs_{}s_seed{}.csv'.format(args.divide_length, args.dataset_seed, args.divide_length, args.dataset_seed))
                cache_path = './data/test_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed)
                print('Test cache_path', cache_path)
        
        self.train_flag = train_flag
        self.transform = transform
        self.args = args

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels
        
        if not os.path.isfile(cache_path):

            # ==========================================================================
            """ get dataset meta information """
            
            label = pd.read_csv(annotation_file, encoding='utf8')
            
            labels = label['label'].values.tolist()
            first_samples = label['sample1'].values.tolist()
            second_samples = label['sample2'].values.tolist()
            
            # for debugging
            '''
            labels = labels[:1000]
            first_samples = first_samples[:1000]
            second_samples = second_samples[:1000]
            '''
            
            self.audio_data = []  # each sample is a tuple with (audio_data, label, metadata)
    
            if print_flag:
                print('*' * 20)  
                print("Extracting individual psychiatry samples..")
    
            self.psy_list = []
            for label, first_sample, second_sample in zip(labels, first_samples, second_samples):
                first_data = get_samples_torchaudio(first_sample, self.sample_rate)
                second_data = get_samples_torchaudio(second_sample, self.sample_rate)
                self.audio_data.append((first_data, second_data, label))
            self.class_nums = np.zeros(args.n_cls)
                
            for sample in self.audio_data:
                self.class_nums[sample[2]] += 1
                
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
            
            """ convert fbank """
            self.audio_images = []
            
            for index in range(len(self.audio_data)): #for the training set, 4142
                first_audio, second_audio, label = self.audio_data[index][0], self.audio_data[index][1], self.audio_data[index][2]
                
                audio_image = []
                for aug_idx in range(self.args.raw_augment+1):
                    if aug_idx > 0:
                        if self.train_flag:
                            audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)                
                        
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                        audio_image.append(image)
                    
                    else: # here
                        if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
            'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
            'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                            first_image = first_audio
                            if args.framework == 'transformers':
                                inputs = self.speech_extractor(audio, sampling_rate=self.sample_rate)
                                first_image = torch.from_numpy(inputs['input_values'][0])
                            
                            second_image = second_audio
                            if args.framework == 'transformers':
                                inputs = self.speech_extractor(audio, sampling_rate=self.sample_rate)
                                second_image = torch.from_numpy(inputs['input_values'][0])
                        else:
                            first_image = generate_fbank(args, first_audio, self.sample_rate, n_mels=self.n_mels)
                            second_image = generate_fbank(args, second_audio, self.sample_rate, n_mels=self.n_mels)
                self.audio_images.append((first_image, second_image, label))
            
            if self.train_flag:
                torch.save(self.audio_images, './data/training_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
            else:
                if valid:
                    torch.save(self.audio_images, './data/valid_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
                else:
                    torch.save(self.audio_images, './data/test_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
        
        else:
            if self.train_flag:
                print('Training data loading')
                self.audio_images = torch.load('./data/training_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
            else:
                if valid:
                    print('Valid data loading')
                    self.audio_images = torch.load('./data/valid_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
                else:
                    print('Test data loading')
                    self.audio_images = torch.load('./data/test_{}s_seed{}.pt'.format(args.divide_length, args.dataset_seed))
    
    def make_num(self, item):
        item = str(item)
        if len(item) == 1:
            item = '00' + item
        elif len(item) == 2:
            item = '0' + item
        return item
        # ==========================================================================

    def __getitem__(self, index):
        first_audio, second_audio, label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
    
        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        
        if self.transform is not None:
            first_audio = self.transform(first_audio)
            second_audio = self.transform(second_audio)
        return first_audio, second_audio, label

    def __len__(self):
        return len(self.audio_images)