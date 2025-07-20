from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.psy_util import get_score

from util.psy_dataset import PsychiatryDataset
from util.augmentation import SpecAugment, RepAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import MetaCL, PatchMixLoss, PatchMixConLoss

from pytorch_metric_learning import losses, miners, samplers, testers, trainers

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save') 
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    # dataset
    parser.add_argument('--dataset', type=str, default='psychiatry')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='psychiatry',
                        help='psychiatry: (before, after)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--m_cls', type=int, default=0,
                        help='set k-way classification problem for domain (meta)')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=5, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--divide_length', type=int,  default=5, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup',  ### check
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--framework', type=str, default='s3prl', 
                        help='using pretrained speech models from s3prl or huggingface', choices=['s3prl', 'transformers'])
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    parser.add_argument('--method', type=str, default='ce')
    
    
    parser.add_argument('--cut_test', action='store_true')
    # for RepAugment
    
    parser.add_argument('--specaug', action='store_true')
    parser.add_argument('--repaug_mask', action='store_true')
    parser.add_argument('--repaug_gen', action='store_true')
    parser.add_argument('--repaug_two', action='store_true')
    parser.add_argument('--repaug_gen_var', type=float, default=0.2)
    parser.add_argument('--repaug_ver2', action='store_true')
    parser.add_argument('--upsampling_count', type=int, default=40000)
    
    #parser.add_argument('--hubert_version', type=int, default=1, help='version1 = flatten, version2 = avgpool for 768')
    
    
    # for AST & SSAST (if we use, we then can clear the comments)
    '''
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')
    '''
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    if args.dataset == 'psychiatry':
        if args.class_split == 'psychiatry':  
            if args.n_cls == 2:
                args.cls_list = ['before', 'after']
            elif args.n_cls == 3:
                args.cls_list = ['same', 'decrease', 'increase']
            else:
                raise NotImplementedError

    return args


def set_loader(args):
    if args.dataset == 'psychiatry':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        '''
        train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)
        
        train_dataset = PsychiatryDataset(train_flag=True, transform=train_transform if args.model not in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 
        'wavlm_base', 'wavlm_base_plus', 'wavlm_large', 'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b'] else None, args=args, print_flag=True)
        val_dataset = PsychiatryDataset(train_flag=False, transform=val_transform if args.model not in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 
        'wavlm_base', 'wavlm_base_plus', 'wavlm_large', 'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b'] else None, args=args, print_flag=True, valid=True)
        '''
        test_dataset = PsychiatryDataset(train_flag=False, transform=val_transform if args.model not in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 
        'wavlm_base', 'wavlm_base_plus', 'wavlm_large', 'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b'] else None, args=args, print_flag=True)
        
    else:
        raise NotImplemented
    
    '''
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    '''
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return None, None, test_loader, args
    

def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls
        model = get_backbone_class(args.model)(**kwargs)
    elif args.model == 'ssast':
        kwargs['label_dim'] = args.n_cls
        kwargs['fshape'], kwargs['tshape'] = args.fshape, args.tshape
        kwargs['fstride'], kwargs['tstride'] = 10, 10
        kwargs['input_tdim'] = 798
        kwargs['task'] = args.ssast_task
        kwargs['pretrain_stage'] = not args.audioset_pretrained
        kwargs['load_pretrained_mdl_path'] = args.ssast_pretrained_type
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls # not debugging yet
        model = get_backbone_class(args.model)(**kwargs)
    elif args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_large', 'wavlm_base_plus',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
        if args.framework == 's3prl':
            from models.speech import PretrainedSpeechModels
            if args.model in ['wav2vec2_base_960', 'hubert_base', 'wavlm_base', 'wavlm_base_plus', 'data2vec_base_960']:
                model = PretrainedSpeechModels(args.model, 768)
            elif args.model in ['wav2vec2_large_ll60k', 'hubert_large_ll60k', 'xls_r_300m', 'wavlm_large', 'data2vec_large_ll60k']:
                model = PretrainedSpeechModels(args.model, 1024)
            elif args.model in ['xls_r_1b']:
                model = PretrainedSpeechModels(args.model, 1280)
            elif args.model in ['xls_r_2b']:
                model = PretrainedSpeechModels(args.model, 1920)
        elif args.framework == 'transformers':
            from transformers import Wav2Vec2Model, HubertModel, AutoFeatureExtractor
            from models.speech import PretrainedSpeechModels_hf
            if args.model in ['facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k']: #hubert-based models
                speech_extractor = HubertModel
                if args.model == 'facebook/hubert-base-ls960':
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 768)
                elif args.model == 'facebook/hubert-large-ll60k':
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 1024)
                elif args.model == 'facebook/hubert-xlarge-ll60k':
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 1280)
            elif args.model in ['facebook/wav2vec2-base', 'facebook/wav2vec2-large', 'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']: #wav2vec2-based models
                speech_extractor = Wav2Vec2Model
                if args.model == 'facebook/wav2vec2-base':
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 768)
                elif args.model in ['facebook/wav2vec2-large', 'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m']:
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 1024)
                elif args.model == 'facebook/wav2vec2-xls-r-1b':
                    model = PretrainedSpeechModels_hf(speech_extractor, args.model, 1280)
            #args.extractor = AutoFeatureExtractor.from_pretrained()
            
    else:
        model = get_backbone_class(args.model)(**kwargs)
    classifier = nn.Linear(model.final_feat_dim * 2, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim * 2, args.proj_dim) if args.method in ['patchmix_cl'] else nn.Identity()
    criterion = nn.CrossEntropyLoss()
           
    if args.model not in ['ast', 'ssast', 'hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')
    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']
        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if args.method == 'ce':
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    classifier.cuda()
    projector.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer


def representation_generation(representations, labels, args): #only for minority class
    noise_sampler = torch.distributions.Normal(0, args.repaug_gen_var)
    noise_masks = noise_sampler.sample(representations.shape).to(device=representations.device)
    
    if not args.repaug_ver2:
        for i in range(len(labels)):
            if labels[i] == 0:
                noise_masks[i] = 0.0
    
    return representations + noise_masks

def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (first_images, second_images, labels) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        first_images = first_images.cuda(non_blocking=True)
        second_images = second_images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                if args.specaug:
                    features = model(args.transforms(first_images), args.transforms(second_images), args=args, alpha=alpha, training=True)
                
                elif args.repaug_mask: ### Representation Masking Only
                    if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                        first_images = torch.squeeze(first_images, 1)
                        second_images = torch.squeeze(second_images, 1)
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    else:
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    masks_for_repaug = args.transforms(features)
                    features = features * masks_for_repaug
                
                elif args.repaug_gen: ### Representation Generation Only (Adding Gaussian Noise with args.repaug_gen_var)
                    if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                        first_images = torch.squeeze(first_images, 1)
                        second_images = torch.squeeze(second_images, 1)
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    else:
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    features = representation_generation(features, labels, args)
                
                elif args.repaug_two: ### Representation Masking + Generation
                    if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                        first_images = torch.squeeze(first_images, 1)
                        second_images = torch.squeeze(second_images, 1)
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    else:
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    masks_for_repaug = args.transforms(features)
                    features = features * masks_for_repaug
                    features = representation_generation(features, labels, args)
                
                else:
                    if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                        first_images = torch.squeeze(first_images, 1)
                        second_images = torch.squeeze(second_images, 1)
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                    else:
                        features = model(first_images, second_images, args=args, alpha=alpha, training=True)
                        
                output = classifier(features)
                loss = criterion[0](output, labels)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    
    labels_all = []
    preds_all = []
    soft = nn.Softmax()
    with torch.no_grad():
    
        end = time.time()
        for idx, (first_images, second_images, labels) in enumerate(val_loader):
            first_images = first_images.cuda(non_blocking=True)
            second_images = second_images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
                if args.model in ['hubert_base', 'hubert_large_ll60k', 'wav2vec2_base_960', 'wav2vec2_large_ll60k', 'xls_r_300m', 'xls_r_1b', 'xls_r_2b', 'wavlm_base', 'wavlm_base_plus', 'wavlm_large',
        'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k', 'facebook/hubert-xlarge-ll60k', 'facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
        'facebook/wav2vec2-large-lv60', 'facebook/wav2vec2-xls-r-300m', 'facebook/wav2vec2-xls-r-1b']:
                    first_images = torch.squeeze(first_images, 1)
                    second_images = torch.squeeze(second_images, 1)
                    features = model(first_images, second_images, args=args, training=False)
                else:
                    features = model(first_images, second_images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            
            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if preds[idx].item() == labels[idx].item():
                    hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    acc = float(top1.avg)
    print('acc', acc)
    if acc > best_acc[0]:
        save_bool = True
        best_acc = [acc, sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * Acc: {:.2f} (Best Acc: {:.2f})'.format(acc, best_acc[0]))
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[1], best_acc[2], best_acc[-1]))
    return best_acc, best_model, save_bool


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    #torch.autograd.set_detect_anomaly(True)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_eval_model = None
    best_test_model = None
    if args.dataset == 'psychiatry':
        #best_val_acc = [0]  # Specificity, Sensitivity, Score
        best_val_score = [0, 0, 0, 0] # Acc, Specificity, Sensitivity, Score
        #best_test_acc = [0]
        best_test_score = [0, 0, 0, 0] # Acc, Specificity, Sensitivity, Score
    
    if args.specaug:
        args.transforms = SpecAugment(args)
    elif args.repaug_mask:
        args.transforms = RepAugment(args)
    elif args.repaug_two:
        args.transforms = RepAugment(args)
        
    train_loader, val_loader, test_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)
    
    
    print('model', model)
    print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
     
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_val_score, best_eval_model, eval_save_bool = validate(val_loader, model, classifier, criterion, args, best_val_score, best_eval_model)
            # save a checkpoint of model and classifier when the best score is updated
            if eval_save_bool:            
                save_file = os.path.join(args.save_folder, 'eval_best_epoch_{}.pth'.format(epoch))
                print('Validation set: Best ckpt is modified with Acc = {:.2f} Score = {:.2f} when Epoch = {}'.format(best_val_score[0], best_val_score[-1], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
                
            print('Evaluate the test set')
            # test for one epoch
            best_test_score, best_test_model, test_save_bool = validate(test_loader, model, classifier, criterion, args, best_test_score, best_test_model)
            print('Test set: Acc = {:.2f} Score = {:.2f} when Epoch = {}'.format(best_test_score[0], best_test_score[-1], epoch))
            
            if test_save_bool:            
                save_file = os.path.join(args.save_folder, 'test_best_epoch_{}.pth'.format(epoch))
                print('Test set: Best ckpt is modified with Acc = {:.2f} Score = {:.2f} when Epoch = {}'.format(best_test_score[0], best_test_score[-1], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
                

        # save a checkpoint of classifier with the best accuracy or score
        eval_save_file = os.path.join(args.save_folder, 'eval_best.pth')
        model.load_state_dict(best_eval_model[0])
        classifier.load_state_dict(best_eval_model[1])
        save_model(model, optimizer, args, epoch, eval_save_file, classifier)
        
        
        # save a checkpoint of classifier with the best accuracy or score
        test_save_file = os.path.join(args.save_folder, 'test_best.pth')
        model.load_state_dict(best_test_model[0])
        classifier.load_state_dict(best_test_model[1])
        save_model(model, optimizer, args, epoch, test_save_file, classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc = [0]
        best_score = [0, 0, 0]
        best_score, _, _  = validate(test_loader, model, classifier, criterion, args, best_score)
        update_json('%s' % args.model_name, best_score, path=os.path.join(args.save_dir, 'results.json'))
    
    print('{} finished'.format(args.model_name))
    update_json('%s' % args.model_name, best_val_score, path=os.path.join(args.save_dir, 'results_valid.json'))
    update_json('%s' % args.model_name, best_test_score, path=os.path.join(args.save_dir, 'results_test.json'))
    
if __name__ == '__main__':
    main()
