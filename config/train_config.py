#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

root_dir  = '/home/jovyan/jupyter/benchmark_images/faceu'
lfw_dir   = osp.join(root_dir, 'face_verfication/lfw')
ms1m_dir  = osp.join(root_dir, 'face_recognition/ms1m_arcface')
cp_dir    = '/home/jovyan/jupyter/checkpoints_zoo/face-recognition/pointEstimate'

def training_args():

    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--workers', type=int,  default=16)

    # -- model
    parser.add_argument('--in_size',    type=tuple,  default=(112, 112))   # FIXED  # (112, 112) | (128, 96)
    parser.add_argument('--offset',     type=int,    default=2)            # FIXED
    parser.add_argument('--t',          type=float,  default=0.2)          # MV
    parser.add_argument('--margin',     type=float,  default=0.5)          # FIXED
    parser.add_argument('--easy_margin',type=bool,   default=True)
    parser.add_argument('--scale',      type=float,  default=64)           # FIXED
    parser.add_argument('--backbone',   type=str,    default='arcres100')  # TODO | iresse50
    parser.add_argument('--use_se',     type=bool,   default=False)        # IRESSE
    parser.add_argument('--use_cbam',   type=bool,   default=False)
    parser.add_argument('--in_feats',   type=int,    default=512)
    parser.add_argument('--drop_ratio', type=float,  default=0.4)          # TODO

    parser.add_argument('--fc_mode',    type=str,    default='arcface',  choices=['softmax', 'sphere', 'cosface', 'arcface', 'mvcos', 'mvarc'])
    parser.add_argument('--hard_mode',  type=str,    default='adaptive', choices=['fixed', 'adaptive']) # MV
    parser.add_argument('--loss_mode',  type=str,    default='ce',       choices=['ce', 'focal_loss', 'hardmining'])
    parser.add_argument('--hard_ratio', type=float,  default=0.9)          # hardmining
    parser.add_argument('--loss_power', type=int,    default=2)            # focal_loss
    parser.add_argument('--classnum',   type=int,    default=85164)        # CASIA (10574, v1-14529, v2-19007, v3-21082, v4-22920) | MS1M (85742, v1-89697)

    # fine-tuning
    parser.add_argument('--resume',     type=str,    default='')           # checkpoint
    parser.add_argument('--fine_tuning',type=bool,   default=False)        # just fine-tuning

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=45)
    parser.add_argument('--batch_size',  type=int,   default=128)      # TODO | 300
    parser.add_argument('--base_lr',     type=float, default=0.1)      # default = 0.1
    parser.add_argument('--lr_adjust',   type=list,  default=[16, 25, 35])
    parser.add_argument('--gamma',       type=float, default=0.1)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED

    # -- dataset
    parser.add_argument('--casia_dir',  type=str, default=ms1m_dir)  # augment_casia_aku8k
    parser.add_argument('--lfw_dir',    type=str, default=lfw_dir)    # TODO
    parser.add_argument('--train_file', type=str, default=osp.join(ms1m_dir, 'anno_file/ms1m_images.txt'))

    # -- verification
    parser.add_argument('--n_folds',   type=int,   default=10)
    parser.add_argument('--thresh_iv', type=float, default=0.005)

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=False)   # TODO
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'reproduce_arcface_paper'))
    parser.add_argument('--print_freq',type=int,   default=300)  # v0 : 454589, v1 : 500396, v2 : 509539, v3 : 513802, v4 : 517673
    parser.add_argument('--save_freq', type=int,   default=3)    # TODO

    args = parser.parse_args()

    return args
