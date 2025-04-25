#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 14:56
# @Author  : Silent
# @File    : data_factory.py
# @Software: PyCharm

import pdb
import numpy as np

from src.data_process.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,  Dataset_Neuro, Dataset_Saugeen_Web,Dataset_gaolu, Dataset_gaolu2,Dataset_Custom2,Dataset_Custom3,Dataset_gaolu3,Dataset_Custom4,Dataset_Custom5,Dataset_gaolu4
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom2':Dataset_Custom2,
    'custom3':Dataset_Custom3,
    'custom4':Dataset_Custom4,
    'custom5':Dataset_Custom5,
    'neuro': Dataset_Neuro,
    'saugeen_web': Dataset_Saugeen_Web,
    'gaolu': Dataset_gaolu,
    'gaolu2': Dataset_gaolu2,
    'gaolu3': Dataset_gaolu3,
    'gaolu4': Dataset_gaolu4
}

def _custom_collate_fn(batch):
    batch_x = torch.tensor([item[0] for item in batch])
    batch_y = torch.tensor([item[1] for item in batch])
    timestamp = [item[2] for item in batch]
    timestamp = np.repeat(timestamp, 7, axis=0)
    return batch_x, batch_y, timestamp, None

def data_provider(args, flag):
    print(args.data)
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        # batch_size = 1
        batch_size = args.batch_size
        freq = args.freq

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    # print('args.root_path',args.root_path)
    # print('args.data_path',args.data_path)
    if Data == Dataset_Custom3:
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target='铁次平均铁水温度',
        timeenc=timeenc,
        freq=freq,
        round=args.round,
    )       # 一共分为三个round，分别对应的test的范围不同
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    print(flag, len(data_set))
    if  Data == Dataset_Custom4:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            collate_fn=_custom_collate_fn
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)
    return data_set, data_loader