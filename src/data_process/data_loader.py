#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 14:57
# @Author  : Silent
# @File    : data_loader.py
# @Software: PyCharm

import pdb
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from . import timefeatures
import warnings
import random
import torch
from .m4 import M4Dataset, M4Meta
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = timefeatures(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = timefeatures(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print(seq_x_mark.shape)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_gaolu(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.__read_data__()
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):

        data = pd.read_csv(self.root_path)
        ot_gctswdcz = data["罐次铁水温度插值"]
        series = ot_gctswdcz
        # 删除第一行
        series = series.drop(series.index[0])
        last_valid_index = series.last_valid_index()
        if last_valid_index is not None:
            # 删除末尾的缺失值
            series = series[:last_valid_index + 1]
        else:
            # 如果全部都是缺失值,则返回空的Series
            series = series
        series.iloc[119] = series[:119].mean()
        series.iloc[120] = series[:120].mean()
        # series.iloc[3098] = series[:3098].mean()
        missing_indices = series.index[series.isna()]

        # print('missing_indices', missing_indices)

        # series2 = pd.concat([series[:missing_indices[0]-1],series[missing_indices[-1]+1:]]).reset_index(drop=True)
        # series2 = series[:missing_indices[0]-1]
        # series2 = series2.drop(series2.index[0])
        # missing_indices2 = series2.index[series2.isna()]

        # print('missing_indices2', missing_indices2)

        self._generate_pairs(series)

        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        self.all_x = np.concatenate(self.all_x, axis=0)
        self.all_y = np.concatenate(self.all_y, axis=0)

    def _generate_pairs(self, df_raw):
        self.scaler = StandardScaler()

        num_train = len(df_raw)  # int(len(df_raw) * 0.5)
        num_test = len(df_raw)  # int(len(df_raw) * 0.2)
        num_vali = 0  # len(df_raw) - num_train - num_test
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border1s = [0, num_train - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values.reshape(-1, 1))
            data = self.scaler.transform(df_data.values.reshape(-1, 1))
        else:
            data = df_data.values.reshape(-1, 1)

        for i in range(border1, border2 - self.pred_len - self.seq_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len].T
            # app = np.stack((batch_x_i, batch_y_i), axis=-1)
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            # self.all_pairs.append(app)

    def __getitem__(self, index):

        seq_x = self.all_x[index]
        seq_y = self.all_y[index]

        return seq_x, seq_y, [], []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_gaolu2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.__read_data__()
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):

        data = pd.read_csv(self.root_path)
        ot_gctswdcz = data["铁次平均温度插值"]
        series = ot_gctswdcz
        # 删除第一行
        # series = series.drop(series.index[0])

        self._generate_pairs(series)

        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        self.all_x = np.concatenate(self.all_x, axis=0)
        self.all_y = np.concatenate(self.all_y, axis=0)

    def _generate_pairs(self, df_raw):
        self.scaler = StandardScaler()

        # num_train = int(len(df_raw) * 0.5)
        # num_test = int(len(df_raw) * 0.2)

        num_train = len(df_raw)
        num_test = len(df_raw)  # int(len(df_raw) * 0.2)

        num_vali = 0  # len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 使用全量数据测试

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values.reshape(-1, 1))
            data = self.scaler.transform(df_data.values.reshape(-1, 1))
        else:
            data = df_data.values.reshape(-1, 1)

        for i in range(border1, border2 - self.pred_len - self.seq_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len].T
            # app = np.stack((batch_x_i, batch_y_i), axis=-1)
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            # self.all_pairs.append(app)

    def __getitem__(self, index):

        seq_x = self.all_x[index]
        seq_y = self.all_y[index]

        return seq_x, seq_y, [], []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_gaolu3(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.__read_data__()


    def __read_data__(self):

        data = pd.read_csv(self.root_path)

        self._generate_pairs(data)

        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        self.all_x = np.concatenate(self.all_x, axis=0)
        self.all_y = np.concatenate(self.all_y, axis=0)

    def _generate_pairs(self, df_raw):
        self.scaler = StandardScaler()

        # num_train = int(len(df_raw) * 0.5)
        # num_test = int(len(df_raw) * 0.2)
        # 删除数据中，prediction target为空的行
         
        
        sampling_related = ['罐次铁水温度','铁次平均-核心温度(方法1)','铁次平均-铁口温度偏差(方法1)','铁次平均-核心温度(方法2)','铁次平均-铁口温度偏差(方法2)','堵口铁水温度','堵口铁水-核心温度(方法1)',
          '堵口铁水-铁口温度偏差(方法1)','堵口铁水-核心温度(方法2)','堵口铁水-铁口温度偏差(方法2)','铁次内硅平均值','出铁时长','相邻铁次出铁间隔','同一铁口出铁间隔','铁次出铁量','铁次出渣量','铁次铁流速度'
          ,'铁水Si','铁次铁流速度'
          ]
        
        training_targets = ['铁水Si', '铁次平均铁水温度']
        

        labels = [
    '罐次温度铁口号','铁口号（平均/堵口温度）'
         ]
        
        columns_to_check = sampling_related + labels


        #df_raw.drop(columns=columns_to_check)


        
        # 在时间上进行划分：

        
        df_raw['作业时刻'] = pd.to_datetime(df_raw['作业时刻'])
        df_raw.set_index('作业时刻', inplace=True)
        
        
        df_raw = df_raw[training_targets[1]]  # 1 for 铁次平均铁水温度

        df_raw = df_raw.dropna()

        closest_timestamp = df_raw.index.asof(pd.Timestamp('2024-5-10 0:00'))
        split_index = df_raw.index.get_loc(closest_timestamp)
        # 在split index之前的作为训练和验证集

        
        num_train = int(split_index * 0.6)
        num_test = len(df_raw) - split_index # 在index之后的全为测试集

        num_vali =  int(split_index*0.4)
        border1s = [0, num_train - self.seq_len, num_train + num_vali-self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        df_data = df_raw
        
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]  
            self.scaler.fit(train_data.values.reshape(-1, 1))
            data = self.scaler.transform(df_data.values.reshape(-1, 1))
        else:
            data = df_data.values.reshape(-1, 1)

        for i in range(border1, border2 - self.pred_len - self.seq_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len].T
            # app = np.stack((batch_x_i, batch_y_i), axis=-1)
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            # self.all_pairs.append(app)

    def __getitem__(self, index):

        seq_x = self.all_x[index]
        seq_y = self.all_y[index]

        return seq_x, seq_y, [], []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_gaolu4(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.__read_data__()


    def __read_data__(self):

        data = pd.read_csv(self.root_path)

        self._generate_pairs(data)

        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        self.all_x = np.stack(self.all_x, axis=0)
        self.all_y = np.stack(self.all_y, axis=0)

    def _generate_pairs(self, df_raw):
        self.scaler = StandardScaler()

        # num_train = int(len(df_raw) * 0.5)
        # num_test = int(len(df_raw) * 0.2)
        # 删除数据中，prediction target为空的行
         
        
        sampling_related = ['罐次铁水温度','铁次平均-核心温度(方法1)','铁次平均-铁口温度偏差(方法1)','铁次平均-核心温度(方法2)','铁次平均-铁口温度偏差(方法2)','堵口铁水温度','堵口铁水-核心温度(方法1)',
          '堵口铁水-铁口温度偏差(方法1)','堵口铁水-核心温度(方法2)','堵口铁水-铁口温度偏差(方法2)','铁次内硅平均值','出铁时长','相邻铁次出铁间隔','同一铁口出铁间隔','铁次出铁量','铁次出渣量','铁次铁流速度'
          ,'铁次铁流速度'
          ]
        

        missing_to_much = ['charge下料间隔']
        training_targets = ['铁水Si', '铁次平均铁水温度']
        

        labels = [
    '罐次温度铁口号','铁口号（平均/堵口温度）'
         ]
        
        columns_to_check = sampling_related + labels + training_targets + missing_to_much


        #df_raw.drop(columns=columns_to_check)


        
        # 在时间上进行划分：

        
        df_raw['作业时刻'] = pd.to_datetime(df_raw['作业时刻'])
        df_raw.set_index('作业时刻', inplace=True)
        
        
        df_raw = df_raw[[columni for columni in df_raw.columns if  columni not in columns_to_check]+[training_targets[1]]]  # 1 for 铁次平均铁水温度


        df_raw = df_raw.dropna()

        closest_timestamp = df_raw.index.asof(pd.Timestamp('2024-5-10 0:00'))
        split_index = df_raw.index.get_loc(closest_timestamp)
        # 在split index之前的作为训练和验证集
        num_test = len(df_raw) - split_index # 在index之后的全为测试集

        
        num_train = int(split_index * 0.6)
        #num_train = int(split_index * 0.01)
        num_vali =  int(split_index*0.4)
        #num_vali =  int(split_index*0.01)
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        #border2s = [num_train, num_train + num_vali, len(df_raw)//40]


        border1s = [0, num_train - self.seq_len, num_train + num_vali-self.seq_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]        
        df_data = df_raw

        
       
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            shape = train_data.shape
            # train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
            self.scaler.fit(train_data.values.reshape(-1,train_data.shape[-1]))  # len * channels
            data = self.scaler.transform(df_data.values)
            #data = train_data_scaled.reshape(train_data.shape)
        else:
            data = df_data.values

        for i in range(border1, border2 - self.pred_len - self.seq_len - self.label_len):
            batch_x_i = data[i:i + self.seq_len]
            batch_y_i = data[i + self.seq_len - self.label_len:i + self.pred_len + self.seq_len+self.label_len]
            # app = np.stack((batch_x_i, batch_y_i), axis=-1)
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            # self.all_pairs.append(app)

    def __getitem__(self, index):

        seq_x = self.all_x[index]
        seq_y = self.all_y[index]

        return seq_x, seq_y, [], []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
     
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # print(df_raw)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # import pdb
            # pdb.set_trace()
            data_stamp = timefeatures(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # print(f'x_begin:{s_begin} x_end:{s_end}, y_begin:{r_begin} y_end:{r_end}\n')
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print('data index', s_begin, s_end, r_begin, r_end)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
class Dataset_Custom2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.timestamp = []
        self.all_prompt_bank = []
        self.__read_data__()
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):
        self.scaler = StandardScaler()

        data = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print("===========loading file===========,",os.path.join(self.root_path, self.data_path))
        self._generate_pairs(data)
        
        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        #self.all_x = np.concatenate(self.all_x, axis=0)
        #self.all_y = np.concatenate(self.all_y, axis=0)
        #self.timestamp = np.concatenate(self.timestamp, axis=0)
    def _generate_pairs(self, df_raw):
        scaler = StandardScaler()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            time_col = df_raw.columns[:1]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            time_data = df_raw[time_col].values
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values

        for i in range(border1, border2 - self.pred_len - self.seq_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
            time_x_i  = time_data[i:i+self.seq_len]
            time_x_i = [str(arr[0]) for arr in time_x_i]
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            self.timestamp.append(time_x_i)


    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x = self.all_x[index]
        seq_y = self.all_y[index]
        time_stamp = self.timestamp[index]
        # seq_x = self.all_pairs[index].T[0]
        # seq_y = self.all_pairs[index].T[1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y,time_stamp, []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom3(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', round=0):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        # 0 代表第一段，1代表第二段，2代表第三段
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.timestamp = []
        self.all_prompt_bank = []
        self.__read_data__()
        self.round = round
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):
        self.scaler = StandardScaler()

        data = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print("===========loading file===========,",os.path.join(self.root_path, self.data_path))
        self._generate_pairs(data)
        
        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)
        self.all_x = np.concatenate(self.all_x, axis=0)
        self.all_y = np.concatenate(self.all_y, axis=0)
        #self.timestamp = np.concatenate(self.timestamp, axis=0)
    def _generate_pairs(self, df_raw):
        
        sampling_related = ['罐次铁水温度', '铁次平均铁水温度','铁次平均-铁口温度偏差(方法1)','铁次平均-核心温度(方法2)','铁次平均-铁口温度偏差(方法2)','堵口铁水温度','堵口铁水-核心温度(方法1)',
          '堵口铁水-铁口温度偏差(方法1)','堵口铁水-核心温度(方法2)','堵口铁水-铁口温度偏差(方法2)','铁次内硅平均值','出铁时长','相邻铁次出铁间隔','同一铁口出铁间隔','铁次出铁量','铁次出渣量','铁次铁流速度'
          ,'铁水Si','铁次铁流速度'
          ]

        labels = [
            '罐次温度铁口号','铁口号（平均/堵口温度）'
                ]
        
        
        test_intervals = [
            (pd.Timestamp('2022-05-03'), pd.Timestamp('2022-06-03')),
            (pd.Timestamp('2023-02-03'), pd.Timestamp('2023-03-03')),
            (pd.Timestamp('2024-05-20'), pd.Timestamp('2024-06-10'))
        ]
        columns_to_check = sampling_related + labels
        columns_to_check2 = [each for each in columns_to_check if each not in ['铁次平均铁水温度']]  # 保留铁次平均铁水温度


        df_raw = df_raw.drop(columns=columns_to_check2)
        # if self.round == 0:
        #     if self.set_type == '0':
        #         subset = df_raw[(df_raw['作业时刻'] <= test_intervals[0][0])]
        #         df_raw = subset.copy()  # 初始化训练集为整个子集
        # elif self.round==1:
        #     if self.set_type == '0':
        #         subset1 = df_raw[(df_raw['作业时刻'] <= test_intervals[0][0])]
        #         subset2 = df_raw[(df_raw['作业时刻'] <= test_intervals[1][0]) & (df_raw['作业时刻'] >= test_intervals[0][1])]
        #         subset = pd.concat([subset1, subset2], ignore_index=True)
        #         df_raw = subset.copy()  # 初始化训练集为整个子集
        #     elif self.set_type == '1':
        #         subset1 = df_raw[(df_raw['作业时刻'] >= test_intervals[1][0]) & df_raw['作业时刻'] <= test_intervals[1][1]]
        #         df_raw = subset1.copy()  # 初始化训练集为整个子集
        #     else:
        #         subset1 = df_raw[(df_raw['作业时刻'] <= test_intervals[0][0])]
        #         subset2 = df_raw[(df_raw['作业时刻'] <= test_intervals[1][0]) & (df_raw['作业时刻'] >= test_intervals[0][1])]
        #         subset = pd.concat([subset1, subset2], ignore_index=True)
        #         len_subset = len(subset.values)
        #         df_raw = subset.copy()  # 初始化训练集为整个子集
        # else:

        #     subset1 = df_raw[(df_raw['作业时刻'] <= test_intervals[0][0])]
        #     subset2 = df_raw[(df_raw['作业时刻'] <= test_intervals[1][0]) & (df_raw['作业时刻'] >= test_intervals[0][1])]
        #     subset3 = df_raw[(df_raw['作业时刻'] <= test_intervals[2][0]) & (df_raw['作业时刻'] >= test_intervals[1][1])]
        #     subset = pd.concat([subset1, subset2, subset3], ignore_index=True)
        #     df_raw = subset.drop(columns=columns_to_check2)  # 删除指标和观察项
        
        

        df_raw['作业时刻'] = pd.to_datetime(df_raw['作业时刻'])
        condition1 = df_raw['作业时刻'] <= test_intervals[0][0]
        condition2 = (df_raw['作业时刻'] >= test_intervals[1][0]) & (df_raw['作业时刻'] <= test_intervals[1][1])
        condition3 = (df_raw['作业时刻'] >= test_intervals[2][0]) & (df_raw['作业时刻'] <= test_intervals[2][1])

        # 使用按位或运算符组合条件
        combined_condition = condition1 | condition2 | condition3

        # 过滤数据框
        df_raw  = df_raw[combined_condition]

        #df_raw = df_raw[(df_raw['作业时刻'] <= test_intervals[0][0]) | (df_raw['作业时刻'] >=test_intervals[1][0] & df_raw['作业时刻'] <=test_intervals[1][1]) | (df_raw['作业时刻']<=test_intervals[2][1] & df_raw['作业时刻']>=test_intervals[2][0])]
        #df_test_sub = df_raw[~((df_raw['作业时刻'] <= test_intervals[0][0]) or (df_raw['作业时刻'] >=test_intervals[1][0] and df_raw['作业时刻'] <=test_intervals[1][1]) or (df_raw['作业时刻']<=test_intervals[2][1] and df_raw['作业时刻']>=test_intervals[2][0]))]


        valid_index = []
        # 直接判断数据是否有效
        for index, row in df_raw.iterrows():
            if not row.isnull().any():
                valid_index.append(row)
        
        df_raw = pd.DataFrame(valid_index)
            
        # 保留格式，第一列为时间，最后一列为预测的目标，最后一列也参与归一化
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('作业时刻')
        
        df_raw = df_raw[['作业时刻'] + cols + [self.target]]   
        num_train = int(len(df_raw) * 0.6)
        num_test = 0
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        for i in range(border1, border2 - self.pred_len - self.seq_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)


    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x = self.all_x[index]
        seq_y = self.all_y[index]
        #time_stamp = self.timestamp[index]
        # seq_x = self.all_pairs[index].T[0]
        # seq_y = self.all_pairs[index].T[1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y,[], []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = timefeatures(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Neuro(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))

        train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
        val_data_reshaped = val_data.reshape(-1, val_data.shape[-1])
        test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])

        if self.scale:
            self.scaler.fit(train_data_reshaped)
            train_data_scaled = self.scaler.transform(train_data_reshaped)
            val_data_scaled = self.scaler.transform(val_data_reshaped)
            test_data_scaled = self.scaler.transform(test_data_reshaped)

        train_scaled_orig_shape = train_data_scaled.reshape(train_data.shape)
        val_scaled_orig_shape = val_data_scaled.reshape(val_data.shape)
        test_scaled_orig_shape = test_data_scaled.reshape(test_data.shape)

        if self.set_type == 0:  # TRAIN
            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
            self.data_x = train_x
            self.data_y = train_y

        elif self.set_type == 1:  # VAL
            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
            self.data_x = val_x
            self.data_y = val_y

        elif self.set_type == 2:  # TEST
            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
            self.data_x = test_x
            self.data_y = test_y

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        pdb.set_trace()
        return self.scaler.inverse_transform(data)


class Dataset_Saugeen_Web(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        data_x = np.load(os.path.join(self.root_path, 'all_x_original.npy'))
        data_y = np.load(os.path.join(self.root_path, 'all_y_original.npy'))

        data_x_sensors_last = data_x
        data_y_sensors_last = data_y

        data_x_reshaped = data_x_sensors_last.reshape(-1, data_x_sensors_last.shape[-1])
        data_y_reshaped = data_y_sensors_last.reshape(-1, data_y_sensors_last.shape[-1])

        if self.scale:
            self.scaler.fit(data_x_reshaped)  # scaling based off of x --> for ltf this is very similar to y
            data_x_scaled = self.scaler.transform(data_x_reshaped)
            data_y_scaled = self.scaler.transform(data_y_reshaped)

        data_x_scaled_orig_shape = data_x_scaled.reshape(data_x_sensors_last.shape)
        data_y_scaled_orig_shape = data_y_scaled.reshape(data_y_sensors_last.shape)

        self.data_x = data_x_scaled_orig_shape
        self.data_y = data_y_scaled_orig_shape

        print(self.set_type, len(self.data_x), len(self.data_y), self.data_x[0].shape, self.data_y[0].shape)

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom4(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.timestamp = []
        self.all_prompt_bank = []
        self.__read_data__()
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        data = pd.read_csv(file_path)
        print("===========loading file===========,",os.path.join(self.root_path, self.data_path))
        csv_file = file_path.split('.')[0].split('/')[-1]
        self._generate_pairs(data, csv_file)
        
        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)

        # self.all_x = np.concatenate(self.all_x, axis=0)
        # self.all_y = np.concatenate(self.all_y, axis=0)
        # self.timestamp = np.concatenate(self.timestamp, axis=0)
        self.all_x = np.stack(self.all_x, axis=0)
        self.all_y = np.stack(self.all_y, axis=0)
        self.timestamp = np.stack(self.timestamp, axis=0)
        #self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need. 
    def _generate_pairs(self, df_raw, csv_file):
        scaler = StandardScaler()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        if csv_file in ['ETTh1', 'ETTh2']:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif csv_file in ['ETTm1', 'ETTm2']:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            time_col = df_raw.columns[:1]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            time_data = df_raw[time_col].values
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values
        # if csv_file in ['ETTh1', 'ETTh2', 'ETTm1' , 'ETTm2']:
        #     for i in range(border1, border2):
        #         batch_x_i = data[i:i + self.seq_len].T
        #         batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
        #         time_x_i  = time_data[i:i+self.seq_len]
        #         time_x_i = [str(time_x_i[0][0]),str(time_x_i[-1][0])]
        #         self.all_x.append(batch_x_i)
        #         self.all_y.append(batch_y_i)
        #         self.timestamp.append(time_x_i)
        # else:
        for i in range(border1, border2 - self.pred_len - self.seq_len - self.label_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
            time_x_i  = time_data[i:i+self.seq_len]
            time_x_i = [str(time_x_i[0][0]),str(time_x_i[-1][0])]
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            self.timestamp.append(time_x_i)


    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x = self.all_x[index]
        seq_y = self.all_y[index]
        time_stamp = self.timestamp[index]
        # seq_x = self.all_pairs[index].T[0]
        # seq_y = self.all_pairs[index].T[1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y,time_stamp, []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Custom5(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_x = []
        self.all_y = []
        self.timestamp = []
        self.all_prompt_bank = []
        self.__read_data__()
        # self._shuffle()

    def _shuffle(self):
        random.shuffle(self.all_pairs)

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        data = pd.read_csv(file_path)
        print("===========loading file===========,",os.path.join(self.root_path, self.data_path))
        csv_file = file_path.split('.')[0].split('/')[-1]
        self._generate_pairs(data, csv_file)
        
        # self.all_pairs = np.concatenate(self.all_pairs,axis=0)

        # self.all_x = np.concatenate(self.all_x, axis=0)
        # self.all_y = np.concatenate(self.all_y, axis=0)
        # self.timestamp = np.concatenate(self.timestamp, axis=0)
        self.all_x = np.stack(self.all_x, axis=0)
        self.all_y = np.stack(self.all_y, axis=0)
        self.timestamp = np.stack(self.timestamp, axis=0)
        #self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need. 
    def _generate_pairs(self, df_raw, csv_file):
        scaler = StandardScaler()
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        if csv_file in ['ETTh1', 'ETTh2']:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif csv_file in ['ETTm1', 'ETTm2']:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            time_col = df_raw.columns[:1]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            time_data = df_raw[time_col].values
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values
        # if csv_file in ['ETTh1', 'ETTh2', 'ETTm1' , 'ETTm2']:
        #     for i in range(border1, border2):
        #         batch_x_i = data[i:i + self.seq_len].T
        #         batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
        #         time_x_i  = time_data[i:i+self.seq_len]
        #         time_x_i = [str(time_x_i[0][0]),str(time_x_i[-1][0])]
        #         self.all_x.append(batch_x_i)
        #         self.all_y.append(batch_y_i)
        #         self.timestamp.append(time_x_i)
        # else:
        for i in range(border1, border2 - self.pred_len - self.seq_len - self.label_len):
            batch_x_i = data[i:i + self.seq_len].T
            batch_y_i = data[i + self.seq_len:i + self.pred_len + self.seq_len + self.label_len].T
            time_x_i  = time_data[i:i+self.seq_len]
            time_x_i = [str(time_x_i[0][0]),str(time_x_i[-1][0])]
            self.all_x.append(batch_x_i)
            self.all_y.append(batch_y_i)
            self.timestamp.append(time_x_i)


    def __getitem__(self, index):
        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x = self.all_x[index]
        seq_y = self.all_y[index]
        time_stamp = self.timestamp[index]
        # seq_x = self.all_pairs[index].T[0]
        # seq_y = self.all_pairs[index].T[1]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y,time_stamp, []  # seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.all_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)