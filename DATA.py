#coding:utf-8
import random
import math
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch import optim
from sklearn import metrics
import time
import numpy as np
import os
from torch.autograd import Variable
from sklearn import preprocessing
from torch.utils import data
from sklearn.cluster import KMeans

from torch.optim.lr_scheduler import CosineAnnealingLR
# torch.set_default_tensor_type(torch.FloatTensor)


from para import configs

Embedding_size=configs.Embedding_size
timestep=configs.timestep
num_factor=configs.num_factor
batchsize=configs.batchsize
dp=configs.dp
M=configs.M
epoch=configs.epoch



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
min_max_scaler = preprocessing.MinMaxScaler()

drops21=['id','cycle_norm','setting1','setting2','setting3']
drops6=['id','cycle_norm','setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19','s2','s3','s8','s9','s13','s14','s17','s20']
# drops14=['id','cycle_norm','s1','s5','s6','s10','s16','s18','s19']
drops3=['id','cycle_norm','setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19','s2','s3','s8','s9','s13','s14','s17','s20','s7','s15','s21']
drops14=['id','cycle_norm','setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19']

def data_pre(name):

    train_df = pd.read_csv(name, sep=" ", header=None)  # train_dr.shape=(20631, 28)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # 去掉26,27列并用新生成的数组替换原数组
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                        's18', 's19', 's20', 's21']
    # print('asdasdas:',train_df)
    # 先按照'id'列的元素进行排序，当'id'列的元素相同时按照'cycle'列进行排序
    train_df = train_df.sort_values(['id', 'cycle'])
    """Data Labeling - generate column RUL"""
    # 按照'id'来进行分组，并求出每个组里面'cycle'的最大值,此时它的索引列将变为id
    # 所以用reset_index()将索引列还原为最初的索引
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    # 将rul通过'id'合并到train_df上，即在相同'id'时将rul里的max值附在train_df的最后一列
    train_df = train_df.merge(rul, on=['id'], how='left')
    # 加一列，列名为'RUL'
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    # 将'max'这一列从train_df中去掉
    train_df.drop('max', axis=1, inplace=True)
    """MinMax normalization train"""
    # 将'cycle'这一列复制给新的一列'cycle_norm'
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])

    # train_df.drop(labels=['id','cycle_norm','setting1','setting2','setting3','s1','s2','s3','s5','s6','s8','s9','s10','s13','s14','s16','s17','s18','s19','s20'], axis=1, inplace=True)
    # 对剩下名字的每一列分别进行特征放缩
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=train_df.index)
    # 将之前去掉的再加回特征放缩后的列表里面
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)
    train_df.drop("cycle", axis=1, inplace=True)
    train_df.drop(labels=drops14, axis=1, inplace=True)

    for i in range(len(train_df['RUL'])):
        if train_df['RUL'][i] > 125:
            train_df['RUL'][i] = 125
    return cols_normalize,train_df



cols_normalize=data_pre('train_FD001.txt')[0]
source_df = data_pre('train_FD001.txt')[1]
target_df = data_pre('train_FD002.txt')[1]


print('111111111111111111111111111111111111')

def data_pre2(name2):

    test_df = pd.read_csv(name2, sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                       's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                       's18', 's19', 's20', 's21']
    truth_df = pd.read_csv('model1/datas/RUL_FD001.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    # 在列名里面去掉'id', 'cycle', 'RUL'这三个列名

    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

    """generate column max for test data"""
    # 第一列是id，第二列是同一个id对应的最大cycle值
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    # 将列名改为id和max
    rul.columns = ['id', 'max']
    # 给rul文件里的数据列命名为'more'
    truth_df.columns = ['more']
    # 给truth_df增加id列，值为truth_df的索引加一
    truth_df['id'] = truth_df.index + 1
    # 给truth_df增加max列，值为rul的max列值加truth_df的more列,
    # truth_df['max']的元素是测试集里面每个id的最大cycle值加rul里每个id的真实剩余寿命
    truth_df['max'] = rul['max'] + truth_df['more']
    # 将'more'这一列从truth_df中去掉
    truth_df.drop('more', axis=1, inplace=True)
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    test_df.drop('cycle',axis=1,inplace=True)

    test_df.drop(labels=drops14, axis=1, inplace=True)
    for i in range(len(test_df['RUL'])):
        if test_df['RUL'][i]>125:
            test_df['RUL'][i]=125
    return test_df

test_df = data_pre2('test_FD002.txt')



# Embedding_size=64
embedding_dim=64
# timestep=40
# num_factor=14
# batchsize=64
# dp=0.7
# M=8
# epoch=301

def create_data(Data,timestep,num_factor):
    inputs = []
    outputs= []
    for i in range(Data.shape[0] - timestep + 1):
        if i == Data.shape[0] - timestep + 1:
            input = Data[i:, 0:num_factor]
            output = Data[i + timestep - 1, -1]
            inputs.append(input)
            outputs.append(output)
        else:
            input = Data[i:i + timestep, 0:num_factor]
            output = Data[i + timestep - 1, -1]
            inputs.append(input)
            outputs.append(output)
    #list转换为torch
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    outputs = np.array(outputs)
    outputs = torch.from_numpy(outputs)
    outputs=outputs.unsqueeze(1)
    return inputs,outputs



# input=train_df.head(49152+12+timestep-1)
# input=train_df.head(19200+timestep-1)
# input=train_df.head(24576+timestep-1)
source_Data=np.array(source_df.head(19200+timestep-1))
input,output = create_data(source_Data,timestep,num_factor)
print(input.shape,output.shape)
class sourceDataset(data.Dataset):
    def __init__(self):
        self.Data=input.detach().numpy()
        self.Label=output.detach().numpy()
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)
data1=sourceDataset()
SOURCELOADER=data.DataLoader(data1,batch_size=batchsize,shuffle=True,num_workers=0)
print('SOURCELOADER:',len(SOURCELOADER))

# for i,traindata in enumerate(TRAINLOADER):
#     print('i:',i)
#     data2,label1=traindata
#     print('data:',data2.shape)
#     print('Label:',label1.shape)


# test_data=np.array(test_df[0:16423])


target_data=np.array(target_df.head(25600+timestep-1))
input,output = create_data(target_data,timestep,num_factor)
class targetDataset(data.Dataset):
    def __init__(self):
        self.Data=input.detach().numpy()
        self.Label=output.detach().numpy()
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)
data1=targetDataset()
TARGETLOADER=data.DataLoader(data1,batch_size=batchsize,shuffle=True,num_workers=0)
print('TARGETLOADER:',len(TARGETLOADER))



print(test_df.shape)
test_data=np.array(test_df.head(33920+timestep-1))


# test_data=np.array(test_df.head(13056+timestep-1))
# test_data=np.array(train_df[49152+timestep-1:49152+2*timestep+2046])
# test_data=np.array(train_df[18432+timestep-1:18432+2*timestep+2046])
test_input,test_output=create_data(test_data,timestep,num_factor)
class TestDataset(data.Dataset):
    def __init__(self):
        self.Data=test_input.detach().numpy()
        self.Label=test_output.detach().numpy()
    def __getitem__(self,index):
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label
    def __len__(self):
        return len(self.Data)
testdata=TestDataset()
TESTLOADER=data.DataLoader(testdata,batch_size=batchsize,shuffle=True,num_workers=0)
print('TESTLOADER:',len(TESTLOADER))