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
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
min_max_scaler = preprocessing.MinMaxScaler()
torch.manual_seed(1)
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
    # print('train_df.columns:',train_df.columns)
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])

    # train_df.drop(labels=['id','cycle_norm','setting1','setting2','setting3','s1','s2','s3','s5','s6','s8','s9','s10','s13','s14','s16','s17','s18','s19','s20'], axis=1, inplace=True)
    # 对剩下名字的每一列分别进行特征放缩
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=train_df.index)
    # print('norm_train_df:',norm_train_df)
    # 将之前去掉的再加回特征放缩后的列表里面
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)
    train_df.drop("cycle", axis=1, inplace=True)
    train_df.drop(labels=drops14, axis=1, inplace=True)

    for i in range(len(train_df['RUL'])):
        if train_df['RUL'][i] > 125:
            train_df['RUL'][i] = 125
    return cols_normalize,train_df



# cols_normalize=data_pre('model1/datas/train_FD001.txt')[0]
# print('cols_normalize:',cols_normalize)
source_df = data_pre('model1/datas/train_FD002.txt')[1]
target_df = data_pre('model1/datas/train_FD002.txt')[1]


print('111111111111111111111111111111111111')

def data_pre2(name2,name3):

    test_df = pd.read_csv(name2, sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                       's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                       's18', 's19', 's20', 's21']
    truth_df = pd.read_csv(name3, sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    # 在列名里面去掉'id', 'cycle', 'RUL'这三个列名

    test_df['cycle_norm'] = test_df['cycle']
    cols_normalize = test_df.columns.difference(['id', 'cycle'])
    # norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
    #                             columns=cols_normalize,
    #                             index=test_df.index)
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=test_df.index)
    # print('norm_test_df:',norm_test_df)
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
print('xxxxxxxxxx')
test_df = data_pre2('model1/datas/test_FD002.txt','model1/datas/RUL_FD002.txt')
print('final_test_df:',test_df)

Embedding_size=64
embedding_dim=64
timestep=40
num_factor=14
batchsize=64
dp=0.7
M=8
epoch=100

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
source_Data=np.array(source_df.head(20480+timestep-1))
input,output = create_data(source_Data,timestep,num_factor)
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

# test_data=np.array(train_df[16413:20538])
# test_data=np.array(train_df[44061:52282])
# test_data=np.array(train_df[20529+2048:24674])
# test_data=np.array(test_df[0:16423])


# target_data=np.array(target_df.head(20480+timestep-1))
# target_data=np.array(test_df.head(12800+timestep-1)) #fd001
target_data=np.array(test_df.head(33920+timestep-1))   #fd002
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



class CNN1(nn.Module):
    def __init__(self,k_size):
        super(CNN1, self).__init__()
        self.k_size = k_size
        self.embedding=nn.Linear(num_factor,embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=self.k_size, stride=1, padding=2, ),
            nn.ReLU(),    # activation

            # nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        # self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
        #     nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(2),  # output shape (32, 7, 7)
        # )
        # self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        # self.out = nn.Linear(16*(embedding_dim-2), 1)
        self.out = nn.Linear((timestep-self.k_size+1) * (embedding_dim - self.k_size+1)*batchsize, embedding_dim)
    def forward(self, x):
        x = self.embedding(x)
        embedding_node = x
        # print('after_embedding:',x.shape)
        x = x.unsqueeze(2).reshape(batchsize,1,timestep,embedding_dim)
        # print('after_adddim: ',x.shape)
        x = self.conv1(x)
        # print('after_conv: ',x.shape)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x =x.squeeze(1)
        x = x+embedding_node
        output = x
        # print('this is test1:   ', output.shape)
        # output = self.out(x)
        # print('after_linnear:  ',output.shape)
        return output

class CNN2(nn.Module):
    def __init__(self,k_size):
        super(CNN2, self).__init__()
        self.k_size = k_size
        self.embedding=nn.Linear(num_factor,embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=self.k_size, stride=1, padding=5 ),
            nn.ReLU(),    # activation

            # nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        # self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
        #     nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(2),  # output shape (32, 7, 7)
        # )
        # self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        # self.out = nn.Linear(16*(embedding_dim-2), 1)
        self.out = nn.Linear((timestep-self.k_size+1) * (embedding_dim - self.k_size+1)*batchsize, embedding_dim)
    def forward(self, x):
        x = self.embedding(x)
        embedding_node = x
        # print('after_embedding:',x.shape)
        x = x.unsqueeze(2).reshape(batchsize,1,timestep,embedding_dim)
        # print('after_adddim: ',x.shape)
        x = self.conv1(x)
        # print('after_conv: ',x.shape)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = x.squeeze(1)
        # print('after_flatten:  ',x.shape)
        x = x + embedding_node
        # print('this is test1:   ', x.shape)
        # output = self.out(x)
        # print('after_linnear:  ',output.shape)
        output = x
        return output

class CNN3(nn.Module):
    def __init__(self,k_size):
        super(CNN3, self).__init__()
        self.k_size = k_size
        self.embedding=nn.Linear(num_factor,embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=self.k_size, stride=1, padding=7 ),
            nn.ReLU(),    # activation

            # nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        # self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
        #     nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
        #     nn.ReLU(),  # activation
        #     nn.MaxPool2d(2),  # output shape (32, 7, 7)
        # )
        # self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        # self.out = nn.Linear(16*(embedding_dim-2), 1)
        self.out = nn.Linear((timestep-self.k_size+1) * (embedding_dim - self.k_size+1)*batchsize, embedding_dim)
    def forward(self, x):
        x = self.embedding(x)
        embedding_node = x
        # print('after_embedding:',x.shape)
        x = x.unsqueeze(2).reshape(batchsize,1,timestep,embedding_dim)
        # print('after_adddim: ',x.shape)
        x = self.conv1(x)
        # print('after_conv: ',x.shape)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = x.squeeze(1)
        # print('after_flatten:  ',x.shape)
        x = x + embedding_node
        # print('kafagdbaD:',x.shape)
        # print('this is test1:   ', x.shape)
        # output = self.out(x)
        # print('after_linnear:  ',output.shape)
        output = x
        return output



class Encoder(nn.Module):
    def __init__(self,embedding_size,M):
        super().__init__()
        self.embedding_size = embedding_size
        self.M = M
        # self.selu = selu(x)
        self.dk = 1/((embedding_size // M)**0.5)
        self.wq = nn.Linear(embedding_size, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.WK = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)
        self.WV = nn.Linear(embedding_size, embedding_size)
        self.embedding = nn.Linear(num_factor, Embedding_size)

        self.w = nn.Linear(embedding_size, embedding_size)

        self.FFw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.dp = nn.Dropout(dp)
        self.FFb1 = nn.Linear(embedding_size * 4, embedding_size)
        # self.out = nn.Linear(embedding_size,)
        self.BN11 = nn.LayerNorm(embedding_size)
        self.BN12 = nn.LayerNorm(embedding_size)

        self.FFW2 = nn.Linear(embedding_size, embedding_size * 4)
        self.FFb2 = nn.Linear(embedding_size * 4, embedding_size)
        self.fc = nn.Linear(batchsize * embedding_size * timestep, batchsize)

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)
    def forward(self, embedding_node):  ##embedding_node is (batch,seq_len,embedding_size)
        # print(type(embedding_node))
        # print('embedding_node:',embedding_node.shape)
        # embedding_node=self.embedding(embedding_node)
        x_p = embedding_node      #输入(20,4,10)

        # q = self.wq(embedding_node)  # (batch,seq_len,embedding_size)
        # k = self.wk(embedding_node)  # (batch,seq_len,embedding)
        # v = self.wv(embedding_node)  # (batch,seq_len,embedding)
        # scores = torch.matmul(q, k.transpose(-2, -1))
        # x = F.softmax(scores, dim=-1)
        # x = torch.matmul(x, v)          #1. a * b 要求两个矩阵输入维度一致，即矩阵对应元素相乘
        #                                             #2. 当输入是二维矩阵时，torch.mm(a,b)和torch.matul(a,b)是一样的
        #                                 #3. torch.matul(a,b)可计算高维矩阵相乘，此时，把多出的一维作为batch提出来，其他部分做矩阵乘法
        # # print("x*v:",x.shape)       #(20,4,10)
        # print('asdasdas:',embedding_node.shape)
        q = self.wq(embedding_node)
        # print('q.shape:',q.shape)
        q = self.wq(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        k = self.wk(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        v = self.wv(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)

        dist=torch.matmul(q,k.transpose(2,3))*self.dk

        dist=torch.softmax(dist,dim=-1)
        att=torch.matmul(dist,v)

        x=att.transpose(1,2).reshape(batchsize,timestep,Embedding_size)


        # 第一层第一个BN
        # x = x.permute(0, 2, 1)
        # print("qqq:",x.shape)
        x = self.BN11(x)
        x = x + x_p
        # print("www:",x.shape)
        # x = x.permute(0, 2, 1)
        # print("eee:", x.shape)
        # x = t.tanh(x)
        #####################
        # 第一层FF
        x1 = self.FFw1(x)
        # print("eee:",x1.shape)
        # x1=self.dp(x1)
        # x1 = self.dp(x1)
        # x1 = F.relu(x1,inplace=True)
        x1 = self.selu(x1)
        # print("eee:", x1.shape)

        x1 = self.FFb1(x1)
        x1 = self.BN12(x1)


        x = x + x1
        # output = x.view(1, -1)
        # output = abs(self.fc(output))
        # output = F.relu(output)
        #####################
        # 第一层第二个BN
        # x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        return x

class Encoder1(nn.Module):
    def __init__(self,embedding_size,M):
        super().__init__()
        self.embedding_size = embedding_size
        self.M = M
        # self.selu = selu(x)
        self.dk = 1/((embedding_size // M)**0.5)
        self.wq = nn.Linear(embedding_size, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.WK = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)
        self.WV = nn.Linear(embedding_size, embedding_size)
        self.embedding = nn.Linear(num_factor, Embedding_size)

        self.w = nn.Linear(embedding_size, embedding_size)

        self.FFw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.dp = nn.Dropout(dp)
        self.FFb1 = nn.Linear(embedding_size * 4, embedding_size)
        # self.out = nn.Linear(embedding_size,)
        self.BN11 = nn.LayerNorm(embedding_size)
        self.BN12 = nn.LayerNorm(embedding_size)

        self.FFW2 = nn.Linear(embedding_size, embedding_size * 4)
        self.FFb2 = nn.Linear(embedding_size * 4, embedding_size)
        self.fc = nn.Linear(batchsize * embedding_size * timestep, batchsize)

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)
    def forward(self, embedding_node):  ##embedding_node is (batch,seq_len,embedding_size)
        # print(type(embedding_node))
        # print('embedding_node:',embedding_node.shape)
        # embedding_node=self.embedding(embedding_node)
        x_p = embedding_node      #输入(20,4,10)

        # q = self.wq(embedding_node)  # (batch,seq_len,embedding_size)
        # k = self.wk(embedding_node)  # (batch,seq_len,embedding)
        # v = self.wv(embedding_node)  # (batch,seq_len,embedding)
        # scores = torch.matmul(q, k.transpose(-2, -1))
        # x = F.softmax(scores, dim=-1)
        # x = torch.matmul(x, v)          #1. a * b 要求两个矩阵输入维度一致，即矩阵对应元素相乘
        #                                             #2. 当输入是二维矩阵时，torch.mm(a,b)和torch.matul(a,b)是一样的
        #                                 #3. torch.matul(a,b)可计算高维矩阵相乘，此时，把多出的一维作为batch提出来，其他部分做矩阵乘法
        # # print("x*v:",x.shape)       #(20,4,10)
        # print('asdasdas:',embedding_node.shape)
        q = self.wq(embedding_node)
        # print('q.shape:',q.shape)
        q = self.wq(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        k = self.wk(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        v = self.wv(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)

        dist=torch.matmul(q,k.transpose(2,3))*self.dk

        dist=torch.softmax(dist,dim=-1)
        att=torch.matmul(dist,v)

        x=att.transpose(1,2).reshape(batchsize,timestep,Embedding_size)


        # 第一层第一个BN
        # x = x.permute(0, 2, 1)
        # print("qqq:",x.shape)
        x = self.BN11(x)
        x = x + x_p
        # print("www:",x.shape)
        # x = x.permute(0, 2, 1)
        # print("eee:", x.shape)
        # x = t.tanh(x)
        #####################
        # 第一层FF
        x1 = self.FFw1(x)
        # print("eee:",x1.shape)
        # x1=self.dp(x1)
        # x1 = self.dp(x1)
        # x1 = F.relu(x1,inplace=True)
        x1 = self.selu(x1)
        # print("eee:", x1.shape)

        x1 = self.FFb1(x1)
        x1 = self.BN12(x1)


        x = x + x1
        # output = x.view(1, -1)
        # output = abs(self.fc(output))
        # output = F.relu(output)
        #####################
        # 第一层第二个BN
        # x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        return x

class Encoder2(nn.Module):
    def __init__(self,embedding_size,M):
        super().__init__()
        self.embedding_size = embedding_size
        self.M = M
        self.dk = 1/((embedding_size // M)**0.5)
        self.wq = nn.Linear(embedding_size, embedding_size)
        self.wk = nn.Linear(embedding_size, embedding_size)
        self.WK = nn.Linear(embedding_size, embedding_size)
        self.wv = nn.Linear(embedding_size, embedding_size)
        self.WV = nn.Linear(embedding_size, embedding_size)
        self.embedding = nn.Linear(num_factor, Embedding_size)

        self.w = nn.Linear(embedding_size, embedding_size)

        self.FFw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.dp = nn.Dropout(dp)
        self.FFb1 = nn.Linear(embedding_size * 4, embedding_size)
        # self.out = nn.Linear(embedding_size,)
        self.BN11 = nn.LayerNorm(embedding_size)
        self.BN12 = nn.LayerNorm(embedding_size)

        self.FFW2 = nn.Linear(embedding_size, embedding_size * 4)
        self.FFb2 = nn.Linear(embedding_size * 4, embedding_size)

        #Finetune 加载参数时使用
        for p in self.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.fc0 = nn.Linear(batchsize * embedding_size * timestep, batchsize)
    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * F.elu(x, alpha)

    def forward(self, embedding_node):  ##embedding_node is (batch,seq_len,embedding_size)
        # print(type(embedding_node))
        # print('embedding_node:',embedding_node.shape)
        # embedding_node=self.embedding(embedding_node)
        x_p = embedding_node      #输入(20,4,10)

        # q = self.wq(embedding_node)  # (batch,seq_len,embedding_size)
        # k = self.wk(embedding_node)  # (batch,seq_len,embedding)
        # v = self.wv(embedding_node)  # (batch,seq_len,embedding)
        # scores = torch.matmul(q, k.transpose(-2, -1))
        # x = F.softmax(scores, dim=-1)
        # x = torch.matmul(x, v)          #1. a * b 要求两个矩阵输入维度一致，即矩阵对应元素相乘
        #                                             #2. 当输入是二维矩阵时，torch.mm(a,b)和torch.matul(a,b)是一样的
        #                                 #3. torch.matul(a,b)可计算高维矩阵相乘，此时，把多出的一维作为batch提出来，其他部分做矩阵乘法
        # # print("x*v:",x.shape)       #(20,4,10)
        # print('asdasdas:',embedding_node.shape)
        q = self.wq(embedding_node)
        # print('q.shape:',q.shape)
        q = self.wq(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        k = self.wk(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)
        v = self.wv(embedding_node).reshape(batchsize, timestep, self.M, Embedding_size // self.M).transpose(1, 2)

        dist=torch.matmul(q,k.transpose(2,3))*self.dk

        dist=torch.softmax(dist,dim=-1)
        att=torch.matmul(dist,v)

        x=att.transpose(1,2).reshape(batchsize,timestep,Embedding_size)


        # 第一层第一个BN
        # x = x.permute(0, 2, 1)
        # print("qqq:",x.shape)
        x = self.BN11(x)
        x = x + x_p
        # print("www:",x.shape)
        # x = x.permute(0, 2, 1)
        # print("eee:", x.shape)
        # x = t.tanh(x)
        #####################
        # 第一层FF
        x1 = self.FFw1(x)
        # print("eee:",x1.shape)
        # x1=self.dp(x1)
        # x1 = self.dp(x1)
        x1 = self.selu(x1)
        # print("eee:", x1.shape)

        x1 = self.FFb1(x1)
        x1 = self.BN12(x1)


        x = x + x1
        x=x.view(1,-1)
        mmd_in = self.fc0(x)
        output = abs(self.fc(mmd_in))
        #####################
        # 第一层第二个BN
        # x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        return output,mmd_in


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.Encoder=Encoder(10,2)
        self.CNN1 = CNN1(5)
        self.CNN2 = CNN2(11)
        self.CNN3 = CNN3(15)
        # self.fcx = nn.Linear(1,2)
        # self.fc = nn.Linear(3,4)
        self.Encoder = Encoder(Embedding_size,8)
        self.Encoder1 = Encoder1(Embedding_size,8)
        self.Encoder2 = Encoder2(Embedding_size, 8)
        # self.pre_layer = pre_layer(Embedding_size)
    def forward(self,input):
        x1 = self.CNN1(input)
        x2 = self.CNN2(input)
        x3 = self.CNN3(input)

        x = x1+x2+x3

        # x =self.fcx(x)
        # print('duytfi:',x.shape)
        # x = x.view(1, -1)
        # x = self.fc(x)
        # x = F.relu(x)
        x = self.Encoder(x)
        x = self.Encoder1(x)
        mmd_in = self.Encoder2(x)[1]
        x = self.Encoder2(x)[0]
        return  x,mmd_in


model=Net()
learning_rate=0.0001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0000001)
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model).to(DEVICE)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
model.to(DEVICE)
loss_func = nn.MSELoss()



  ############修改网络层##############
# print(model.fc)
# flag=0
# flagx=0
# model.load_state_dict(torch.load('c1.pt',map_location='cpu'))
# print(model.fc)
# print('before:        ',model)
# for param in model.parameters():
#     param.requires_grad = True
# print(model.fc)
# num_fc_in = model.Encoder2.fc.in_features #获取到fc层的输入
# model.Encoder2.fc = nn.Linear(num_fc_in, 32) # 定义一个新的FC层
# model.Encoder2.fc2 = nn.Linear(32, 64)
# model=model.to(DEVICE)# 放到设备中
# print(model.fc)
# print('after:        ',model) # 最后再打印一下新的模型



# swa_model = AveragedModel(model)
# scheduler = CosineAnnealingLR(optimizer, T_max=100) # 使用学习率策略（余弦退火）
# swa_start = 5  # 设置SWA开始的周期，当epoch到该值的时候才开始记录模型的权重
# swa_scheduler = SWALR(optimizer, swa_lr=0.9) # 当SWA开始的时候，使用的学习率策略

# model.load_state_dict(torch.load('11.pt'))
# print("training here")
# for epoch in range(100):
#     model.train()
#     total_loss=0
#     # print('learning rate:',optimizer.param_groups[0]['lr'])
#     for i,datas in enumerate(TARGETLOADER,0):
#         inputs,labels = datas
#         inputs,label = inputs.to(DEVICE),labels.to(DEVICE)
#         inputs=inputs.to(torch.float32)
#         outputs=model(inputs)[0]
#         label = label.view(1,batchsize)
#         label = label.to(torch.float32)
#         loss = loss_func(outputs,label)
#         loss = torch.sqrt(loss)
#         running_loss = loss.cpu().detach().numpy()
#         total_loss += loss
#         optimizer.zero_grad()
#         # loss= loss.to(torch.float32)
#         loss.backward()
#         optimizer.step()
#
#         if i % 100==1:
#             print('[%d,  %5d] train_loss: %.3f'  %(epoch +1, i-1, loss))
#     swa_model.update_parameters(model)
#     swa_scheduler.step()
#     test_loss = total_loss / len(TARGETLOADER)
#     print('[%d] test_loss: %.3f' % (epoch + 1, test_loss))
#     if 21 < test_loss < 22:
#         torch.save(swa_model.state_dict(), os.path.join('t2_22.pt'))
#     if 20.5 < test_loss < 21:
#         torch.save(swa_model.state_dict(), os.path.join('t2_21.pt'))
#     if 20<test_loss < 20.5:
#         torch.save(swa_model.state_dict(), os.path.join('t2_20.5.pt'))
#
# print('finished train')



# print('testing here')
# for epoch in range(3):
#     total_loss = []
#     model.load_state_dict(torch.load('t1_20.5.pt'))
#     for i, datas in enumerate(TARGETLOADER, 0):
#         # total_loss=[]
#         # model.eval()
#         inputs, labels = datas
#         inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#         label = labels
#         # label = np.array(labels)
#         # label = torch.from_numpy(label)
#         # inputs, label = inputs.to(DEVICE), label.to(DEVICE)
#         inputs = inputs.float()
#         # print(inputs.shape)
#         outputs = model(inputs)[0]
#         # loss=torch.sqrt(loss_func(label,outputs))
#         label = label.view(1, batchsize)
#         label = label.cpu().detach().numpy()
#         outputs = outputs.cpu().detach().numpy()
#         # print(type(label), type(outputs))
#         print(label.shape,outputs.shape)
#         loss = np.sqrt(metrics.mean_squared_error(label, outputs))
#         total_loss.append(loss)
#
#         # if i==0:
#         #     print('i:', i, 'loss:', loss)
#     # if np.mean(total_loss)-1>loss:
#     #     flag=1
#     print('test_loss:               ', np.mean(total_loss))





class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)   #在第一个维度上堆叠,将源域与目标域拼接。
        # print('total:  ',total)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # print('total0:  ', total0)
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # print('total1:  ', total1)
        L2_distance = ((total0-total1)**2).sum(2)
        # print(L2_distance)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


A=MMD_loss()
source=torch.randn(2,3)
target=torch.randn(1,3)
print("source:",source)
print('target:',target)
print(torch.cat([source, target], dim=0).shape)
b=A(source,target)
print(b)
print('next000000000000')
pd.set_option('display.max_columns', None)

# MMD 计算 test
# for i, datas in enumerate(TRAINLOADER, 0):
#     inputs, labels = datas
#     print('inputs:',inputs.shape)
#     print('iteration:', i)
#     inputs = inputs.view(inputs.shape[0],timestep*num_factor)
#     print('inputs:', inputs.shape)
#     # print('labels:',labels.shape)
#     inputs, label = inputs.to(DEVICE), labels.to(DEVICE)
#     inputs = inputs.to(torch.float32)
#     outputs = A(inputs,inputs)
#     print(outputs)
#####微调模型中的某几层
address = 'F:/桌面内容/CT model/DA'  # 模型保存路径
# address='D:/xjy/untitled11/transformer model'
model=Net()
model.to(DEVICE)
learning_rate=0.0000001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0000001)
loss_func = nn.MSELoss()





class TransferNet(nn.Module):
    def __init__(self,
                 # num_class,
                 base_net='MSCNNTE',
                 transfer_loss='mmd',
                 use_bottleneck=False,
                 bottleneck_width=256,
                 width=1024):
        super(TransferNet, self).__init__()
        if base_net == 'MSCNNTE':
            self.base_network = Net()
        else:
            # Your own basenet
            return

    def forward(self, source, target):



        source_mmd = self.base_network(source)[1]    #源域特征
        target_mmd = self.base_network(target)[1]    #目标与特征
        source_clf = self.base_network(source)[0]    #RUL预测值source
        # target_clf = self.base_network(target)[1]    #RUL预测值target
        y_pre = source_clf
        # y_pre = source_clf+target_clf
        print('source_mmd:',source_mmd.shape)
        print('target_mmd:',target_mmd.shape)
        transfer_loss = self.adapt_loss(source_mmd, target_mmd, 'mmd')  # 域Loss

        TRANSFER_LOSS = transfer_loss
        return y_pre, TRANSFER_LOSS

    #     print('source.shape:', source.shape)
    #     print('target.shape:', target.shape)
    #     source = self.base_network(source)
    #     print('source.shape:', source.shape)
    #     target = self.base_network(target)
    #     print('target.shape:', target.shape)
    #     # source_clf = self.classifier_layer(source)
    #     source_clf = self.base_network(source)[1]
    #     if self.use_bottleneck:
    #         source = self.bottleneck_layer(source)
    #         target = self.bottleneck_layer(target)
    #     transfer_loss = self.adapt_loss(source_mmd, target_mmd, self.transfer_loss)
    #     print('source:',source_clf)
    #     print('transfer_loss:', transfer_loss)
    #     return source_clf, transfer_loss
    #
    # def predict(self, x):
    #     features = self.base_network(x)
    #     clf = self.classifier_layer(features)
    #     return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        # elif adapt_loss == 'coral':
        #     loss = CORAL(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss

model=TransferNet()
model.to(DEVICE)
learning_rate=0.0001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_func = nn.MSELoss()
lamb = 0.1

def train(model, optimizer):
    # model.load_state_dict(torch.load('pretrain_pram.pt'))
    source_loader, target_train_loader = SOURCELOADER, TARGETLOADER
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    stop = 0
    n_batch = min(len_source_loader, len_target_loader)
    for e in range(100):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        model.train()
        print('epoch_num:',e)
        for (src, tar) in zip(source_loader, target_train_loader):
            data_source, label_source = src
            data_target, _ = tar
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            data_source = data_source.float()
            label_source = label_source.float()
            data_target = data_target.float()
            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)  # TransferNet先后输出预测值，域loss
            clf_loss = loss_func(label_source_pred, label_source)  # 标签loss
            loss_y = torch.sqrt(clf_loss)
            print('[%d] train_loss: %.3f' % (e + 1, loss_y))
            loss = clf_loss + lamb * transfer_loss
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), os.path.join('pram.pt'))

train(model, optimizer)




embedding_size=64
class finetuneNet(nn.Module):
    def __init__(self, base_net='MSCNNTE',):
        super(finetuneNet, self).__init__()
        # if base_net == 'MSCNNTE':
        self.base_network = Net()
        # for param in self.base_network.parameters():
        #     param.requires_grad=False
        # # self.base_network.Encoder2.fc0=nn.Linear(batchsize * embedding_size * timestep, batchsize)
        # # self.base_network.Encoder2.fc =nn.Linear(embedding_size, embedding_size)
    def forward(self,target):
        target_clf = self.base_network(target)[0]    #RUL预测值target
        return target_clf
########finetune###################

# model.load_state_dict(torch.load('t1_20.5.pt'),strict=False)
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model).to(DEVICE)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
model.to(DEVICE)
loss_func = nn.MSELoss()

def finetunemodel(model,optimizer):
    # model.load_state_dict(torch.load('pretrain_pram_1.pt'))
    for epoch in range(100):
        model.train()
        total_loss = 0
        for i, datas in enumerate(TARGETLOADER, 0):
            inputs, labels = datas
            inputs, label = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)[0]
            label = label.view(1, batchsize)
            label = label.to(torch.float32)
            loss = loss_func(outputs, label)
            loss = torch.sqrt(loss)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 1:
                print('[%d,  %5d] train_loss: %.3f' % (epoch + 1, i - 1, loss))
        test_loss = total_loss / len(TARGETLOADER)
        print('[%d] test_loss: %.3f' % (epoch + 1, test_loss))
        if 19 < test_loss < 20:
            torch.save(model.state_dict(), os.path.join('pram2_19.0.pt'))
# optimizer = torch.optim.Adam([{'params':[model.base_network.Encoder2.fc0.weight],'lr':0.0001},
#                               {'params':[model.base_network.Encoder2.fc.weight],'lr':0.00001}])


model=finetuneNet()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0000001)
model.to(DEVICE)
print('finetuning!!')
# finetunemodel(model,optimizer)



#测试保存的Finetune模型
def finetunetest():
    print('testing here')
    model.load_state_dict(torch.load('pram1_13.5.pt'))
    for epoch in range(3):
        total_loss = []
        for i, datas in enumerate(TARGETLOADER, 0):
            # total_loss=[]
            # model.eval()
            inputs, labels = datas
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            label = labels
            # label = np.array(labels)
            # label = torch.from_numpy(label)
            # inputs, label = inputs.to(DEVICE), label.to(DEVICE)
            inputs = inputs.float()
            # print(inputs.shape)
            outputs = model(inputs)[0]
            # loss=torch.sqrt(loss_func(label,outputs))
            label = label.view(1, batchsize)
            label = label.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            # print(type(label), type(outputs))
            label=label.squeeze()
            loss = np.sqrt(metrics.mean_squared_error(label, outputs))
            total_loss.append(loss)

        print('test_loss:               ', np.mean(total_loss))
# finetunetest()