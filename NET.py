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

from para import configs

Embedding_size=configs.Embedding_size
embedding_dim = configs.Embedding_size
timestep=configs.timestep
num_factor=configs.num_factor
batchsize=configs.batchsize
dp=configs.dp
M=configs.M
epoch=configs.epoch


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