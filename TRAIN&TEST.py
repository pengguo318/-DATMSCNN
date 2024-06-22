import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch import optim
from sklearn import metrics
import time
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
import os
from torch.autograd import Variable
from sklearn import preprocessing
from torch.utils import data


from DATA import sourceDataset
from DATA import targetDataset
from DATA import TestDataset

from NET import Net
from NET import MMD_loss
from NET import TransferNet


print("111111111")

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

from para import configs
Embedding_size=configs.Embedding_size
timestep=configs.timestep
num_factor=configs.num_factor
batchsize=configs.batchsize
dp=configs.dp
M=configs.M
epoch=configs.epoch
learning_rate=configs.learning_rate
weight_decay_value=configs.weight_decay_value
lamb = configs.lamb




model=TransferNet()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_func = nn.MSELoss()


data1=sourceDataset()
SOURCELOADER=data.DataLoader(data1,batch_size=batchsize,shuffle=True,num_workers=0)

data2=targetDataset()
TARGETLOADER=data.DataLoader(data2,batch_size=batchsize,shuffle=True,num_workers=0)


data3=TestDataset()
TESTLOADER=data.DataLoader(data3,batch_size=batchsize,shuffle=True,num_workers=0)




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

# train(model, optimizer)



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

        torch.save(model.state_dict(), os.path.join('pram2.pt'))
# optimizer = torch.optim.Adam([{'params':[model.base_network.Encoder2.fc0.weight],'lr':0.0001},
#                               {'params':[model.base_network.Encoder2.fc.weight],'lr':0.00001}])


model=finetuneNet()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay_value)
model.to(DEVICE)
# finetunemodel(model,optimizer)



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