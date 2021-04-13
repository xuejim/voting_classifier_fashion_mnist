#!/usr/bin/env python
# coding: utf-8

# In[177]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import visdom
import numpy as np
import os.path
import torch.nn.functional as F


# In[178]:
'''
1. 加载数据：pytorch datasets自带Fashion-mnist数据集
2. 建立模型：四个基础模型，一个votingclassifier
3. 训练模型：每个模型训练50个epoch
4. 评估模型：测试准确率
'''

# 定义参数
batch_size = 50
category = 10
epoches = 50
path_1 = './model_state/test_acc_76.pth'
path_2 = './model_state/test_acc_89.pth'
path_3 = './model_state/test_acc_91.pth'
path_4 = './model_state/test_acc_92.pth'

# 加载fashion-mnist数据到dataloader
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_dl = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_set,batch_size=batch_size,shuffle=True)


# In[179]:

# 定义模型
class MLP(nn.Module):
    
    def __init__(self):
        super(MLP,self).__init__()
        self.ln1 = nn.Linear(28*28*1,1000)
        self.ln2 = nn.Linear(1000,500)
        self.ln3 = nn.Linear(500,200)
        self.ln4 = nn.Linear(200,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,input):
        input = input.reshape(batch_size,784)
        x = self.ln1(input)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.ln4(x)
        x = self.softmax(x)
        
        return x
    
class MyConv(nn.Module):
    
    def __init__(self):
        super(MyConv,self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,1,2)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.maxpool = nn.MaxPool2d(2)
        self.ln1 = nn.Linear(32*7*7,64)
        self.ln2 = nn.Linear(64,32)
        self.ln3 = nn.Linear(32,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,input):
        # input batch_size*1*28*28
        x = self.conv1(input)# batch_size*16*28*28 
        x = self.relu(x)
        x = self.maxpool(x)# batch_size*16*14*14
        
        x = self.conv2(x)# batch_size*32*14*14
        x = self.relu(x)
        x = self.maxpool(x)# batch_size*32*7*7
        
        x = x.view(x.shape[0],-1)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.ln3(x)
        x = self.softmax(x)
        
        return x
    
class MyConv1(nn.Module):
    
    def __init__(self):
        super(MyConv1,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) 
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))# bacth_size*20*24*24
        x = F.max_pool2d(x, 2, 2)# bacth_size*20*12*12
        x = F.relu(self.conv2(x))# bacth_size*50*8*8
        x = F.max_pool2d(x, 2, 2)# bacth_size*50*4*4
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MyConv2(nn.Module):
    
    def __init__(self):
        super(MyConv2,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1) 
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)


    def forward(self,x):
        x = self.conv1(x)# bacth_size*64*30*30
        x = self.conv2(x)# bacth_size*64*30*30
        x = self.pool1(x)# bacth_size*64*15*15
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)# bacth_size*128*15*15
        x = self.conv4(x)# bacth_size*128*15*15
        x = self.pool2(x)# bacth_size*128*8*8
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(-1,128*8*8)# flatten
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x


# In[180]:

# votingclassifier，评估各个模型在一个类别上的概率
def voting_classifier(batch_size,category,pred_list):
    sum_pred = 0
    cdx_list = []
    idx_list = []
    name_dict = {}
    new_pred_list = []
    
    for index,pred in enumerate(pred_list):
        name_dict['pred_'+str(index)] = pred
        
    for values in name_dict.values():
        sum_pred = sum_pred+values
        
    return sum_pred/len(pred_list)


# In[181]:

# 初始化模型、定义损失函数（交叉熵）
model_mlp = MLP()
model_myconv = MyConv()
model_myconv1 = MyConv1()
model_myconv2 = MyConv2()
loss_fn = nn.CrossEntropyLoss()


# In[182]:

# 训练模型函数
loss = 0
def train_model(epoches):
    for epoch in range(1,epoches+1):
        for index,(x,y) in enumerate(train_dl):
            pred_1 = model_mlp(x)
            pred_2 = model_myconv(x)
            pred_3 = model_myconv1(x)
            pred_4 = model_myconv2(x)

            optim_mlp.zero_grad()
            optim_myconv.zero_grad()
            optim_myconv1.zero_grad()
            optim_myconv2.zero_grad()

            loss_mlp = loss_fn(pred_1,y)
            loss_myconv = loss_fn(pred_2,y) 
            loss_myconv1 = loss_fn(pred_3,y) 
            loss_myconv2 = loss_fn(pred_4,y)
            loss_mlp.backward()
            loss_myconv.backward()
            loss_myconv1.backward()
            loss_myconv2.backward()

            optim_mlp.step()
            optim_myconv.step()
            optim_myconv1.step()
            optim_myconv2.step()


# In[183]:

# 模型参数文件若存在则直接加载参数，否则进行训练
if(os.path.isfile(path_1)):
    model_mlp.load_state_dict(torch.load(path_1))
    model_myconv.load_state_dict(torch.load(path_2))
    model_myconv1.load_state_dict(torch.load(path_3))
    model_myconv2.load_state_dict(torch.load(path_4))
else:
    optim_mlp = torch.optim.Adam(model_mlp.parameters(),lr=1e-3)
    optim_myconv = torch.optim.Adam(model_myconv.parameters(),lr=1e-3)
    optim_myconv1 = torch.optim.Adam(model_myconv1.parameters(),lr=1e-3)
    train_model(epoches)


# In[195]:

# 评估测试准确率（一个bacth里的测试准确率）
def eval_acc(y_pred,y):   
    return np.array(torch.max(y_pred,dim=1).indices == y).astype('int32').mean()
# 评估测试准确率（一个epoch里的测试准确率）
def eval_total_acc(list_total_pred,round_num):
    return np.round(np.array(list_total_pred).mean(),round_num)


# In[210]:

# pred_list存放每个模型预测的结果，list_acc存放一个epoch的预测准确率
list_acc = []
pred_list = []
list_acc_1 = []
list_acc_2 = []
list_acc_3 = []
list_acc_4 = []
opts = {
    'title':'mlp vs myconv vs myconv1 vs myconv2 vs vc',
    'x_label':'batch_idx',
    'y_label':'test_acc',
    'legend':['mlp','myconv','myconv1','myconv2','vc']
    
}
# 使用visdom可视化
vs = visdom.Visdom(env='test-vc')
# 评估模型
with torch.no_grad():
    for index,(x,y) in enumerate(test_dl):
        pred_mlp = model_mlp(x)
        pred_myconv = model_myconv(x)
        pred_myconv1 = model_myconv1(x)
        pred_myconv2 = model_myconv2(x)
        pred_list.append(pred_mlp)
        pred_list.append(pred_myconv)
        pred_list.append(pred_myconv1)
        pred_list.append(pred_myconv2)
        y_pred = voting_classifier(batch_size,category,pred_list)
        
        y_acc_1 = eval_acc(pred_mlp,y)
        y_acc_2 = eval_acc(pred_myconv,y)
        y_acc_3 = eval_acc(pred_myconv1,y)
        y_acc_4 = eval_acc(pred_myconv2,y)
        y_acc = eval_acc(y_pred,y)
        pred_list.clear()
        if(index%10 == 0):
            vs.line(
                X=[index],
                Y=[[y_acc_1,y_acc_2,y_acc_3,y_acc_4,y_acc]],
                opts=opts,
                win='test-vc',
                update='append'
            )
        list_acc.append(y_acc)
        list_acc_1.append(y_acc_1)
        list_acc_2.append(y_acc_2)
        list_acc_3.append(y_acc_3)
        list_acc_4.append(y_acc_4)
    print('[vc:',eval_total_acc(list_acc,3),
          'mlp:',eval_total_acc(list_acc_1,3),
          'myconv:',eval_total_acc(list_acc_2,3),
          'myconv1:',eval_total_acc(list_acc_3,3),
         'myconv2:',eval_total_acc(list_acc_4,3),']')


# [vc: 0.933 mlp: 0.885 myconv: 0.892 myconv1: 0.907 myconv2: 0.925 ]




