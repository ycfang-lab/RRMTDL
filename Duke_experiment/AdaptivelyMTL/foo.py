# from __future__ import unicode_literals, print_function, division
# from io import open
# import glob
# import os
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# def findFiles(path): return glob.glob(path)

# print(findFiles('C:\\Users\\CSR\\Desktop\\data\\names\\*.txt'))

# import unicodedata
# import string

# all_letters = string.ascii_letters + " .,;'"
# n_letters = len(all_letters)

# # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#         and c in all_letters
#     )

# print(unicodeToAscii('Ślusàrski'))

# category_lines = {}
# all_categories = []

# def readLines(filename):
#     with  open(filename,encoding='utf-8') as f:
#         lines = f.read().strip().split('\n')
#         return [unicodeToAscii(line) for line in lines]
    
# for filename in findFiles('C:\\Users\\CSR\\Desktop\\data\\names\\*.txt'):
#     category = os.path.splitext(os.path.basename(filename))[0]
#     all_categories.append(category)
#     lines = readLines(filename)
#     category_lines[category] = lines
    
# n_categories = len(all_categories)

# print(category_lines['Italian'][:5])

# import torch
# def letterToIndex(letter):
#     return all_letters.find(letter)

# def letterToTensor(letter):
#     tensor = torch.zeros(1, n_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor

# def lineToTensor(line):
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][letterToIndex(letter)] = 1
#     return tensor

# print(letterToTensor('J'))
# print(lineToTensor('Jone').size())

# import torch.nn as nn

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.h2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.h2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)

# n_hidden = 128
# rnn = RNN(n_letters, n_hidden, n_categories)

# # input = letterToTensor('A')
# # hidden = torch.zeros(1, n_hidden)

# # output, next_hidden = rnn(input, hidden)

# input = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)

# output, next_hidden = rnn(input[0], hidden)
# print(output)

# def categoryFromOutput(output):
#     top_n, top_i = output.topk(1)
#     category_i = top_i[0].item()
#     return all_categories[category_i], category_i

# import random

# def randomChoice(l):
#     return l[random.randint(0, len(l) - 1)]

# def randomTrainingExample():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
#     line_tensor = lineToTensor(line)
#     return category, line, category_tensor, line_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)


# learning_rate = 0.005

# criterion = nn.NLLLoss()

# def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

#     rnn.zero_grad()

#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)

#     loss = criterion(output, category_tensor)
#     loss.backward()

#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)

#     return output, loss.item()

# import time
# import math

# n_iters = 100000
# print_every = 5000
# plot_every = 1000

# current_loss = 0
# all_losses = []

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# start = time.time()

# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss

#     # Print iter number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0

# plt.figure()
# plt.plot(all_losses)
# plt.show()

# import torch
# from torch import nn
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--echo', default='fuck')
# arg = parser.parse_args()
# print(arg.echo)

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred


# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Construct our model by instantiating the class defined above
# model = TwoLayerNet(D_in, H, D_out)

# print(list(model.parameters()))





# class net2(nn.Module):
#     def __init__(self):
#         super(net2, self).__init__()
#         #self.linears = [nn.Linear(10,10) for i in range(2)]
#         self.linear1 = torch.nn.Linear(10,10)
#         self.linear2 = torch.nn.Linear(10,10)
#         self.linears = [self.linear1, self.linear2]
#     def forward(self, x):
#         for m in self.linears:
#             x = m(x)
#         return x

# net = net2()
# print(net)
# # net2()
# print(list(net.parameters())
# import copy,torch
# import torchvision.models as models
# pretrained = True
# A = models.resnet50()
# a = A.layer4
# b = A.layer4
# c = copy.deepcopy(A.layer4)

# print(a is b)
# print(a is c)
# print(c)

# a = {'1':1,'2':2}
# b = a
# c = copy.deepcopy(a)
# a['1'] = 3
# print(b)
# print(c)

# print

# import torch
# a = torch.unbind(torch.tensor([[1, 2, 3],
#                            [4, 5, 6],
#                            [7, 8, 9]]), 1)
# print(a)

# import torch,copy
# import torchvision.models as models

# model = models.resnet50()
# print(model)

# with open('C:\\Users\\CSR\\Desktop\\scmtl extended experiment\\MCNN-AUX\\label_txt\\Black_Hair_Straight_Hair_rest.txt','r') as f_rest:
#     list_rest = f_rest.readlines()[8000]
#     print(list_rest)
import torch
import numpy as np
import Networks
import ReadData
import Util
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# w = torch.empty(3, 5)
# nn.init.constant_(w, 0.3)
# print(w.requires_grad_())
# a = [torch.tensor([[1,2],[3,4]]), torch.tensor([[1,2],[3,4]])]
# print(sum(a))
task_group = {'group_1':['Black_Hair'], 'group_2':['Straight_Hair']}
task_list = ['Brown_Hair', 'Straight_Hair']
task_num = 2
device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")
root_dir = 'F:\\datasetandbackup\\CelebA\\Img\\img_align_celeba'
label_dir = 'C:\\Users\\CSR\\Desktop\\scmtl_extended_experiment\\MCNN-AUX\\label_txt'
data_transform = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])

batch_size = 64

validation_set = ReadData.CelebASpecializedDataset(root_dir, label_dir, task_list[0], task_list[1], 'validation_1', data_transform)
validation_dataloader =  DataLoader(validation_set, batch_size=batch_size, shuffle=True)

print(len(validation_set))

# a = iter(validation_dataloader)
# print(next(a))
# try:
#     for i in range(1000):
#         print(next(a).keys())
# except StopIteration:
#     a = iter(validation_dataloader)
#     print('fuck')
#     print(next(a))

# import scipy
# import numpy as np
from scipy.stats import norm
from math import log

# x_norm = np.random.normal(2,1,(10,))
# x_mean, x_std = norm.fit(x_norm)
# print(x_mean)
# print(x_std)

#该函数用来进行极大似然估计，返回值是四个正态分布的均值与标准差(mean与std)
# p_eeg_score EEG的分类器对于正类的样本的输出值，一个N维的numpy数组，N指样本个数 
# n_eeg_score EEG的分类器对于负类的样本的输出值，一个N维的numpy数组，N指样本个数 
# p_face_score 人脸的分类器对于正类的样本的输出值，一个N维的numpy数组，N指样本个数 
# n_face_score 人脸的分类器对于负类的样本的输出值，一个N维的numpy数组，N指样本个数 
def baye_fusion(p_eeg_score, n_eeg_score, p_face_score, n_face_score):
    p_eeg_mean, p_eeg_std = norm.fit(p_eeg_score)
    n_eeg_mean, n_eeg_std = norm.fit(n_eeg_score)
    p_face_mean, p_face_std = norm.fit(p_face_score)
    n_face_mean, n_face_std = norm.fit(n_face_score)  
    return (p_eeg_mean, p_eeg_std), (n_eeg_mean, n_eeg_std), (p_face_mean, p_face_std), (n_face_mean, n_face_std)

#贝叶斯融合后的分类器
# eeg_score 待分类样本的EEG分类器输出值 
# face_score 待分类样本的face分类器输出值
# p_eeg_para, 概率密度函数p(eeg_score|positive)的输出值
# n_eeg_para, 概率密度函数p(eeg_score|negative)的输出值
# p_face_para, 概率密度函数p(face_score|positive)的输出值
# n_face_para 概率密度函数p(face_score|negative)的输出值
def fusion_classifer(eeg_score, face_score, p_eeg_para, n_eeg_para, p_face_para, n_face_para):
    #norm.pdf(x, mu, sigma)
    threshold = 0 
    p_eeg_prob = norm.pdf(eeg_score, p_eeg_para[0], p_eeg_para[1])
    n_eeg_prob = norm.pdf(eeg_score, n_eeg_para[0], n_eeg_para[1])
    p_face_prob = norm.pdf(face_score, p_face_para[0], p_face_para[1])
    n_face_prob = norm.pdf(face_score, n_face_para[0], n_face_para[1])
    score = -1*(log(p_eeg_prob)+log(p_face_prob)) + (log(n_eeg_prob)+log(n_face_prob))
    if score >= threshold:
        return 1
    else:
        return 0
