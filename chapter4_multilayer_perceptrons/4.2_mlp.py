""" MLP 多层感知机
    输入->隐藏层1(激活函数1)->隐藏层2(激活函数2)->输出 (->Softmax置信度，用于分类)
    每一个隐藏层后都要带非线性的激活函数(Sigmoid,ReLU...)
    多层感知机的输出是分数，如果要分类就在输出后多加一层Softmax
"""

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_hidden1, num_outputs = 784, 256, 10

W1 = nn.Parameter(torch.rand(num_inputs, num_hidden1, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden1, requires_grad=True))
W2 = nn.Parameter(torch.rand(num_hidden1, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    zeros = torch.zeros_like(X)
    return torch.max(X,zeros)

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(torch.matmul(X,W1) + b1)
    return relu(torch.matmul(H1,W2) + b2)

loss = nn.CrossEntropyLoss(reduction='none')
#
# def train(net, train_iter, test_iter, loss, num_epochs):
#     for epoch in range(num_epochs):
#

# nn.init.xavier_uniform_ 适用于 激活函数tanh或sigmoid
# nn.init.kaiming_uniform_ 适用于 激活函数ReLU