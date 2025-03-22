""" 卷积能够将(h,w)形状的输出变成(h',w')形状
    使用同样超参数，转置卷积能够将(h',w')形状的输入变成(h,w)形状
    Y[i:i+h, j:j+w] = X[i,j] * K
    对于卷积 H = X.shape[0] - h + 1
    对于转置卷积 H = X.shape[0] + h - 1 （由于形状改变，运算符号改变）"""
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as f
from d2l import torch as d2l

def trans_conv(X, K):
    """ 转置卷积运算 """
    h, w = K.shape
    Y = torch.zeros((X.shape[0]+h-1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Y[i:i+h, j:j+w] += X[i,j] * K
    return Y

# 验证转置卷积操作
# X = torch.tensor([[0.0, 1.0],[2.0, 3.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(trans_conv(X, K))

# # 使用nn的API进行转置卷积
# X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
# tconv.weight.data = K
# print(tconv(X))
