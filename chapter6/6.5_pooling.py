import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size # 解包
    Y = torch.zeros(size=(X.shape[0] - p_h +1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max': # 最大池化
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg': # 平均池化
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

X = torch.tensor([[0., 1., 2.],[3., 4., 5.], [6., 7., 8.]])

print(f'max pooling:\n{pool2d(X, (2,2), "max")}')
print(f'mean pooling:\n{pool2d(X, (2,2), "avg")}')

pool2d_frame1 = nn.MaxPool2d(3) # 3x3的池化窗口, stride默认为kernal_size（就是参数3）
pool2d_frame2 = nn.MaxPool2d((2, 3), padding=(1, 2), stride=(2, 3))

