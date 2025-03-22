""" FCN用于语义分割，会保留空间信息
    使用转置卷积替换CNN最后的全连接(用于还原空间信息，用于像素级别预测)
"""
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as f

pretrained_net = torchvision.models.resnet18(pretrained=True) # pretrained=True下载预训练参数
# print(list(pretrained_net.children())[-3:]) # 找到最后的三层

net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 去掉最后两层 全局平均池化层AdaptiveAvgPool2d和线性层Linear
 # AdaptiveAvgPool2d提取特征，Linear产生预测

# X = torch.rand(size=(1, 3, 320, 480))
# print(net(X).shape) # 可以先看一下去掉最后两层后从X到抽取特征时的维度 (1, 512, 10, 15)

num_classes = 21 # voc2012有20类，再加一个背景类
net.add_module('final_conv',
               nn.Conv2d(512, num_classes, kernel_size=1)) # 1x1 Conv
net.add_module('transposed_conv',
               nn.ConvTranspose2d(
                   num_classes, num_classes,
                   kernel_size=64, padding=16,
                   stride=32
               )) # 上一层特征高宽为(10，15)要还原成(320, 480)，就要扩大32倍，stride=32
                #

""" 生成转置卷积核 用于双线性插值（用于上采样，也就是放大图像） """
def bilinear_kernel(in_channles, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2 # 缩放因子
    if factor % 2 == 1:
        centor = factor - 0.5
    else:
        centor = factor -1
    og = (torch.arange(kernel_size).reshape(-1, 1), # 列向量，存放每一行的index
          torch.arange(kernel_size).reshape(1,-1))  # 行向量，存放每一列的index
    # og相当于用一个二维表格把kernel的每个元素行列index存起来
    w = (1 - torch.abs(og[0] - centor)/factor) * (1- torch.abs(og[1] - centor)/factor)

    # w(x,y) = (1- abs(x-centor)/factor) * (1- abs(y-centor)/factor)
    weight = torch.zeros((in_channles, out_channels, kernel_size, kernel_size))
    weight[range(in_channles), range(out_channels), :,:] = w
    return weight

# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2)
# conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
# conv_trans.weight 就是卷积核
# copy_是原地操作，将bilinear_kernel生成的权重放入卷积核

W = bilinear_kernel(num_classes, num_classes, 64) # 最后一层转置卷积见22-27行
net[-1].weight.data.copy_(W)

batch_size, crop_size = 32, (320, 480) # crop_size 裁剪输入图像的尺寸
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

""" 训练过程 """
def loss(inputs, targets):
    return f.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    # 第一个mean(1)是 (batch_size, height, width) --> (batch_size, width)
    # 第二个mean(1)是 (batch_size, width) --> (batch_size)
num_epochs, lr, weight_decay, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

""" 预测 """
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0) # normalize_image将图片归一化
                                                # unsqueeze(0) 对img增加一个第0维，用于存放batch_size
    pred = net(X.to(devices[0])).argmax(dim=1) # 此时形状是(batch_size, channels, height, width)
        # dim=1 就在每个channels选出预测值最大的那个channel的index，表示预测为该类
        # pred是整张图片上对每个像素类别的预测，形状是(batch_size, height, width)
    return pred.reshape(pred.shape[1], pred.shape[2])
    # 去掉batch_size 形状和图片高宽完全一样