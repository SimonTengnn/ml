""" fine-tuning 实际上就是：
    1. 复制一个已经训练好的模型(此处利用ResNet18)
    2. 随机初始化最后一层fc层
    3. 重新训练fc层
"""
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from d2l import torch as d2l
from torch import nn

d2l.DATA_HUB['hotdog']  = (d2l.DATA_URL + 'hotdog.zip',
                           'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# RGB三个通道
normalize = torchvision.transforms.Normalize([.845, .456, .406], [.229, .224, .225])
""" 图像预处理管道 """
train_augs = torchvision.transforms.Compose([ # 训练集数据增强augmentation
    torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪，调整到224x224
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),              # 转为tensor
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),  # 先resize到256x256
    torchvision.transforms.CenterCrop(224),     # 再直接取中间的224x224
    torchvision.transforms.ToTensor(),
    normalize
])

""" 使用ResNet-18作为源模型 (在ImageNet数据集上预训练过) """
pretrained_net = torchvision.models.resnet18(pretrained = True) # pretrained=True下载预训练参数

# 目标模型
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) # 最后的FullConnection，输出最终结果(2类)
    # 由于finetune_net使用的是预训练好的resnet18, 已经带有了resnet18的所有层和参数，直接用finetune_net.fc就可以调用
    # 此处是重新定义来替换它的fc
nn.init.xavier_uniform(finetune_net.fc.weight)

""" 微调训练 """
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs
    ), batch_size=batch_size, shuffle=True) # 训练集，打乱
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs
    ), batch_size=batch_size)

    # devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group: # param_group==True就使用十倍学习率
        params_1x  = [param for name, param in net.named_parameters() # 遍历模型所有参数
                      if name not in ["fc.weight", "fc.bias"]]  # 只有name不为fc.weight, fc.bias的param会被放入params_1x
                                                                # （也就是fc层以外的所有参数保留）
        trainer = torch.optim.SGD( # 把网络分成两部分，fc参数用10倍学习率(局部学习率，fc是随机初始化的，需要快速收敛)其他参数一倍学习率（全局学习率）
            [{'params': params_1x},
            {'params': net.fc.parameters(),
             'lr': learning_rate * 10}], # 局部学习率，用于重新随机初始化的fc层，更快收敛，加快训练速度
            lr=learning_rate, #全局学习率
            weight_decay=0.001
        )
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 'mps') # apple silicon


train_fine_tuning(finetune_net, 5e-5)


