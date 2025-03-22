# 数据增广/图像增广

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from d2l import torch as d2l

# d2l.set_figsize()
# img = d2l.Image.open('../img/cat1.jpg')
# d2l.plt.imshow(img)

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows*num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale) # 增广出num_rows行， num_cols列个图片

# apply(img, torchvision.transforms.RandomHorizontalFlip()) # 0.5的概率水平翻转
# apply(img, torchvision.transforms.RandomVerticalFlip()) # 0.5的概率上下翻转
# apply(img, torchvision.transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1), ratio=(0.5, 2))) # 随机裁剪出size=(200, 200), scale为占原图的百分比大小，再按比例缩放回原始宽高
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)) # 改变亮度， 对比度， 饱和度， 颜色色调

# all_images = torchvision.datasets.CIFAR10(train=True, root='./data', download=True)    # CIFAR10 把IMAGENET按32x32存储
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

train_augs = torchvision.transforms.Compose([   # Compose可以按顺序放入多个transforms变换，会依次执行
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor() # 原始输入是PIL.Image, 转换为tensor进行深度学习后续操作
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(train=is_train, root='./data', transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=4
    )
    return dataloader

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if type(X) == list:
        X = [(x.to(devices[0]) for x in X)] # devices[0]是主GPU
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.sum().backward()  # 每个batch， 损失l都是一个向量
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_num = d2l.accuracy(y_hat, y)
    return train_loss_sum, train_acc_num

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epoch, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epoch], )
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) # 模型并行，把模型分到多个GPU上
    for epoch in range(num_epoch):
        metric = d2l.Accumulator(4) # 损失，正确数量，样本数量，标签元素总数
        for i, (features, labels) in enumerate(train_iter): # enumerate会记录index, value
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(100)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight) # 根据m.weight重新随机均匀分布

net.apply(init_weights)

def train_with_data_aug(train_augs, net, lr=0.001):
    train_iter = load_cifar10(is_train=True, augs=train_augs, batch_size=batch_size)
    test_iter = load_cifar10(is_train=False, augs=test_augs, batch_size=batch_size)
    loss = nn.CrossEntropyLoss(reduction='none') # reduction='none' 不对损失做处理，直接返回损失tensor （mean返回平均，sum返回总和）
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


