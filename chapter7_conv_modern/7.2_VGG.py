import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels # 下一层的输入等于本层输出
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers) # *代表解包，layers是列表，解出列表的每个元素依次放入Sequential

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # 定义每个阶段的(num_convs, out_channels)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_channels, out_channels
        ))
        in_channels = out_channels # 将下一次的输入通道更新为本次输出通道数

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 10)
    )

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """ 使用GPU计算模型在数据集上的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval() # 神经网络切换到评估模式
        if not device:
            device = next(iter(net.parameters())).device # 从parameters中找到gpu设备
    metric = d2l.Accumulator(2) # [0]存放正确预测数 [1]存放总数
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X] # 移动tensor到设备上
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel()) # accuracy()返回的是预测正确的元素数量 y.numel()返回y的元素总数
    return metric[0] / metric[1]


def train_ch6(device, lr, num_epochs, net, train_iter, test_iter):
    """ 在GPU上进行训练 """
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight) # xavier_uniform_ 使用均匀分布初始化，保持每一层激活和梯度的方差尽量相同

    net.apply(init_weights) # 对每个module（也就是每一层）都应用init_weights这个方法
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
                                            # x轴范围                  # 每条线的意义（会在图表中区分）
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter): # enumerate 返回index和元素值
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                            # 当前batch损失总数=损失率*batch_size
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')



