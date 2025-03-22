""" 样式迁移 (基于CNN)
    将样式图片的样式迁移到目标图片"""

import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
content_img = d2l.Image.open('/Users/simondeng/Desktop/ml/pytorch/img/rainier.jpg')
d2l.plt.show(content_img)

style_img = d2l.Image.open('/Users/simondeng/Desktop/ml/pytorch/img/autumn-oak.jpg')
d2l.plt.show(style_img)

""" 预处理(图片RGB三通道分别标准化)和后处理(还原) """
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape), # Resize是对PIL图片的操作
        torchvision.transforms.ToTensor(), # torchvision的ToTensor会自动归一化
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ]) # 归一 (X-mean)/std
    return transform(img).unsqueeze(0) # 多加一个batch_size维度

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
        # 反归一 X'=(X-mean)/std --> X=X'*std + mean
        # clamp会在操作后将比min小的数都变成min 比max大的数都变成max
        # 从(C, H, W) -->(H, W, C)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
    # 从Tensor(要求格式为C,H,W)变回PIL图片， 从(H,W,C) --> (C,H,W)

""" 抽取图像特征 """
pretrained_net = torchvision.models.vgg19(pretrained=True)
style_layers, content_layers = [0, 5, 10, 19, 28], [25] # 用来匹配风格/内容的层的index，越小越靠近输入，越大越靠近输出
net = nn.Sequential(*[
    pretrained_net.features[i] for i in range(max(style_layers+content_layers) + 1)
    # net.features会输出一个nn.Sequential容器，包含每一层
])

def extract_features(X, content_layers, style_layers):
    """ 找出匹配层的content和style特征"""
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in content_layers:
            contents.append(X)
        elif i in style_layers:
            styles.append(X)
    return contents, styles

def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device) # content在原始图像的值
    content_Y, _ = extract_features(content_X, content_layers, style_layers) # content在匹配层的feature
    return content_X, content_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y

def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean() # Y.detach()生成一个新的tensor用于计算，但是不计算梯度（切断Y与计算图的联系）
    # .mean()得到一个标量

def gram(X):
    """ gram矩阵中，元素x(i,j)就是通道i和通道j的相关性"""
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape(num_channels, n)
    return torch.matmul(X, X.T) / (num_channels * n)  # gram矩阵
    # /(num_channels*n)也就是/X.numel() = 元素总数c*h*w，这样每次计算gram矩阵都不会因为输入图像的尺寸不同而产生变化

def style_loss(Y_hat, gram_Y):
    return torch.square(Y_hat - gram_Y.detach()).mean()
    # .mean()得到一个标量

def tv_loss(Y_hat):
    """ 全变分损失降低噪声 total variation denoising"""
    return 0.5 * (torch.abs(Y_hat[:,:,1:,:] - Y_hat[:,:,:-1,:]) + torch.abs(Y_hat[:,:,:,1:] - Y_hat[:,:,:,:-1]))
    # 求和|X(i,j)-X(i,j+1)|+|X(i,j)-X(i+1,j)|

""" 损失函数 """
content_weight ,style_weight, tv_weight = 1, 1e3, 10 # 使用时广播
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    """ 计算内容损失、风格损失、全变分损失"""
    # 是多层特征，需要遍历每一层计算
    contents_l = [content_loss(Y_hat, Y)*content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(10 * styles_l + contents_l + [tv_l]) # tv_l 是标量
    return contents_l, styles_l, tv_l, l

""" 初始化合成图像 """
class SynthesisedImage(nn.Module):
    def __init__(self, image_shape, **kwargs):
        super().__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(image_shape))

    def forward(self):
        return self.weight

def get_init(X, device, lr, styles_Y):
    gen_image = SynthesisedImage(X.shape).to(device)
    gen_image.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_image.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    # gram是对单独的一个图求的(每个Y都代表一个不同风格特征)，所以要单独对每个Y求gram
    return gen_image, trainer, styles_Y_gram

""" 训练 """
def train(X, contents_Y, styles_Y, device, lr, lr_decay_epoch, epochs):
    X, trainer, styles_Y_gram = get_init(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    # 经过lr_decay_epoch就更新学习率lr, 新lr=旧lr * 0.8

    for epoch in range(epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X,contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()

    return X


