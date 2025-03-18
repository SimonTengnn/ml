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



