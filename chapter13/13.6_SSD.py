import torch
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F


# img = plt.imread('/Users/simondeng/Desktop/ml/pytorch/img/catdog.jpg')
# h, w = img.shape[:2]
# print(f'height:\t{h}\nwidth:\t{w}')
#
# def display_anchors(fmap_w, fmap_h, s):
#     d2l.set_figsize()
#     fmap = torch.zeros((1, 10, fmap_h, fmap_w)) # batchsize, channels, height, width
#     anchors = d2l.multibox_prior(
#         fmap, sizes=s, ratios=[1, 2, 0.5]
#     )   # 每个像素都生成size大小（占原始图片的百分比） 高宽比例为ratios(高/宽=1或2或0.5)的不同锚框
#         # ratios每有一个元素，就按该比例生成一个锚框，所以此处是生成三个锚框
#     bbox_scale = torch.tensor((w, h, w, h))
#     d2l.show_bboxes(d2l.plt.imshow(img).axes,
#                     anchors[0] * bbox_scale)
#     plt.show()

def cls_predictor(num_inputs, num_anchors, num_classes):
    """ 预测锚框类别 """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes +1),
                     kernel_size=3, padding=1)  # 输入通道数， 输出通道数（+1为背景类，即没有物体的区域）， 卷积核， 步幅

def bbox_predictor(num_inputs, num_anchors):
    """ 预测锚框对应边界框(真实的框)位置 """
    return nn.Conv2d(num_inputs,
                     num_anchors*4,# 对每个锚框都需要预测出边界框，而每个边界框由(xmin,ymin,xmax,ymax)坐标构成
                     kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

# # 检查输出通道数
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))  # 输出通道应该为5*(10+1)
Y2 = forward(torch.zeros(2, 16, 10, 10), cls_predictor(16, 3, 10))  # 输出通道应该为3*(10+1)
# print(f'shape of Y1:\t{Y1.shape}\nshape of Y2:\t{Y2.shape}')

# 将通道放到最后一个维度，这样每个像素对应的所有锚框(在不同通道下的全部锚框)都可以一次性找到
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # permute按维度index重新排列维度
                                    # permute(0,2,3,1) 即 (batch_size, height, width, channels)
                                    # 从第start_dim维度开始，把后面的维度压缩成一个维度
                                    # (batch_size, h, w, c) --> (batch_size, h*w*c)
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
    # 在第1维拼接，最终拼接成一个(batch_size, sum(h*w*c))

# print(f'{concat_preds([Y1, Y2]).shape}')

def down_sample_blk(in_channels, out_channels):
    """ 高宽减半块 （提取特征）"""
    blk = []
    for _ in range(2):
        blk.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2)) #
    return nn.Sequential(*blk)
# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) -1 ):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

def get_blk(i):
    """ 完整SSD模块构成 """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1)) # 最后一层，全局最大池化(把高和宽都降为1)
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward():
    
    ...