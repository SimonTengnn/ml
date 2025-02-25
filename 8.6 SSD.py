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
    return nn.Conv2d()