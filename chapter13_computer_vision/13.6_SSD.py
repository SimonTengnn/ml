""" Single Shot MultiBox Detector
    目标检测
    基础网络-多尺度检测-默认框-边界框回归-非极大值抑制NMS
"""

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

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """ 每个块的前向(除了得到Y,还要得到锚框及预测)"""
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)  # 每个像素都生成anchor
    cls_preds = cls_predictor(Y)    # 预测锚框类别
    bbox_preds = bbox_predictor(Y)  # 预测锚框偏移
    return (Y, anchors, cls_preds, bbox_preds)

# 设置超参数
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]] # 锚框覆盖原图片的占比
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) -1 # n+m-1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}',
                    cls_predictor(idx_to_in_channels[i],
                                  num_anchors,
                                  num_classes))
            setattr(self,f'bbox_{i}',
                    bbox_predictor(idx_to_in_channels[i],
                                   num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5 ,[None] * 5 ,[None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X,
                                                                     getattr(self, f'blk_{i}'),
                                                                     sizes[i],
                                                                     ratios[i],
                                                                     getattr(self, f'cls_{i}'),
                                                                     getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1) # 所有层的anchors合并，拼接在一起
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

""" 创建一个模型实例 然后用它执行forward前向计算"""
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print(f'anchors size: {anchors.shape}')
print(f'class size: {cls_preds.shape}')
print(f'boudingbox size: {bbox_preds.shape}')

""" 读取数据集 """
batch_size = 32
train_iter, test_iter = d2l.load_data_bananas(batch_size)

device, net = d2l.try_all_gpus(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

""" 损失函数和评价函数 """
cls_loss = nn.CrossEntropyLoss(reduction="none")
bbox_loss = nn.L1Loss(reduction="none") # 绝对误差损失，|bbox_pred - bbox_label| / n

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_class = cls_preds.shape[0], cls_preds.shpae[2] # cls_preds一般是(batch_size, num_anchors,  num_classes+1)
    cls = cls_loss(cls_preds.reshape(-1, num_class),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1) # cls_labels一般是(batch_size, num_anchors)
            # CrossEntyopyLoss需要(input, target) input的维度是二维(N, C) target的维度是一维(N,)
            # 第i个样本对第j个类的预测 和 第i个样本的真实类别 做交叉熵输出维度为(N,)
            # reshape成(batch_size, num_anchors) .mean(dim=1)按照num_anchors求出每个样本的分类损失
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1) # *bbox_masks去掉背景类
    return cls + bbox # SSD综合损失 = 分类损失cls + 边界框回归损失bbox

def cls_eval(cls_preds, cls_labels): # 计算分类准确率
    # 类别预测放在最后一个维度 argmax要指定dim=-1 (argmax找到最大值的索引就是预测出来的类别 将这个索引和label中比较)
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum()) # .type就是转换成对应的dtype

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 边界框误差总和
    return float((torch.abs((bbox_preds - bbox_labels) * bbox_masks)).sum())

""" 训练过程 """
num_epochs, timer = 20, d2l.Timer()

net = net.to(device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad() # 梯度清零
        X, Y = features.to(device), target.to(device) # 放到gpu
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y) # 为每个锚框标注类型和偏移
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()

        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel()) # numel就是求总和
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')


