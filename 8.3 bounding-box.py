import matplotlib.pyplot as plt

import torch
from d2l import torch as d2l
from numpy.lib.npyio import savez

d2l.set_figsize()
img = plt.imread('/Users/simondeng/Desktop/ml/pytorch/img/catdog.jpg')
plt.imshow(img)
# plt.show() # pycharm一定要加上这一句，才会显示图片

""" 注意： 原点在左上角！！！ x轴正方向向右----->
                          y轴正方向向下
                          |
                          |
                          |
"""
def box_corner_to_center(boxes): # boxes是N个框，[第i个，两个顶点的xy轴坐标]
                            # 一般按照x1, y1（左上角）, x2, y2（右下角）顺序存储
    """ 用定位框的（左上，右下）坐标 转换成 （中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1+x2) / 2
    cy = (y1+y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1) # 把多个[cx, cy, w, h]拼在一个tensor中，
                                                        # 且原来的维度在新维度的-1处
                                                        # 拼成 tensor([cx1, cy1, w1, h1],
                            #                                         [cx2, cy2, w2, h2]...)
    return boxes

def box_center_to_corner(boxes):
    """ 从定位框的 （中间，宽度，高度） 转换成 （左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx- w/2
    y1 = cy- h/2
    x2 = cx+ w/2
    y2 = cy+ h/2
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0] #目测的边界

# 验证两个转换函数的正确性
boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

def bbox_to_rectangle(bbox, color):
    # 将定位框（左上x，左上y，右下x，右下y） 转换换成matplotlib格式（(左上x，左上y)，宽，高）
    return d2l.plt.Rectangle(
        xy=(bbox[0],bbox[1]), width=bbox[2]-bbox[0], height= bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

fig = plt.imshow(img) # 让图片加入matplotlib
fig.axes.add_patch(bbox_to_rectangle(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rectangle(cat_bbox, 'red'))
plt.show()  # 让matplotlib渲染出来

