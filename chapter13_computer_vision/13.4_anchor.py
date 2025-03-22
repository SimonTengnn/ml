""" IoU Intersection over Union 交集/并集
    对边缘框(训练样本，是读入的图片上已经有的边缘框)寻找最相近的锚框(生成)
    使用NMS非极大值抑制(就是每一类保留最大预测值，去掉所有和它IoU>某个threshold的预测)
"""

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

torch.set_printoptions(2) # 调整tensor的输出精度为小数点后2位

def multibox_prior(data, sizes, ratios):
    """
    :param data:   输入的图片
    :param sizes:  缩放比
    :param ratios: 宽高比
    :return:
    # 通常生成锚框，会给出缩放比s1...sn(n个) 和 宽高比r1...rm(m个)
    #             会用 (s1,r1),(s1,r2)...(s1,rm)  --缩放比s1不变，宽高比从r1到rm
    #             和           (s2,r1)...(sn,r1)  --缩放比从r2到rn, 宽高比r1不变
    # 同一像素为中心，一共n+m-1个锚框
    # 对于整个输入图像，输入一共有wh个像素，就会一共生成wh(n+m-1)个锚框
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios -1
    size_tensor = torch.tensor(sizes, device=device)    # 从list转换成tensor
    ratio_tensor = torch.tensor(ratios, device=device)

    #将锚点移动到像素中心
    offset_h, offset_w = 0.5, 0.5 #像素宽高都为1，所以偏移中心0.5
    steps_h = 1.0 / in_height # 表示一个像素(步长)占整个图片的百分比
    steps_w = 1.0 / in_width

    # 生成锚框所有中心点
    center_h = (torch.arange(in_height, device=device)+offset_h)*steps_h # arange生成[0,1,2,...in_height-1]
                                            # 加上offset每个像素都偏移到中心
                                            # 乘上steps_h就能转换为每个中心在整张图片中的相对位置
    center_w = (torch.arange(in_width, device=device)+offset_w)*steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij') # 扩展为len(center_h)*len(center_w)的矩阵
                                    #原点在左上角 x轴往右--->
                                            #   y轴往下
                                            #   |
                                            #   |
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1) # reshape成一维，后面便于以[xmin,ymin,xmax,ymax]形式和anchor_manipulations对应

    """ 生成'boxes_per_pixel'个高和宽（也就是每个像素点都会有这么多个锚框，生成每个锚框的高和宽）"""
    # 用来创建锚框顶点坐标
    """ wa=s*h*sqrt(r)/w   ha=s/sqrt(r) """
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),   # r不变，从s1到sn
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
                                                                # s不变，从r1到rm
                                                            # ratios_tensor[0]已经在上一行计算过，所以此处从1开始切片
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                  sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 锚框中心点像素坐标anchor_manipulations
    anchor_manipulations = torch.stack((-w,-h,w,h)).T.repeat(in_width*in_height,1) / 2 # 相对中心偏移，所以要/2才是对于图像中点的偏移量
                                    # stack堆叠：[4]-->[4,boxes]
                                    # .T转置: [4,boxes]-->[boxes,4]
                                    # repeat重复: [boxes, 4] -->boxes重复in_width*in_height次，4重复1次--> [in_width*in_height*boxes, 4]
    # 锚框左上角和右下角相对中心点的偏移
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
                                                                                    # 把第0维重复boxes_per_pixel次
                                                                                    # 每个像素点要对应多个锚框
    # 边界=中心点+偏移
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0) #在第0维增加一个维度（batch_size）

img = plt.imread('/Users/simondeng/Desktop/ml/pytorch/img/catdog.jpg')
h, w = img.shape[:2]

print(f'h: {h}, w: {w}')
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(f'Y.shape: {Y.shape}')

boxes = Y.reshape(h, w, 5, 4)   # 像素高，像素宽，每像素锚框数（3+3-1）,锚框顶点xyxy归一化值（占整张图片的比率）
print(boxes[250,250,0,:])   #打印出(250,250)为中心的第0个锚框四个顶点的归一化值(占整张图片的比率)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """ 画出所有边界框 """
    def _make_list(obj, default_values = None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (tuple, list)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h)) # 整张图片的宽高，后面乘上归一化坐标就是实际坐标
fig = d2l.plt.imshow(img)
# 要在matplotlib画出来，那就要把上面归一化的坐标换回实际坐标
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])

"""!!重点 mark!!"""
def box_iou(boxes1, boxes2):
    """ 计算交并比IoU """        # 对每个边界框都按(xmin,ymin,xmax,ymax)顺序，宽w=xmax-xmin,高h=ymax-ymin
    box_area = lambda boxes: ((boxes[:, 2]-boxes[:, 0]) * (boxes[:,3]-boxes[:,1]))
    areas1=box_area(boxes1)
    areas2=box_area(boxes2)

    """ 取交集，以(boxes1数量，boxes2数量，2)的形状输出每两个锚框之间的重合区域的宽高 """
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # None让boxes1增加一个维度,从(boxes1数量，4)变成(boxes1数量，1，4)
                                                                # :2表示取到第2个前（不包含2），从(xmin,ymin,xmax,ymax)取出xmin,ymin
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) # 把负值都变成0，相当于把所有无交集的情况都变成0
    # 用重合区域的坐标相减(xmax-xmin, ymax-ymin)  得到的是宽和高

    inter_areas = inters[:, :, 0] * inters[:, :, 1] # 重合面积=宽*高，维度是（boxes1数量，boxes2数量）
    union_areas = areas1[:, None] + areas2 - inter_areas   # 并集面积=两集合面积-重合面积
                            # areas1形状是（boxes1的数量,）,areas2形状是(boxes2的数量,)，两者形状不一致
                            # areas1[:, None]增加一个维度后形状为(boxes1的数量，1), 就可以跟areas2广播了！第二个维度复制到boxes2就能跟areas2计算
    return inter_areas / union_areas    # 返回每个锚框和每个真实边界的交并比,形状是（num_anchors, num_ground_truth）

# 把 真实边界框 分配给 IoU最大(最接近真实值)的锚框
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    jaccard = box_iou(anchors, ground_truth) # 每个锚框和真实边框都做IoU
    anchor_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)   # 初始化所有值为-1，表示没有匹配上真实边框

    # 根据阈值threshold判断是否分配真实边框
    max_ious, indices = torch.max(jaccard, dim=1)   # torch.max在dim=1(跨每一列)上找最大值，依次返回 IoU值 和 真实边界对应的index
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 以一维返回比threshold大的锚框index(在140行基础上找出满足条件的)
    box_j = indices[max_ious >= iou_threshold] # 真实边界index在140行已经求出，直接取比threshold大的（在140行基础上找出满足条件的）
    anchor_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1) # 把满足所在列（锚框列）都删除
    row_discard = torch.full((num_gt_boxes,), -1) # 把满足所在行（真实边界行）都删除

    #遍历每个真实边框，每次都会把一个真实边框匹配到一个锚框
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # argmax会将tensor展平后返回最大值的索引，max会返回值和索引
            # 返回展平成一维的jaccard中IoU最大的索引 jaccard是每个锚框对每个真实边界框的IoU，形状是(num_anchors, num_ground_truth)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchor_bbox_map[anc_idx] = box_idx  # anc_idx号的锚框就对应box_idx号的真实边界框了
        # 已经相互对应的锚框和边界框，在jaccard 形状为(num_anchors, num_ground_truth) 中的行和列要清空
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchor_bbox_map


