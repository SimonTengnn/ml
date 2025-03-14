""" 语义分割 将图片中的每个像素都分配label（分配到对应类别）

    使用数据集 Pascal VOC2012
    (应用场景：背景虚化、路面分割)
"""
import os
import torch
import torchvision
import requests
from torchvision.transforms import functional as F
from d2l import torch as d2l

#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True):
    """ 读取所有voc图片并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB # 使用torchvision.io.read_image()的时候可以作为mode参数

    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for index, name in enumerate(images):
        features.append(
            torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{name}.jpg'), mode=mode)
        )
        labels.append(
            torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{name}.png'), mode=mode)
        ) # 由于整张图片每个像素都有一个对应label，因此也用图片形式存储
    return features, labels
train_features, train_labels = read_voc_images(voc_dir, True)

# n = 5
# imgs = train_features[:n] + train_labels[:n]
# imgs = [img.permute(1, 2, 0) for img in imgs]
#@save 数据集自带的标注
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 重点
def voc_colormap_to_label():
    """ (映射单像素点)从RGB 映射到 VOC类别 """
    colormap_to_label = torch.zeros(256**3, dtype=torch.long)
    for index, colormap in enumerate(VOC_COLORMAP):
        colormap_to_label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = index
    return colormap_to_label

# 重点
def voc_label_indices(img, colormap_to_label):
    """ （映射整张图像）将VOC标签中的RGB 映射到 类别索引 img是(3, H, W)"""
    img = img.permute(1, 2, 0).numpy().astype('int32') # 从(3,H,W)换成(H,W,3) 这样下一步可以分别把每个通道计算
    idx = (img[:, :, 0] * 256 + img[:, :, 1]) *256 + img[:, :, 2]
    return colormap_to_label[idx]

def voc_rand_crop(feature, label, height, width):
    """ 随机裁剪特征和标签图像 """
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))# 裁剪出(height,width)大小的区域，返回(top,left,h,w)
    # 裁剪出的目标区域可以表示为[:, top:top+h, left:left+w]
    feature = F.crop(feature, *rect)
    label = F.crop(label, *rect)
    return feature, label

# 在数据集的第一张图片上尝试随机裁剪
# imgs = []
# for _ in range(5):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap_to_label = voc_colormap_to_label()
        print(f'read {len(self.features)} examples')


    def normalize_image(self, img):
        return self.transform(img.float() / 255) # 原始像素值在[0,255]之间，归一到[0,1]之间

    def filter(self, imgs):
        return [
            img for img in imgs if
            ( img[1]>= self.crop_size[0] and
              img[2]>= self.crop_size[1] )
        ]

    def __getitem__(self, index):
        """ 返回裁剪后的图像 和 转换后的标签(拿到标签对应的类别名)"""
        feature, label = voc_rand_crop(self.features[index], self.labels[index], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap_to_label))

    def __len__(self):
        return  len(self.features)


""" 读取语义分割数据集 """
# crop_size = (320, 480)
# voc_train = VOCSegDataset(True, crop_size, voc_dir)
# voc_test = VOCSegDataset(False, crop_size, voc_dir)
#
# batch_size = 64
# train_iter = torch.utils.data.DataLoader(
#     voc_train,
#     batch_size,
#     shuffle=True,
#     drop_last=True, # drop_last丢弃最后一个不足batch_size的batch
#     num_workers=d2l.get_dataloader_workers() # 加载数据的进程数
# )
#
# for X, Y in train_iter:
#     print(X.shape)
#     print(Y.shape)
#     break
def load_data_voc(batch_size, crop_size):
    """ 加载VOC语义分割数据集 """
    voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit','VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    return train_iter, test_iter