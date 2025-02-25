import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)   # batch_size, channels, height, width
print(f'X:\n {X}')

rois = torch.Tensor([[0,0,0,20,20], [0,0,10,30,30]]) # 区域目标类别（样本类别），左上x，左上y，右下x，右下y

output = torchvision.ops.roi_pool(X, rois, output_size=(2,2), spatial_scale=0.1)
print(f'output: \n{output}')