import os
import numpy as np
import cv2
import random
import tqdm

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_points(mat, k):
    res = []
    H, W = mat.shape[:2]
    i = 0
    flag = 0
    while i < k:
        h = random.randint(0, H-1)
        w = random.randint(0, W-1)
        if np.sum(mat[h, w]) != 0:
            i += 1
            res.append(mat[h, w] / np.linalg.norm(mat[h, w]))
        flag += 1
        if flag >= 500:
            break
    return res


# 加载模型
weights = 'model/efficientnet-b4_ucf101_RGB_se_resnext50_32x4d_avg_segment8_best-0.1752.pth.tar'
model = smp.Unet(
    encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    )
checkpoint = torch.load(weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
res = model.load_state_dict(base_dict)
print(res)
model = model.cuda()
model.eval()
last_layer = 'segmentation_head'
setattr(model, last_layer, nn.Identity())

# dummy_input = torch.rand(1, 3, 256, 256).cuda().float()
# res = model(dummy_input)
# print(res.shape)
# 加载数据
train_data_root = 'E:\\python\\Data\\SDdataset\\DSS_training_data\\training_data'
train_data = os.path.join(train_data_root, 'blendall')
train_label = os.path.join(train_data_root, 'gt_blendall')
mean = np.array([0.485, 0.456, 0.406],
            dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225],
            dtype=np.float32).reshape(1, 1, 3)
images = sorted([os.path.join(train_data, f) for f in os.listdir(train_data)])
labels = sorted([os.path.join(train_label, f) for f in os.listdir(train_label)])
k = 5  # 每个图像上随机选择k个数据点入库
select_image_nums = 1000
indexes = list(range(len(images)))
random.shuffle(indexes)
indexes = indexes[:select_image_nums]
all_points = []
for i in tqdm.tqdm(indexes):
    image = cv2.imread(images[i])
    lab = cv2.imread(labels[i], 0)
    lab = np.where(lab==255, 0, 1)

    process_image = (image.astype(np.float32) / 255.)
    process_image = (process_image - mean) / std
    process_image = process_image.transpose(2, 0, 1).reshape(1,3,process_image.shape[0],process_image.shape[1])
    process_image = torch.from_numpy(process_image)
    process_image = process_image.cuda()
    seg_output, _ = model(process_image)
    mat = lab * seg_output.squeeze(0).detach().cpu().numpy()
    mat = mat.transpose(1,2,0)
    points = get_points(mat, k)
    all_points.extend(points)

nums = len(all_points)
length = all_points[0].shape[0]
mat = np.zeros((nums, length))
for i in range(nums):
    mat[i] = all_points[i]
np.save('neg_samples.npy', mat)
    


