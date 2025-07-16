import os
import numpy as np
import cv2
import random
import tqdm
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# 加载模型
weights = 'model/efficientnet-b4_ucf101_RGB_se_resnext50_32x4d_avg_segment8_best.pth.tar'
model = smp.Unet(
    encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation='sigmoid'
    )
checkpoint = torch.load(weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
res = model.load_state_dict(base_dict)
print(model)
print(res)
model = model.cuda()
model.eval()


# dummy_input = torch.rand(1, 3, 256, 256).cuda().float()
# res = model(dummy_input)
# print(res.shape)
# 加载数据
train_data_root = 'E:\\python\\Data\\SDdataset\\DSS_training_data\\training_data'
train_concen_label = os.path.join(train_data_root, 'gt_prediction')

train_file = 'dataset_split/SDDataset_train-0.1.txt'
with open(train_file, 'r') as f:
    data_info = f.readlines()
images = [i.strip() for i in data_info]
labels = [i.strip().replace('blendall', 'gt_blendall').replace('.jpg', '.png') for i in data_info]
#train_data = os.path.join(train_data_root, 'blendall')
#train_label = os.path.join(train_data_root, 'gt_blendall')
#images = sorted([os.path.join(train_data, f) for f in os.listdir(train_data)])
#labels = sorted([os.path.join(train_label, f) for f in os.listdir(train_label)])

# val_data_root = '../SD01/'
# images = [os.path.join(val_data_root, '{:04d}.png'.format(f)) for f in range(1, 1001)]
# labels = sorted(glob.glob(val_data_root + '*Alpha.png'))


if not os.path.exists(train_concen_label):
    os.makedirs(train_concen_label)

mean = np.array([0.485, 0.456, 0.406],
            dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225],
            dtype=np.float32).reshape(1, 1, 3)

# 加载特征数据
feathre_path = 'neg_samples.npy'
features = np.load(feathre_path)
features_num = features.shape[0]
features = torch.from_numpy(features).cuda().float()


for k, img_path in enumerate(tqdm.tqdm(images)):
    image = cv2.imread(img_path)
    lab = cv2.imread(labels[k], 0)
    lab = np.where(lab>0, 1, 0)
    #cv2.imwrite('test.png',lab*255) ########################################
    h, w = image.shape[:2]
    con_lab = np.zeros((h, w))

    process_image = (image.astype(np.float32) / 255.)
    process_image = (process_image - mean) / std
    process_image = process_image.transpose(2, 0, 1).reshape(1,3,process_image.shape[0],process_image.shape[1])
    process_image = torch.from_numpy(process_image)
    process_image = process_image.cuda()
    prediction, features = model(process_image)
    b, c, h, w = prediction.shape
    prediction = prediction.squeeze(0).permute(1, 2, 0)  # 256 * 256 * 16

    prediction = prediction.view(w, h)
    con_lab = prediction.detach().cpu().numpy()
    #print(con_lab.max())
    #print(con_lab.min())
    

    con_lab = con_lab / np.max(con_lab)####################################
    #con_lab = cv2.blur(con_lab,(5,5))####################################
    con_lab = con_lab * lab
    con_lab = np.array(con_lab * 255, dtype=np.uint8)
    file_name = os.path.basename(labels[k])
    cv2.imwrite(os.path.join(train_concen_label, file_name), con_lab)
    #cv2.imwrite('res.png', con_lab)


