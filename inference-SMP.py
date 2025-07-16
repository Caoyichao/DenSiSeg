from os.path import basename
import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import glob
import os
import shutil
import time
import tqdm

from models import TSN
import transforms
from processor import ctdet_post_process, ctdet_decode
from utils import _sigmoid
import segmentation_models_pytorch as smp

mean = np.array([0.485, 0.456, 0.406],
            dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225],
            dtype=np.float32).reshape(1, 1, 3)

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def display(dets, frame_path, save_dir):
    thresh = 0.2
    data = dets[1]
    index = data[:, -1] > thresh
    data = data[index]

    img = cv2.imread(frame_path)
    for info in data:
        cv2.rectangle(img, (min(int(info[0]), 224), min(int(info[1]), 224)),
                            (min(int(info[2]), 224), min(int(info[3]), 224)), (255,0,0), 2)

    save_name = os.path.basename(frame_path)
    cv2.imwrite(os.path.join(save_dir, save_name), img)

def seg_display(seg_img, frame_path, save_dir):
    save_name = os.path.splitext(os.path.basename(frame_path))[0] + '.png'
    cv2.imwrite(os.path.join(save_dir, save_name), seg_img)

#trained model path
weights = 'model/efficientnet-b4_ucf101_RGB_se_resnext50_32x4d_avg_segment8_best.pth.tar'
test_imgs_txt= 'dataset_split/SDDataset_train-1.txt'
results_folder = 'results'

if __name__ == '__main__':
    args = get_args()
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
    print(res)
    model = model.cuda()
    model.eval()
    test_imgs = []
    with open(test_imgs_txt) as file_object:
        test_imgs = file_object.readlines()

    with torch.no_grad():
        try:
            for test_img in tqdm.tqdm(test_imgs):
                image = cv2.imread(test_img.rstrip())
                process_image = (image.astype(np.float32) / 255.)
                process_image = (process_image - mean) / std
                process_image = process_image.transpose(2, 0, 1).reshape(1,3,process_image.shape[0],process_image.shape[1])
                process_image = torch.from_numpy(process_image)
                process_image = process_image.cuda()

                seg_output, _ = model(process_image)

                #seg_output = F.softmax(seg_output)
                #seg = seg_output.sigmoid_()
                # im_hm = np.array(seg_output[:,0,:,:].squeeze().cpu().numpy()*255, dtype=np.uint8)
                im_hm = np.array(seg_output[:,0,:,:].squeeze().cpu().numpy()*255, dtype=np.uint8)
            
                base_name = os.path.basename(test_img.strip())
                cv2.imwrite(os.path.join(results_folder, base_name), im_hm)
                # src_img_path = test_img.rstrip().replace('E:/python/Data/SDdataset/SD01',results_folder)
                # res_img_path = src_img_path.replace('.png','_res.png')
                # ann_img_path = test_img.rstrip().replace('.png','Alpha.png') #annotation img
                # ann_image = cv2.imread(ann_img_path)
                # ann_img_path = src_img_path.replace('.png','_Alpha.png') #src img
                # cv2.imwrite(src_img_path,image)
                # cv2.imwrite(res_img_path,im_hm)
                # cv2.imwrite(ann_img_path,ann_image)

        except Exception as e:
            print(e)