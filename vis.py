import torch.utils.data as d                          

import cv2
import os
import math
import os.path
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from numpy.random import randint
import torchvision
import torch
import transforms
import glob
import random
import time

from image import flip, color_aug
from image import get_affine_transform, affine_transform
from image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from image import draw_dense_reg

def show_cam_on_image(img, mask, file_name):
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET) #激活后*255
    heatmap = np.float32(heatmap) 
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # cv2.imwrite('heatmap.jpg', np.uint8(255 * heatmap))
    # shutil.copyfile(file_name, os.path.join(output_dir, os.path.basename(file_name)))
    #cam_file = os.path.join(output_dir, os.path.basename(file_name).split('.')[0] + '.jpg')
    cv2.imwrite(file_name, np.uint8(255 * cam))
    
#显示权重图
def show_attention():
    srcimgpath = 'E:\\python\\Data\\8frames'
    maskpath = 'output_cor_prob2'
    respath = 'visualization_output_cor_prob_paper'
    np.set_printoptions(suppress=True)

    files = os.listdir(maskpath)
    for file in files:
        print(file)
        mask = os.path.join(maskpath,file)
        mask_data = np.load(mask)
        #print(mask_data)
        mask_data = abs(mask_data)
        mask_data = 1- mask_data
        mask_data = mask_data * 255 
        #maxval = np.max(mask_data)
        #mask_data = mask_data * 255 / maxval
        #mask_data[mask_data< 0] = 0
        #print(mask_data)

        #读原始图像
        src = []
        seq_name = file.split('.')[0][:-5]  #序列编号  
        srclist = os.listdir(os.path.join(srcimgpath,seq_name))
        for srcimg in srclist:
            img_path = os.path.join(srcimgpath,seq_name,srcimg)
            img = cv2.imread(img_path)
            src.append(img)
    
        #准备可视化
        visimg = []
        for i in range(8):
            mask = mask_data[i]
            #print(mask)
            img_tmp = cv2.resize(mask,(224,224))
            img_path = os.path.join(respath,seq_name)
            if not os.path.exists(img_path):
                print('creating folder ' + img_path)
                os.mkdir(img_path)
            img_path = os.path.join(img_path,str(i+1)+'.jpg')
            show_cam_on_image(src[i], img_tmp, img_path)
            #visimg.append(img_tmp)


'''
    mask_data = cv2.resize(mask_data,(224,224))
    #print(mask_data)
    img_name = file.replace('npy','jpg')
    img_name = img_name.replace('0001','0008')
    img_path = os.path.join(srcimgpath,img_name.split('.')[0][:-5],img_name.split('.')[0]+'.jpg')
    img = cv2.imread(img_path)

    #for i in range(8):
    #    img_path = 

    show_cam_on_image(img, mask_data, img_path)

'''

#显示权重图
def show_source_label():
    srcimgpath = 'E:\\python\\Data\\8frames'
    maskpath = 'E:\\python\\Data\\mask_output'
    respath = 'visulization_output_source_label'
    np.set_printoptions(suppress=True)

    files = os.listdir(maskpath)
    for file in files:
        mask = os.path.join(maskpath,file)
        mask_data = (np.load(mask)*255).astype(np.uint8)
        #img_tmp = cv2.resize(mask,(224,224))

        #读原始图像
        seq_name = file.split('.')[0]  #序列编号
        try:
            srclist = os.listdir(os.path.join(srcimgpath,seq_name))
            img_path = os.path.join(srcimgpath,seq_name,srclist[0])
        except Exception:
            break;
        img = cv2.imread(img_path)
        if(img.size > 0):
            #准备可视化
            #img_tmp = cv2.resize(mask,(224,224))
            img_path = os.path.join(respath,seq_name)+'.jpg'
            mask_img_path = os.path.join(respath,seq_name)+'_m.jpg'
            cv2.imwrite(mask_img_path,mask_data)
            show_cam_on_image(img, mask_data, img_path)
'''
a = np.load('E:\\tmp\\i3d-features-ca778843c6bb459d25daa3bc6df8b587.npy')
b = np.load('E:\\tmp\\vggish-feature-ca778843c6bb459d25daa3bc6df8b587.npy')
print(a.shape)
print(b.shape)
a = np.load('E:\\tmp\\0004bf664eee696a9032c4a2770e12dc.npy')
b = np.load('E:\\tmp\\audio\\0004bf664eee696a9032c4a2770e12dc.npy')
print(a.shape)
print(b.shape)
a = np.load('E:\\tmp\\0005d29679ec9fd487ce2ec032c52eb3.npy')
b = np.load('E:\\tmp\\audio\\0005d29679ec9fd487ce2ec032c52eb3.npy')
print(a.shape)
print(b.shape)
a = np.load('E:\\tmp\\00078280e36d079bd76f1420ec527c43.npy')
b = np.load('E:\\tmp\\audio\\00078280e36d079bd76f1420ec527c43.npy')
print(a.shape)
print(b.shape)
'''


show_source_label()