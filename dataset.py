import torch.utils.data as data

from PIL import Image
import cv2
import os
import os.path
import numpy as np
from numpy.random import randint
import torchvision
import torch
import transforms
import glob
import random
import time

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def label(self):
        return int(self._data[2])

    @property
    def confidence(self):
        return float(self._data[3])

'''
class TSNDataSet(data.Dataset):
    def __init__(self, split, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.split = split
        self.mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

        self._parse_list()
        self.total_list = self.image_list

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        self.image_list = [x.strip().split(' ') for x in open(self.list_file)]
        #self.image_list = [VideoRecord(item) for item in tmp]
        print('image number:%d'%(len(self.image_list)))

    def __getitem__(self, index):
        record = self.image_list[index]
        # check this is a legit video folder
        #while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        str_path = record[0]
        while not os.path.exists(str_path):
            print(str_path)
            index = np.random.randint(len(self.image_list))
            str_path = self.image_list[index][0]
        if self.split == 'train':
            seg_img_path = str_path.replace(".jpg",".png")
            seg_img_path = seg_img_path.replace("blendall","gt_blendall")    
        if self.split == 'val':
            seg_img_path = str_path.replace(".png","Alpha.png")

        return self.get(str_path, seg_img_path)

    def get(self, record, seg_img_path):
        image = cv2.imread(record)
        process_image = (image.astype(np.float32) / 255.)
        process_image = (process_image - self.mean) / self.std
        process_image = process_image.transpose(2, 0, 1)

        seg_img = cv2.imread(seg_img_path)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        # label = np.where(seg_img>0,255,0)
        seg_img = (seg_img.astype(np.float32) / 255.)

        #process_image = self.transform(process_image)

        return process_image, seg_img

    def __len__(self):
        return len(self.image_list)
'''
    

class TSNDataSet(data.Dataset):
    def __init__(self, split, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.split = split
        self.mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

        self._parse_list()
        self.total_list = self.image_list

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        self.image_list = [x.strip().split(' ') for x in open(self.list_file)]
        #self.image_list = [VideoRecord(item) for item in tmp]
        print('image number:%d'%(len(self.image_list)))

    def __getitem__(self, index):
        record = self.image_list[index]
        # check this is a legit video folder
        #while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        str_path = record[0]
        while not os.path.exists(str_path):
            print(str_path)
            index = np.random.randint(len(self.image_list))
            str_path = self.image_list[index][0]
        if self.split == 'train':
            seg_img_path = str_path.replace(".jpg",".png")
            #seg_img_path = seg_img_path.replace("blendall","gt_prediction")
            seg_img_path = seg_img_path.replace("blendall","gt_blendall")
            #seg_img_path = str_path.replace(".png","Alpha.png")
        if self.split == 'val':
            seg_img_path = str_path.replace(".png","Alpha.png")

        return self.get(str_path, seg_img_path)

    def get(self, record, seg_img_path):
        image = cv2.imread(record)
        process_image = (image.astype(np.float32) / 255.)
        process_image = (process_image - self.mean) / self.std
        process_image = process_image.transpose(2, 0, 1)

        seg_img = cv2.imread(seg_img_path, 0)
        # seg_img = (seg_img.astype(np.float32) / 255.)
        seg_img = seg_img.astype(np.float32) / np.max(seg_img)
        #print(seg_img_path)
        return process_image, seg_img

    def __len__(self):
        return len(self.image_list)
