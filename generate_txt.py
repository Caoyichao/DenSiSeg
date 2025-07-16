import os
import glob
import random 
import torch
import torchvision
import torchvision.datasets as datasets
import cv2

def generate_SDdataset(split_ratio = 0.8):
    root_path = 'data/SDdataset/SD01/'
    train_path = 'dataset_split/SD01_train.txt'
    val_path = 'dataset_split/SD01_val.txt'
    segs = sorted(glob.glob(root_path + '*Alpha.png'))
    random.shuffle(segs)
    imgs = [{i[1].replace('Alpha','')} for i in enumerate(segs)]
    segs_len = len(segs)
    imgs_len = len(imgs)
    assert segs_len == imgs_len, "imgs length != segs length"
    train_imgs = imgs[0:int(split_ratio*imgs_len)]
    train_segs = segs[0:int(split_ratio*imgs_len)]
    val_imgs = imgs[int(split_ratio*imgs_len):-1]
    val_segs = segs[int(split_ratio*imgs_len):-1]
    with open(train_path, 'w') as file_object:
        for path in train_imgs:
            file_object.write(list(path)[0]+'\n')
    with open(val_path, 'w') as file_object:
        for path in val_imgs:
            file_object.write(list(path)[0]+'\n')

            
#no split
def generate_SDdataset_train():
    root_path = 'E:/python/Data/SDdataset/DSS_training_data/training_data/blendall/'
    train_path = 'dataset_split/SDDataset_train.txt'
    imgs = sorted(glob.glob(root_path + '*.jpg'))
    random.shuffle(imgs)
    imgs_len = len(imgs)
    with open(train_path, 'w') as file_object:
        for path in imgs:
            file_object.write(path+'\n')
#no split
def generate_SDdataset_test():
    root_path = 'E:/python/Data/SDdataset/SD01/'
    test_path = 'dataset_split/SDDataset_test_SD01.txt'
    segs = sorted(glob.glob(root_path + '*Alpha.png'))
    random.shuffle(segs)
    imgs = [{i[1].replace('Alpha','')} for i in enumerate(segs)]
    segs_len = len(segs)
    imgs_len = len(imgs)
    assert segs_len == imgs_len, "imgs length != segs length"
    with open(test_path, 'w') as file_object:
        for path in imgs:
            file_object.write(list(path)[0]+'\n')


if __name__ == '__main__':
    generate_SDdataset_test()