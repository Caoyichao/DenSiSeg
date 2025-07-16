import os
import torch
import torchvision
import torchvision.datasets as datasets

ROOT_DATASET = 'dataset_split'

def return_dataset(dataset, modality):

    file_imglist_train = os.path.join(ROOT_DATASET, "SDDataset_train-0.01.txt")
    file_imglist_val = os.path.join(ROOT_DATASET, "SDDataset_test_SD01.txt")
    #file_imglist_train = os.path.join(ROOT_DATASET, "SDDataset_train-0.5.txt")
    #file_imglist_val = os.path.join(ROOT_DATASET, "SDDataset_test_SD01.txt")
    root_data = ""
    prefix = ""

    return file_imglist_train, file_imglist_val, root_data, prefix
