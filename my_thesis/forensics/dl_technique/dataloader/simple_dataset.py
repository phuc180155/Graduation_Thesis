import os, sys

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.interpolate import griddata

import glob
import numpy as np
import cv2
import torch

from dataloader.utils import azimuthalAverage
from PIL import Image, ImageEnhance
import copy

"""
    Class for make dual (spatial and spectrum) image dataset
"""
class SingleRGBStreamDataset(Dataset):
    def __init__(self, path, image_size, transform=None, shuffle=True):
        self.transform = transform
        self.image_size =image_size
        self.shuffle = shuffle
        self.data_path = path
        # print("sample: ", self.data_path[:10])
        # print("len: ", len(self.data_path))
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
            
    def __getitem__(self, index):
        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        
        # Convert to PIL Image instance and transform spatial image
        PIL_img = Image.fromarray(img)
        if self.transform is not None:
            PIL_img = self.transform(PIL_img)
            
        ############ Make FFT image ############
        # Make label
        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[index] or '1_nt' in self.data_path[index] or '1_fake' in self.data_path[index]:
            y = 1
        return PIL_img, y

    def __len__(self):
        return int(len(self.data_path))