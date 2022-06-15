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
class DualFFTMagnitudeImageDataset(Dataset):
    def __init__(self, path,image_size, transform=None, transform_fft = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None, dset=None, highpass_filter=None):
        self.path = path
        self.transform = transform
        self.image_size =image_size
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        if dset is None:
            data_path = data_path + glob.glob(path + "/*/*.jpg")
            data_path = data_path + glob.glob(path + "/*/*.jpeg")
            data_path = data_path + glob.glob(path + "/*/*.png")
        else:
            data_path = copy.deepcopy(dset)
        self.data_path = data_path
        self.highpass = highpass_filter
        # print("sample: ", self.data_path[:10])
        # print("len: ", len(self.data_path))
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
            
    def __getitem__(self, index):
        # Read image in RGB and resize to (image_size, image_size)
        # print("idx: ", index)
        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        
        # Adjust brightness and contrast if have corresponding parameters
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)

        # Convert to PIL Image instance and transform spatial image
        PIL_img = Image.fromarray(img)
        if self.transform is not None:
            PIL_img = self.transform(PIL_img)
            
        ############ Make FFT image ############
        # Make another instance of image to do fourier transform
        img2 = transforms.ToPILImage()(PIL_img)
        img2 = np.array(img2)
        # 2D-Fourier transform, needs convert to gray-scale image to do transform
        f = np.fft.fft2(cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY))
        # Shift by spectral image
        fshift = np.fft.fftshift(f)
        fshift += 1e-8
        # Generate magnitude spectrum image
        magnitude_spectrum = np.log(np.abs(fshift))
        if self.highpass is not None:
            magnitude_spectrum = self.highpass_filter(fshift=fshift, img=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY))
        magnitude_spectrum = cv2.resize(magnitude_spectrum, (self.image_size,self.image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1, 2, 0))    # From C, H, W =>  H, W, C
        if self.transform_fft is not None:
            magnitude_spectrum = self.transform_fft(magnitude_spectrum)

        # Make label
        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[index] or '1_nt' in self.data_path[index] or '1_fake' in self.data_path[index]:
            y = 1
        return PIL_img, magnitude_spectrum, y

    def highpass_filter(self, fshift: np.ndarray, img=None):
        H, W = img.shape[:2]
        crow, ccol = self.image_size // 2, self.image_size // 2

        # Create a mask with two channels because DFT result in 2D
        mask = np.ones((H, W), dtype=np.uint8)

        # Circular radius from center
        r = H // int(self.highpass)
        center = [crow, ccol]

        # @return x: np.ndarray([[0], [1]...[H-1]], shape=(H, 1))
        # @return y: np.ndarray([[0, 1... W-1]], shape=(1, W))
        x, y = np.ogrid[:H, :W]

        # Tạo mask_area: Lọc các tần số thấp <=> mask = 0
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        f_shift = fshift * mask
        f_shift += 1e-8
        filtered_magnitude_spectrum = np.log(np.abs(f_shift))
        return filtered_magnitude_spectrum

    def __len__(self):
        return int(len(self.data_path))

class TripleFFTMagnitudePhaseDataset(Dataset):
    def __init__(self, path,image_size, transform=None, transform_fft = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None):
        self.path = path
        self.transform = transform
        self.image_size =image_size
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
            
    def __getitem__(self, index):
        # Read image in RGB and resize to (image_size, image_size)
        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        
        # Adjust brightness and contrast if have corresponding parameters
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)

        # Convert to PIL Image instance and transform spatial image
        PIL_img = Image.fromarray(img)
        if self.transform is not None:
            PIL_img = self.transform(PIL_img)
            
        ############ Make FFT Magnitude image ############
        # Make another instance of image to do fourier transform
        img2 = transforms.ToPILImage()(PIL_img)
        img2 = np.array(img2)
        # 2D-Fourier transform, needs convert to gray-scale image to do transform
        f = np.fft.fft2(cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY))
        # Shift by spectral image
        fshift = np.fft.fftshift(f)
        fshift += 1e-8
        # Generate magnitude spectrum image
        magnitude_spectrum = np.log(np.abs(fshift))
        magnitude_spectrum = cv2.resize(magnitude_spectrum, (self.image_size,self.image_size))
        magnitude_spectrum = np.array([magnitude_spectrum])
        magnitude_spectrum = np.transpose(magnitude_spectrum, (1, 2, 0))    # From C, H, W =>  H, W, C
        
        if self.transform_fft is not None:
            magnitude_spectrum = self.transform_fft(magnitude_spectrum)

        ############ Make FFT Phase image ############
        # Make another instance of image to do fourier transform
        img3 = transforms.ToPILImage()(PIL_img)
        img3 = np.array(img3)
        # 2D-Fourier transform, needs convert to gray-scale image to do transform
        f = np.fft.fft2(cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY))
        # Shift by spectral image
        fshift = np.fft.fftshift(f)
        # Generate phase spectrum image
        phase_spectrum = np.angle(fshift)
        phase_spectrum = cv2.resize(phase_spectrum, (self.image_size,self.image_size))
        phase_spectrum = np.array([phase_spectrum])
        phase_spectrum = np.transpose(phase_spectrum, (1, 2, 0))    # From C, H, W =>  H, W, C
        
        if self.transform_fft is not None:
            phase_spectrum = self.transform_fft(phase_spectrum)

        # Make label
        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[index] or '1_nt' in self.data_path[index] or '1_fake' in self.data_path[index]:
            y = 1
        return PIL_img, magnitude_spectrum, phase_spectrum, y

    def __len__(self):
        return int(len(self.data_path))

class DualFFTMagnitudeFeatureDataset(Dataset):
    def __init__(self, path,image_size, transform=None, transform_fft = None, should_invert=True,shuffle=True,adj_brightness=None, adj_contrast=None):
        self.path = path
        self.transform = transform
        self.image_size =image_size
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)
            
    def __getitem__(self, index):
        # Read image in RGB and resize to (image_size, image_size)
        img = cv2.imread(self.data_path[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        
        # Adjust brightness and contrast if have corresponding parameters
        if self.adj_brightness is not None and self.adj_contrast is not None:
            PIL_img1 = Image.fromarray(img)
            enhancer = ImageEnhance.Brightness(PIL_img1)
            img_adj = enhancer.enhance(self.adj_brightness)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(self.adj_contrast)
            img = np.array(img_adj)

        # Convert to PIL Image instance and transform spatial image
        PIL_img = Image.fromarray(img)
        if self.transform is not None:
            PIL_img = self.transform(PIL_img)
            
        ############ Make FFT image ############
        # Make another instance of image to do fourier transform
        img2 = transforms.ToPILImage()(PIL_img)
        img2 = np.array(img2)
        # 2D-Fourier transform, needs convert to gray-scale image to do transform
        f = np.fft.fft2(cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY))
        # Shift by spectral image
        fshift = np.fft.fftshift(f)
        fshift += 1e-8
        # Generate magnitude spectrum image
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        psd1D = azimuthalAverage(magnitude_spectrum)
        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = (psd1D-np.min(psd1D))/(np.max(psd1D)-np.min(psd1D))

        # psd1D = np.expand_dims(psd1D, axis=0)
        # if self.transform_fft is not None:
        #     psd1D = self.transform_fft(psd1D)
        
        psd1D = torch.Tensor(psd1D)
        # Make label
        y = 0
        if '0_real' in self.data_path[index]:
            y = 0
        elif '1_df' in self.data_path[index] or '1_f2f' in self.data_path[index] or '1_fs' in self.data_path[index] or '1_nt' in self.data_path[index] or '1_fake' in self.data_path[index]:
            y = 1
        return PIL_img, psd1D, y

    def __len__(self):
        return int(len(self.data_path))