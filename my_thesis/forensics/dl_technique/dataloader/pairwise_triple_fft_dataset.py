import glob
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import cv2
from PIL import Image, ImageEnhance
import torch

class PairwiseTripleFFTMagnitudePhaseDataset(Dataset):
    def __init__(self, path=None, image_size=128, transform=None, transform_fft=None, should_invert=True,shuffle=True, adj_brightness=None, adj_contrast=None):
        self.path = path
        self.imageFolderDataset = ImageFolder(path)
        self.image_size = image_size
        self.transform = transform
        self.transform_fft = transform_fft
        self.should_invert = should_invert
        self.shuffle = shuffle
        self.adj_brightness = adj_brightness
        self.adj_contrast = adj_contrast
        data_path = []
        data_path = data_path + glob.glob(path + "/*/*.jpg")
        data_path = data_path + glob.glob(path + "/*/*.jpeg")
        data_path = data_path + glob.glob(path + "/*/*.png")
        self.data_path = data_path
        np.random.shuffle(self.data_path)
        self.indexes = range(len(self.data_path))
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.data_path)

    def find_label(self, img_path):
        y = 0
        if '0_real' in img_path:
            y = 0
        elif '1_df' in img_path or '1_f2f' in img_path or '1_fs' in img_path or '1_nt' in img_path or '1_fake' in img_path:
            y = 1
        return y

    def __getitem__(self, index):
        img0_path = self.data_path[index]
        #we need to make sure approx 50% of images are in the same class
        img0_label = self.find_label(img0_path)
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_index = random.choice(self.indexes)
                if index == img1_index:
                    continue
                img1_path = self.data_path[img1_index]
                img1_label = self.find_label(img1_path)
                if img0_label == img1_label:
                    break
        else:
            while True:
                #keep looping till the same class image is found
                img1_index = random.choice(self.indexes)
                if index == img1_index:
                    continue
                img1_path = self.data_path[img1_index]
                img1_label = self.find_label(img1_path)
                if img0_label != img1_label:
                    break

        PIL_img0, PIL_img1, mag_img0, phase_img0, mag_img1, phase_img1 = self.__data_generation(img0_path, img1_path)
        # print(img0_path, img1_path)
        # print(img0_label, img1_label)

        if img0_label != img1_label:
            label_contrastive = 1
        else:
            label_contrastive = 0
        return PIL_img0, mag_img0, phase_img0, img0_label, PIL_img1, mag_img1, phase_img1, img1_label, label_contrastive

    def __data_generation(self, img0_path, img1_path):
        PIL_img0, mag_img0, phase_img0 = self.__get_triple_images(img0_path)
        PIL_img1, mag_img1, phase_img1 = self.__get_triple_images(img1_path)
        return PIL_img0, PIL_img1, mag_img0, phase_img0, mag_img1, phase_img1


    def __get_triple_images(self, img_path):
       # Read image in RGB and resize to (image_size, image_size)
        img = cv2.imread(img_path)
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
        return PIL_img, magnitude_spectrum, phase_spectrum

    def __len__(self):
        return int(np.floor(len(self.data_path)))