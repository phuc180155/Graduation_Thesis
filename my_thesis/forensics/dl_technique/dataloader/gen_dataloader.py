import os, sys
from random import shuffle
import os.path as osp
sys.path.append(osp.dirname(__file__))

import torch
from torchvision import datasets, transforms

from .utils import make_weights_for_balanced_classes
from gen_dual_fft import ImageGeneratorDualFFT

"""
    Make dataloader for train and validation in trainning phase
    @info: 
        - Data Transformation for phase "train":
            Input image => Resize (target_size, target_size)
                        => Horizontal Flip with prob 0.5
                        => Rotation with angle <degrees> in [min, max] or [-degrees, degrees] with prob 0.5
                        => Affine (Rotation + Scale + Translate), here only uses Rotation and scale
                        => Convert to tensor and normalize with mean = (0.485, 0.456, 0.406) and std = (0.229, 0.224, 0.225)
        - Data Transformation for phase "test": 
            Input image => Resize (target_size, target_size)
                        => Convert to tensor and normalize with mean = (0.485, 0.456, 0.406) and std = (0.229, 0.224, 0.225)

    @Some used method:
        - dataset = datasets.ImageFolder(data_dir, transform): Make a dataset with input param data_dir (has structure below) and a transformation for each image in data_dir
            @info: structural hierachy:
                <data_dir>:
                    * <folder_contains_image_class_0>
                    * <folder_contains_image_class_1>
                    ....
                    * <folder_contains_image_class_n>
            @info <some method in this dataset Class>:
                - dataset.imgs: return [(img_path_0, class_label), (img_path_1, class_label)...]. Eg: [('/content/.../img_0.jpg', 0), ...]
                - dataset.classes: return [<folder_class_0>, <folder_class_1>, ...]. Eg: ['0_real', '1_fake']

            @return: A dataset with an item in form (tranformed_image, class_label), <class_label> base on order of respective folder in <data_dir>
        - sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True):
            @info "WeightedRandomSampler" có mục đích lấy các samples trong 1 batch_size phải thoả mãn số class lấy được phải tỷ lệ thuận với class_weights.
            @example:   Hàm <make_weights_for_balanced_classes> trả về class_weight = <num_samples>/<samples_per_class>. Class nào càng ít, class_weight càng lớn,
                        tỉ lệ lấy ra được càng lớn => sampler đảm bảo cho trong 1 batch, số lượng các class phải gần xấp xỉ nhau
"""
def generate_dataloader_image_stream(train_dir, val_dir, image_size, batch_size, num_workers, sampler_type='weight_random_sampler'):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)), \
                                        transforms.RandomHorizontalFlip(p=0.5), \
                                        transforms.RandomApply([ \
                                            transforms.RandomRotation(5),\
                                            transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)) \
                                        ], p=0.5), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                            std=[0.229, 0.224, 0.225]) \

                                        ])
    
    transform_fwd_test = transforms.Compose([transforms.Resize((image_size, image_size)), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                             std=[0.229, 0.224, 0.225]) \
                                        ])
    
    dataset_train = datasets.ImageFolder(train_dir, transform=transform_fwd)
    assert dataset_train, "Train Dataset is empty"
    print("Train image dataset: ", dataset_train.__len__())

    weights, num_samples = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    if sampler_type == 'none':
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    print(sampler_type)
    dataset_val = datasets.ImageFolder(val_dir, transform=transform_fwd_test)
    assert dataset_val, "Val Dataset is empty"
    print("Val image dataset: ", dataset_val.__len__())
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader_train, dataloader_val, num_samples


"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_dual_stream(train_dir, val_dir, image_size, batch_size, num_workers, sampler_type='weight_random_sampler'):
    # Transform for image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                            std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectrum image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    ############## TRain dataset #############
    fft_train_dataset = ImageGeneratorDualFFT(path=train_dir, image_size=image_size,\
                                              transform=transform_fwd, transform_fft=transform_fft,\
                                              should_invert=False,shuffle=True)
    
    print("fft dual train len :   ", fft_train_dataset.__len__())
    assert fft_train_dataset, "Dataset is empty!"
    
    ##### Use ImageFolder for only calculate the weights for each sample, and use it for dual_fft dataset
    dataset_train = datasets.ImageFolder(train_dir, transform=transform_fwd)
    assert dataset_train
    # Calculate weights for each sample
    weights, num_samples = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    ############## Val dataset #############
    fft_val_dataset = ImageGeneratorDualFFT(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    print("fft dual val len :   ", fft_val_dataset.__len__())
    assert fft_val_dataset
    # Make val dataloader
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

"""
    Make test dataloader for single (spatial) image stream
"""
def generate_test_dataloader_image_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.Lambda(lambda img :transforms.functional.adjust_brightness(img,adj_brightness)),\
                                        transforms.Lambda(lambda img :transforms.functional.adjust_contrast(img,adj_contrast)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Make dataset using built-in ImageFolder function of torch
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_fwd)
    assert test_dataset, "Dataset is empty!"
    print("Test dataset: ", test_dataset.__len__())
    # Make dataloader
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test

"""
    Make test dataloader for dual (spatial and frequency) stream
"""
def generate_test_dataloader_dual_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for spatial image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset
    test_dual_dataset = ImageGeneratorDualFFT(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("Test (dual) dataset: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    print("Test dataset: ", test_dual_dataset.__len__())

    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test