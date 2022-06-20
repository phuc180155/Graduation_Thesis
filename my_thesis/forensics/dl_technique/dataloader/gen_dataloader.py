import os, sys
from random import shuffle
import os.path as osp
sys.path.append(osp.dirname(__file__))

import torch
from torchvision import datasets, transforms

from .utils import make_weights_for_balanced_classes, make_weights_for_balanced_classes_2
from .dual_fft_dataset import DualFFTMagnitudeFeatureDataset, DualFFTMagnitudeImageDataset, TripleFFTMagnitudePhaseDataset
from .pairwise_dual_fft_dataset import PairwiseDualFFTMagnitudeFeatureDataset, PairwiseDualFFTMagnitudeImageDataset
from .pairwise_triple_fft_dataset import PairwiseTripleFFTMagnitudePhaseDataset
from .triplewise_dual_fft_dataset import TriplewiseDualFFTMagnitudeImageDataset
from .transform import transform_method


#################################################################################################################
########################################## SINGLE FOR RGB IMAGE STREAM ##########################################
#################################################################################################################
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
def generate_dataloader_single_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=False, sampler_type='weight_random_sampler'):
    # Add augmentation:
    if not augmentation:
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
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)

    # Transformation for val phase
    transform_fwd_val = transforms.Compose([transforms.Resize((image_size, image_size)), \
                                             transforms.ToTensor(), \
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                                  std=[0.229, 0.224, 0.225]) \
                                             ])
    # Make dataloader train
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
    # Make dataloader val
    dataset_val = datasets.ImageFolder(val_dir, transform=transform_fwd_val)
    assert dataset_val, "Val Dataset is empty"
    print("Val image dataset: ", dataset_val.__len__())
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

"""
    Make test dataloader for single (spatial) image stream
"""
def generate_test_dataloader_single_cnn_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.Lambda(lambda img :transforms.functional.adjust_brightness(img,adj_brightness)),\
                                        transforms.Lambda(lambda img :transforms.functional.adjust_contrast(img,adj_contrast)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Make dataset using built-in ImageFolder function of torch
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_fwd)
    assert test_dataset, "Test Dataset is empty!"
    print("Test image dataset: ", test_dataset.__len__())
    # Make dataloader
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test

###########################################################################################################################################
########################################## DUAL CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM ##########################################
###########################################################################################################################################

"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_dual_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Make dataloader train:
    fft_train_dataset = DualFFTMagnitudeImageDataset(path=train_dir, image_size=image_size,\
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
    
    # Make dataloader val
    fft_val_dataset = DualFFTMagnitudeImageDataset(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    assert fft_val_dataset
    print("fft dual val len :   ", fft_val_dataset.__len__())
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

"""
    Make test dataloader for dual (spatial and frequency) stream
"""
def generate_test_dataloader_dual_cnn_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for RGB image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset and make test dataloader
    test_dual_dataset = DualFFTMagnitudeImageDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("fft dual test len: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test

###########################################################################################################################################################
########################################## DUAL CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM  FOR TRIPLEWISE ##########################################
###########################################################################################################################################################

"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_dual_cnn_stream_for_triplewise(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Make dataloader train:
    fft_train_dataset = TriplewiseDualFFTMagnitudeImageDataset(path=train_dir, image_size=image_size,\
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
    
    # Make dataloader val
    fft_val_dataset = TriplewiseDualFFTMagnitudeImageDataset(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    assert fft_val_dataset
    print("fft dual val len :   ", fft_val_dataset.__len__())
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val

"""
    Make test dataloader for dual (spatial and frequency) stream
"""
def generate_test_dataloader_dual_cnn_stream_for_triplewise(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for RGB image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset and make test dataloader
    test_dual_dataset = TriplewiseDualFFTMagnitudeImageDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("fft dual test len: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test


###########################################################################################################################################
########################################## TRIPLE CNN-CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM ##########################################
###########################################################################################################################################

"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_triple_cnn_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Make dataloader train:
    fft_train_dataset = TripleFFTMagnitudePhaseDataset(path=train_dir, image_size=image_size,\
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
    
    # Make dataloader val
    fft_val_dataset = TripleFFTMagnitudePhaseDataset(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    assert fft_val_dataset
    print("fft dual val len :   ", fft_val_dataset.__len__())
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

"""
    Make test dataloader for triple (spatial and frequency) stream
"""
def generate_test_dataloader_triple_cnn_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for RGB image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset and make test dataloader
    test_dual_dataset = TripleFFTMagnitudePhaseDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("fft dual test len: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test

##############################################################################################################################################################
########################################## TRIPLE CNN-CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM FOR PAIRWISE ##########################################
##############################################################################################################################################################

"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_triple_cnn_stream_for_pairwise(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Make dataloader train:
    fft_train_dataset = PairwiseTripleFFTMagnitudePhaseDataset(path=train_dir, image_size=image_size,\
                                              transform=transform_fwd, transform_fft=transform_fft,\
                                              should_invert=False,shuffle=True)
    
    print("fft triple train len :   ", fft_train_dataset.__len__())
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
    
    # Make dataloader val
    fft_val_dataset = PairwiseTripleFFTMagnitudePhaseDataset(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    assert fft_val_dataset
    print("fft triple val len :   ", fft_val_dataset.__len__())
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val

"""
    Make test dataloader for triple (spatial and frequency) stream
"""
def generate_test_dataloader_triple_cnn_stream_for_pairwise(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for RGB image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset and make test dataloader
    test_dual_dataset = PairwiseTripleFFTMagnitudePhaseDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("fft dual test len: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test


###################################################################################################################################################
########################################## DUAL CNN-FEEDFORWARD FOR RGB IMAGE AND FREQUENCY IMAGE STREAM ##########################################
###################################################################################################################################################

def generate_dataloader_dual_cnnfeedforward_stream(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = None
    
    # Make train dataloader
    fft_train_dataset = DualFFTMagnitudeFeatureDataset(path=train_dir, image_size=image_size,\
                                              transform=transform_fwd, transform_fft=transform_fft,\
                                              should_invert=False,shuffle=True)
    print("fft dual train len :   ", fft_train_dataset.__len__())
    assert fft_train_dataset, "Dataset is empty!"
    dataset_train = datasets.ImageFolder(train_dir, transform=transform_fwd)
    assert dataset_train
    weights, num_samples = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    # Make val dataloader
    fft_val_dataset = DualFFTMagnitudeFeatureDataset(path=val_dir,image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False)
    print("fft dual val len :   ", fft_val_dataset.__len__())
    assert fft_val_dataset
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

def generate_test_dataloader_dual_cnnfeedforward_stream(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    # Transform for spatial image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = None
    
    # Generate dataset
    test_dual_dataset = DualFFTMagnitudeFeatureDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
    print("fft dual test len : ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test


####################################################################################################################################################
########################################## PAIRWISE DUAL CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM ##########################################
####################################################################################################################################################
def generate_dataloader_dual_cnn_stream_for_pairwise(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for training phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.RandomHorizontalFlip(p=0.5),\
                                            transforms.RandomApply([
                                                transforms.RandomRotation(5),\
                                                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05))
                                            ], p=0.5),
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])

    # Make train dataloader
    train_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path=train_dir, image_size=image_size, transform=transform_fwd, transform_fft=transform_fft)
    dataset_train = datasets.ImageFolder(train_dir, transform=transform_fwd)
    assert dataset_train
    # Calculate weights for each sample
    weights, num_samples = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    assert train_pairwise_dualfft_dataset, "Train dataset is None!"

    # Make val dataloader:
    transform_val_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    val_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path=val_dir, image_size=image_size, transform=transform_val_fwd, transform_fft=transform_fft)
    val_dataloader  = torch.utils.data.DataLoader(val_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert val_pairwise_dualfft_dataset, "Val dataset is None!"
    return train_dataloader, val_dataloader

def generate_test_dataloader_dual_cnn_stream_for_pairwise(test_dir, image_size, batch_size, num_workers):
    transform_test_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(), \
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    transform_fft = transforms.Compose([transforms.ToTensor()])                
    test_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path=test_dir, image_size=image_size, transform=transform_test_fwd, transform_fft=transform_fft)
    test_dataloader  = torch.utils.data.DataLoader(test_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert test_pairwise_dualfft_dataset, "Val dataset is None!"
    return test_dataloader

############################################################################################################################################################
########################################## PAIRWISE DUAL CNN-FEEDFORWARD FOR RGB IMAGE AND FREQUENCY IMAGE STREAM ##########################################
############################################################################################################################################################
def generate_dataloader_dual_cnnfeedforward_stream_for_pairwise(train_dir, val_dir, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for trainning phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.RandomHorizontalFlip(p=0.5),\
                                            transforms.RandomApply([
                                                transforms.RandomRotation(5),\
                                                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05))
                                            ], p=0.5),
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    # Transform for spectrum image
    transform_fft = None
    train_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeFeatureDataset(path=train_dir, image_size=image_size, transform=transform_fwd, transform_fft=transform_fft)

    dataset_train = datasets.ImageFolder(train_dir, transform=transform_fwd)
    assert dataset_train
    # Calculate weights for each sample
    weights, num_samples = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    assert train_pairwise_dualfft_dataset, "Train dataset is None!"
    # Transform for val dataset:
    transform_val_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    val_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeFeatureDataset(path=val_dir, image_size=image_size, transform=transform_val_fwd, transform_fft=transform_fft)
    val_dataloader  = torch.utils.data.DataLoader(val_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert val_pairwise_dualfft_dataset, "Val dataset is None!"
    return train_dataloader, val_dataloader

def generate_test_dataloader_dual_cnnfeedforward_stream_for_pairwise(test_dir, image_size, batch_size, num_workers):
    transform_test_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(), \
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    transform_fft = None       
    test_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeFeatureDataset(path=test_dir, image_size=image_size, transform=transform_test_fwd, transform_fft=transform_fft)
    test_dataloader  = torch.utils.data.DataLoader(test_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert test_pairwise_dualfft_dataset, "Test dataset is None!"
    return test_dataloader

######################################################################################################################################################
########################################## DUAL CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM FOR K-FOLD ##########################################
######################################################################################################################################################

"""
    Make  dataloader for both spatial image and spectrum image in training phase

"""
def generate_dataloader_dual_cnn_stream_for_kfold(train_dir, train_set, val_set, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler', highpass=None):
    # Transform for train phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                                std=[0.229, 0.224, 0.225]),\
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Make dataloader train:
    fft_train_dataset = DualFFTMagnitudeImageDataset(path='', image_size=image_size,\
                                              transform=transform_fwd, transform_fft=transform_fft,\
                                              should_invert=False,shuffle=True, dset=train_set, highpass_filter=highpass)
    
    print("fft dual train len :   ", fft_train_dataset.__len__())
    assert fft_train_dataset, "Dataset is empty!"
    ##### Use ImageFolder for only calculate the weights for each sample, and use it for dual_fft dataset
    # Calculate weights for each sample
    weights, num_samples = make_weights_for_balanced_classes_2(fft_train_dataset.data_path, 2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(fft_train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    # Make dataloader val
    fft_val_dataset = DualFFTMagnitudeImageDataset(path='',image_size=image_size,\
                                            transform=transform_fwd, transform_fft=transform_fft,\
                                            should_invert=False,shuffle=False, dset=val_set, highpass_filter=highpass)
    assert fft_val_dataset
    print("fft dual val len :   ", fft_val_dataset.__len__())
    dataloader_val = torch.utils.data.DataLoader(fft_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader_train, dataloader_val, num_samples

"""
    Make test dataloader for dual (spatial and frequency) stream
"""
def generate_test_dataloader_dual_cnn_stream_for_kfold(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0, highpass=None):
    # Transform for RGB image
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Transform for spectral image
    transform_fft = transforms.Compose([transforms.ToTensor()])
    
    # Generate dataset and make test dataloader
    test_dual_dataset = DualFFTMagnitudeImageDataset(path=test_dir, image_size=image_size,\
                                        transform=transform_fwd, transform_fft=transform_fft,\
                                        should_invert=False, shuffle=False, adj_brightness=adj_brightness, adj_contrast=adj_contrast, highpass_filter=highpass)
    print("fft dual test len: ", test_dual_dataset.__len__())
    assert test_dual_dataset, "Dataset is empty!"
    dataloader_test = torch.utils.data.DataLoader(test_dual_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test

##############################################################################################################################################################
########################################## PAIRWISE DUAL CNN-CNN FOR RGB IMAGE AND FREQUENCY IMAGE STREAM FOR KFOLD ##########################################
##############################################################################################################################################################
def generate_dataloader_dual_cnn_stream_for_kfold_pairwise(train_dir, train_set, val_set, image_size, batch_size, num_workers, augmentation=True, sampler_type='weight_random_sampler'):
    # Transform for training phase:
    if not augmentation:
        transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.RandomHorizontalFlip(p=0.5),\
                                            transforms.RandomApply([
                                                transforms.RandomRotation(5),\
                                                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05))
                                            ], p=0.5),
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)
    transform_fft = transforms.Compose([transforms.ToTensor()])

    # Make train dataloader
    train_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path='', image_size=image_size, transform=transform_fwd, transform_fft=transform_fft, dset=train_set)
    weights, num_samples = make_weights_for_balanced_classes_2(train_pairwise_dualfft_dataset.data_path, 2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_pairwise_dualfft_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    assert train_pairwise_dualfft_dataset, "Train dataset is None!"

    # Make val dataloader:
    transform_val_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    val_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path='', image_size=image_size, transform=transform_val_fwd, transform_fft=transform_fft, dset=val_set, shuffle=False)
    val_dataloader  = torch.utils.data.DataLoader(val_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert val_pairwise_dualfft_dataset, "Val dataset is None!"
    return train_dataloader, val_dataloader, num_samples

def generate_test_dataloader_dual_cnn_stream_for_kfold_pairwise(test_dir, image_size, batch_size, num_workers):
    transform_test_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(), \
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    transform_fft = transforms.Compose([transforms.ToTensor()])                
    test_pairwise_dualfft_dataset = PairwiseDualFFTMagnitudeImageDataset(path=test_dir, image_size=image_size, transform=transform_test_fwd, transform_fft=transform_fft)
    test_dataloader  = torch.utils.data.DataLoader(test_pairwise_dualfft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    assert test_pairwise_dualfft_dataset, "Val dataset is None!"
    return test_dataloader

###########################################################################################################################
########################################## SINGLE FOR RGB IMAGE STREAM FOR KFOLD ##########################################
###########################################################################################################################
from dataloader.simple_dataset import SingleRGBStreamDataset
def generate_dataloader_single_cnn_stream_for_kfold(train_dir, train_set, val_set, image_size, batch_size, num_workers, augmentation=False, sampler_type='weight_random_sampler'):
    # Add augmentation:
    if not augmentation:
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
    else:
        transform_fwd = transform_method(image_size=image_size, mean_noise=0.1, std_noise=0.08)

    # Make dataloader train
    dataset_train = SingleRGBStreamDataset(path=train_set, transform=transform_fwd, image_size=image_size, shuffle=True)
    assert dataset_train, "Train Dataset is empty"
    weights, num_samples = make_weights_for_balanced_classes_2(dataset_train.data_path, 2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # Make dataloader with WeightedRandomSampler
    if sampler_type == 'none':
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    # Make val dataloader:
    transform_val_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                            transforms.ToTensor(),\
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    dataset_val = SingleRGBStreamDataset(path=val_set, transform=transform_val_fwd, image_size=image_size, shuffle=False)
    assert dataset_val, "Val dataset is None!"
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dataloader, dataloader_val, num_samples

"""
    Make test dataloader for single (spatial) image stream
"""
def generate_test_dataloader_single_cnn_stream_for_kfold(test_dir, image_size, batch_size, num_workers, adj_brightness=1.0, adj_contrast=1.0):
    transform_fwd = transforms.Compose([transforms.Resize((image_size,image_size)),\
                                        # transforms.Lambda(lambda img :transforms.functional.adjust_brightness(img,adj_brightness)),\
                                        # transforms.Lambda(lambda img :transforms.functional.adjust_contrast(img,adj_contrast)),\
                                        transforms.ToTensor(),\
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                             std=[0.229, 0.224, 0.225]),\
                                        ])
    # Make dataset using built-in ImageFolder function of torch
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_fwd)
    assert test_dataset, "Test Dataset is empty!"
    print("Test image dataset: ", test_dataset.__len__())
    # Make dataloader
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test