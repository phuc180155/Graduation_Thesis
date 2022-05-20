import torch
import torchvision.transforms as transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def transform_method(image_size=128, mean_noise=0.1, std_noise=0.08):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomApply([
            transforms.RandomRotation(10, expand=False),
            transforms.RandomAffine(degrees=10, scale=(0.95, 1.05))
        ], p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.0),
        transforms.ToTensor(), 
        transforms.RandomErasing(p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([
            AddGaussianNoise(mean=mean_noise, std=std_noise)
        ],p=0.1),
    ])
