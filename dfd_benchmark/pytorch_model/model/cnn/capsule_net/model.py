"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for training Capsule-Forensics-v2 on FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import random
import numpy as np
from torch.optim import Adam
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics

import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models

NO_CAPS = 10

class StatsNet(nn.Module):
    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.data.shape[3])

        mean = torch.mean(x, 2)
        std = torch.std(x, 2)

        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class VggExtractor(nn.Module):
    def __init__(self):
        super(VggExtractor, self).__init__()

        self.vgg_1 = self.Vgg(models.vgg19(pretrained=False), 0, 18)
        for param in self.vgg_1.parameters():
            param.requires_grad = True
            
    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin:(end + 1)])
        return features

    def forward(self, input):
        return self.vgg_1(input)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                StatsNet(),

                nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(8),
                nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(1),
                View(-1, 8),
            )
            for _ in range(NO_CAPS)]
        )

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x):
        outputs = [capsule(x.detach()) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)

        return self.squash(output, dim=-1)


class RoutingLayer(nn.Module):
    def __init__(self, device,  num_input_capsules, num_output_capsules, data_in, data_out, num_iterations):
        # gpu_id = 0, num_input_capsules = 10, num_output_capsules = 2, data_in = 8, data_out = 4, num_iterations = 2)
        super(RoutingLayer, self).__init__()

        self.device = device
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(num_output_capsules, num_input_capsules, data_out, data_in))

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x, random, dropout):
        # x[b, data, in_caps] => x[b, in_caps, data]
        x = x.transpose(2, 1)
        if random:
            noise = Variable(0.01 * torch.randn(*self.route_weights.size()))
            if self.device is not None:
                noise = noise.to(self.device)
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights

        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]
        # print(priors.shape)     # torch.Size([2, 2, 10, 4, 1])
        # route_weights [out_caps , 1 , in_caps , data_out , data_in]
        # x             [   1     , b , in_caps , data_in ,    1    ]
        # priors        [out_caps , b , in_caps , data_out,    1    ]

        priors = priors.transpose(1, 0)
        # priors[b, out_caps, in_caps, data_out, 1]

        if dropout > 0.0:
            drop = Variable(torch.FloatTensor(*priors.size()).bernoulli(1.0 - dropout))
            if self.device is not None:
                drop = drop.to(self.device)
            priors = priors * drop

        logits = Variable(torch.zeros(*priors.size()))
        # logits[b, out_caps, in_caps, data_out, 1]

        if self.device is not None:
            logits = logits.to(self.device)

        num_iterations = self.num_iterations

        for i in range(num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)

            if i != self.num_iterations - 1:
                delta_logits = priors * outputs
                logits = logits + delta_logits

        # outputs[b, out_caps, 1, data_out, 1]
        outputs = outputs.squeeze()

        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        # outputs[b, data_out, out_caps]

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, num_class, device=None):
        super(CapsuleNet, self).__init__()

        self.num_class = num_class
        self.fea_ext = FeatureExtractor()
        self.fea_ext.apply(self.weights_init)

        self.routing_stats = RoutingLayer(device=device,num_input_capsules=NO_CAPS, num_output_capsules=num_class,
                                          data_in=8, data_out=4, num_iterations=2)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x, random=False, dropout=0.0):

        z = self.fea_ext(x)
        z = self.routing_stats(z, random, dropout=dropout)
        # z[b, data, out_caps]

        classes = F.softmax(z, dim=-1)
        class_ = classes.detach()
        class_ = class_.mean(dim=1)

        # class_ = classes
        # class_ = class_.mean(dim=1)[0][1:].unsqueeze_(1)
        # print(class_)
        return classes, class_

class Args:
    dataset = "/data/tam/kaggle"
    train_set = 'train_imgs'
    val_set = 'test_imgs'
    workers = 1
    batchSize =32
    imageSize = 128
    niter =25
    lr =0.005
    beta1=0.9
    # gpu_id=0
    resume=0
    outf='checkpoints/binary_faceforensicspp'
    disable_random = False
    dropout = 0.05
    manualSeed=0

opt = Args()
opt.random = not opt.disable_random