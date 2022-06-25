import torch.nn as nn
from torch import einsum
import torch
from torchsummary import summary
from einops import rearrange

import sys
from model.backbone.efficient_net.model import EfficientNet

import re
import torch.nn.functional as F

import re, math
from model.vision_transformer.vit.vit import ViT, Transformer
from model.vision_transformer.vit.kvit import kNNTransformer
from model.vision_transformer.cnn_vit.efficient_vit import EfficientViT
import torch.nn.functional as F
from pytorchcv.model_provider import get_model

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, topk_rate=0.5):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.channel = channel
        self.topk_rate = topk_rate
        self.topk = int(channel * topk_rate)

    def forward(self, x):

        # x: B, C, W, H
        # y: B, C, 1, 1
        y = self.avg_pool(x)
        attn_weight = self.conv_du(y)   # B, C, 1, 1
        attn = attn_weight * x
        attnw_idx = torch.topk(input=attn_weight,k=self.topk,dim=1,largest=True, sorted=False).indices  # B, k, 1, 1
        attnw_idx = torch.sort(attnw_idx, dim=1).values
        attnw_idx = attnw_idx.expand(-1, -1, x.shape[2], x.shape[3])     
        attn = torch.gather(attn, dim=1, index=attnw_idx)
        return attn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class DAB(nn.Module):
    def __init__(self, n_feat: int, reduction: int, topk_rate=0.5, act_dab=None, dab_modules='sa-ca'):
        super(DAB, self).__init__()
        self.use_sa = True if 'sa' in dab_modules else False
        self.use_ca = True if 'ca' in dab_modules else False
        self.dab_modules = dab_modules
        self.SA = spatial_attn_layer() if self.use_sa else None             ## Spatial Attention
        self.CA = CALayer(n_feat, reduction, topk_rate)  if self.use_ca else None      ## Channel Attention
        self.conv1x1_1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)
        # self.conv1x1_2 = nn.Conv2d(n_feat, n_feat, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(int(n_feat * (1 + topk_rate)), n_feat, kernel_size=1)
        self.act = act_dab
        self.topk_rate = topk_rate

    def forward(self, ifreq, rgb):
        if self.use_sa:
            sa_branch = self.SA(ifreq)
        if self.use_ca:
            ca_branch = self.CA(ifreq)

        if self.use_sa and self.use_ca:
            attn = torch.cat([sa_branch, ca_branch], dim=1)
        if self.use_sa and not self.use_ca:
            attn = sa_branch
        if not self.use_sa and self.use_ca:
            attn = ca_branch

        # print(attn.shape)
        if '-' in self.dab_modules:
            attn = self.conv1x1_1(attn)

        # print("        attn shape: ", rgb.shape, attn.shape)
        res = torch.cat([rgb, attn], dim=1)
        res = self.conv1x1_3(res)
        if self.act:
            res = self.act(res)
        return res
    
class DualDabCNN(nn.Module):
    def __init__(self, image_size=224, num_classes=1, \
                mlp_dim=2048,\
                backbone='xception_net', pretrained=True,unfreeze_blocks=-1,\
                features_at_block='10', \
                dropout_in_mlp=0.0,
                act_dab='none', topk_channels=0.5, dab_modules='sa-ca', dabifft_normalize='none', dab_blocks='1_3_6_9'):  

        super(DualDabCNN, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.backbone = backbone
        if 'efficient_net' in backbone:
            dab_blocks = '-1_' + dab_blocks
        self.dab_blocks = sorted(list(map(int, dab_blocks.split('_'))))
        print("dab blocks: ", self.dab_blocks)
        self.dabifft_normalize = dabifft_normalize
        self.last_block = int(features_at_block) if features_at_block != 'final' else 15

        self.features_size = {
            'efficient_net': {
                '0': (16, 64, 64),
                '1': (24, 32, 32),
                '2': (24, 32, 32),
                '3': (40, 16, 16),
                '4': (40, 16, 16),
                '5': (80, 8, 8),
                '6': (80, 8, 8),
                '7': (80, 8, 8),
                '8': (112, 8, 8),
                '9': (112, 8, 8),
                '10': (112, 8, 8),
                '11': (192, 4, 4),
                '12': (192, 4, 4),
                '13': (192, 4, 4),
                '14': (192, 4, 4),
                '15': (320, 4, 4),
                'final': (1280, 4, 4)
            },
            'xception_net': {
                'final': (2048, 4, 4)
            }
        }
        self.out_ext_channels = self.features_size[backbone][features_at_block][0]
        self.out_ext_size = self.features_size[backbone][features_at_block][1]
        self.dab_activation = self.get_activation(act_dab)

        self.pretrained = pretrained
        self.features_at_block = features_at_block
        self.rgb_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=3)   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)
        self.freq_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)     

        # DAB Block:
        num_dab = len(self.dab_blocks) - 1
        self.dab = nn.ModuleList([])
        for i in range(num_dab):
            at_block = self.dab_blocks[i+1]
            in_features = self.features_size[backbone][str(at_block)][0]
            self.dab.append(DAB(n_feat=in_features, reduction=1, topk_rate=topk_channels, act_dab=self.dab_activation, dab_modules=dab_modules))

        # Multi ViT:
        # print(type(useKNN))
        self.mlp_relu = nn.ReLU(inplace=True)
        self.mlp_dropout = nn.Dropout(dropout_in_mlp)
        self.mlp_head_hidden = nn.Linear(2 * self.out_ext_channels, mlp_dim)
        self.mlp_head_out = nn.Linear(mlp_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def get_activation(self, act):
        if act == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act == 'tanh':
            activation = nn.Tanh()
        elif act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'leakyrely':
            activation = nn.LeakyReLU()
        elif act == 'selu':
            activation = nn.SELU()
        elif act == 'gelu':
            activation = nn.GELU()
        else:
            activation = None
        return activation

    def get_feature_extractor(self, architecture="efficient_net", unfreeze_blocks=-1, pretrained=False, num_classes=1, in_channels=3):
        extractor = None
        if architecture == "efficient_net":
            extractor = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes,in_channels = in_channels, pretrained=bool(pretrained))
            if unfreeze_blocks != -1:
                # Freeze the first (num_blocks - 3) blocks and unfreeze the rest 
                for i in range(0, len(extractor._blocks)):
                    for index, param in enumerate(extractor._blocks[i].parameters()):
                        if i >= len(extractor._blocks) - unfreeze_blocks:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        
        if architecture == 'xception_net':
            xception = get_model("xception", pretrained=bool(pretrained))
            extractor = nn.Sequential(*list(xception.children())[:-1])
            extractor[0].final_block.pool = nn.Identity()
            if in_channels != 3:
                extractor[0].init_block.conv1.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

            if unfreeze_blocks != -1:
                blocks = len(extractor[0].children())
                print("Number of blocks in xception: ", len(blocks))
                for i, block in enumerate(extractor[0].children()):
                    if i >= blocks - unfreeze_blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                    else:
                        for param in block.parameters():
                            param.requires_grad = False
        print("Pretrained backbone: ", bool(pretrained))
        return extractor

    def ifft(self, freq_feature, norm_type='none'):
        ifreq_feature = torch.log(torch.abs(torch.fft.ifft2(torch.fft.ifftshift(freq_feature))) + 1e-10)  # Hơi ảo???
        if norm_type == 'none':
            pass
        elif norm_type == 'batchnorm':
            ifreq_feature = self.batchnorm_ifft(ifreq_feature)
        elif norm_type == 'layernorm':
            ifreq_feature = self.layernorm_ifft(ifreq_feature)
        elif norm_type == 'normal':
            ifreq_feature = F.normalize(ifreq_feature)
        elif norm_type == 'no_ifft':
            return freq_feature
        return ifreq_feature


    def extract_feature(self, rgb_imgs, freq_imgs):
        if self.backbone == 'efficient_net':
            # Conv stem:
            rgb_features = self.rgb_extractor.extract_features_convstem(rgb_imgs)                 
            freq_features = self.freq_extractor.extract_features_convstem(freq_imgs)
            # print("Features shape: ", rgb_features.shape)
            # DAB Block:
            for i in range(len(self.dab_blocks) - 1):   # -1_1_3_6_9
                                                        #  0 1 2 3 4
                # print("dab_blocks: ", self.dab_blocks[i]+1, ' -> ', self.dab_blocks[i+1])
                rgb_features = self.rgb_extractor.extract_features_block_inrange(rgb_features, from_block=self.dab_blocks[i]+1, to_block=self.dab_blocks[i+1])
                freq_features = self.freq_extractor.extract_features_block_inrange(freq_features, from_block=self.dab_blocks[i]+1, to_block=self.dab_blocks[i+1])
                # Attention, concat and reduce channels:
                ifreq_features = self.ifft(freq_features, norm_type=self.dabifft_normalize)
                # print(" Shape: ", rgb_features.shape, ifreq_features.shape)
                rgb_features = self.dab[i](ifreq_features, rgb_features)

            # Last block:
            rgb_features = self.rgb_extractor.extract_features_block_inrange(rgb_features, from_block=self.dab_blocks[-1]+1, to_block=self.last_block)
            freq_features = self.freq_extractor.extract_features_block_inrange(freq_features, from_block=self.dab_blocks[-1]+1, to_block=self.last_block)
            
            # Convhead:
            # print("After last block Features shape: ", rgb_features.shape, freq_features.shape)
            if self.features_at_block == 'final':
                rgb_features = self.rgb_extractor.extract_features_convhead(rgb_features)
                freq_features = self.freq_extractor.extract_features_convhead(freq_features)
        else:
            rgb_features = self.rgb_extractor(rgb_imgs)
            freq_features = self.freq_extractor(freq_imgs)
        return rgb_features, freq_features

    def forward(self, rgb_imgs, freq_imgs):
        rgb_features, freq_features = self.extract_feature(rgb_imgs, freq_imgs)
        # print("Features shape: ", rgb_features.shape, freq_features.shape)

        ##### Forward to ViT
        # x = self.multi_transformer(rgb_features, freq_features, ifreq_features)     # B, number_of_patch * D
        x = torch.cat([rgb_features, freq_features], dim=1)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(dim=-1).squeeze(dim=-1)

        x = self.mlp_dropout(x)         # B, number_of_patch * D
        x = self.mlp_head_hidden(x)     # B, number_of_patch * D => B, mlp_dim
        x = self.mlp_relu(x)
        x = self.mlp_dropout(x)
        x = self.mlp_head_out(x)        # B, mlp_dim => B, 1
        x = self.sigmoid(x)
        return x

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(2, 3, 128, 128)
    y = torch.ones(2, 1, 128, 128)
    model_ = DualDabCNN(image_size=128, num_classes=1, \
                mlp_dim=2048,\
                backbone='efficient_net', pretrained=True,unfreeze_blocks=-1,\
                features_at_block='11', \
                act_dab='none', topk_channels=0.5, dab_modules='ca', dabifft_normalize='normal', dab_blocks='0_3_5_6_10')
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    x, y = x.to(device), y.to(device)
    model_ = model_.to(device)

    out = model_(x, y)
    print(out.shape)