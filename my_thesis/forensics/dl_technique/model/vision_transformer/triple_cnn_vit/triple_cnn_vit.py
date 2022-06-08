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
from model.vision_transformer.cnn_vit.efficient_vit import EfficientViT
from pytorchcv.model_provider import get_model

class CrossAttention(nn.Module):
    def __init__(self, in_dim, inner_dim=0, prj_out=True, qkv_embed=True):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.qkv_embed = qkv_embed
        self.to_out = nn.Identity()
        if self.qkv_embed:
            inner_dim = self.in_dim if inner_dim == 0 else inner_dim
            self.to_k = nn.Linear(in_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(in_dim, inner_dim, bias = False)
            self.to_q = nn.Linear(in_dim, inner_dim, bias = False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, in_dim),
                nn.Dropout(p=0.0)
            ) if prj_out else nn.Identity()

    def forward(self, x, y):
        """
            x ~ rgb_vectors: (b, n, in_dim)
            y ~ freq_vectors: (b, n, in_dim)
            z ~ freq_vectors: (b, n, in_dim)
            Returns:
                attn_weight: (b, n, n)
                attn_output: (b, n, in_dim)
        """
        if self.qkv_embed:
            q = self.to_q(x)
            k = self.to_k(y)
            v = self.to_v(y)
        else:
            q, k, v = x, y, y
        out, attn = self.scale_dot(q, k, v, dropout_p=0.00)
        out = self.to_out(out)
        return out, attn

    """
        Get from torch.nn.MultiheadAttention
        scale-dot: https://github.com/pytorch/pytorch/blob/1c5a8125798392f8d7c57e88735f43a14ae0beca/torch/nn/functional.py#L4966
        multi-head: https://github.com/pytorch/pytorch/blob/1c5a8125798392f8d7c57e88735f43a14ae0beca/torch/nn/functional.py#L5059
    """
    def scale_dot(self, q, k, v, attn_mask=None, dropout_p=0):
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

class TripleCNNViT(nn.Module):
    def __init__(self, image_size=224, num_classes=1, dim=1024,\
                depth=6, heads=8, mlp_dim=2048,\
                dim_head=64, dropout=0.15,\
                backbone='xception_net', pretrained=True,\
                normalize_ifft='batchnorm',\
                flatten_type='patch',\
                freq_combine='add', act='none', prj_out=True, \
                patch_size=7, \
                version='ca-fcat-0.5', unfreeze_blocks=-1, \
                inner_ca_dim=0, \
                dropout_in_mlp=0.0, classifier='mlp', share_weight=False):  
        super(TripleCNNViT, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        
        self.backbone = backbone
        self.features_size = {
            'efficient_net': (1280, 4, 4),
            'xception_net': (2048, 4, 4),
        }
        self.out_ext_channels = self.features_size[backbone][0]
        
        self.flatten_type = flatten_type # in ['patch', 'channel']
        self.version = version  # in ['ca-rgb_cat-0.5', 'ca-freq_cat-0.5']
        self.activation = self.get_activation(act)
        self.share_weight = share_weight

        self.pretrained = pretrained
        self.rgb_extractor = self.get_feature_extractor(architecture=backbone, pretrained=True, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=3)   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)
        if not self.share_weight:
            self.mag_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)
            self.phase_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)  
        else:
            self.freq_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)
        self.normalize_ifft = normalize_ifft
        if self.normalize_ifft == 'batchnorm':
            self.batchnorm_ifft = nn.BatchNorm2d(num_features=self.out_ext_channels)
        if self.normalize_ifft == 'layernorm':
            self.layernorm_ifft = nn.LayerNorm(normalized_shape=self.features_size[self.backbone])
        self.freq_combine = freq_combine
        if self.freq_combine == 'add':
            self.balance_magphase = nn.Parameter(torch.ones(1))
        if self.freq_combine == 'cat':
            self.convfreq = nn.Conv2d(in_channels=2*self.out_ext_channels, out_channels=self.out_ext_channels, kernel_size=1)
        ############################# PATCH CONFIG ################################
        
        if self.flatten_type == 'patch':
            # Kích thước của 1 patch
            self.patch_size = patch_size
            # Số lượng patches
            self.num_patches = int((self.features_size[backbone][1] * self.features_size[backbone][2]) / (self.patch_size * self.patch_size))
            # Patch_dim = P^2 * C
            self.patch_dim = self.out_ext_channels * (self.patch_size ** 2)

        ############################# CROSS ATTENTION #############################
        if self.flatten_type == 'patch':
            self.in_dim = self.patch_dim
        else:
            self.conv = nn.Conv2d(self.out_ext_channels, self.out_ext_channels // 4, kernel_size=1)
            self.in_dim = int(self.features_size[backbone][1] * self.features_size[backbone][2])

        self.CA = CrossAttention(in_dim=self.in_dim, inner_dim=inner_ca_dim, prj_out=prj_out)

        ############################# VIT #########################################
        # Giảm chiều vector sau concat 2*patch_dim về D:
        if 'cat' in self.version:
            self.embedding = nn.Linear(2 * self.in_dim, self.dim)
        else:
            self.embedding = nn.Linear(self.in_dim, self.dim)

        # Thêm 1 embedding vector cho classify token:
        self.classifier = classifier
        self.num_vecs = self.num_patches if self.flatten_type == 'patch' else self.out_ext_channels
        if 'vit' in self.classifier:
            self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)
            self.batchnorm = nn.BatchNorm1d(self.num_vecs)
        if 'vit_aggregate' in self.classifier:
            gamma = float(self.classifier.split('_')[-1])
            if gamma == -1:
                self.gamma = nn.Parameter(torch.ones(1))
            else:
                self.gamma = gamma

        self.mlp_relu = nn.ReLU(inplace=True)
        self.mlp_head_hidden = nn.Linear(self.dim, self.mlp_dim)
        self.mlp_dropout = nn.Dropout(dropout_in_mlp)
        self.mlp_head_out = nn.Linear(self.mlp_dim, self.num_classes)
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
        # if not pretrained:
        #     self.init_conv_weight(extractor)
        return extractor

    def flatten_to_vectors(self, feature):
        vectors = None
        if self.flatten_type == 'patch':
            vectors = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        elif self.flatten_type == 'channel':
            feature = self.conv(feature)
            vectors = rearrange(feature, 'b c h w -> b c (h w)')
        else:
            pass
        return vectors

    def reconstruct(self, mag_feature, phase_feature, norm_type='none', freq_combine='add'):
        freq_feature = None
        if freq_combine == 'add':
            freq_feature = mag_feature + self.balance_magphase * phase_feature
        if freq_combine == 'cat':
            freq_feature = torch.cat([mag_feature, phase_feature], dim=1)
            freq_feature = self.convfreq(freq_feature)

        combined = torch.multiply(mag_feature, torch.exp(1j*phase_feature))
        fftx = torch.fft.ifftshift(combined)
        ffty = torch.fft.ifft2(fftx)
        reconstructed = torch.abs(ffty)
        if norm_type == 'none':
            ifreq_feature = reconstructed
        elif norm_type == 'batchnorm':
            ifreq_feature = self.batchnorm_ifft(reconstructed)
        elif norm_type == 'layernorm':
            ifreq_feature = self.layernorm_ifft(reconstructed)
        elif norm_type == 'normal':
            ifreq_feature = F.normalize(reconstructed)
        
        return freq_feature, ifreq_feature

    def fusion(self, rgb, out_attn):
        """
        Arguments:
            rgb --      b, n, d
            out_attn -- b, n, d
        """
        weight = float(self.version.split('-')[-1])
        if 'cat' in self.version:
            out = torch.cat([rgb, weight * out_attn], dim=2)
        elif 'add' in self.version:
            out = torch.add(rgb, weight * out_attn)
        return out

    def extract_feature(self, rgb_imgs, mag_imgs, phase_imgs):
        if self.backbone == 'efficient_net':
            rgb_features = self.rgb_extractor.extract_features(rgb_imgs)                 # shape (batchsize, 1280, 8, 8)
            if not self.share_weight:
                mag_features = self.mag_extractor.extract_features(mag_imgs)              # shape (batchsize, 1280, 4, 4)
                phase_features = self.phase_extractor.extract_features(phase_imgs)              # shape (batchsize, 1280, 4, 4)
            else:
                mag_features = self.freq_extractor.extract_features(mag_imgs)              # shape (batchsize, 1280, 4, 4)
                phase_features = self.freq_extractor.extract_features(phase_imgs)              # shape (batchsize, 1280, 4, 4)  
        else:
            rgb_features = self.rgb_extractor(rgb_imgs)
            if not self.share_weight:
                mag_features = self.mag_extractor(mag_imgs)
                phase_features = self.phase_extractor(phase_imgs)
            else:
                mag_features = self.freq_extractor(mag_imgs)
                phase_features = self.freq_extractor(phase_imgs)
        return rgb_features, mag_features, phase_features

    def forward(self, rgb_imgs, mag_imgs, phase_imgs):
        rgb_features, mag_features, phase_features = self.extract_feature(rgb_imgs, mag_imgs, phase_imgs)
        freq_features, ifreq_features = self.reconstruct(mag_features, phase_features, norm_type=self.normalize_ifft, freq_combine=self.freq_combine)
        # print("Features shape: ", rgb_features.shape, freq_features.shape, ifreq_features.shape)

        # Turn to q, k, v if use conv-attention, and then flatten to vector:
        # print("Q K V shape: ", rgb_query.shape, freq_value.shape, ifreq_key.shape, ifreq_value.shape)
        if self.freq_combine != 'none':
            rgb_vectors = self.flatten_to_vectors(rgb_features)
            freq_vectors = self.flatten_to_vectors(freq_features)
            ifreq_vectors = self.flatten_to_vectors(ifreq_features)
            ##### Cross attention and fusion:
            _, attn_weight = self.CA(rgb_vectors, ifreq_vectors)
            attn_freq = torch.bmm(attn_weight, freq_vectors)
            fusion_out = self.fusion(rgb_vectors, attn_freq)
            if self.activation is not None:
                fusion_out = self.activation(fusion_out)
            # print("Fusion shape: ", fusion_out.shape)
            embed = self.embedding(fusion_out)
        else:
            rgb_vectors = self.flatten_to_vectors(rgb_features)
            ifreq_vectors = self.flatten_to_vectors(ifreq_features)
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print("ifreq: ", ifreq_vectors)
            # print("rgb: ", rgb_vectors)
            # sys.stdout = sys.__stdout__
            ##### Cross attention and fusion:
            _, attn_weight = self.CA(rgb_vectors, ifreq_vectors)
            attn_ifreq = torch.bmm(attn_weight, ifreq_vectors)
            fusion_out = self.fusion(rgb_vectors, attn_ifreq)
            if self.activation is not None:
                fusion_out = self.activation(fusion_out)
            embed = self.embedding(fusion_out)
        # print("Inner ViT shape: ", embed.shape)

        ##### Forward to ViT
        if self.classifier == 'mlp':
            x = embed.mean(dim = 1).squeeze(dim=1)     # B, N, D => B, 1, D
            x = self.mlp_dropout(x)         
            x = self.mlp_head_hidden(x) # B, 1, D => 
            x = self.mlp_relu(x)
            x = self.mlp_dropout(x)
            x = self.mlp_head_out(x)

        if self.classifier == 'vit':
            x = self.transformer(embed)
            # sys.stdout = open('/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/check.txt', 'w')
            # print(x[0])
            # sys.stdout = sys.__stdout__
            x = x.mean(dim = 1).squeeze(dim=1)
            x = self.mlp_dropout(x)         
            x = self.mlp_head_hidden(x) # B, 1, D => 
            x = self.mlp_relu(x)
            x = self.mlp_dropout(x)
            x = self.mlp_head_out(x)

        if 'vit_aggregate' in self.classifier:
            x = self.transformer(embed)
            # x = self.batchnorm(x)    
            x = embed + self.gamma * x
            x = x.mean(dim = 1).squeeze(dim=1)
            x = self.mlp_dropout(x)         
            x = self.mlp_head_hidden(x) # B, 1, D => 
            x = self.mlp_relu(x)
            x = self.mlp_dropout(x)
            x = self.mlp_head_out(x)
        x = self.sigmoid(x)
        return x

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(32, 3, 128, 128)
    y = torch.ones(32, 1, 128, 128)
    model_ = TripleCNNViT(  image_size=128, num_classes=1, dim=1024,\
                                depth=6, heads=8, mlp_dim=2048,\
                                dim_head=64, dropout=0.15, \
                                backbone='efficient_net', pretrained=False,\
                                normalize_ifft='batchnorm',\
                                flatten_type='patch',\
                                inner_ca_dim=0, freq_combine='add',  act='none', prj_out=True, \
                                patch_size=2, \
                                version='ca-fcat-0.5', unfreeze_blocks=-1, classifier='vit_aggregate_0.3', share_weight=False)
    out = model_(x, y, y)
    print(out.shape)