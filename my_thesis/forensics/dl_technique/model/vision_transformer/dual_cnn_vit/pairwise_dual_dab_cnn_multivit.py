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
from pytorchcv.model_provider import get_model

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
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

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

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
        self.CA = CALayer(n_feat, reduction)  if self.use_ca else None      ## Channel Attention
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

        topk = int(self.topk_rate * ifreq.shape[1])
        attn = torch.topk(input=attn,k=topk,dim=1,largest=True,sorted=False).values

        if '-' in self.dab_modules:
            attn = self.conv1x1_1(attn)

        # print("        attn shape: ", rgb.shape, attn.shape)
        res = torch.cat([rgb, attn], dim=1)
        res = self.conv1x1_3(res)
        if self.act:
            res = self.act(res)
        return res

class CrossAttention(nn.Module):
    def __init__(self, version='ca-fcat-0.5', in_dim=1024, activation=None, inner_dim=0, prj_out=False, qkv_embed=True):
        super().__init__()
        self.version = version
        self.use_freq = True if self.version.split('-')[1][0] == 'f' else False
        self.in_dim = in_dim
        self.qkv_embed = qkv_embed
        self.to_out = nn.Identity()
        self.activation = activation
        if self.qkv_embed:
            inner_dim = self.in_dim if inner_dim == 0 else inner_dim
            self.to_k = nn.Linear(in_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(in_dim, inner_dim, bias = False)
            self.to_q = nn.Linear(in_dim, inner_dim, bias = False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, in_dim),
                nn.Dropout(p=0.1)
            ) if prj_out else nn.Identity()
        print("freq combine: ", self.use_freq)

    def forward(self, rgb, freq, ifreq):
        """
            x ~ rgb_vectors: (b, n, in_dim)
            y ~ freq_vectors: (b, n, in_dim)
            z ~ freq_vectors: (b, n, in_dim)
            Returns:
                attn_weight: (b, n, n)
                attn_output: (b, n, in_dim)
        """
        if self.qkv_embed:
            q = self.to_q(rgb)
            k = self.to_k(freq)
            v = self.to_v(ifreq)
        else:
            q, k, v = rgb, freq, ifreq
        attn_rgb_to_ifreq, attnweight_rgb_to_ifreq = self.scale_dot(q, k, v, dropout_p=0.00)
        if self.use_freq:
            attn_rgb_to_freq = torch.bmm(attnweight_rgb_to_ifreq, freq)
            attn_out = self.to_out(attn_rgb_to_freq)
        else:
            attn_out = self.to_out(attn_rgb_to_ifreq)
            
        fusion_out = self.fusion(rgb, attn_out)
        if self.activation is not None:
            fusion_out = self.activation(fusion_out)
        return fusion_out

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

class MultiscaleViT(nn.Module):
    def __init__(self, in_channels=112, in_size=8, patch_reso='1-2-4-8', gamma_reso='0.8_0.4_0.2_0.1', residual=True,\
                qkv_embed=True, prj_out=True, activation=None, fusca_version='ca-fcat-0.5', \
                useKNN=0, depth=6, heads=8, dim=1024, mlp_dim=2048, dim_head=64, dropout=0.15, share_weight=True):
        super(MultiscaleViT, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout_value = dropout

        self.fusca_version = fusca_version
        self.residual = residual
        self.patch_reso = patch_reso
        self.gamma_reso = gamma_reso

        self.patch_size = list(map(int, patch_reso.split('-')))
        self.gamma_reso = list(map(float, gamma_reso.split('_')))
        self.gamma = []
        self.g0 = nn.Parameter(torch.ones(1))
        self.g1 = nn.Parameter(torch.ones(1))
        self.g2 = nn.Parameter(torch.ones(1))
        self.g3 = nn.Parameter(torch.ones(1))
        cnt = 0
        if residual:
            for g in self.gamma_reso:
                if g != 0:
                    self.gamma.append(g)
                else:
                    if cnt == 0:
                        self.gamma.append(self.g0)
                    if cnt == 1:
                        self.gamma.append(self.g1)
                    if cnt == 2:
                        self.gamma.append(self.g2)
                    if cnt == 3:
                        self.gamma.append(self.g3)
                    cnt += 1

        self.num_patches = [int((in_size // p)** 2) for p in self.patch_size]
        self.patch_dim = [int(in_channels * (p ** 2)) for p in self.patch_size]

        ############################# CROSS ATTENTION #############################
        self.cross_attention = nn.ModuleList([])
        for p_dim in self.patch_dim:
            self.cross_attention.append(CrossAttention(version=fusca_version, in_dim=p_dim, activation=activation, inner_dim=0, prj_out=prj_out, qkv_embed=qkv_embed))

        ############################# VIT #########################################
        # Giảm chiều vector sau concat 2*patch_dim về D:
        self.embedding = nn.ModuleList([])
        for p_dim in self.patch_dim:
            if 'cat' in self.fusca_version:
                self.embedding.append(nn.Linear(2 * p_dim, self.dim))
            else:
                self.embedding.append(nn.Linear(p_dim, self.dim))
        # transformer:
        self.share_weight = share_weight
        if not share_weight:
            self.transformers = nn.ModuleList([])
            for _ in range(len(self.patch_size)):
                if useKNN == 0:
                    print("use vanilla attention")
                    self.transformers.append(Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value))
                elif useKNN > 0:
                    print("use KNN attention: topK ratio: ", useKNN)
                    self.transformers.append(kNNTransformer(self.dim, self.depth, self.heads, self.mlp_dim, self.dropout_value, useKNN))
                else:
                    print("error when use attention...")
        else:
            if useKNN == 0:
                print("use vanilla attention")
                self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)
            elif useKNN > 0:
                print("use KNN attention: topK ratio: ", useKNN)
                self.transformer = kNNTransformer(self.dim, self.depth, self.heads, self.mlp_dim, self.dropout_value, useKNN)
            else:
                print("error when use attention...")
        
    def forward(self, rgb_features, freq_features, ifreq_features):
        outputs = []
        for i in range(len(self.patch_size)):
            # Flatten to vectors:
            rgb_vectors = self.flatten_to_vectors(feature=rgb_features, p_size=self.patch_size[i])      # B, num_patch, patch_dim
            freq_vectors = self.flatten_to_vectors(feature=freq_features, p_size=self.patch_size[i])    # B, num_patch, patch_dim
            ifreq_vectors = self.flatten_to_vectors(feature=ifreq_features, p_size=self.patch_size[i])  # B, num_patch, patch_dim
            # print("patchsize: ", self.patch_size[i])
            # print("     Vectors shape: ", rgb_vectors.shape, freq_vectors.shape, ifreq_vectors.shape)

            # Cross attention:
            attn_out = self.cross_attention[i](rgb_vectors, freq_vectors, ifreq_vectors)    # B, num_patch, patch_dim/2*patch_dim
            # print("     attn out shape: ", attn_out.shape)

            # ViT:
            embed = self.embedding[i](attn_out)                # B, num_patch, dim
            if not self.share_weight:       
                output = self.transformers[i](embed)                # B, num_patch, dim
            else:
                output = self.transformer(embed)
            if self.residual:
                output = embed + self.gamma[i] * output        # B, num_patch, dim
            # print("     output shape: ", output.shape)
            output = output.mean(dim = 1).squeeze(dim=1)          # B, 1, dim
            outputs.append(output)
        
        out = torch.cat(outputs, dim=1)
        # print("multi shape: ", out.shape)
        return out

    def flatten_to_vectors(self, feature=None, p_size=1):
        return rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p_size, p2=p_size)

    
class PairwiseDualDabCNNMultiViT(nn.Module):
    def __init__(self, image_size=224, num_classes=1, \
                dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head=64, dropout=0.15,\
                backbone='xception_net', pretrained=True,unfreeze_blocks=-1,\
                normalize_ifft='batchnorm',\
                qkv_embed=True, prj_out=False, act='none',\
                patch_reso='1-2-4-8', gammaagg_reso='0.8_0.4_0.2_0.1',\
                fusca_version='ca-fcat-0.5',\
                features_at_block='10', \
                dropout_in_mlp=0.0, residual=True, transformer_shareweight=True, \
                act_dab='none', topk_channels=0.5, dab_modules='sa-ca', dabifft_normalize='none', dab_blocks='1_3_6_9', \
                embedding_return='mlp_hidden', useKNN=0):  

        super(PairwiseDualDabCNNMultiViT, self).__init__()

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
        self.fusca_version = fusca_version  # in ['ca-rgb_cat-0.5', 'ca-freq_cat-0.5']
        self.activation = self.get_activation(act)
        self.dab_activation = self.get_activation(act_dab)

        self.pretrained = pretrained
        self.features_at_block = features_at_block
        self.rgb_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=3)   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)
        self.freq_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)     
        self.normalize_ifft = normalize_ifft
        if self.normalize_ifft == 'batchnorm':
            self.batchnorm_ifft = nn.BatchNorm2d(num_features=self.out_ext_channels)
        if self.normalize_ifft == 'layernorm':
            self.layernorm_ifft = nn.LayerNorm(normalized_shape=self.features_size[backbone][features_at_block])
        
        # DAB Block:
        num_dab = len(self.dab_blocks) - 1
        self.dab = nn.ModuleList([])
        for i in range(num_dab):
            at_block = self.dab_blocks[i+1]
            in_features = self.features_size[backbone][str(at_block)][0]
            self.dab.append(DAB(n_feat=in_features, reduction=1, topk_rate=topk_channels, act_dab=self.dab_activation, dab_modules=dab_modules))

        # Multi ViT:
        self.multi_transformer = MultiscaleViT(in_channels=self.out_ext_channels, in_size=self.out_ext_size, patch_reso=patch_reso, gamma_reso=gammaagg_reso,\
                                          qkv_embed=qkv_embed, prj_out=prj_out, activation=self.activation, fusca_version=fusca_version,\
                                          useKNN=useKNN, depth=depth, heads=heads, dim=dim, mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout, residual=residual, share_weight=transformer_shareweight)

        self.mlp_relu = nn.ReLU(inplace=True)
        self.mlp_head_hidden = nn.Linear(len(patch_reso.split('-')) * dim, mlp_dim)
        self.mlp_dropout = nn.Dropout(dropout_in_mlp)
        self.mlp_head_out = nn.Linear(mlp_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        #
        self.embedding_return = embedding_return

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

    def forward_once(self, rgb_imgs, freq_imgs):
        rgb_features, freq_features = self.extract_feature(rgb_imgs, freq_imgs)
        ifreq_features = self.ifft(freq_features, norm_type=self.normalize_ifft)
        # print("Features shape: ", rgb_features.shape, freq_features.shape, ifreq_features.shape)

        ##### Forward to ViT
        e1 = self.multi_transformer(rgb_features, freq_features, ifreq_features)     # B, number_of_patch * D

        x = self.mlp_dropout(e1)         # B, number_of_patch * D
        e2 = self.mlp_head_hidden(x)     # B, number_of_patch * D => B, mlp_dim
        x = self.mlp_relu(e2)
        x = self.mlp_dropout(x)
        e3 = self.mlp_head_out(x)        # B, mlp_dim => B, 1
        x = self.sigmoid(e3)
        e = None
        if self.embedding_return=='mlp_before':
            e = e1
        if self.embedding_return=='mlp_hidden':
            e = e2
        if self.embedding_return=='mlp_out':
            e = e3
        return e, x

    def forward(self, rgb_imgs0, freq_imgs0, rgb_imgs1, freq_imgs1):
        embedding_0, out_0 = self.forward_once(rgb_imgs0, freq_imgs0)
        embedding_1, out_1 = self.forward_once(rgb_imgs1, freq_imgs1)
        # print("embed: ", embedding_0.shape, "   out: ", out_0.shape)
        return embedding_0, out_0, embedding_1, out_1

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(2, 3, 128, 128)
    y = torch.ones(2, 1, 128, 128)
    model_ = PairwiseDualDabCNNMultiViT(image_size=128, num_classes=1, \
                dim=1024, depth=6, heads=8, mlp_dim=2048, dim_head=64, dropout=0.15,\
                backbone='efficient_net', pretrained=True,unfreeze_blocks=-1,\
                normalize_ifft='batchnorm',\
                qkv_embed=True, prj_out=False, act='selu',\
                patch_reso='1-2-4', gammaagg_reso='0.8_0.4_0.2',\
                fusca_version='ca-fcat-0.5',\
                features_at_block='11', \
                dropout_in_mlp=0.0, residual=True, transformer_shareweight=True, \
                act_dab='selu', topk_channels=0.5, dab_modules='ca', dabifft_normalize='normal', dab_blocks='0_3_5_6_10', \
                embedding_return="mlp_out")
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    x, y = x.to(device), y.to(device)
    model_ = model_.to(device)

    # extractor = model_.rgb_extractor
    # print(len(extractor._blocks))
    # for idx in range(17):
    #     print("after block ", idx, " shape: ", extractor.extract_features_at_block(x, selected_block=idx).shape)
    ### CHECK @@@
    # print("\nCheck @@@")
    # for idx in range(16):
    #     print("after block ", idx, " shape: ", extractor.extract_features_block_inrange(x, from_block=idx, to_block=idx+1).shape)
    embed, out, _, _ = model_(x, y, x, y)
    print(embed.shape)
    print(out.shape)

