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
from model.vision_transformer.vit.vit import Transformer
from pytorchcv.model_provider import get_model

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, output_dim=1024, dropout=0.3):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights_()

    def _init_weights_(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class DualCNNFeedForwardViT(nn.Module):
    def __init__(self, \
                image_size=224, num_classes=1, dim=1024,\
                depth=6, heads=8, mlp_dim=2048,\
                dim_head=64, dropout=0.15, emb_dropout=0.15,\
                backbone='xception_net', pretrained=True, unfreeze_blocks=-1,\
                conv_reduction_channels=False, ratio_reduction=1,\
                flatten_type='patch', patch_size=2,\
                input_freq_dim=88, hidden_freq_dim=256,\
                position_embed=False, pool='cls',\
                aggregation="cat-0.8",\
                init_weight=False, init_linear="xavier", init_layernorm="normal", init_conv="kaiming", \
                dropout_in_mlp=0.0):  
        super(DualCNNFeedForwardViT, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout
        
        self.backbone = backbone
        self.features_size = {
            'efficient_net': (1280, 4, 4),
            'xception_net': (2048, 4, 4),
        }
        self.out_ext_channels = self.features_size[backbone][0]
        
        self.flatten_type = flatten_type # in ['patch', 'channel']
        self.position_embed = position_embed
        self.pool = pool
        self.conv_reduction_channels = conv_reduction_channels
        self.aggregation = aggregation

        self.pretrained = pretrained
        self.rgb_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=3)   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)
        self.freq_extractor = FeedforwardNeuralNetModel(input_dim=input_freq_dim, hidden_dim=hidden_freq_dim, output_dim=dim, dropout=0.2)
        ############################# PATCH CONFIG ################################
        
        if self.flatten_type == 'patch':
            # Kích thước của 1 patch
            self.patch_size = patch_size
            # Số lượng patches
            self.num_patches = int((self.features_size[backbone][1] * self.features_size[backbone][2]) / (self.patch_size * self.patch_size))
            # Patch_dim = P^2 * C
            self.patch_dim = self.out_ext_channels//ratio_reduction * (self.patch_size ** 2)

        if self.conv_reduction_channels:
            self.reduction_conv = nn.Conv2d(in_channels=self.out_ext_channels, out_channels=self.out_ext_channels//ratio_reduction, kernel_size=1)

        ############################# CROSS ATTENTION #############################
        if self.flatten_type == 'patch':
            self.in_dim = self.patch_dim
        else:
            self.in_dim = int(self.features_size[backbone][1] * self.features_size[backbone][2])

        ############################# VIT #########################################
        # Number of vectors:
        self.num_vecs = self.num_patches if self.flatten_type == 'patch' else self.out_ext_channels // ratio_reduction
        # Embed vị trí cho từng vectors (nếu chia theo patch):
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_vecs+1, self.dim))
        # Giảm chiều vector sau concat 2*patch_dim về D:
        self.embedding = nn.Linear(self.in_dim, self.dim)

        # Thêm 1 embedding vector cho classify token:
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)
        self.to_cls_token = nn.Identity()

        if 'cat' in self.aggregation:
            self.mlp_head = nn.Sequential(
                nn.Dropout(dropout_in_mlp),
                nn.Linear(2 * self.dim, self.mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout_in_mlp),
                nn.Linear(self.mlp_dim, self.num_classes)
            )
        elif 'add' in self.aggregation:
            self.mlp_head = nn.Sequential(
                nn.Dropout(dropout_in_mlp),
                nn.Linear(self.dim, self.mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout_in_mlp),
                nn.Linear(self.mlp_dim, self.num_classes)
            )   

        self.sigmoid = nn.Sigmoid()
        self.init_linear, self.init_layernorm, self.init_conv = init_linear, init_layernorm, init_conv
        if init_weight:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # print("Linear: ", module)
            if self.init_linear == 'normal':
                module.weight.data.normal_(mean=0.0, std=1.0)
            elif self.init_linear == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            else:
                pass
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # print("Layer norm: ", module)
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d) and self.pretrained == 0:
            # print("Conv: ", module)
            if self.init_conv == 'kaiming':
                nn.init.kaiming_normal_(module.weight, a=1)
            elif self.init_conv == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                pass

            if not module.bias is None:
                nn.init.constant_(module.bias, 0)

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
            vectors = rearrange(feature, 'b c h w -> b c (h w)')
        else:
            pass
        return vectors

    def fusion(self, out_1, out_2):
        """
        Arguments:
            rgb --      b, n, d
            out_attn -- b, n, d
        """
        # print("out_1: ", out_1.shape)
        # print("out_2: ", out_2.shape)
        weight = float(self.aggregation.split('-')[-1])
        # print("weight: ", weight)
        if 'cat' in self.aggregation:
            out = torch.cat([out_1, weight * out_2], dim=1)
        elif 'add' in self.aggregation:
            out = torch.add(out_1, weight * out_2)
        return out

    def extract_feature(self, rgb_imgs):
        if self.backbone == 'efficient_net':
            rgb_features = self.rgb_extractor.extract_features(rgb_imgs)                 # shape (batchsize, 1280, 8, 8)
        else:
            rgb_features = self.rgb_extractor(rgb_imgs)
        return rgb_features


    def forward(self, rgb_imgs, freq_features):
        ################## Stream for RGB image:
        # Extract features
        rgb_features = self.extract_feature(rgb_imgs)
        if self.conv_reduction_channels:
            rgb_features = self.reduction_conv(rgb_features)
        # Turn to feature vector
        rgb_feature_vectors = self.flatten_to_vectors(rgb_features)
        embedding_vectors = self.embedding(rgb_feature_vectors)
        # Forward to ViT
        # Expand classify token to batchsize and add to patch embeddings:
        cls_tokens = self.cls_token.expand(embedding_vectors.shape[0], -1, -1)
        x = torch.cat((cls_tokens, embedding_vectors), dim=1)   # (batchsize, in_dim+1, dim)
        if self.position_embed:
            x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x.mean(dim = 1) if self.pool == 'mean' else x[:, 0])

        ################### Stream for frequency features:
        y = self.freq_extractor(freq_features)
        ## Fusion and forward to MLP:
        z = self.fusion(x, y)
        z = self.mlp_head(z)
        out = self.sigmoid(z)
        return out

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(32, 3, 128, 128)
    y = torch.ones(32, 1, 128, 128)
    model_ = DualCNNFeedForwardViT(  image_size=128, num_classes=1, dim=1024,\
                                depth=6, heads=8, mlp_dim=2048,\
                                dim_head=64, dropout=0.15, emb_dropout=0.15,\
                                backbone='xception_net', pretrained=False, unfreeze_blocks=-1,\
                                conv_reduction_channels=False, ratio_reduction=1,\
                                flatten_type='patch', patch_size=2,\
                                input_freq_dim=88, hidden_freq_dim=256,\
                                position_embed=False, pool='cls',\
                                aggregation="cat-0.8",\
                                init_weight=False, init_linear="xavier", init_layernorm="norm", init_conv="kaiming")
    out = model_(x, y)
    print(out.shape)