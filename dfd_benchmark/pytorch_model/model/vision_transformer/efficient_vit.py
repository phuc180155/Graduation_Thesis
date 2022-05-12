

"""
https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/efficient-vit/efficient_vit.py

Coccomini, Davide, Nicola Messina, Claudio Gennaro, and Fabrizio Falchi.
"Combining efficientnet and vision transformers for video deepfake detection."
arXiv preprint arXiv:2107.02612 (2021).



"""

import torch
from torch import nn
from einops import rearrange
# from efficientnet_pytorch import EfficientNet
from ..backbone.efficient_net.model import EfficientNet
import cv2
import re
# from utils import resize
import numpy as np
from torch import einsum
from random import randint
from .vit import Transformer

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class EfficientViT(nn.Module):
    def __init__(self, channels=1280, selected_efficient_net=0,
                 image_size=224,patch_size=7,num_classes=1,dim=1024,
                 depth=6,heads=8,mlp_dim=2048,
                 emb_dim=32, dim_head=64,dropout=0.15,emb_dropout=0.15, freeze=0, pool='cls'):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.emb_dim = emb_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout
        self.pool = pool

        self.output_features_size = {
            128: (4, 4),
            224: (7, 7),
            256: (8, 8)
        }

        # assert self.image_size % self.patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.selected_efficient_net = selected_efficient_net

        # Nếu không sử dụng pretrain model, sử dụng features là efficient_net-b0
        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, in_channels=3, pretrained=False)
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes, in_channels=3, pretrained=False)
            checkpoint = torch.load("weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
                                    map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()},
                                               strict=False)
        if freeze:
            for i in range(0, len(self.efficient_net._blocks)):
                for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                    if i >= len(self.efficient_net._blocks) - 3:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # Kích thước của 1 patch
        self.patch_size = patch_size
        # Số lượng patches
        self.num_patches = int((self.output_features_size[image_size][0] * self.output_features_size[image_size][1]) / (self.patch_size * self.patch_size))
        # Patch_dim = P^2 * C
        patch_dim = channels * (self.patch_size ** 2)

        # print("Num patches: ", num_patches)
        # print("Patch dim: ", patch_dim)

        # Embed vị trí cho từng batch
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, self.dim))

        # Đưa flatten vector của feature maps về chiều cố định của vector trong transformer.
        self.patch_to_embedding = nn.Linear(patch_dim, self.dim)

        # Thêm 1 embedding vector cho classify token:
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.efficient_net.extract_features(img)  # 1280x7x7
        # print("Features shape: ", x.shape)

        # Tách feature maps (b, c, H, W) thành các patch:
        # Ví dụ: Với (H, W) là kích thước của ảnh đầu vào, C là số channels, P là kích thước mỗi gói sau khi chia. 
        # Ta biểu diễn lại ảnh đầu vào có kích thước x ∈ R^(b* H*W*C) thành x ∈ R^{b * N * (P^2 * C)}.
        # Số gói ta có thể chia từ ảnh đầu vào là N = HW / P^2 
        # y.shape = (B, N, T), T = P^2 * C - đại diện cho 1 patch trên 1 ảnh với đầy đủ channels.
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p) 

        # Do VIT vẫn sử dụng hằng số kích thước embedding size D trong mô hình do đó chúng ta cần một phép biển đổi tuyến tính biến các gói về kích thước D.
        # Phép biến đổi này còn gọi là patch embedding.
        # y.shape = (B, N, dim)
        y = self.patch_to_embedding(y)

        # Expand classify token to batchsize and add to patch embeddings:
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)

        # x:                shape(batchsize, num_patches+1, embed_dim)
        # pos_embedding:    shape(1, num_patches+1, embed_dim)
        x += self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x.mean(dim = 1) if self.pool == 'mean' else x[:, 0])
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = EfficientViT(image_size=256,patch_size=2 )
    import torchsummary
    torchsummary.summary(model,(3,256,256))

    x = torch.ones((32, 3, 256, 256))
    out = model(x)
    print(out.shape)