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
from model.backbone.efficient_net.utils import Conv2dStaticSamePadding

class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True, gamma_cma=-1):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        if gamma_cma == -1:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = gamma_cma

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y, z):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(y).view(
            B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        if self.cross_value:
            proj_value = self.value_conv(z).view(
                B, -1, H*W)  # B , C , HW
        else:
            proj_value = self.value_conv(z).view(
                B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        if self.activation is not None:
            out = self.activation(out)
        # print("out: ", out.shape)
        return out  # , attention

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

    def forward(self, x):
        b, c, h, w = x.size()   # 32, 1280, 8, 8
        # print("x size:", x.size())
        d_k = c // len(self.patchsize)  # 320
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        attentions = []
        # print("_query: ", _query.shape)
        for (width, height), query, key, value in zip(
            self.patchsize,
            torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1),
        ):
            # print('query: ', query.shape)   # (B, )
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = (
                query.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = (
                key.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = (
                value.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )

            y, _ = self.attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            attentions.append(y)
            output.append(y)

        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention

class MultiHeadedAttentionv2(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

    def forward(self, x, y):
        b, c, h, w = x.size()   # 32, 1280, 8, 8
        # print("x size:", x.size())
        d_k = c // len(self.patchsize)  # 320
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(y)
        _value = self.value_embedding(y)
        attentions = []
        # print("_query: ", _query.shape)
        for (width, height), query, key, value in zip(
            self.patchsize,
            torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1),
        ):
            # print('query: ', query.shape)   # (B, )
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = (
                query.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = (
                key.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = (
                value.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )

            out, attn = self.attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            out = out.view(b, out_h, out_w, d_k, height, width)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            attentions.append(attn)
            output.append(out)

        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)
        return self_attention

class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class PatchTrans(nn.Module):
    def __init__(self, in_channel, in_size, patch_self_resolution="1-2-4-8", gamma_self_patchtrans=-1):
        super().__init__()
        self.in_size = in_size

        patchsize = []
        reso = map(float, patch_self_resolution.split("-"))
        for r in reso:
            patchsize.append((int(in_size//r), int(in_size//r)))
        # print(patchsize)
        self.transform_ = TransformerBlock(patchsize, in_channel=in_channel, gamma_self_patchtrans=gamma_self_patchtrans)
        # print(in_channel)

    def forward(self, enc_feat):
        output = self.transform_(enc_feat)
        return output

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, in_channel=256, gamma_self_patchtrans=-1):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )
        if gamma_self_patchtrans == -1:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = gamma_self_patchtrans


    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self.gamma * self_attention
        output = output + self.feed_forward(output)
        return output

class TransformerBlockv2(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, in_channel=256, gamma_patchtrans=-1):
        super().__init__()
        self.attention = MultiHeadedAttentionv2(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )
        if gamma_patchtrans == -1:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = gamma_patchtrans

    def forward(self, rgb, freq):
        self_attention = self.attention(rgb, freq)
        output = rgb + self.gamma * self_attention
        output = output + self.feed_forward(output)
        return output

class PatchTransv2(nn.Module):
    def __init__(self, in_channel, in_size, patch_crossattn_resolution="1-2-4-8", gamma_patchtrans=-1):
        super().__init__()
        self.in_size = in_size

        patchsize = []
        reso = map(float, patch_crossattn_resolution.split("-"))
        for r in reso:
            patchsize.append((int(in_size//r), int(in_size//r)))
        # print(patchsize)
        self.transform_ = TransformerBlockv2(patchsize, in_channel=in_channel, gamma_patchtrans=gamma_patchtrans)
        # print(in_channel)

    def forward(self, rgb_fea, freq_fea):
        output = self.transform_(rgb_fea, freq_fea)
        return output

class PairwiseDualPatchCNNCMAViT(nn.Module):
    def __init__(self, image_size=224, num_classes=1, depth_block4=2, \
                backbone='xception_net', pretrained=True, unfreeze_blocks=-1, \
                normalize_ifft='batchnorm',\
                act='none',\
                init_type="xavier_uniform", \
                gamma_cma=-1, gamma_crossattn_patchtrans=-1, patch_crossattn_resolution='1-2', \
                gamma_self_patchtrans=-1, patch_self_resolution='1-2', \
                flatten_type='patch', patch_size=2, \
                dim=1024, depth_vit=2, heads=3, dim_head=64, dropout=0.15, emb_dropout=0.15, mlp_dim=2048, dropout_in_mlp=0.0, \
                classifier='mlp', in_vit_channels=64, embedding_return='mlp_hidden'):  
        super(PairwiseDualPatchCNNCMAViT, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.depth_block4 = depth_block4

        self.depth_vit = depth_vit
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout
        self.flatten_type = flatten_type
        self.patch_size = patch_size

        self.backbone = backbone
        self.features_size = {
            'efficient_net': (1280, 8, 8),
            'xception_net': (2048, 8, 8),
        }
        self.out_ext_channels = self.features_size[backbone][0]
        self.before_vit_channels = self.out_ext_channels if classifier == 'mlp' else 320
        self.activation = self.get_activation(act)

        self.pretrained = pretrained
        self.rgb_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=3)   # efficient_net-b0, return shape (1280, 8, 8) or (1280, 7, 7)
        self.freq_extractor = self.get_feature_extractor(architecture=backbone, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks, num_classes=num_classes, in_channels=1)     
        self.normalize_ifft = normalize_ifft
        if self.normalize_ifft == 'batchnorm':
            self.batchnorm_ifft = nn.BatchNorm2d(num_features=self.before_vit_channels)
        if self.normalize_ifft == 'layernorm':
            self.layernorm_ifft = nn.LayerNorm(normalized_shape=self.features_size[self.backbone])
        ############################# PATCH CONFIG ################################

        # self.CA = CrossAttention(in_dim=self.in_dim, inner_dim=inner_ca_dim, prj_out=prj_out, qkv_embed=qkv_embed, init_weight=init_ca_weight)
        self.cma = CrossModalAttention(in_dim=self.before_vit_channels, activation=self.activation, ratio=4, cross_value=True, gamma_cma=gamma_cma)

        # Thêm 1 embedding vector cho classify token:
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer_block_4 = nn.ModuleList([])
        for _ in range(depth_block4):
            self.transformer_block_4.append(PatchTrans(in_channel=40, in_size=16, patch_self_resolution=patch_self_resolution, gamma_self_patchtrans=gamma_self_patchtrans))
        self.transformer_block_10_rgb = PatchTransv2(in_channel=112, in_size=8, patch_crossattn_resolution=patch_crossattn_resolution, gamma_patchtrans=gamma_crossattn_patchtrans)
        self.transformer_block_10_freq = PatchTransv2(in_channel=112, in_size=8, patch_crossattn_resolution=patch_crossattn_resolution, gamma_patchtrans=gamma_crossattn_patchtrans)

        # Classifier:
        self.classifier = classifier
        self.embedding_return = embedding_return
        if 'mlp' in self.classifier:
            self.mlp_relu = nn.ReLU(inplace=True)
            self.mlp_head_hidden = nn.Linear(1280, self.mlp_dim)
            self.mlp_dropout = nn.Dropout(dropout_in_mlp)
            self.mlp_head_out = nn.Linear(self.mlp_dim, self.num_classes)

        if 'vit' in self.classifier:
            self.convr = nn.Conv2d(in_channels=320, out_channels=in_vit_channels, kernel_size=1)
            self.in_dim = self.patch_size*self.patch_size *in_vit_channels if flatten_type=='patch' else 16
            if self.flatten_type == 'channel':
                self.dim = 32
                self.mlp_dim = 64
            self.embedding = nn.Linear(self.in_dim, self.dim)
            self.transformer = Transformer(self.dim, self.depth_vit, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)
            self.mlp_relu = nn.ReLU(inplace=True)
            self.mlp_head_hidden = nn.Linear(self.dim, self.mlp_dim)
            self.mlp_dropout = nn.Dropout(dropout_in_mlp)
            self.mlp_head_out = nn.Linear(self.mlp_dim, self.num_classes)

        if 'vit_aggregate' in self.classifier:
            gamma = float(self.classifier.split('_')[-1])
            if gamma == -1:
                self.gamma = nn.Parameter(torch.ones(1))
            else:
                self.gamma = gamma

        self.sigmoid = nn.Sigmoid()
        # self.init_weights(init_type=init_type)

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
            # extractor._blocks[11]._depthwise_conv = Conv2dStaticSamePadding(in_channels=672, out_channels=672, kernel_size=(5, 5), stride=(1, 1), groups=672, image_size=224)
            # extractor._conv_head = nn.Identity()
            if unfreeze_blocks != -1:
                # Freeze the first (num_blocks - 3) blocks and unfreeze the rest 
                for i in range(0, len(extractor._blocks)):
                    for index, param in enumerate(extractor._blocks[i].parameters()):
                        if i >= len(extractor._blocks) - unfreeze_blocks:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            # print(extractor)
        
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

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1
            ):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type
                    )
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

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
        return ifreq_feature

    def flatten_to_vectors(self, feature):
        vectors = None
        if self.flatten_type == 'patch':
            vectors = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        elif self.flatten_type == 'channel':
            vectors = rearrange(feature, 'b c h w -> b c (h w)')
        else:
            pass
        return vectors

    def extract_feature(self, rgb_imgs, freq_imgs):
        if self.backbone == 'efficient_net':
            #
            rgb_features = self.rgb_extractor.extract_features_block_4(rgb_imgs)                 # shape (batchsize, 1280, 8, 8)
            for attn in self.transformer_block_4:
                rgb_features = attn(rgb_features)
            freq_features = self.freq_extractor.extract_features_block_4(freq_imgs)              # shape (batchsize, 1280, 4, 4)
            #
            rgb_features = self.rgb_extractor.extract_features_block_10(rgb_features)
            freq_features = self.freq_extractor.extract_features_block_10(freq_features)
            rgb_features_1 = self.transformer_block_10_rgb(rgb_features, freq_features)
            freq_features_1 = self.transformer_block_10_freq(freq_features, rgb_features)
            rgb_features = self.rgb_extractor.extract_features_last_block(rgb_features_1, classifier=self.classifier)
            freq_features = self.freq_extractor.extract_features_last_block(freq_features_1, classifier=self.classifier)
        else:
            rgb_features = self.rgb_extractor(rgb_imgs)
            freq_features = self.freq_extractor(freq_imgs)
        return rgb_features, freq_features

    def forward_once(self, rgb_imgs, freq_imgs):
        rgb_features, freq_features = self.extract_feature(rgb_imgs, freq_imgs)
        ifreq_features = self.ifft(freq_features, norm_type=self.normalize_ifft)
        # print("Features shape: ", rgb_features.shape, freq_features.shape, ifreq_features.shape)
        out = self.cma(rgb_features, ifreq_features, freq_features)     # B, 1280, 4, 4

        if self.classifier == 'mlp':
            x = F.adaptive_avg_pool2d(out, (1, 1))
            e1 = x.squeeze().squeeze()
            x = self.mlp_dropout(e1)         
            e2 = self.mlp_head_hidden(x)
            x = self.mlp_relu(e2)
            x = self.mlp_dropout(x)
            e3 = self.mlp_head_out(x)

        if self.classifier == 'vit':
            x = self.convr(out)
            x = self.flatten_to_vectors(x)
            x = self.embedding(x)
            x = self.transformer(x)

            e1 = x.mean(dim = 1).squeeze(dim=1)
            x = self.mlp_dropout(e1)         
            e2 = self.mlp_head_hidden(x)
            x = self.mlp_relu(e2)
            x = self.mlp_dropout(x)
            e3 = self.mlp_head_out(x)

        if 'vit_aggregate' in self.classifier:
            x = self.convr(out)
            x = self.flatten_to_vectors(x)
            embed = self.embedding(x)
            x = self.transformer(embed)
            x = embed + self.gamma * x
            e1 = x.mean(dim = 1).squeeze(dim=1)
            x = self.mlp_dropout(e1)         
            e2 = self.mlp_head_hidden(x) # B, 1, D => 
            x = self.mlp_relu(e2)
            x = self.mlp_dropout(x)
            e3 = self.mlp_head_out(x) 

        if self.embedding_return=='mlp_before':
            e = e1
        if self.embedding_return=='mlp_hidden':
            e = e2
        if self.embedding_return=='mlp_out':
            e = e3
        out = self.sigmoid(e3)
        return e, out

    def forward(self, rgb_imgs0, freq_imgs0, rgb_imgs1, freq_imgs1):
        embedding0, out0 = self.forward_once(rgb_imgs0, freq_imgs0)
        embedding1, out1 = self.forward_once(rgb_imgs1, freq_imgs1)
        return embedding0, out0, embedding1, out1

from torchsummary import summary
if __name__ == '__main__':
    x = torch.ones(32, 3, 128, 128)
    y = torch.ones(32, 1, 128, 128)
    model_ = PairwiseDualPatchCNNCMAViT(image_size=128, num_classes=1, depth_block4=2,\
                backbone='efficient_net', pretrained=True, unfreeze_blocks=-1,\
                normalize_ifft='batchnorm',\
                act='selu',\
                init_type="xavier_uniform",\
                gamma_cma=-1, gamma_crossattn_patchtrans=-1, patch_crossattn_resolution='1-2',\
                gamma_self_patchtrans=-1, patch_self_resolution='1-2', \
                flatten_type='channel', patch_size=2, \
                dim=1024, depth_vit=2, heads=3, dim_head=64, dropout=0.15, emb_dropout=0.15, mlp_dim=2048, dropout_in_mlp=0.0, \
                classifier='mlp', in_vit_channels=64, embedding_return='mlp_out')
    out1, out2, _, _ = model_(x, y,x, y)
    print(out1.shape)
    print(out2.shape)