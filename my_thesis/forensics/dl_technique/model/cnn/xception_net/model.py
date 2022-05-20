from pytorchcv.model_provider import get_model
from torch import dropout
""" 
    @info: List of available model:
        vgg, resnet, pyramidnet, diracnet, densenet, condensenet, wrn, drn, dpn, darknet, fishnet, squeezenet, squeezenext, shufflenet, menet, mobilenet, igcv3, mnasnet, darts, xception, inception, polynet, nasnet, pnasnet    
    @detail, see in https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
"""

import torch
import torch.nn as nn
from torchsummary import summary
import sys, os
import os.path as osp
sys.path.append(osp.dirname(__file__))

class ClassifierBlock(nn.Module):
    def __init__(self, in_features, out_features, drop_out=0.5):
        super(ClassifierBlock, self).__init__()
        # Flatten
        self.flatten = nn.Flatten()
        self.batchnorm_1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(drop_out)

        # FC1
        self.linear_1 = nn.Linear(in_features, 512)
        self.activate = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm1d(512)

        # FC2
        self.linear_2 = nn.Linear(512, out_features)
        print(out_features)

    def forward(self, x):
        x = self.flatten(x)
        x = self.batchnorm_1(x)
        x = self.dropout(x)

        x = self.linear_1(x)
        x = self.activate(x)
        x = self.batchnorm_2(x)
        x = self.dropout(x)

        out = self.linear_2(x)
        # 1/(1+e^-out)
        out = nn.Sigmoid()(out) # Tensor shape: (N, 1)
        return out

class Xception(nn.Module):
    def __init__(self, base_net, in_features):
        super(Xception, self).__init__()
        self.base_net = base_net
        self.classifier = ClassifierBlock(in_features=in_features, out_features=1)
    
    def forward(self, x):
        x = self.base_net(x)
        x = self.classifier(x)
        return x

def xception(pretrained=True):
    """
        Original Xception Structure:
        (features):
            (init_block): <--- in_channels = 3, out_channels = 64
                (conv_1):   ConvBlock(  Conv2D(3, 32, (3, 3), stride=(2, 2)) => BatchNorm(32) => Relu   )
                (conv_2):   ConvBlock(  Conv2D(32, 64, (3, 3), stride=(1, 1)) => BatchNorm(64) => Relu  )
            
            (stage_1):  <--- in_channels = 64, out_channels = 128
                (unit_1):
                    (identity_conv): ConvBlock(   Conv2D(64, 128, (1, 1), stride=(2, 2)) => BatchNorm(128)  )
                    (body):
                        (block_1) <DwsConvBlock>:
                            (DwsConv):
                                (dw_conv): Conv2D(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
                                (pw_conv): Conv2D(64, 128, (1, 1), (1, 1))
                            (bn): BatchNorm(128)
                        (block_2):
                            (Relu) => DwsConv() => (bn)
                        (pool): MaxPool(ksize=3, stride=2)

            (stage_2): <--- in_channels = 128, out_channels = 256
                (unit_1)

            (stage_3):
                (unit_1) (in = 128, out = 728)
                    (identity_conv)
                    (body)
                (unit_2) (in = 728, out = 728):
                    (block_1) <DwsConvBlock>
                    (block_2) <Relu + DwsConvBlock>
                    (block_3) <Relu + DwsConvBlock>
                (unit_3) (in = 728, out = 728) (same as unit_2)
                (unit_4) (in = 728, out = 728) (same as unit_2)
                (unit_5) (in = 728, out = 728) (same as unit_2)       
                (unit_6) (in = 728, out = 728) (same as unit_2)       
                (unit_7) (in = 728, out = 728) (same as unit_2)  
                (unit_8) (in = 728, out = 728) (same as unit_2)       
                (unit_9) (in = 728, out = 728) (same as unit_2)   

            (stage_4): <-- in = 728, out = 1024
                (unit_1)  (same as (body) in stage_1)

            (final_block):  <-- in = 1024, out = 2048
                (conv_1) <DwsConvBlock>
                (conv_2) <Relu + DwsConvBlock>
                (activ): Relu
                # AvgPool2d does the average operation per channel to obtain the single scalar value in each channel
                (pool): AvgPool2d(kernel_size=10, stride=1, padding=0)

        (output) <classifier block>: Linear(in_features=2048, out_features=1000, bias=True)
    """
    model = get_model("xception", pretrained=pretrained)
    # Remove original output layer: Linear(in_features=2048, out_features=1000, bias=True)
    # model.children() => (features module) and (output module)
    model = nn.Sequential(*list(model.children())[:-1])
    # Replace AvgPool2D by AdaptiveAvgPool
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))   # output_shape: (2048, 1, 1)
    model = Xception(model, 2048)
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xception(pretrained=False).to(device)
    summary(model, input_size=(3, 128, 128))


