import torch
import torch.nn as nn

from model.backbone.efficient_net.model import EfficientNet

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class DualEfficient(nn.Module):
    def __init__(self, pretrained=False):
        super(DualEfficient, self).__init__()
        self.efficient3 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1,in_channels = 3,pretrained=pretrained)
        self.efficient3._dropout = Identity()
        self.efficient3._fc = Identity()
        self.efficient1 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1,in_channels = 1,pretrained=pretrained)
        self.efficient1._dropout = Identity()
        self.efficient1._fc = Identity()
        self.fc = nn.Linear(1280+1280,1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, input,input_fft):
        x3 = self.efficient3(input)
        # x3 = self.flatten(x3)
        x1 = self.efficient1(input_fft)
        # x1 = self.flatten(x1)
        # print(x3)
        # print(x1)
        x = torch.cat([x3,x1],1)
        # x = torch.cat([x3,x1],1)
        # x = x3+x1
        # print(x.size())
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1,in_channels = 1)
    # model = EfficientDual()
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1,in_channels = 3)
    model = DualEfficient()
    import torchsummary
    # torchsummary.summary(model,(1,128,128))
    # model2 = nn.Sequential(*(list(model.children())[:-3]))
    # model2 = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1, bias=False),
    #                        model)

    # torchsummary.summary(model2,(3,128,128))
    # model._dropout = Identity()
    # model._fc = Identity()
    # print(model)
    torchsummary.summary(model, [(3, 128, 128),(1, 128, 128)])
    # torchsummary.summary(model, (3, 128, 128))
