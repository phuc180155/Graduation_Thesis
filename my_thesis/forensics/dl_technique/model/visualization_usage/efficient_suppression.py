import torch
import torch.nn as nn

from model.backbone.efficient_net.model import EfficientNet

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class EfficientSuppression(nn.Module):
    def __init__(self, pretrained=False, features_at_block='8'):
        super(EfficientSuppression, self).__init__()

        self.features_size = {
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
        }

        self.efficient = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2, in_channels = 3,pretrained=pretrained)
        # print(self.efficient)
        self.features_at_block = features_at_block
        self.final = True if features_at_block == 'final' else False
        if not self.final:
            self._conv_head = self.efficient.get_conv(in_channel=self.features_size[features_at_block][0], out_channel=1280)
            self._bn1 = self.efficient._bn1
            self._avg_pooling = self.efficient._avg_pooling
            self._dropout = self.efficient._dropout
            self._fc = self.efficient._fc
            self._swish = self.efficient._swish

            for i in range(int(self.features_at_block) + 1, 16):
                self.efficient._blocks[i] = Identity()

        # print(self.efficient)

    def forward(self, rgb):
        if not self.final:
            x = self.efficient.extract_features_at_block(rgb, selected_block=int(self.features_at_block))
            x = self._conv_head(x)
            x = self._bn1(x)
            x = self._avg_pooling(x)
            x = x.squeeze(dim=-1).squeeze(dim=-1)
            x = self._dropout(x)
            x = self._fc(x)
        else:
            x = self.efficient(rgb)
        return x


if __name__ == "__main__":
    loss = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    for t in [str(i) for i in range(5, 6)]:
        model = EfficientSuppression(pretrained=True, features_at_block=t)
        x = torch.rand(8, 3, 128, 128)
        label = torch.randint(low=0, high=2, size=(8,))
        out = model(x)
        values, preds = torch.max(out, dim=1)
        print(out)
        print(label.data)
        print(preds)
        accurate = torch.mean((label.data == preds), dtype=torch.float32).item()
        print(accurate)
        break
