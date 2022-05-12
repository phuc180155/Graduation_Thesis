import torch.nn as nn

class Meso4(nn.Module):
    
    """
	Pytorch Implemention of Meso4
	Autor: Honggu Liu
	Date: July 4, 2019
	"""
    def __init__(self, num_classes=1,image_size=256):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        #flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*int(image_size/32)*int(image_size/32), 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input) #(8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)
        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x
    
def mesonet(image_size=256):
    model = Meso4(image_size=image_size)
    return model

if __name__ == '__main__':
    image_size = 128
    model = mesonet(image_size=image_size)
    import torchsummary
    torchsummary.summary(model, (3, image_size, image_size))