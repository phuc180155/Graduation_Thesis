import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weights = weights      # [class_0, class_1] và class_0 + class_1 = 1
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, target):    # pred: (batch_size, ), target: (batch_size, )
        pred = pred.clamp(self.eps, 1 - self.eps)
        if self.weights is not None:
            loss = self.weights[1] * ((1 - pred) ** self.gamma) * target * torch.log(pred) + self.weights[0] * (pred ** self.gamma) * (1 - target) * torch.log(1 - pred)
        else:
            loss = ((1 - pred) * self.gamma) * target * torch.log(pred) + (pred ** self.gamma) * (1 - target) * torch.log(1 - pred)
        return torch.mean(torch.neg(loss))