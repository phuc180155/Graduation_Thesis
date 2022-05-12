import torch.nn as nn
import torch

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, weights=None, eps=1e-7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.weights = weights      # [class_0, class_1]
        self.eps = eps
        
    def forward(self, pred, target):    # target: (batch_size, ), pred: (batch_size, )
        pred = pred.clamp(self.eps, 1-self.eps)
        if self.weights is not None:
            loss = self.weights[1] * (target * torch.log(pred)) + self.weights[0] * ((1- target) * torch.log(1-pred))
        else:
            loss = target * torch.log(pred) + (1 - target) * torch.log(1-pred)
        return torch.mean(torch.neg(loss))