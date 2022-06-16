import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self,device, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, label):
        out1_magnitude = torch.norm(output1, dim=1, keepdim=True)
        out2_magnitude = torch.norm(output2, dim=1, keepdim=True)
        max_magnitude = torch.maximum(out1_magnitude, out2_magnitude)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = euclidean_distance / max_magnitude
        # print("Before norm: ", euclidean_distance)
        # euclidean_distance 
        # print("After norm: ", )
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.max(torch.tensor(0.0).to(self.device), torch.pow(torch.tensor(self.margin).to(self.device) - euclidean_distance, 2)))
        return loss_contrastive