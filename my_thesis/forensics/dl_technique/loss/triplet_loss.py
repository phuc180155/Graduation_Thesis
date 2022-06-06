import torch
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor_magnitude = torch.norm(anchor, dim=1, keepdim=True)
        positive_magnitude = torch.norm(positive, dim=1, keepdim=True)
        negative_magnitude = torch.norm(negative, dim=1, keepdim=True)
        max1 = torch.maximum(anchor_magnitude, positive_magnitude)
        max2 = torch.maximum(anchor_magnitude, negative_magnitude)
        #
        pos_dist = F.pairwise_distance(anchor, positive, keepdim=True) / max1
        neg_dist = F.pairwise_distance(anchor, negative, keepdim=True) / max2
        # print(pos_dist, neg_dist)
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
        return triplet_loss.mean()