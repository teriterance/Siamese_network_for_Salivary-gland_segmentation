#######################################
### Autor: teriterance(Gabin FODOP)####
#######################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, alpha = 1, beta = 1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta= beta

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = self.alpha * torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      self.beta * (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
if __name__ =="__main__":
    """Test contrastiveloss function"""
    loss = ContrastiveLoss()
    val1 = torch.tensor([[1., -1.], [1., -1.]])
    val2 = torch.tensor([[1., -1.], [1., -1.]])
    lab = torch.tensor(1)
    out = loss(val1, val2, lab)
    print("test output", out)