import torch
from torch.nn import functional as f


class LabelSmoothLoss(torch.nn.Module):
    def __init__(self, smooth_eps=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.conf = 1.0 - smooth_eps
        self.smooth_eps = smooth_eps

    def forward(self, x, target):
        log_softmax = f.log_softmax(x, dim=-1)
        loss = - log_softmax.gather(dim=-1, index=target.unsqueeze(1))
        loss = loss.squeeze(1)
        smooth_loss = - log_softmax.mean(dim=-1)
        loss = self.conf * loss + self.smooth_eps * smooth_loss
        return loss.mean()
