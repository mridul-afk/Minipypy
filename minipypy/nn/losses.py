from .modules import Module
from .functional import mse_loss


class MSELoss(Module):
    def forward(self, pred, target):
        return mse_loss(pred, target)
