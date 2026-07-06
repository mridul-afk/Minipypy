from .modules import Module
from .functional import mse_loss, hinge_loss


class MSELoss(Module):
    def forward(self, pred, target):
        return mse_loss(pred, target)


class HingeLoss(Module):
    def forward(self, pred, target):
        return hinge_loss(pred, target)
