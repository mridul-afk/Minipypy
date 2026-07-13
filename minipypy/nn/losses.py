from .modules import Module
from .functional import mse_loss, hinge_loss, cross_entropy


class MSELoss(Module):
    def forward(self, pred, target):
        return mse_loss(pred, target)


class HingeLoss(Module):
    def forward(self, pred, target):
        return hinge_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        return cross_entropy(logits, target)
