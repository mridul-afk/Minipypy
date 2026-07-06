def mse_loss(pred, target):
    error = pred - target
    return (error * error).mean()

def relu(x):
    return x.relu()

def hinge_loss(pred, target):
    """
    Hinge loss for binary classification.

    Expected target values:
        -1 or +1

    Formula:
        mean(relu(1 - target * pred))
    """
    margin = 1.0 - target * pred
    return relu(margin).mean()

def softmax(x, dim = -1):
    return x.softmax(dim)
