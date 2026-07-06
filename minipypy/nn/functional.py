def mse_loss(pred, target):
    error = pred - target
    return (error * error).mean()

def relu(x):
    return x.relu()
