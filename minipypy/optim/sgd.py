class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self.model.step(self.lr)
