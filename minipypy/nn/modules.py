import random
import minipypy as mini


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        scale = 0.01

        w_data = []
        for _ in range(in_features):
            row = []
            for _ in range(out_features):
                row.append((random.random() * 2.0 - 1.0) * scale)
            w_data.append(row)

        b_data = [[0.0 for _ in range(out_features)]]

        self.W = mini.Tensor(w_data, requires_grad=True)
        self.b = mini.Tensor(b_data, requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

    def step(self, lr):
        self.W = (self.W - self.W.grad() * lr).detach().requires_grad_(True)
        self.b = (self.b - self.b.grad() * lr).detach().requires_grad_(True)

class ReLU(Module):
    def forward(self, x):
        return x.relu()
