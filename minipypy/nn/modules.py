import random
import minipypy as mini


class Module:
    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def step(self, lr):
        for owner, name, param in self.named_parameters():
            updated = (param - param.grad() * lr).detach().requires_grad_(True)
            setattr(owner, name, updated)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Module subclasses must implement forward()")

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

    def named_parameters(self):
        return [
            (self, "W", self.W),
            (self, "b", self.b),
        ]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def named_parameters(self):
        named_params = []

        for layer in self.layers:
            named_params.extend(layer.named_parameters())

        return named_params

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]
