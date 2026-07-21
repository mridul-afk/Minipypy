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

class TensorFoldLinear(Module):
    """
    TensorFold low-rank Linear layer.

    Dense Linear:
        y = x @ W + b

    TensorFoldLinear:
        W ≈ U @ V
        y = (x @ U) @ V + b

    This layer never stores the full dense W matrix.
    It stores only U, V, and b.
    """

    def __init__(self, in_features, out_features, rank):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if rank <= 0:
            raise ValueError("rank must be greater than 0")

        max_useful_rank = (
            in_features * out_features
        ) / (
            in_features + out_features
        )

        if rank >= max_useful_rank:
            print(
                "Warning: TensorFoldLinear rank may not compress this layer. "
                f"rank={rank}, useful rank should be less than {max_useful_rank:.2f}"
            )

        scale = 0.05

        u_data = []
        for _ in range(in_features):
            row = []
            for _ in range(rank):
                row.append((random.random() * 2.0 - 1.0) * scale)
            u_data.append(row)

        v_data = []
        for _ in range(rank):
            row = []
            for _ in range(out_features):
                row.append((random.random() * 2.0 - 1.0) * scale)
            v_data.append(row)

        b_data = [[0.0 for _ in range(out_features)]]

        self.U = mini.Tensor(u_data, requires_grad=True)
        self.V = mini.Tensor(v_data, requires_grad=True)
        self.b = mini.Tensor(b_data, requires_grad=True)

    def forward(self, x):
        hidden = x @ self.U
        out = hidden @ self.V
        return out + self.b

    def parameters(self):
        return [self.U, self.V, self.b]

    def named_parameters(self):
        return [
            (self, "U", self.U),
            (self, "V", self.V),
            (self, "b", self.b),
        ]

    def parameter_count(self):
        return (
            self.in_features * self.rank
            + self.rank * self.out_features
            + self.out_features
        )

    def dense_parameter_count(self):
        return (
            self.in_features * self.out_features
            + self.out_features
        )

    def compression_ratio(self):
        return self.dense_parameter_count() / self.parameter_count()

class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Softmax(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)


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
