class Adam:
    def __init__(
        self,
        model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        self.model = model
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.t = 0

        # Optimizer state.
        #
        # Since MiniPyPy currently updates parameters by replacing Tensor
        # objects, we store state by parameter name, not by Python object id.
        self.m = {}
        self.v = {}

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self.t += 1

        for owner, name, param in self.model.named_parameters():
            grad = param.grad()

            key = f"{id(owner)}.{name}"

            if key not in self.m:
                # Initialize first and second moment estimates with zeros.
                self.m[key] = (grad * 0.0).detach()
                self.v[key] = (grad * 0.0).detach()

            m = self.m[key]
            v = self.v[key]

            # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            m = (m * self.beta1 + grad * (1.0 - self.beta1)).detach()

            # v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            v = (v * self.beta2 + (grad * grad) * (1.0 - self.beta2)).detach()

            self.m[key] = m
            self.v[key] = v

            # Bias correction.
            m_hat = m / (1.0 - (self.beta1 ** self.t))
            v_hat = v / (1.0 - (self.beta2 ** self.t))

            update = m_hat / (v_hat.sqrt() + self.eps)

            updated = (param - update * self.lr).detach().requires_grad_(True)

            setattr(owner, name, updated)
