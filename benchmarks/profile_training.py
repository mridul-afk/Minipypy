import minipypy as mini
import time


# Large enough to generate meaningful CUDA activity,
# but reasonable for your RTX 3050 4 GB.
BATCH = 1024
IN_FEATURES = 512
OUT_FEATURES = 256

WARMUP = 10
STEPS = 100
LR = 0.001


def make_data():
    # Deterministic data; profiling performance, not model quality.
    x_data = [
        [(i + j) % 10 / 10.0 for j in range(IN_FEATURES)]
        for i in range(BATCH)
    ]

    y_data = [
        [(i + j) % 5 / 5.0 for j in range(OUT_FEATURES)]
        for i in range(BATCH)
    ]

    x = mini.Tensor(x_data)
    y = mini.Tensor(y_data)

    w_data = [
        [0.01 for _ in range(OUT_FEATURES)]
        for _ in range(IN_FEATURES)
    ]

    W = mini.Tensor(w_data, requires_grad=True)

    return x, y, W


def training_step(x, y, W):
    pred = x @ W
    error = pred - y
    loss = (error * error).mean()

    loss.backward()

    W = (W - W.grad() * LR).detach().requires_grad_(True)

    return W


def main():
    print("Creating tensors...")
    x, y, W = make_data()

    print("Warmup...")

    for _ in range(WARMUP):
        W = training_step(x, y, W)

    print("Profiling workload...")

    start = time.perf_counter()

    for _ in range(STEPS):
        W = training_step(x, y, W)

    # Force completion before stopping timer.
    W.cpu()

    elapsed = time.perf_counter() - start

    print(f"Steps: {STEPS}")
    print(f"Total: {elapsed:.4f} s")
    print(f"Average: {elapsed / STEPS * 1000:.4f} ms/step")


if __name__ == "__main__":
    main()
