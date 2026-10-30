import time
import minipypy as mp


SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
]

WARMUP = 10
ITERATIONS = 100


def benchmark_clone(size):
    if ITERATIONS <= 0:
        raise ValueError("ITERATIONS must be greater than 0")

    data = [1.0] * size
    x = mp.Tensor(data)

    # Warmup
    for _ in range(WARMUP):
        x.clone()

    y = x.clone()

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        y = x.clone()

    # Synchronize by copying the final result to CPU
    y.cpu()

    end = time.perf_counter()

    return (end - start) / ITERATIONS * 1000

def main():
    print("MiniPyPy Tensor.clone() benchmark")
    print("-" * 45)

    for size in SIZES:
        avg_ms = benchmark_clone(size)

        mb = size * 4 / (1024 * 1024)

        print(
            f"{size:>10,} elements "
            f"({mb:>7.2f} MiB): "
            f"{avg_ms:>9.4f} ms"
        )


if __name__ == "__main__":
    main()
