import time
import statistics

import minipypy as mp


SHAPES = [
    [16, 32, 64],
    [32, 64, 128],
    [64, 128, 128],
    [128, 128, 256],
]

WARMUP = 10
REPEATS = 30


def benchmark(shape):
    numel = 1

    for dim in shape:
        numel *= dim

    # Construct input once, outside the timed region.
    x = mp.Tensor(
        [1.0] * numel,
        shape,
        requires_grad=True,
    )

    # Build the computation graph once.
    y = x.permute([2, 0, 1])
    loss = y.sum()

    # Warm up GPU and autograd.
    for _ in range(WARMUP):
        loss.backward()

    timings = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        loss.backward()

        # Force GPU synchronization through a CPU read.
        # This also adds transfer overhead to the timing.
        _ = x.grad().cpu()

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    return {
        "shape": shape,
        "numel": numel,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main():
    print("\nMiniPyPy PERMUTE Backward Benchmark")
    print("=" * 75)

    for shape in SHAPES:
        result = benchmark(shape)

        print(
            f"Shape: {result['shape']}\n"
            f"Elements: {result['numel']:,}\n"
            f"Mean:   {result['mean_ms']:.4f} ms\n"
            f"Median: {result['median_ms']:.4f} ms\n"
            f"Min:    {result['min_ms']:.4f} ms\n"
            f"Max:    {result['max_ms']:.4f} ms\n"
        )


if __name__ == "__main__":
    main()
