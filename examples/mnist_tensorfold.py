import random
import minipypy as mini


def load_mnist_from_torchvision(batch_size=32, train=True, limit=None, seed = 123):
    """
    Uses torchvision only for dataset loading.

    MiniPyPy handles:
      - Tensor creation
      - Forward pass
      - CrossEntropyLoss
      - Backward pass
      - Optimizer step
    """

    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
    except ImportError:
        raise ImportError(
            "This example requires torchvision for loading MNIST.\n"
            "Install it with:\n"
            "pip install torchvision"
        )

    dataset = MNIST(
        root="./data",
        train=train,
        download=True,
        transform=ToTensor(),
    )

    indices = list(range(len(dataset)))

    if train:
        rng = random.Random(seed)
        rng.shuffle(indices)

    if limit is not None:
        indices = indices[:limit]

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        x_batch = []
        y_batch = []

        for idx in batch_indices:
            image, label = dataset[idx]

            # image shape from torchvision: [1, 28, 28]
            # flatten to [784]
            flat = image.view(-1).tolist()

            x_batch.append(flat)

            # MiniPyPy currently stores labels as float tensors.
            # CrossEntropyLoss casts them to integer class indices internally.
            y_batch.append(float(label))

        yield mini.Tensor(x_batch), mini.Tensor(y_batch)


def accuracy(logits, labels):
    """
    CPU-side accuracy helper.

    logits shape: [batch, classes]
    labels shape: [batch]
    """

    shape = logits.shape()

    batch_size = shape[0]
    num_classes = shape[1]

    logits_cpu = logits.cpu()
    labels_cpu = labels.cpu()

    correct = 0

    for i in range(batch_size):
        row_start = i * num_classes
        row = logits_cpu[row_start:row_start + num_classes]

        pred_class = 0
        pred_value = row[0]

        for j in range(1, num_classes):
            if row[j] > pred_value:
                pred_value = row[j]
                pred_class = j

        true_class = int(labels_cpu[i])

        if pred_class == true_class:
            correct += 1

    return correct / batch_size


def evaluate(model, batch_size=64, limit=512):
    loss_fn = mini.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for x, y in load_mnist_from_torchvision(
        batch_size=batch_size,
        train=False,
        limit=limit,
    ):
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.cpu()[0]
        total_acc += accuracy(logits, y)
        batches += 1

    return total_loss / batches, total_acc / batches


def train_tensorfold_rank(rank, batch_size, epochs, train_limit, test_limit, lr):
    random.seed(0)

    model = mini.nn.TensorFoldLinear(784, 10, rank=rank, init="xavier")

    loss_fn = mini.nn.CrossEntropyLoss()
    optimizer = mini.optim.SGD(model, lr=lr)

    print()
    print(f"TensorFoldLinear rank={rank}")
    print("-" * 40)
    print(f"params       = {model.parameter_count()}")
    print(f"dense params = {model.dense_parameter_count()}")
    print(f"compression  = {model.compression_ratio():.4f}x")
    print()

    final_train_loss = None
    final_train_acc = None

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for x, y in load_mnist_from_torchvision(
            batch_size=batch_size,
            train=True,
            limit=train_limit,
        ):
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu()[0]
            total_acc += accuracy(logits, y)
            batches += 1

            if batches % 10 == 0:
                print(
                    f"epoch {epoch + 1} "
                    f"batch {batches} "
                    f"loss {total_loss / batches:.4f} "
                    f"acc {total_acc / batches:.4f}"
                )

        final_train_loss = total_loss / batches
        final_train_acc = total_acc / batches

        test_loss, test_acc = evaluate(
            model,
            batch_size=batch_size,
            limit=test_limit,
        )

        print()
        print(
            f"epoch {epoch + 1} summary: "
            f"train_loss={final_train_loss:.4f} "
            f"train_acc={final_train_acc:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )
        print()

    final_test_loss, final_test_acc = evaluate(
        model,
        batch_size=batch_size,
        limit=test_limit,
    )

    return {
        "rank": rank,
        "params": model.parameter_count(),
        "dense_params": model.dense_parameter_count(),
        "compression": model.compression_ratio(),
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "test_loss": final_test_loss,
        "test_acc": final_test_acc,
    }


def print_results_table(results):
    print()
    print("TensorFoldLinear MNIST Benchmark Summary")
    print("=" * 80)
    print(
        f"{'Rank':>6} "
        f"{'Params':>10} "
        f"{'Compression':>14} "
        f"{'Train Loss':>12} "
        f"{'Train Acc':>10} "
        f"{'Test Loss':>10} "
        f"{'Test Acc':>9}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['rank']:>6} "
            f"{result['params']:>10} "
            f"{result['compression']:>13.2f}x "
            f"{result['train_loss']:>12.4f} "
            f"{result['train_acc']:>10.4f} "
            f"{result['test_loss']:>10.4f} "
            f"{result['test_acc']:>9.4f}"
        )

    print("=" * 80)


def main():
    batch_size = 32
    epochs = 3
    train_limit = 2048
    test_limit = 512
    lr = 0.1
    init = "xavier"

    ranks = [2, 4, 8, 10]

    print("MiniPyPy TensorFoldLinear MNIST Benchmark")
    print("-----------------------------------------")
    print(f"batch_size  = {batch_size}")
    print(f"epochs      = {epochs}")
    print(f"train_limit = {train_limit}")
    print(f"test_limit  = {test_limit}")
    print(f"optimizer   = SGD")
    print(f"lr          = {lr}")
    print(f"ranks       = {ranks}")
    print(f"init        = {init}")

    results = []

    for rank in ranks:
        result = train_tensorfold_rank(
            rank=rank,
            batch_size=batch_size,
            epochs=epochs,
            train_limit=train_limit,
            test_limit=test_limit,
            lr=lr,
        )

        results.append(result)

    print_results_table(results)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
