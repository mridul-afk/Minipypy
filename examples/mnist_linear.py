import random
import minipypy as mini


def load_mnist_from_torchvision(batch_size=32, train=True, limit=None):
    """
    Uses torchvision only for dataset loading.

    MiniPyPy still does:
      - Tensor creation
      - Forward pass
      - Loss
      - Backward
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
        random.shuffle(indices)

    if limit is not None:
        indices = indices[:limit]

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        x_batch = []
        y_batch = []

        for idx in batch_indices:
            image, label = dataset[idx]

            # image shape from torchvision:
            # [1, 28, 28]
            #
            # Flatten to [784]
            flat = image.view(-1).tolist()

            x_batch.append(flat)

            # MiniPyPy has float tensors only for now.
            # CrossEntropyLoss casts labels to int internally.
            y_batch.append(float(label))

        yield mini.Tensor(x_batch), mini.Tensor(y_batch)


def accuracy(logits, labels):
    """
    CPU-side accuracy helper.

    logits: mini.Tensor with shape [batch, classes]
    labels: mini.Tensor with shape [batch]
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
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    loss_fn = mini.nn.CrossEntropyLoss()

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


def main():
    random.seed(0)

    batch_size = 32
    epochs = 3
    train_limit = 2048
    test_limit = 512
    lr = 0.1

    model = mini.nn.TensorFoldLinear(784, 10, rank=8)

    loss_fn = mini.nn.CrossEntropyLoss()
    optimizer = mini.optim.SGD(model, lr=lr)

    print("MiniPyPy MNIST Linear Classifier")
    print("--------------------------------")
    print(f"batch_size  = {batch_size}")
    print(f"epochs      = {epochs}")
    print(f"train_limit = {train_limit}")
    print(f"test_limit  = {test_limit}")
    print(f"lr          = {lr}")
    print("params:", model.parameter_count())
    print("dense params:", model.dense_parameter_count())
    print("compression:", model.compression_ratio())
    print()

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

        train_loss = total_loss / batches
        train_acc = total_acc / batches

        test_loss, test_acc = evaluate(
            model,
            batch_size=batch_size,
            limit=test_limit,
        )

        print()
        print(
            f"epoch {epoch + 1} summary: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
