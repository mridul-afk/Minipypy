import minipypy as mini


def accuracy(logits, labels):
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

        if pred_class == int(labels_cpu[i]):
            correct += 1

    return correct / batch_size


def test_tiny_classification_training_reduces_loss():
    x = mini.Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ])

    y = mini.Tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    model = mini.nn.Linear(4, 3)
    loss_fn = mini.nn.CrossEntropyLoss()
    optimizer = mini.optim.SGD(model, lr=0.1)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(150):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_logits = model(x)
    final_loss = loss_fn(final_logits, y).cpu()[0]
    final_acc = accuracy(final_logits, y)

    assert final_loss < initial_loss
    assert final_acc >= 0.66
