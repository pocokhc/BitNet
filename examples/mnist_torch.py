import itertools
import os
import sys
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bitnet"))

from bitnet_torch import BitLinear

# use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _train_mnist(
    model: torch.nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    times = []

    print("train start")
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            y = model(images)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        times.append(time.time() - t0)
        accuracy = _valid_model(model, test_loader)
        history.append(accuracy)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.5f}, Acc: {accuracy:.3%}")
    total_times = list(itertools.accumulate(times))

    accuracy = _valid_model(model, test_loader)
    return history, total_times, accuracy


def _valid_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def simple_test():
    model = nn.Sequential(
        nn.Flatten(),
        BitLinear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    print(model)
    _, _, score = _train_mnist(model, 5, 512, 0.001)
    print(f"Accuracy on test data: {score:.3%}")  # Accuracy on test data: 96.00%


def compare(units: int, layers: int, lr: float, epochs: int):
    models = []

    # --- dense
    m = nn.Sequential()
    m.add_module("flatten", nn.Flatten())
    m.add_module("in_layer", nn.Linear(28 * 28, units))
    for _ in range(layers):
        m.add_module("norm", nn.LayerNorm(units))
        m.add_module("linear", nn.Linear(units, units, bias=False))
        m.add_module("relu", nn.ReLU())
    m.add_module("linear", nn.Linear(units, 10))
    models.append(["Dense", m])

    # --- 1bit
    m = nn.Sequential()
    m.add_module("flatten", nn.Flatten())
    m.add_module("in_layer", nn.Linear(28 * 28, units))
    for _ in range(layers):
        m.add_module("bitnet", BitLinear(units, units, "1bit", flg_before_linear=False))
        m.add_module("relu", nn.ReLU())
    m.add_module("linear", nn.Linear(units, 10))
    models.append(["BitLinear 1bit", m])

    # --- 1.58bit
    m = nn.Sequential()
    m.add_module("flatten", nn.Flatten())
    m.add_module("in_layer", nn.Linear(28 * 28, units))
    for _ in range(layers):
        m.add_module("bitnet", BitLinear(units, units, "1.58bit"))
        m.add_module("relu", nn.ReLU())
    m.add_module("linear", nn.Linear(units, 10))
    models.append(["BitLinear 1.58bit", m])

    for name, m in models:
        history, times, _ = _train_mnist(m, epochs, 512, lr)
        plt.plot(times, history, label=name)

    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simple_test()
    # compare(units=64, layers=5, lr=0.0001, epochs=20)
