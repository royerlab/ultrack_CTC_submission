from typing import Callable, Optional, Set

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dexp_dl.loss.mas import MAS


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: th.Tensor, y: Optional[th.Tensor] = None) -> th.Tensor:
        return self.model.forward(x)


class BiasedMNIST(MNIST):
    def __init__(self, classes: Set[int], **kwargs):
        super().__init__(**kwargs)

        self.data = [d for d, t in zip(self.data, self.targets) if t.item() in classes]
        self.targets = th.Tensor([t for t in self.targets if t.item() in classes])


def transform(image: Image) -> th.Tensor:
    tensor = to_tensor(image)
    return tensor


def train(
    model: nn.Module,
    loader: DataLoader,
    optim: th.optim.Optimizer,
    loss_fn: Callable,
    use_cuda: bool,
    aux_loss_fn: Optional[Callable] = None,
    gamma: float = 0.0,
) -> None:

    for _ in range(5):  # epochs
        with tqdm(loader) as pbar:
            for x, y in pbar:
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()

                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                if aux_loss_fn is not None:
                    loss += gamma * aux_loss_fn(model)

                optim.zero_grad()
                loss.backward()
                optim.step()

                pbar.set_postfix({"loss": loss.item()})


def val(model: nn.Module, loader: DataLoader, use_cuda: bool) -> float:
    results = [[]] * 10
    with th.no_grad():
        for x, y in tqdm(loader):
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            y_hat = model(x).argmax(dim=1)
            for p, t in zip(y, y_hat):
                results[t.item()] = (p == t).item()

    results = np.array([np.array(acc).mean().item() if acc else 0.0 for acc in results])

    for i, acc in enumerate(results):
        print(f"{i}: {acc:0.4f}")
    print()
    return results.mean().item()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_MAS():
    use_cuda = False

    train_1_ds = BiasedMNIST(
        root="/tmp/mnist",
        train=True,
        download=True,
        transform=transform,
        classes=set(range(5)),
    )

    train_2_ds = BiasedMNIST(
        root="/tmp/mnist",
        train=True,
        download=True,
        transform=transform,
        classes=set(range(5, 10)),
    )

    print()
    print("Content of DS 1", th.unique(train_1_ds.targets, return_counts=True))
    print("Content of DS 2", th.unique(train_2_ds.targets, return_counts=True))

    test_ds = MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    net = Net()
    model = ModelWrapper(net)
    if use_cuda:
        model = model.cuda()

    optim = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = nn.NLLLoss()

    loader = DataLoader(train_1_ds, batch_size=64, shuffle=True)
    train(model, loader, optim, loss_fn, use_cuda)

    aux_loss = MAS(model, loader)
    aux_loss.fit(1000, "cpu" if not use_cuda else "cuda:0")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print("\n---- first half results ----")
    avg_before = val(model, loader, use_cuda)

    loader = DataLoader(train_2_ds, batch_size=64, shuffle=True)
    train(model, loader, optim, loss_fn, use_cuda, aux_loss, 1.0)

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print("\n---- second half results ----")
    avg_after = val(model, loader, use_cuda)

    # for omega, (name, _) in zip(aux_loss._omegas, model.named_parameters()):
    #     print(name)
    #     print(omega)

    # it has to be better my a large margin
    assert avg_after > avg_before + 0.25
