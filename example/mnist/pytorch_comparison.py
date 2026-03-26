"""PyTorch MNIST example aligned with the athena Fortran example.

This script uses the same text datasets, network shape, optimiser settings,
batch size, and epoch count as example/mnist/src/main.f90 so code speed can be
compared directly.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (SCRIPT_DIR / "../../../../DMNIST").resolve()
DEFAULT_TRAIN_FILE = "MNIST_train.txt"
DEFAULT_TEST_FILE = "MNIST_test.txt"


class AthenaMnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if module.out_features == 10:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        return self.classifier(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST benchmark aligned with athena example/mnist.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(name)


def set_reproducibility(seed: int, threads: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(threads, 4)))
        except RuntimeError:
            pass


def load_mnist_text(file_path: Path, limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"MNIST data file not found: {file_path}. "
            f"Pass --data-dir to point at the same dataset used by the Fortran example."
        )

    raw = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if limit is not None:
        raw = raw[:limit]

    labels = raw[:, 0].astype(np.int64, copy=False)
    pixels = raw[:, 1:]
    image_size = int(math.isqrt(pixels.shape[1]))
    if image_size * image_size != pixels.shape[1]:
        raise ValueError(f"Input size {pixels.shape[1]} does not describe a square image.")

    images = pixels.reshape(-1, image_size, image_size) / 255.0
    images = np.ascontiguousarray(images[:, np.newaxis, :, :])

    return torch.from_numpy(images), torch.from_numpy(labels)


def build_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = torch.argmax(logits, dim=1)
    return int((predictions == targets).sum().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> tuple[float, float, float]:
    model.train()
    start_time = time.perf_counter()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += accuracy_from_logits(logits, labels)
        total += batch_size

        if log_every > 0 and step % log_every == 0:
            avg_loss = running_loss / total
            avg_accuracy = correct / total
            print(
                f"epoch={epoch:02d} batch={step:04d}/{len(loader):04d} "
                f"loss={avg_loss:.5f} accuracy={avg_accuracy:.5f}"
            )

    elapsed = time.perf_counter() - start_time
    return running_loss / total, correct / total, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    start_time = time.perf_counter()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += accuracy_from_logits(logits, labels)
        total += batch_size

    elapsed = time.perf_counter() - start_time
    return running_loss / total, correct / total, elapsed


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed, args.threads)
    device = resolve_device(args.device)

    train_path = (args.data_dir / args.train_file).resolve()
    test_path = (args.data_dir / args.test_file).resolve()

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    train_images, train_labels = load_mnist_text(train_path, limit=args.limit_train)
    test_images, test_labels = load_mnist_text(test_path, limit=args.limit_test)
    load_time = time.perf_counter() - load_start

    train_loader = build_loader(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = build_loader(
        test_images,
        test_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = AthenaMnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print("PyTorch MNIST comparison")
    print(f"device={device}")
    print(f"threads={torch.get_num_threads()}")
    print(f"train_samples={len(train_loader.dataset)}")
    print(f"test_samples={len(test_loader.dataset)}")
    print(f"batch_size={args.batch_size}")
    print(f"epochs={args.epochs}")
    print(f"learning_rate={args.learning_rate}")
    print(f"momentum={args.momentum}")
    print(f"parameter_count={parameter_count}")
    print(f"data_load_time_s={load_time:.6f}")

    train_time = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_accuracy, epoch_time = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
        )
        train_time += epoch_time
        print(
            f"epoch={epoch:02d} train_loss={epoch_loss:.5f} "
            f"train_accuracy={epoch_accuracy:.5f} epoch_time_s={epoch_time:.6f}"
        )

    test_loss, test_accuracy, test_time = evaluate(model, test_loader, criterion, device)
    total_time = time.perf_counter() - total_start

    print(f"test_loss={test_loss:.5f}")
    print(f"test_accuracy={test_accuracy:.5f}")
    print(f"training_time_s={train_time:.6f}")
    print(f"testing_time_s={test_time:.6f}")
    print(f"total_time_s={total_time:.6f}")


if __name__ == "__main__":
    main()
