"""Compare a normal CNN with a simple curved-metric CNN.

The goal of this script is not to claim a benchmark win. It is a compact,
hackable demo for the idea:

    data/task -> local metric G(h) -> distances/logits in curved feature space
    X_{l+1} = G_theta(X_l, g_theta(X_l), kappa_theta(X_l))

The script supports synthetic data plus public torchvision datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class SyntheticPatternDataset(Dataset):
    """Tiny 28x28 image dataset with four pattern classes.

    Classes:
      0: vertical bar
      1: horizontal bar
      2: main diagonal
      3: anti diagonal
    """

    def __init__(
        self,
        samples: int = 4000,
        image_size: int = 28,
        noise_std: float = 0.22,
        seed: int = 7,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.noise_std = noise_std
        self.seed = seed
        label_generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 4, (samples,), generator=label_generator)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        label = int(self.labels[index])
        size = self.image_size
        img = torch.zeros(size, size)
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)

        shift = int(torch.randint(-4, 5, (1,), generator=generator))
        thickness = int(torch.randint(1, 4, (1,), generator=generator))
        center = size // 2 + shift
        rows = torch.arange(size)

        if label == 0:
            lo, hi = self._band(center, thickness, size)
            img[:, lo:hi] = 1.0
        elif label == 1:
            lo, hi = self._band(center, thickness, size)
            img[lo:hi, :] = 1.0
        elif label == 2:
            cols = (rows + shift).clamp(0, size - 1)
            self._draw_diagonal(img, rows, cols, thickness)
        else:
            cols = (size - 1 - rows + shift).clamp(0, size - 1)
            self._draw_diagonal(img, rows, cols, thickness)

        img = img + torch.randn(size, size, generator=generator) * self.noise_std
        img = img.clamp(0.0, 1.0).unsqueeze(0)
        return img, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _band(center: int, thickness: int, size: int) -> Tuple[int, int]:
        lo = max(0, center - thickness)
        hi = min(size, center + thickness + 1)
        return lo, hi

    def _draw_diagonal(self, img: Tensor, rows: Tensor, cols: Tensor, thickness: int) -> None:
        size = self.image_size
        for offset in range(-thickness, thickness + 1):
            shifted_cols = (cols + offset).clamp(0, size - 1)
            img[rows, shifted_cols] = 1.0


@dataclass(frozen=True)
class DataConfig:
    classes: int
    input_channels: int


class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, feature_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    """Plain residual block used as a fairer non-geometric baseline."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_act(x + self.net(x))


class ResidualCNN(nn.Module):
    """CNN with standard residual blocks but no metric or curvature fields."""

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        classes: int = 4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualConvBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResidualConvBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ResidualConvBlock(128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        return self.classifier(h)


class TraditionalCNN(nn.Module):
    """Plain CNN encoder plus a linear classifier."""

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(input_channels, feature_dim)
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        return self.classifier(h)


class CurvedMetricHead(nn.Module):
    """Prototype classifier with a sample-dependent diagonal metric.

    The head learns class prototypes p_c and predicts a positive local metric
    diag(G(h)) for each sample. Logits are negative squared distances:

        logit_c = -sum_j G_j(h) * (h_j - p_cj)^2 + b_c
    """

    def __init__(self, feature_dim: int = 64, classes: int = 4) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(classes, feature_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(classes))
        self.metric = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        metric_diag = F.softplus(self.metric(h)) + 1e-4
        metric_diag = metric_diag / metric_diag.mean(dim=-1, keepdim=True).clamp_min(1e-4)
        diff = h.unsqueeze(1) - self.prototypes.unsqueeze(0)
        distances = (metric_diag.unsqueeze(1) * diff.square()).sum(dim=-1)
        return -distances + self.bias


class CurvedMetricCNN(nn.Module):
    """Same encoder, but classification happens in a learned local metric."""

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(input_channels, feature_dim)
        self.head = CurvedMetricHead(feature_dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        return self.head(h)


class GeometricFlowBlock(nn.Module):
    """Layer update driven by a learned metric and curvature.

    This block implements a stable residual version of:

        X_{l+1} = G_theta(X_l, g_theta(X_l), kappa_theta(X_l))

    `g_theta` is a positive local metric gate over channels and spatial
    positions. `kappa_theta` is a bounded channel-wise curvature signal. The
    update uses a first-order vector field plus a small curvature-dependent
    second-order correction.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.vector_field = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.metric = nn.Conv2d(channels, channels, kernel_size=1)
        self.curvature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.step_logit = nn.Parameter(torch.tensor(-2.0))
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_act = nn.SiLU(inplace=True)
        self.current_metric: Optional[Tensor] = None
        self.current_curvature: Optional[Tensor] = None
        self.last_stats: Dict[str, float] = {}

    def forward(self, x: Tensor) -> Tensor:
        vector = self.vector_field(x)

        metric = F.softplus(self.metric(x)) + 1e-4
        metric = metric / metric.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)

        curvature = self.curvature(x)
        self.current_metric = metric
        self.current_curvature = curvature
        first_order = metric * vector
        second_order = 0.5 * curvature * first_order * torch.tanh(first_order)
        step = torch.sigmoid(self.step_logit)

        self.last_stats = {
            "metric_mean": float(metric.detach().mean().item()),
            "metric_std": float(metric.detach().std(unbiased=False).item()),
            "curvature_abs": float(curvature.detach().abs().mean().item()),
            "step": float(step.detach().item()),
        }
        return self.out_act(self.out_norm(x + step * (first_order + second_order)))


class GeometricFlowCNN(nn.Module):
    """CNN whose internal feature maps evolve through geometric flow blocks."""

    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 64,
        classes: int = 4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            GeometricFlowBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            GeometricFlowBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            GeometricFlowBlock(128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        return self.classifier(h)

    def geometric_regularization(self, curvature_weight: float, metric_weight: float) -> Tensor:
        penalties = []
        for module in self.modules():
            if isinstance(module, GeometricFlowBlock):
                if curvature_weight and module.current_curvature is not None:
                    penalties.append(curvature_weight * module.current_curvature.square().mean())
                if metric_weight and module.current_metric is not None:
                    metric_centered = module.current_metric - 1.0
                    penalties.append(metric_weight * metric_centered.square().mean())
        if not penalties:
            return self.classifier.weight.new_zeros(())
        return torch.stack(penalties).sum()

    def geometric_diagnostics(self) -> Dict[str, float]:
        stats: Dict[str, List[float]] = {
            "metric_mean": [],
            "metric_std": [],
            "curvature_abs": [],
            "step": [],
        }
        for module in self.modules():
            if isinstance(module, GeometricFlowBlock) and module.last_stats:
                for key in stats:
                    stats[key].append(module.last_stats[key])
        return {
            f"geo_{key}": sum(values) / len(values)
            for key, values in stats.items()
            if values
        }


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    reg_loss: float = 0.0
    diagnostics: Optional[Dict[str, float]] = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    label_smoothing: float = 0.0,
    curvature_reg: float = 0.0,
    metric_reg: float = 0.0,
) -> EpochStats:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_samples = 0
    diagnostic_sums: Dict[str, float] = {}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(training):
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            reg_loss = ce_loss.new_zeros(())
            if training and hasattr(model, "geometric_regularization"):
                reg_loss = model.geometric_regularization(curvature_reg, metric_reg)
            loss = ce_loss + reg_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        total_loss += float(ce_loss.item()) * batch_size
        total_reg_loss += float(reg_loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_samples += batch_size
        if hasattr(model, "geometric_diagnostics"):
            for key, value in model.geometric_diagnostics().items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value * batch_size

    return EpochStats(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        reg_loss=total_reg_loss / total_samples,
        diagnostics={
            key: value / total_samples
            for key, value in diagnostic_sums.items()
        } or None,
    )


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    scheduler_name: str,
    label_smoothing: float,
    curvature_reg: float,
    metric_reg: float,
    device: torch.device,
) -> List[Dict[str, float]]:
    model.to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler_name == "cosine"
        else None
    )
    history: List[Dict[str, float]] = []

    print(f"\n== {name} ==")
    print(f"parameters: {params:,}")
    for epoch in range(1, epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=label_smoothing,
            curvature_reg=curvature_reg,
            metric_reg=metric_reg,
        )
        test = run_epoch(model, test_loader, None, device)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": float(epoch),
            "lr": float(current_lr),
            "train_loss": train.loss,
            "train_acc": train.accuracy,
            "train_reg_loss": train.reg_loss,
            "test_loss": test.loss,
            "test_acc": test.accuracy,
            "parameters": float(params),
        }
        if train.diagnostics:
            row.update(train.diagnostics)
        history.append(row)
        line = (
            f"epoch {epoch:02d} | "
            f"train loss {train.loss:.4f} acc {train.accuracy:.3f} | "
            f"test loss {test.loss:.4f} acc {test.accuracy:.3f}"
        )
        if train.reg_loss:
            line += f" | reg {train.reg_loss:.5f}"
        if train.diagnostics:
            line += (
                f" | g {train.diagnostics.get('geo_metric_mean', 0.0):.3f}"
                f"+/-{train.diagnostics.get('geo_metric_std', 0.0):.3f}"
                f" |k| {train.diagnostics.get('geo_curvature_abs', 0.0):.3f}"
                f" step {train.diagnostics.get('geo_step', 0.0):.3f}"
            )
        print(line)
        if scheduler is not None:
            scheduler.step()

    return history


def maybe_plot(results: Dict[str, List[Dict[str, float]]], output_path: Optional[str]) -> None:
    if output_path is None:
        return

    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for name, history in results.items():
        epochs = [row["epoch"] for row in history]
        test_acc = [row["test_acc"] for row in history]
        plt.plot(epochs, test_acc, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"\nSaved plot to {path}")


def maybe_save_metrics(results: Dict[str, List[Dict[str, float]]], output_path: Optional[str]) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {path}")


def deterministic_subset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def build_synthetic_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, DataConfig]:
    dataset = SyntheticPatternDataset(
        samples=args.samples,
        noise_std=args.noise,
        seed=args.seed,
    )
    train_size = math.floor(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed + 101)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=split_generator)
    return train_set, test_set, DataConfig(classes=4, input_channels=1)


def build_public_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, DataConfig]:
    from torchvision import datasets, transforms

    dataset_name = args.dataset.lower()
    data_root = Path(args.data_dir)

    if dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ) if args.augment else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_set = datasets.MNIST(data_root, train=True, download=args.download, transform=train_transform)
        test_set = datasets.MNIST(data_root, train=False, download=args.download, transform=test_transform)
        config = DataConfig(classes=10, input_channels=1)
    elif dataset_name == "fashion-mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        ) if args.augment else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        train_set = datasets.FashionMNIST(
            data_root,
            train=True,
            download=args.download,
            transform=train_transform,
        )
        test_set = datasets.FashionMNIST(
            data_root,
            train=False,
            download=args.download,
            transform=test_transform,
        )
        config = DataConfig(classes=10, input_channels=1)
    elif dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        ) if args.augment else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_set = datasets.CIFAR10(data_root, train=True, download=args.download, transform=train_transform)
        test_set = datasets.CIFAR10(data_root, train=False, download=args.download, transform=test_transform)
        config = DataConfig(classes=10, input_channels=3)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_set = deterministic_subset(train_set, args.train_samples, args.seed + 401)
    test_set = deterministic_subset(test_set, args.test_samples, args.seed + 402)
    return train_set, test_set, config


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataConfig]:
    if args.dataset == "synthetic":
        train_set, test_set, config = build_synthetic_datasets(args)
    else:
        train_set, test_set, config = build_public_datasets(args)

    train_generator = torch.Generator().manual_seed(args.seed + 202)
    test_generator = torch.Generator().manual_seed(args.seed + 303)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )
    return train_loader, test_loader, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["synthetic", "mnist", "fashion-mnist", "cifar10"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument(
        "--models",
        type=str,
        default="traditional,residual,curved-head,geometric-flow",
        help="Comma-separated subset: traditional, residual, curved-head, geometric-flow",
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--curvature-reg", type=float, default=1e-4)
    parser.add_argument("--metric-reg", type=float, default=1e-5)
    parser.add_argument("--noise", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def choose_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def selected_model_builders(model_names: str) -> Dict[str, Type[nn.Module]]:
    registry: Dict[str, Tuple[str, Type[nn.Module]]] = {
        "traditional": ("Traditional CNN", TraditionalCNN),
        "residual": ("Residual CNN", ResidualCNN),
        "curved-head": ("Curved Metric CNN", CurvedMetricCNN),
        "geometric-flow": ("Geometric Flow CNN", GeometricFlowCNN),
    }
    selected = [name.strip() for name in model_names.split(",") if name.strip()]
    unknown = [name for name in selected if name not in registry]
    if unknown:
        valid = ", ".join(registry)
        raise ValueError(f"Unknown model(s): {unknown}. Valid options: {valid}")
    return {registry[name][0]: registry[name][1] for name in selected}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    results = {}
    for name, model_cls in selected_model_builders(args.models).items():
        set_seed(args.seed)
        train_loader, test_loader, data_config = build_loaders(args)
        results[name] = train_model(
            name=name,
            model=model_cls(
                input_channels=data_config.input_channels,
                feature_dim=args.feature_dim,
                classes=data_config.classes,
            ),
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_name=args.scheduler,
            label_smoothing=args.label_smoothing,
            curvature_reg=args.curvature_reg,
            metric_reg=args.metric_reg,
            device=device,
        )

    maybe_save_metrics(results, args.metrics_json)
    maybe_plot(results, args.plot)


if __name__ == "__main__":
    main()
