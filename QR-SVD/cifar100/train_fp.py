"""FP32-SGD fine-tuning for the SGD fine-tune -> PTQ baseline.

This is intentionally separate from train_sam.py: no quantization modules,
SAM/QSAM perturbation, or QAT clipping parameters are constructed here. It
starts from the same r20_w32a32 FP32 checkpoint as QAT-SGD and SAQ.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import core.models as fp_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="FP32 SGD fine-tuning for the PreResNet-20 -> PTQ baseline"
    )
    parser.add_argument("--data_path", required=True, help="Parent directory containing cifar/")
    parser.add_argument("--pretrained", required=True, help="Shared FP32 r20_w32a32 checkpoint")
    parser.add_argument("--output", required=True, help="New directory for FP32 artifacts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--milestones", default="80,120")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--print_frequency", type=int, default=50)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(args):
    mean = [0.50705882, 0.48666667, 0.44078431]
    std = [0.26745098, 0.25568627, 0.27607843]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    root = str(Path(args.data_path) / "cifar")
    train_set = datasets.CIFAR100(
        root=root, train=True, transform=train_transform, download=args.download
    )
    test_set = datasets.CIFAR100(
        root=root, train=False, transform=test_transform, download=args.download
    )
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )


def load_pretrained(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError("pretrained checkpoint must be a state_dict or contain a 'model' state_dict")
    state = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FP32 checkpoint mismatch. Missing: {}; unexpected: {}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )


def accuracy(logits, targets):
    prediction = logits.topk(5, dim=1).indices
    top1 = prediction[:, :1].eq(targets[:, None]).sum().item()
    top5 = prediction.eq(targets[:, None]).sum().item()
    return top1, top5


def train_epoch(model, loader, criterion, optimizer, device, epoch, print_frequency):
    model.train()
    total_loss = total_top1 = total_top5 = total_examples = 0
    start = time.perf_counter()
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1
        total_top5 += top5
        total_examples += batch_size
        if step % print_frequency == 0:
            print(
                "Epoch {:03d} step [{}/{}] loss {:.4f} top1 {:.4f}".format(
                    epoch,
                    step,
                    len(loader),
                    total_loss / total_examples,
                    100.0 * total_top1 / total_examples,
                ),
                flush=True,
            )
    return {
        "loss": total_loss / total_examples,
        "top1": 100.0 * total_top1 / total_examples,
        "top5": 100.0 * total_top5 / total_examples,
        "seconds": time.perf_counter() - start,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_top1 = total_top5 = total_examples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            top1, top5 = accuracy(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += top1
            total_top5 += top5
            total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "top1": 100.0 * total_top1 / total_examples,
        "top5": 100.0 * total_top5 / total_examples,
        "examples": total_examples,
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    set_reproducible(args.seed)
    device = torch.device(args.device)
    milestones = [int(item) for item in args.milestones.split(",")]

    train_loader, test_loader = make_loaders(args)
    model = fp_models.preresnet20(num_classes=100)
    load_pretrained(model, args.pretrained)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.gamma
    )

    history = []
    best_top1 = -float("inf")
    best_epoch = None
    total_start = time.perf_counter()
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.print_frequency
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(
            "Epoch {:03d}: lr {:.6f}; test Top-1 {:.4f}; Top-5 {:.4f}".format(
                epoch, lr, test_metrics["top1"], test_metrics["top5"]
            ),
            flush=True,
        )
        if test_metrics["top1"] > best_top1:
            best_top1 = test_metrics["top1"]
            best_epoch = epoch
            torch.save(model.state_dict(), output / "best_fp32_model.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
            },
            output / "last_fp32_checkpoint.pth",
        )
        scheduler.step()

    result = {
        "method": "FP32-SGD fine-tuning",
        "dataset": "CIFAR-100",
        "network": "preresnet20",
        "source_checkpoint": str(Path(args.pretrained).resolve()),
        "seed": args.seed,
        "optimizer": "Nesterov SGD",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "scheduler": "MultiStepLR",
            "milestones": milestones,
            "gamma": args.gamma,
            "augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip",
        },
        "best_epoch": best_epoch,
        "best_test": max(history, key=lambda item: item["test"]["top1"])["test"],
        "last_test": history[-1]["test"],
        "total_seconds": time.perf_counter() - total_start,
        "history": history,
    }
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(
        "Best FP32 result: epoch {} | Top-1 {:.4f} | Top-5 {:.4f}".format(
            best_epoch, result["best_test"]["top1"], result["best_test"]["top5"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
