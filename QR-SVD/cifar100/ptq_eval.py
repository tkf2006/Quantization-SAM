"""Post-training quantization baseline for SAQ's CIFAR-100 ResNet-20 setup.

This script deliberately does *not* create an optimizer or run backward. It
loads the same full-precision checkpoint used by the SAQ experiment, applies
static uniform PTQ on its original weights, calibrates activation ranges on
training images, and evaluates W8A32 / W4A4 / W8A8 inference on CIFAR-100.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import core.models as fp_models
from quan_models.qmodel_analyse import QModelAnalyse


class StaticQuantConv2d(nn.Conv2d):
    """Inference-only Conv2d with raw-weight uniform PTQ.

    SAQ's LIQ_wn QAT layers standardize weights before their forward pass. That
    is trainable during QAT but cannot be applied unchanged to an FP32 model in
    a PTQ comparison: it changes the FP32 convolution scale. This module keeps
    the pretrained weights in their original scale and quantizes them directly.
    """

    def __init__(self, source, bits_weights, bits_activations):
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
        )
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if source.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.register_buffer("weight_clip_value", torch.ones(1))
        self.register_buffer("activation_clip_value", torch.ones(1))

    def forward(self, inputs):
        quantized_inputs = (
            inputs
            if self.bits_activations >= 32
            else quantize_unsigned(inputs, self.activation_clip_value.abs(), self.bits_activations)
        )
        quantized_weight = (
            self.weight
            if self.bits_weights >= 32
            else quantize_symmetric(self.weight, self.weight_clip_value.abs(), self.bits_weights)
        )
        return F.conv2d(
            quantized_inputs,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class StaticQuantLinear(nn.Linear):
    """Inference-only Linear layer with raw-weight uniform PTQ."""

    def __init__(self, source, bits_weights, bits_activations):
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if source.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.register_buffer("weight_clip_value", torch.ones(1))
        self.register_buffer("activation_clip_value", torch.ones(1))

    def forward(self, inputs):
        quantized_inputs = (
            inputs
            if self.bits_activations >= 32
            else quantize_unsigned(inputs, self.activation_clip_value.abs(), self.bits_activations)
        )
        quantized_weight = (
            self.weight
            if self.bits_weights >= 32
            else quantize_symmetric(self.weight, self.weight_clip_value.abs(), self.bits_weights)
        )
        return F.linear(quantized_inputs, quantized_weight, self.bias)


QUANT_LAYER_TYPES = (StaticQuantConv2d, StaticQuantLinear)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PTQ baseline aligned with SAQ-W4A4-PreResNet20-CIFAR100"
    )
    parser.add_argument("--data_path", required=True, help="Parent directory containing cifar/")
    parser.add_argument("--pretrained", required=True, help="FP32 r20_w32a32 checkpoint")
    parser.add_argument("--output", required=True, help="New directory for PTQ artifacts")
    parser.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--calib_seed", type=int, default=1)
    parser.add_argument("--observer_samples_per_batch", type=int, default=32768)
    parser.add_argument("--threshold_candidates", type=int, default=81)
    parser.add_argument("--threshold_min_ratio", type=float, default=0.20)
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-100")
    return parser.parse_args()


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    """Use inference preprocessing for both calibration and test.

    Calibration uses only training-set images, but omits random crop/flip so
    that its distribution and its result are deterministic and deployment-like.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.50705882, 0.48666667, 0.44078431],
                std=[0.26745098, 0.25568627, 0.27607843],
            ),
        ]
    )
    root = str(Path(args.data_path) / "cifar")
    train_set = datasets.CIFAR100(
        root=root, train=True, transform=transform, download=args.download
    )
    test_set = datasets.CIFAR100(
        root=root, train=False, transform=transform, download=args.download
    )
    if args.calib_size <= 0 or args.calib_size > len(train_set):
        raise ValueError("calib_size must be in [1, {}]".format(len(train_set)))

    generator = torch.Generator().manual_seed(args.calib_seed)
    indices = torch.randperm(len(train_set), generator=generator)[: args.calib_size]
    calibration_set = Subset(train_set, indices.tolist())
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return (
        DataLoader(calibration_set, shuffle=False, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
        indices.tolist(),
    )


def unwrap_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state_dict or a dict containing 'model'")
    return {key.replace("module.", "", 1) if key.startswith("module.") else key: value for key, value in state.items()}


def build_fp_model(state):
    model = fp_models.preresnet20(num_classes=100)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FP32 checkpoint mismatch. Missing: {}; unexpected: {}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    return model


def load_fp_model(state, device):
    return build_fp_model(state).to(device).eval()


def replace_with_static_quant_layers(module, prefix=""):
    for child_name, child in list(module.named_children()):
        full_name = "{}.{}".format(prefix, child_name) if prefix else child_name
        if isinstance(child, nn.Conv2d):
            bits_weights, bits_activations = (8, 32) if full_name == "conv" else (4, 4)
            setattr(
                module,
                child_name,
                StaticQuantConv2d(child, bits_weights, bits_activations),
            )
        elif isinstance(child, nn.Linear):
            bits_weights, bits_activations = (8, 8) if full_name == "fc" else (4, 4)
            setattr(
                module,
                child_name,
                StaticQuantLinear(child, bits_weights, bits_activations),
            )
        else:
            replace_with_static_quant_layers(child, full_name)


def load_quant_model(state, device):
    model = build_fp_model(state)
    replace_with_static_quant_layers(model)
    return model.to(device).eval()


def quantize_symmetric(x, clip, bits):
    levels = 2 ** int(bits) - 1
    normalized = (x / clip).clamp(-1.0, 1.0)
    return ((torch.round((normalized + 1.0) * 0.5 * levels) / levels) * 2.0 - 1.0) * clip


def quantize_unsigned(x, clip, bits):
    levels = 2 ** int(bits) - 1
    normalized = (x / clip).clamp(0.0, 1.0)
    return torch.round(normalized * levels) / levels * clip


def mse_clip_threshold(samples, bits, signed, args, device):
    """Choose a static threshold by grid-searching quantization MSE."""
    samples = samples.to(device=device, dtype=torch.float32)
    max_value = samples.abs().max() if signed else samples.max()
    if not torch.isfinite(max_value) or max_value <= 0:
        return 1.0
    ratios = torch.linspace(
        args.threshold_min_ratio,
        1.0,
        args.threshold_candidates,
        device=device,
    )
    candidates = max_value * ratios
    best_clip, best_error = float(candidates[-1]), float("inf")
    for candidate in candidates:
        quantized = (
            quantize_symmetric(samples, candidate, bits)
            if signed
            else quantize_unsigned(samples, candidate, bits)
        )
        error = torch.mean((samples - quantized) ** 2).item()
        if error < best_error:
            best_clip, best_error = float(candidate), error
    return best_clip


def calibrate_weight_clips(model, args, device):
    records = {}
    for name, module in model.named_modules():
        if not isinstance(module, QUANT_LAYER_TYPES):
            continue
        bits = int(module.bits_weights)
        if bits >= 32:
            continue
        clip = mse_clip_threshold(module.weight.detach(), bits, True, args, device)
        module.weight_clip_value.data.fill_(clip)
        records[name] = {"weight_bits": bits, "weight_clip_value": clip}
    return records


def forward_quant_layer_order(model, example):
    order, seen, handles = [], set(), []

    def capture(module, _inputs):
        if id(module) not in seen:
            seen.add(id(module))
            order.append(module)

    for module in model.modules():
        if isinstance(module, QUANT_LAYER_TYPES):
            handles.append(module.register_forward_pre_hook(capture))
    with torch.no_grad():
        model(example)
    for handle in handles:
        handle.remove()
    return order


def collect_input_samples(model, module, calibration_loader, args, device):
    samples = []

    def capture(_module, inputs):
        values = inputs[0].detach().reshape(-1)
        if values.numel() > args.observer_samples_per_batch:
            stride = max(1, values.numel() // args.observer_samples_per_batch)
            values = values[::stride][: args.observer_samples_per_batch]
        samples.append(values.float().cpu())

    handle = module.register_forward_pre_hook(capture)
    with torch.no_grad():
        for images, _targets in calibration_loader:
            model(images.to(device, non_blocking=True))
    handle.remove()
    if not samples:
        raise RuntimeError("activation observer did not receive input")
    return torch.cat(samples)


def calibrate_activation_clips(model, calibration_loader, args, device, records):
    example, _ = next(iter(calibration_loader))
    ordered_modules = forward_quant_layer_order(model, example.to(device, non_blocking=True))
    module_names = {id(module): name for name, module in model.named_modules()}

    for module in ordered_modules:
        bits = int(module.bits_activations)
        if bits >= 32:
            continue
        samples = collect_input_samples(model, module, calibration_loader, args, device)
        clip = mse_clip_threshold(samples, bits, False, args, device)
        module.activation_clip_value.data.fill_(clip)
        name = module_names[id(module)]
        records.setdefault(name, {"weight_bits": int(module.bits_weights)})
        records[name].update(
            {
                "activation_bits": bits,
                "activation_clip_value": clip,
                "observer_values": int(samples.numel()),
            }
        )
    for name, module in model.named_modules():
        if isinstance(module, QUANT_LAYER_TYPES):
            records.setdefault(name, {"weight_bits": int(module.bits_weights)})
            records[name].setdefault("activation_bits", int(module.bits_activations))
            records[name].setdefault("weight_clip_value", float(module.weight_clip_value.abs()))
            records[name].setdefault("activation_clip_value", float(module.activation_clip_value.abs()))
    return records


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = total_top1 = total_top5 = total_examples = 0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            prediction = logits.topk(5, dim=1).indices
            total_top1 += prediction[:, :1].eq(targets[:, None]).sum().item()
            total_top5 += prediction.eq(targets[:, None]).sum().item()
            total_examples += targets.size(0)
    return {
        "top1": 100.0 * total_top1 / total_examples,
        "top5": 100.0 * total_top5 / total_examples,
        "loss": total_loss / total_examples,
        "examples": total_examples,
    }


def compute_bops_mb(model, device):
    """Measure BOPs with SAQ's official QModelAnalyse implementation."""
    analyser = QModelAnalyse(model, logger=None)
    with torch.no_grad():
        bops = analyser.bops_compute(torch.zeros(1, 3, 32, 32, device=device))
    return bops / 1e6


def ensure_output_dir(output):
    output.mkdir(parents=True, exist_ok=False)
    return output


def main():
    args = parse_args()
    if args.threshold_candidates < 2 or not 0 < args.threshold_min_ratio <= 1:
        raise ValueError("threshold search requires >=2 candidates and 0 < min ratio <= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    output = ensure_output_dir(Path(args.output))
    set_reproducible(args.calib_seed)

    calibration_loader, test_loader, calibration_indices = make_loaders(args)
    state = unwrap_checkpoint(args.pretrained)

    fp_model = load_fp_model(state, device)
    fp_metrics = evaluate(fp_model, test_loader, device)

    ptq_model = load_quant_model(state, device)
    for parameter in ptq_model.parameters():
        parameter.requires_grad_(False)

    calibration_start = time.perf_counter()
    records = calibrate_weight_clips(ptq_model, args, device)
    records = calibrate_activation_clips(
        ptq_model, calibration_loader, args, device, records
    )
    calibration_seconds = time.perf_counter() - calibration_start

    test_start = time.perf_counter()
    ptq_metrics = evaluate(ptq_model, test_loader, device)
    test_seconds = time.perf_counter() - test_start
    bops_mb = compute_bops_mb(ptq_model, device)

    torch.save(ptq_model.state_dict(), output / "ptq_model.pth")
    with (output / "calibration_indices.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration_indices, handle)

    result = {
        "method": "PTQ-MSE (raw-weight uniform)",
        "source_checkpoint": str(Path(args.pretrained).resolve()),
        "dataset": "CIFAR-100",
        "network": "preresnet20 converted to static PTQ",
        "quantizer": "per-tensor symmetric weight / unsigned activation uniform quantization",
        "bits": {"first": "W8A32", "body_and_downsample": "W4A4", "fc": "W8A8"},
        "bops_mb": bops_mb,
        "calibration": {
            "source": "CIFAR-100 training set only",
            "preprocessing": "ToTensor + CIFAR-100 Normalize; no random augmentation",
            "images": args.calib_size,
            "seed": args.calib_seed,
            "method": "per-layer MSE clip search",
            "threshold_candidates": args.threshold_candidates,
            "threshold_min_ratio": args.threshold_min_ratio,
            "observer_samples_per_batch": args.observer_samples_per_batch,
            "seconds": calibration_seconds,
        },
        "fp32_test": fp_metrics,
        "ptq_test": ptq_metrics,
        "top1_drop": fp_metrics["top1"] - ptq_metrics["top1"],
        "test_seconds": test_seconds,
        "layer_calibration": records,
    }
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("FP32  Top-1: {:.4f}, Top-5: {:.4f}".format(fp_metrics["top1"], fp_metrics["top5"]))
    print("PTQ   Top-1: {:.4f}, Top-5: {:.4f}".format(ptq_metrics["top1"], ptq_metrics["top5"]))
    print("Top-1 drop: {:.4f}".format(result["top1_drop"]))
    print("BOPs: {:.6f} MBOPs".format(bops_mb))
    print("Saved PTQ artifacts to {}".format(output.resolve()))


if __name__ == "__main__":
    main()
