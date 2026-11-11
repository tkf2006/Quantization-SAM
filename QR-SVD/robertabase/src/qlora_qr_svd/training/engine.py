from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_scheduler

from qlora_qr_svd.config import ProjectConfig
from qlora_qr_svd.evaluation.glue_metrics import compute_glue_metrics, selection_metric
from qlora_qr_svd.qr_svd.perturbation import capture_rng_state, restore_rng_state
from qlora_qr_svd.utils import append_jsonl, ensure_dir, write_json


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_batch(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, task_name: str, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(**batch)
        losses.append(float(output.loss.detach().float().item()))
        predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())
    metrics = compute_glue_metrics(task_name, predictions, labels)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics


def _save_adapter(
    model,
    tokenizer,
    output_dir: Path,
    name: str,
    save_tokenizer: bool = True,
) -> Path:
    target = ensure_dir(output_dir / name)
    model.save_pretrained(target, safe_serialization=True)
    if save_tokenizer:
        tokenizer.save_pretrained(target)
    return target


def train_encoder(
    model,
    tokenizer,
    train_loader,
    validation_loader,
    config: ProjectConfig,
    perturbation=None,
    comparison_diagnostics=None,
) -> dict:
    requires_perturbation = config.experiment.method != "qlora"
    if requires_perturbation and perturbation is None:
        raise ValueError(
            f"Method {config.experiment.method!r} requires a perturbation manager"
        )
    if not requires_perturbation and perturbation is not None:
        raise ValueError("QLoRA baseline must not receive a perturbation manager")
    if comparison_diagnostics is not None and perturbation is None:
        raise ValueError("Subspace comparisons require a primary perturbation manager")

    device = torch.device(config.runtime.device)
    output_dir = ensure_dir(config.training.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    diagnostics_path = output_dir / "subspace_diagnostics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    if diagnostics_path.exists():
        diagnostics_path.unlink()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = len(train_loader) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = _make_scaler(config.runtime.fp16)

    best_score = -math.inf
    best_epoch = -1
    stale_epochs = 0
    global_step = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory_allocated = torch.cuda.memory_allocated(device)
        initial_memory_reserved = torch.cuda.memory_reserved(device)
    else:
        initial_memory_allocated = 0
        initial_memory_reserved = 0
    started = time.time()

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_clean_losses: list[float] = []
        for batch in train_loader:
            global_step += 1
            comparison_results = None
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if perturbation is None:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model(**batch)
                    update_loss = output.loss
            else:
                rng_state = capture_rng_state()
                perturbation.begin_clean_pass()
                collect_comparisons = (
                    comparison_diagnostics is not None
                    and global_step % config.training.logging_steps == 0
                )
                if collect_comparisons:
                    comparison_diagnostics.begin_clean_pass()
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        clean_output = model(**batch)
                        clean_loss = clean_output.loss
                    # The clean pass only needs gradients used to construct the
                    # SAM perturbation; it must not populate optimizer gradients.
                    # Scaling its backward through fp16 can overflow before the
                    # optimizer's GradScaler has a chance to back off.
                    primary_tensors = perturbation.clean_tensors()
                    comparison_tensors = (
                        comparison_diagnostics.clean_tensors()
                        if collect_comparisons
                        else ()
                    )
                    clean_gradients = torch.autograd.grad(
                        clean_loss, primary_tensors + comparison_tensors
                    )
                    primary_count = len(primary_tensors)
                    diagnostics = perturbation.begin_perturbed_pass(
                        clean_gradients[:primary_count]
                    )
                    if collect_comparisons:
                        comparison_results = comparison_diagnostics.collect(
                            clean_gradients[primary_count:]
                        )
                    optimizer.zero_grad(set_to_none=True)
                    restore_rng_state(rng_state)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        perturbed_output = model(**batch)
                        update_loss = perturbed_output.loss
                    epoch_clean_losses.append(float(clean_loss.detach().float().item()))
                except Exception:
                    perturbation.disable()
                    if comparison_diagnostics is not None:
                        comparison_diagnostics.disable()
                    optimizer.zero_grad(set_to_none=True)
                    raise

            try:
                scaler.scale(update_loss).backward()
            finally:
                if perturbation is not None:
                    # LoRA-SAM mutates LoRA parameters for the perturbed pass;
                    # restore clean parameters before clipping and AdamW.step().
                    perturbation.disable()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.training.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_value = float(update_loss.detach().float().item())
            epoch_losses.append(loss_value)

            if global_step % config.training.logging_steps == 0:
                record = {
                    "type": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss_value,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if perturbation is not None:
                    record.update(
                        {
                            "clean_loss": epoch_clean_losses[-1],
                            "sam_gradient_norm": diagnostics["gradient_norm"],
                            "perturbation_norm": diagnostics["perturbation_norm"],
                        }
                    )
                    if "projected_gradient_norm" in diagnostics:
                        record["projected_gradient_norm"] = diagnostics[
                            "projected_gradient_norm"
                        ]
                    if "weighted_gradient_norm" in diagnostics:
                        record["weighted_gradient_norm"] = diagnostics[
                            "weighted_gradient_norm"
                        ]
                        record["singular_value_exponent"] = diagnostics[
                            "singular_value_exponent"
                        ]
                append_jsonl(metrics_path, record)
                if (
                    perturbation is not None
                    and "per_layer_projected_gradient_norm" in diagnostics
                ):
                    primary_norm = diagnostics["projected_gradient_norm"]
                    comparison_payload = {}
                    for name, result in (comparison_results or {}).items():
                        comparison_norm = result["projected_gradient_norm"]
                        comparison_payload[name] = result | {
                            "norm_ratio_to_primary": comparison_norm
                            / max(primary_norm, 1e-12)
                        }
                    append_jsonl(
                        diagnostics_path,
                        {
                            "type": "subspace_diagnostics",
                            "epoch": epoch,
                            "global_step": global_step,
                            "clean_loss": epoch_clean_losses[-1],
                            "perturbed_loss": loss_value,
                            "loss_increase": loss_value - epoch_clean_losses[-1],
                            "primary": diagnostics,
                            "comparisons": comparison_payload,
                        },
                    )

        validation_metrics = evaluate(
            model, validation_loader, config.data.task_name, device
        )
        score = selection_metric(config.data.task_name, validation_metrics)
        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)),
            "validation": validation_metrics,
            "selection_score": score,
            "elapsed_seconds": time.time() - started,
        }
        if epoch_clean_losses:
            epoch_record["clean_train_loss"] = float(np.mean(epoch_clean_losses))
        append_jsonl(metrics_path, epoch_record)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _save_adapter(
                model,
                tokenizer,
                output_dir,
                "best_adapter",
                save_tokenizer=config.training.save_tokenizer_with_adapter,
            )
            if config.training.save_optimizer_state:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_score": best_score,
                    },
                    output_dir / "best_training_state.pt",
                )
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                break

    if config.training.save_last_adapter:
        _save_adapter(
            model,
            tokenizer,
            output_dir,
            "last_adapter",
            save_tokenizer=config.training.save_tokenizer_with_adapter,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_allocated = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_memory_allocated = 0
        peak_memory_reserved = 0
    elapsed_seconds = time.time() - started
    summary = {
        "experiment": config.experiment.name,
        "method": config.experiment.method,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_completed": epoch,
        "global_steps": global_step,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": elapsed_seconds / global_step if global_step else None,
        "initial_memory_allocated_gib": initial_memory_allocated / (1024**3),
        "initial_memory_reserved_gib": initial_memory_reserved / (1024**3),
        "peak_memory_allocated_gib": peak_memory_allocated / (1024**3),
        "peak_memory_reserved_gib": peak_memory_reserved / (1024**3),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
