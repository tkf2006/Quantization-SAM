from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from qlora_qr_svd.config import DataConfig, TrainingConfig


TASK_FIELDS = {
    "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
}


@dataclass
class GlueData:
    train_dataset: object
    validation_dataset: object
    train_loader: DataLoader
    validation_loader: DataLoader
    selected_indices: list[int]
    indices_path: Path
    validation_source: str
    inner_split_path: Path | None
    train_indices: list[int]
    validation_indices: list[int] | None


def stratified_prefix_indices(
    labels: Sequence[int], fraction: float, seed: int
) -> list[int]:
    """Return a deterministic stratified prefix subset.

    For a fixed seed, a smaller fraction is a subset of a larger fraction.
    This gives the 5%/20%/100% experiments nested data subsets.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)

    selected: list[int] = []
    for label in sorted(by_label):
        candidates = by_label[label].copy()
        random.Random(seed + 104729 * (label + 1)).shuffle(candidates)
        count = len(candidates) if fraction == 1 else max(1, math.floor(len(candidates) * fraction))
        selected.extend(candidates[:count])
    return sorted(selected)


def stratified_holdout_indices(
    labels: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split positions into deterministic stratified train and validation sets."""
    validation = stratified_prefix_indices(labels, validation_fraction, seed)
    validation_set = set(validation)
    train = [index for index in range(len(labels)) if index not in validation_set]
    if not train or not validation:
        raise ValueError("Inner train/validation split produced an empty partition")
    return train, validation


def _load_or_create_indices(dataset, config: DataConfig) -> tuple[list[int], Path]:
    fraction_tag = f"{config.train_fraction:.6f}".rstrip("0").rstrip(".")
    indices_dir = Path(config.indices_dir)
    indices_dir.mkdir(parents=True, exist_ok=True)
    path = indices_dir / f"glue_{config.task_name}_fraction_{fraction_tag}_seed_{config.data_seed}.json"
    fingerprint = getattr(dataset, "_fingerprint", None)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = {
            "dataset_name": config.dataset_name,
            "dataset_revision": config.dataset_revision,
            "task_name": config.task_name,
            "train_fraction": config.train_fraction,
            "data_seed": config.data_seed,
            "dataset_size": len(dataset),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"Saved data indices mismatch for {key}: {path}")
        if fingerprint and payload.get("dataset_fingerprint") not in {None, fingerprint}:
            raise RuntimeError(f"Dataset fingerprint changed; remove or audit {path}")
        return [int(index) for index in payload["indices"]], path

    indices = stratified_prefix_indices(dataset["label"], config.train_fraction, config.data_seed)
    payload = {
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "task_name": config.task_name,
        "train_fraction": config.train_fraction,
        "data_seed": config.data_seed,
        "dataset_size": len(dataset),
        "dataset_fingerprint": fingerprint,
        "indices": indices,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return indices, path


def _load_or_create_inner_split(
    dataset,
    selected_indices: list[int],
    config: DataConfig,
) -> tuple[list[int], list[int], list[int], list[int], Path]:
    fraction_tag = f"{config.train_fraction:.6f}".rstrip("0").rstrip(".")
    validation_tag = (
        f"{config.inner_validation_fraction:.6f}".rstrip("0").rstrip(".")
    )
    indices_dir = Path(config.indices_dir)
    indices_dir.mkdir(parents=True, exist_ok=True)
    path = indices_dir / (
        f"glue_{config.task_name}_fraction_{fraction_tag}_seed_{config.data_seed}_"
        f"innerdev_{validation_tag}_seed_{config.inner_validation_seed}.json"
    )
    fingerprint = getattr(dataset, "_fingerprint", None)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = {
            "dataset_name": config.dataset_name,
            "dataset_revision": config.dataset_revision,
            "task_name": config.task_name,
            "train_fraction": config.train_fraction,
            "data_seed": config.data_seed,
            "inner_validation_fraction": config.inner_validation_fraction,
            "inner_validation_seed": config.inner_validation_seed,
            "selected_size": len(dataset),
            "selected_indices": selected_indices,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"Saved inner-dev split mismatch for {key}: {path}")
        if fingerprint and payload.get("dataset_fingerprint") not in {None, fingerprint}:
            raise RuntimeError(f"Selected dataset fingerprint changed; remove or audit {path}")
        train_positions = [int(index) for index in payload["train_positions"]]
        validation_positions = [int(index) for index in payload["validation_positions"]]
        train_indices = [int(index) for index in payload["train_indices"]]
        validation_indices = [int(index) for index in payload["validation_indices"]]
        return (
            train_positions,
            validation_positions,
            train_indices,
            validation_indices,
            path,
        )

    train_positions, validation_positions = stratified_holdout_indices(
        dataset["label"], config.inner_validation_fraction, config.inner_validation_seed
    )
    train_indices = [selected_indices[index] for index in train_positions]
    validation_indices = [selected_indices[index] for index in validation_positions]
    payload = {
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "task_name": config.task_name,
        "train_fraction": config.train_fraction,
        "data_seed": config.data_seed,
        "inner_validation_fraction": config.inner_validation_fraction,
        "inner_validation_seed": config.inner_validation_seed,
        "selected_size": len(dataset),
        "dataset_fingerprint": fingerprint,
        "selected_indices": selected_indices,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return train_positions, validation_positions, train_indices, validation_indices, path


def build_glue_data(tokenizer, data_config: DataConfig, training_config: TrainingConfig) -> GlueData:
    from datasets import load_dataset, load_from_disk
    from transformers import DataCollatorWithPadding

    if data_config.task_name not in TASK_FIELDS:
        raise ValueError(f"Unsupported GLUE task: {data_config.task_name}")
    if data_config.local_dataset_dir is not None:
        dataset_dir = Path(data_config.local_dataset_dir).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Local dataset directory not found: {dataset_dir}")
        raw = load_from_disk(str(dataset_dir))
    else:
        raw = load_dataset(
            data_config.dataset_name,
            data_config.task_name,
            revision=data_config.dataset_revision,
        )
    for split in ("train", "validation"):
        if split not in raw:
            raise ValueError(f"GLUE dataset is missing required split {split!r}")
    full_train_raw = raw["train"]
    official_validation_raw = raw["validation"]
    indices, indices_path = _load_or_create_indices(full_train_raw, data_config)
    selected_train_raw = full_train_raw.select(indices)
    inner_split_path = None
    validation_indices = None
    if data_config.validation_source == "inner_train":
        (
            train_positions,
            validation_positions,
            train_indices,
            validation_indices,
            inner_split_path,
        ) = _load_or_create_inner_split(selected_train_raw, indices, data_config)
        train_raw = selected_train_raw.select(train_positions)
        validation_raw = selected_train_raw.select(validation_positions)
    else:
        train_raw = selected_train_raw
        validation_raw = official_validation_raw
        train_indices = indices

    first_field, second_field = TASK_FIELDS[data_config.task_name]

    def tokenize(batch):
        second = batch[second_field] if second_field else None
        return tokenizer(
            batch[first_field],
            second,
            truncation=True,
            max_length=data_config.max_length,
        )

    removable_train = [column for column in train_raw.column_names if column != "label"]
    removable_validation = [column for column in validation_raw.column_names if column != "label"]
    train_dataset = train_raw.map(tokenize, batched=True, remove_columns=removable_train)
    validation_dataset = validation_raw.map(
        tokenize, batched=True, remove_columns=removable_validation
    )
    train_dataset.set_format("torch")
    validation_dataset.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_generator = torch.Generator()
    train_generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        pin_memory=training_config.dataloader_pin_memory,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        pin_memory=training_config.dataloader_pin_memory,
    )
    return GlueData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        selected_indices=indices,
        indices_path=indices_path,
        validation_source=data_config.validation_source,
        inner_split_path=inner_split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
