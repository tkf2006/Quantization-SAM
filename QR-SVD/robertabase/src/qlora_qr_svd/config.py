from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    method: str = "qlora"


@dataclass
class ModelConfig:
    name_or_path: str = "FacebookAI/roberta-base"
    revision: str = "main"
    num_labels: int = 2
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    dataset_name: str = "nyu-mll/glue"
    dataset_revision: str = "bcdcba7"
    local_dataset_dir: str | None = None
    task_name: str = "rte"
    train_fraction: float = 0.2
    data_seed: int = 42
    max_length: int = 256
    num_workers: int = 2
    indices_dir: str = "artifacts/data_indices"
    validation_source: str = "official"
    inner_validation_fraction: float = 0.2
    inner_validation_seed: int = 314159


@dataclass
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])
    modules_to_save: list[str] = field(default_factory=lambda: ["classifier"])


@dataclass
class SAMSettings:
    enabled: bool = False
    rho: float = 0.05
    subspace_rank: int = 8
    random_basis_seed: int = 12345
    singular_value_exponent: float = 0.0
    reuse_qr_singular_values: bool = False


@dataclass
class QRSVDSettings:
    enabled: bool = False
    cache_dir: str = "artifacts/cache/qr_svd"
    collect_subspace_comparisons: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/run"
    seed: int = 42
    epochs: int = 20
    train_batch_size: int = 16
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    logging_steps: int = 10
    dataloader_pin_memory: bool = True
    save_last_adapter: bool = True
    save_optimizer_state: bool = True
    save_tokenizer_with_adapter: bool = True


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    fp16: bool = True
    bf16: bool = False
    allow_tf32: bool = False
    deterministic: bool = False


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    lora: LoraSettings
    sam: SAMSettings
    qr_svd: QRSVDSettings
    training: TrainingConfig
    runtime: RuntimeConfig
    source_path: str | None = None

    def validate(self) -> None:
        supported_methods = {"qlora", "lora_sam", "random_subspace", "qr_svd"}
        if self.experiment.method not in supported_methods:
            raise ValueError(
                f"experiment.method must be one of {sorted(supported_methods)}"
            )
        expects_sam = self.experiment.method != "qlora"
        if self.sam.enabled != expects_sam:
            raise ValueError(
                f"sam.enabled must be {str(expects_sam).lower()} "
                f"for method={self.experiment.method!r}"
            )
        expects_qr_svd = self.experiment.method == "qr_svd"
        if self.qr_svd.enabled != expects_qr_svd:
            raise ValueError(
                f"qr_svd.enabled must be {str(expects_qr_svd).lower()} "
                f"for method={self.experiment.method!r}"
            )
        if not 0 < self.data.train_fraction <= 1:
            raise ValueError("data.train_fraction must be in (0, 1]")
        if self.data.task_name not in {"rte", "mrpc", "cola"}:
            raise ValueError("Encoder stage supports only rte, mrpc, and cola")
        if self.data.validation_source not in {"official", "inner_train"}:
            raise ValueError(
                "data.validation_source must be 'official' or 'inner_train'"
            )
        if not 0 < self.data.inner_validation_fraction < 1:
            raise ValueError("data.inner_validation_fraction must be in (0, 1)")
        if self.data.inner_validation_seed < 0:
            raise ValueError("data.inner_validation_seed must be non-negative")
        if self.data.local_dataset_dir is not None and not self.data.local_dataset_dir.strip():
            raise ValueError("data.local_dataset_dir must be a non-empty path")
        if not self.model.load_in_4bit:
            raise ValueError("This experiment line requires model.load_in_4bit=true")
        if self.model.quant_type not in {"nf4", "fp4"}:
            raise ValueError("model.quant_type must be 'nf4' or 'fp4'")
        if self.model.compute_dtype != "float16":
            raise ValueError("RTX 2080 Ti configuration requires compute_dtype=float16")
        if not self.runtime.fp16 or self.runtime.bf16:
            raise ValueError("RTX 2080 Ti configuration requires fp16=true and bf16=false")
        if self.lora.rank <= 0 or self.sam.subspace_rank <= 0:
            raise ValueError("LoRA and SAM subspace ranks must be positive")
        if self.sam.rho < 0:
            raise ValueError("sam.rho must be non-negative")
        if self.sam.random_basis_seed < 0:
            raise ValueError("sam.random_basis_seed must be non-negative")
        if not math.isfinite(self.sam.singular_value_exponent):
            raise ValueError("sam.singular_value_exponent must be finite")
        if self.sam.singular_value_exponent < 0:
            raise ValueError("sam.singular_value_exponent must be non-negative")
        if (
            self.experiment.method == "random_subspace"
            and self.sam.singular_value_exponent > 0
            and not self.sam.reuse_qr_singular_values
        ):
            raise ValueError(
                "Weighted random_subspace requires sam.reuse_qr_singular_values=true"
            )
        if (
            self.sam.reuse_qr_singular_values
            and self.experiment.method != "random_subspace"
        ):
            raise ValueError(
                "sam.reuse_qr_singular_values is valid only for random_subspace"
            )
        if self.training.gradient_accumulation_steps != 1:
            raise ValueError(
                "The current E2-E5 engine requires gradient_accumulation_steps=1"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        return result


def _construct(raw: dict[str, Any], source_path: str) -> ProjectConfig:
    required = {
        "experiment",
        "model",
        "data",
        "lora",
        "sam",
        "qr_svd",
        "training",
        "runtime",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    config = ProjectConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraSettings(**raw["lora"]),
        sam=SAMSettings(**raw["sam"]),
        qr_svd=QRSVDSettings(**raw["qr_svd"]),
        training=TrainingConfig(**raw["training"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        source_path=source_path,
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return _construct(raw, str(config_path))
