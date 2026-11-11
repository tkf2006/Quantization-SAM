from __future__ import annotations

from dataclasses import dataclass

import torch

from qlora_qr_svd.config import ProjectConfig
from qlora_qr_svd.utils import trainable_parameter_summary


@dataclass
class EncoderBundle:
    model: torch.nn.Module
    tokenizer: object
    parameter_summary: dict[str, float | int]


def _device_map(device: str) -> dict[str, int]:
    if not device.startswith("cuda:"):
        raise ValueError("The configured experiment requires a CUDA device such as cuda:0")
    return {"": int(device.split(":", 1)[1])}


def load_quantized_encoder(config: ProjectConfig):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    compute_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.model.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.model.double_quant,
        # Despite the legacy name, Transformers' 4-bit quantizer also uses
        # this list when deciding which modules must remain unquantized.
        llm_int8_skip_modules=config.lora.modules_to_save,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        num_labels=config.model.num_labels,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map=_device_map(config.runtime.device),
    )
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(quantized_model, tokenizer, config: ProjectConfig) -> EncoderBundle:
    import bitsandbytes as bnb
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        quantized_model,
        use_gradient_checkpointing=config.model.gradient_checkpointing,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        modules_to_save=config.lora.modules_to_save,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    quantized_trainable = [
        name
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
        for parameter_name, parameter in module.named_parameters(recurse=False)
        if parameter.requires_grad
        for name in [f"{name}.{parameter_name}"]
    ]
    if quantized_trainable:
        raise RuntimeError(f"Frozen 4-bit weights unexpectedly became trainable: {quantized_trainable}")

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not any("lora_" in name for name in trainable_names):
        raise RuntimeError("No trainable LoRA parameters were created")
    if not any("classifier" in name for name in trainable_names):
        raise RuntimeError("The sequence-classification head is not trainable")

    return EncoderBundle(
        model=model,
        tokenizer=tokenizer,
        parameter_summary=trainable_parameter_summary(model),
    )

