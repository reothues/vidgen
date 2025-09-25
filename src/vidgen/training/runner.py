from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .animatediff_trainer import train_animatediff
from .data import VideoDatasetConfig

__all__ = [
    "TrainingConfig",
    "ValidationInput",
    "ValidationConfig",
    "ModelConfig",
    "run_training",
]


@dataclass(slots=True)
class ModelConfig:
    provider: str = "animatediff"
    base_model: str = "runwayml/stable-diffusion-v1-5"
    motion_lora: Optional[str] = None
    dtype: str = "bf16"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        provider = str(data.get("provider", "animatediff")).lower()
        if provider != "animatediff":
            raise ValueError(f"Unsupported model provider '{provider}'. Only 'animatediff' is implemented.")

        base_model = str(data.get("base_model", "runwayml/stable-diffusion-v1-5"))
        motion_lora_raw = data.get("motion_lora")
        motion_lora = str(motion_lora_raw) if motion_lora_raw else None
        dtype = str(data.get("dtype", "bf16"))

        return cls(
            provider=provider,
            base_model=base_model,
            motion_lora=motion_lora,
            dtype=dtype,
        )


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 1337
    gradient_accumulation_steps: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    checkpoint_interval: Optional[int] = None
    validation_interval: Optional[int] = None
    mixed_precision: str = "bf16"
    compile: bool = False
    dataloader_workers: int = 4
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    gradient_checkpointing: bool = False
    enable_xformers: bool = False
    vae_slicing: bool = True
    vae_tiling: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            seed=int(data.get("seed", 1337)),
            gradient_accumulation_steps=max(1, int(data.get("gradient_accumulation_steps", 1))),
            batch_size=max(1, int(data.get("batch_size", 1))),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            max_train_steps=max(1, int(data.get("max_train_steps", 1000))),
            checkpoint_interval=_maybe_int(data.get("checkpoint_interval")),
            validation_interval=_maybe_int(data.get("validation_interval")),
            mixed_precision=str(data.get("mixed_precision", "bf16")),
            compile=bool(data.get("compile", False)),
            dataloader_workers=max(0, int(data.get("dataloader_workers", 4))),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            adam_beta1=float(data.get("adam_beta1", 0.9)),
            adam_beta2=float(data.get("adam_beta2", 0.999)),
            adam_weight_decay=float(data.get("adam_weight_decay", 1e-2)),
            adam_epsilon=float(data.get("adam_epsilon", 1e-8)),
            lr_scheduler=str(data.get("lr_scheduler", "constant")),
            lr_warmup_steps=max(0, int(data.get("lr_warmup_steps", 0))),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
            enable_xformers=bool(data.get("enable_xformers", False)),
            vae_slicing=bool(data.get("vae_slicing", True)),
            vae_tiling=bool(data.get("vae_tiling", False)),
        )


@dataclass(slots=True)
class ValidationInput:
    prompt: Optional[str] = None
    init_frame: Optional[str] = None


@dataclass(slots=True)
class ValidationConfig:
    inputs: List[ValidationInput] = field(default_factory=list)
    num_frames: int = 16
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ValidationConfig":
        data = data or {}

        inputs: List[ValidationInput] = []
        raw_inputs = data.get("inputs")
        if raw_inputs:
            if isinstance(raw_inputs, dict):
                raw_inputs = [raw_inputs]
            for item in raw_inputs:
                if isinstance(item, str):
                    inputs.append(ValidationInput(prompt=item))
                elif isinstance(item, dict):
                    inputs.append(
                        ValidationInput(
                            prompt=item.get("prompt"),
                            init_frame=item.get("init_frame") or item.get("initial_frame"),
                        )
                    )
        else:
            prompts = data.get("prompts") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            inputs = [ValidationInput(prompt=value) for value in prompts]

        return cls(
            inputs=inputs,
            num_frames=int(data.get("num_frames", 16)),
            guidance_scale=float(data.get("guidance_scale", 7.5)),
            num_inference_steps=int(data.get("num_inference_steps", 50)),
        )


def _build_dataset_config(data: Dict[str, Any]) -> VideoDatasetConfig:
    root_raw = data.get("root")
    if not root_raw:
        raise ValueError("dataset.root must be specified in the configuration")
    root = Path(root_raw)

    num_frames = int(data.get("max_frames") or data.get("num_frames") or 16)
    resolution = int(data.get("resolution", 512))
    prompt = str(data.get("prompt", ""))

    return VideoDatasetConfig(
        root=root,
        num_frames=num_frames,
        resolution=resolution,
        prompt=prompt,
    )


def run_training(
    *,
    config: Dict[str, Any],
    output_dir: Path,
    console: Optional[Console] = None,
) -> None:
    console = console or Console()

    dataset_cfg = _build_dataset_config(config.get("dataset", {}))
    model_cfg = ModelConfig.from_dict(config.get("model", {}))
    training_cfg = TrainingConfig.from_dict(config.get("training", {}))
    validation_cfg = ValidationConfig.from_dict(config.get("validation"))

    console.log("Starting AnimateDiff fine-tuning")

    artifacts = train_animatediff(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        validation_cfg=validation_cfg,
        output_dir=output_dir,
    )

    console.log(
        f"Training complete | steps={artifacts.global_step} | average_loss={artifacts.train_loss:.4f}",
    )


def _maybe_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
