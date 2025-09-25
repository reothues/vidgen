"""AnimateDiff fine-tuning utilities."""
from __future__ import annotations

import math
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AnimateDiffPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from .data import VideoSampleDataset, VideoDatasetConfig, video_collate_fn
if TYPE_CHECKING:
    from .runner import ModelConfig, TrainingConfig, ValidationConfig


@dataclass(slots=True)
class TrainerArtifacts:
    global_step: int
    train_loss: float


def _resolve_weight_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value == "fp16" or value == "float16":
        return torch.float16
    if value == "bf16" or value == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _build_dataloader(
    dataset_cfg: VideoDatasetConfig,
    training_cfg: TrainingConfig,
) -> DataLoader:
    dataset = VideoSampleDataset(dataset_cfg)
    return DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=video_collate_fn,
    )


def _encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: torch.nn.Module,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        embeddings = text_encoder(text_inputs_ids)[0]
    return embeddings.to(dtype)


def _prepare_latents(
    pixel_values: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, int, Tuple[int, int, int, int, int]]:
    batch_size, frames, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * frames, channels, height, width)
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * 0.18215
    latents = latents.to(device=device, dtype=dtype)
    latent_channels = latents.shape[1]
    latent_height = latents.shape[2]
    latent_width = latents.shape[3]

    latents = latents.view(batch_size, frames, latent_channels, latent_height, latent_width)
    original_shape = (batch_size, latent_channels, frames, latent_height, latent_width)
    latents = latents.permute(0, 2, 1, 3, 4).contiguous()
    latents = latents.view(batch_size * frames, latent_channels, latent_height, latent_width)

    return latents, frames, original_shape


def train_animatediff(
    *,
    dataset_cfg: VideoDatasetConfig,
    model_cfg: "ModelConfig",
    training_cfg: "TrainingConfig",
    validation_cfg: "ValidationConfig",
    output_dir: Path,
) -> TrainerArtifacts:
    set_seed(training_cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        mixed_precision=training_cfg.mixed_precision,
        log_with=None,
    )

    weight_dtype = _resolve_weight_dtype(model_cfg.dtype)

    auth_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("huggingface_hub_token")

    try:
        pipeline = AnimateDiffPipeline.from_pretrained(
            model_cfg.base_model,
            motion_module=model_cfg.motion_lora,
            torch_dtype=weight_dtype,
            use_auth_token=auth_token,
        )
    except Exception as err:
        hint = (
            "Unable to download model weights from Hugging Face. Make sure the repository exists, you "
            "have accepted any required licenses, and you are logged in (run `huggingface-cli login`)."
        )
        raise RuntimeError(hint) from err
    pipeline.set_progress_bar_config(disable=True)

    unet = pipeline.unet
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    if training_cfg.gradient_checkpointing:
        try:
            unet.enable_gradient_checkpointing()
        except AttributeError:
            pass

    if training_cfg.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            accelerator.print("[warning] xFormers attention not available; continuing without it.")

    if training_cfg.vae_slicing:
        try:
            vae.enable_slicing()
        except AttributeError:
            pass

    if training_cfg.vae_tiling:
        try:
            vae.enable_tiling()
        except AttributeError:
            pass

    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    train_dataloader = _build_dataloader(dataset_cfg, training_cfg)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=training_cfg.learning_rate,
        betas=(training_cfg.adam_beta1, training_cfg.adam_beta2),
        weight_decay=training_cfg.adam_weight_decay,
        eps=training_cfg.adam_epsilon,
    )

    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    lr_scheduler = get_scheduler(
        training_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=training_cfg.lr_warmup_steps,
        num_training_steps=training_cfg.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    global_step = 0
    running_loss = 0.0

    _ = validation_cfg  # placeholder until validation sampling is implemented

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_cfg.gradient_accumulation_steps)
    num_epochs = math.ceil(training_cfg.max_train_steps / num_update_steps_per_epoch)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device)
                prompts = batch["prompts"]

                latents, frames, original_shape = _prepare_latents(pixel_values, vae, accelerator.device, weight_dtype)
                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                    dtype=torch.long,
                )

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                model_input = scheduler.scale_model_input(noisy_latents, timesteps)

                encoder_hidden_states = _encode_prompt(
                    tokenizer,
                    text_encoder,
                    prompts,
                    accelerator.device,
                    weight_dtype,
                )
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(frames, dim=0)

                model_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = latents

                if model_pred.shape != target.shape:
                    diagnostics = {
                        "model_pred": tuple(model_pred.shape),
                        "target": tuple(target.shape),
                        "original_shape": original_shape,
                    }
                    raise RuntimeError(f"AnimateDiff mismatch -> {diagnostics}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), training_cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.detach().item()

                if (
                    training_cfg.checkpoint_interval
                    and global_step % training_cfg.checkpoint_interval == 0
                    and accelerator.is_main_process
                ):
                    accelerator.save_state(output_dir / f"checkpoint-{global_step:05d}")

            if global_step >= training_cfg.max_train_steps:
                break

        if global_step >= training_cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()

    unet = accelerator.unwrap_model(unet)
    pipeline.unet = unet.to(torch.float32)

    if accelerator.is_main_process:
        pipeline.to("cpu")
        pipeline.save_pretrained(output_dir / "pipeline")

    # Proactively release GPU memory before returning
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    avg_loss = running_loss / max(global_step, 1)

    return TrainerArtifacts(global_step=global_step, train_loss=avg_loss)
