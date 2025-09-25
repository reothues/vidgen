# VidGen — Video Diffusion Fine‑Tuning Toolkit

VidGen is a small, practical scaffold to fine‑tune video generative models (e.g., AnimateDiff on SD1.5 with motion LoRA). This repo gives you a clean starting point: environment setup, a CLI, config files, and space for datasets/models so you can iterate quickly.

## Highlights
- Minimal CLI to validate your setup and config
- `.env` based dataset location resolution
- Config‑driven runs via YAML
- Make targets for install, lint, tests

## Requirements
- Tested on: Intel i9 Gen 10 + NVIDIA RTX 3080
- OS: Linux recommended (CUDA support)
- Python: 3.10+
- Disk: training set accessible via local path or mounted NAS

## Quickstart

1) Set your dataset location and preparation settings in `.env` (root of this repo):

```
training_set_location = '/media/asg/My Book'
vidgen_max_video_duration_seconds = 10
vidgen_target_resolution = '512x512'
vidgen_model_output_resolution = '512x512'
vidgen_target_format = 'mp4'
vidgen_prepared_dataset_subdir = 'processed'
```

2) Create a virtual environment and install (editable):

```
python -m venv vidgen-env
./vidgen-env/bin/pip install --upgrade pip
./vidgen-env/bin/pip install -e .[dev]
```

Alternatively with Make:

```
make install
```

3) Run a dry‑run to validate setup and config:

```
./vidgen-env/bin/python -m vidgen.cli train --config configs/default.yaml --dry-run
```

You should see confirmation of the resolved dataset root (pointing at the processed subset) and the output directory being prepared. Drop `--dry-run` to launch training once everything looks correct.

4) Prepare the dataset (dry run example):

```
./vidgen-env/bin/python -m vidgen.cli prepare-dataset --dry-run
```

Drop `--dry-run` to perform the conversions.

## Project Layout
- `src/vidgen/__init__.py` — package metadata
- `src/vidgen/cli.py` — Typer CLI (`train` command)
- `src/vidgen/utils/env.py` — `.env` loading and key normalization
- `configs/default.yaml` — baseline run configuration
- `Makefile` — install, lint, test, train helpers
- `requirements.txt`, `pyproject.toml` — dependencies and packaging

## Configuration
The default config is in `configs/default.yaml`. Key fields:

- `dataset.root`: may be a literal path or `ENV:TRAINING_SET_LOCATION` to pull from `.env`
- `run.output_dir`: where run artifacts will be stored
- `model.*`: placeholders for provider/base model/motion LoRA
- `training.*`: typical training hyperparameters

Example: configs/default.yaml

```
dataset:
  root: ENV:TRAINING_SET_LOCATION
run:
  name: baseline
  output_dir: runs/baseline
```

## CLI
After installing, you can invoke the CLI:

- Dry run (validate env + config):
  - `./vidgen-env/bin/python -m vidgen.cli train --config configs/default.yaml --dry-run`

- Fine-tune AnimateDiff (runs Accelerate + Diffusers training):
  - `./vidgen-env/bin/python -m vidgen.cli train --config configs/default.yaml`

- Prepare dataset (filters + resize + format):
  - `./vidgen-env/bin/python -m vidgen.cli prepare-dataset --dry-run`
  - Drop `--dry-run` to run the conversions.

Tip: You can also add the venv to your PATH or use the generated console script `vidgen` (entry point in `pyproject.toml`).

## Datasets
The `.env` file now drives dataset preparation. Key fields:

- `training_set_location`: root directory containing your raw videos
- `vidgen_max_video_duration_seconds`: maximum clip length (seconds) to keep
- `vidgen_target_resolution`: resolution to center-crop the dataset to (auto-aligned to model output)
- `vidgen_model_output_resolution`: final resolution expected by the fine-tuned model
- `vidgen_target_format`: container/extension for processed clips (e.g. `mp4`)
- `vidgen_prepared_dataset_subdir`: relative folder under the input root for processed files
- Optional: `vidgen_prep_recursive` (`true`/`false`) and `vidgen_prep_overwrite`

Run the CLI to filter, resize, and transcode videos:

```
./vidgen-env/bin/python -m vidgen.cli prepare-dataset --dry-run
```

Remove `--dry-run` when you're ready to generate the processed dataset.

## AnimateDiff Training
The `train` command now instantiates an Accelerate-backed fine-tuning loop for AnimateDiff on top of the base Stable Diffusion weights and motion module specified in `model.*`.

- The dataset loader expects processed videos under `<TRAINING_SET_LOCATION>/<VIDGEN_PREPARED_DATASET_SUBDIR>` (defaults to `processed/`) and preserves the original folder hierarchy.
- `training.batch_size` is per-device; the effective batch is `batch_size * gradient_accumulation_steps * num_devices`.
- For lower-memory GPUs you can set `training.gradient_checkpointing: true`, keep `training.batch_size: 1`, and leave `training.vae_slicing: true` to reduce the VRAM footprint.
- Optional: `training.enable_xformers: true` (requires xFormers install) further trims memory usage by swapping in memory-efficient attention.
- Checkpoints are written via `accelerator.save_state` every `training.checkpoint_interval` steps if provided. The final fine-tuned pipeline is exported to `<output_dir>/pipeline/`.
- Validation hooks accept optional `inputs` with `init_frame` paths, ready for future image-to-video sampling integration.

Ensure you have the required Hugging Face weights downloaded ahead of time (login if necessary with `huggingface-cli login`) and that PyTorch is built with CUDA enabled for GPU training.

## Models
Currently implemented:
- AnimateDiff (motion‑LoRA on SD1.5)
Future work will add additional providers as needed.

## Development
- Lint and format: `./vidgen-env/bin/ruff check src tests && ./vidgen-env/bin/ruff format src tests`
- Type check: `./vidgen-env/bin/mypy src`
- Tests: `./vidgen-env/bin/pytest`

You can also use:

```
make lint
make test
```

## Notes
- Some model weights may require accepting licenses on Hugging Face. Log in with `huggingface-cli login` if needed before training.
- CUDA setup is required for efficient training; ensure the appropriate PyTorch + CUDA wheels are installed for your GPU.

## Status
AnimateDiff fine-tuning is now wired up end-to-end: the CLI validates configuration, resolves the processed dataset location, and launches a working training loop.
