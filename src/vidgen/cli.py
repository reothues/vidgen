from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import typer
from omegaconf import OmegaConf
from rich import print
from rich.panel import Panel
from rich.console import Console

from .datasets.preparation import (
    DatasetPreparationSettings,
    prepare_dataset as run_dataset_preparation,
)
from .utils.env import load_environment, get_training_root

app = typer.Typer(help="VidGen CLI: train and manage fine-tuning runs.")


def _resolve_env_path(value: str) -> str:
    if isinstance(value, str) and value.startswith("ENV:"):
        key = value.split(":", 1)[1]
        return str(Path(typer.get_app_dir("vidgen"))) if key == "APP_DIR" else (os.environ.get(key) or "")
    return value


@app.command()
def train(
    config: Path = typer.Option(Path("configs/default.yaml"), exists=True, help="Path to YAML config."),
    dry_run: bool = typer.Option(True, help="Validate setup without launching training."),
):
    """Validate environment, resolve dataset path, and prepare a run directory.

    This is a bootstrap training entrypoint. It validates configuration and
    environment and creates the output directory. Replace with actual training
    once models/datasets are integrated.
    """
    load_environment()

    cfg = OmegaConf.load(config)

    # Resolve dataset root from ENV:VAR indirection if used
    dataset_root = cfg.get("dataset", {}).get("root")
    if isinstance(dataset_root, str) and dataset_root.startswith("ENV:"):
        env_key = dataset_root.split(":", 1)[1]
        resolved = (get_training_root() if env_key == "TRAINING_SET_LOCATION" else Path(typer.get_app_dir("vidgen")))
        dataset_root = resolved
    else:
        dataset_root = Path(dataset_root) if dataset_root else get_training_root()

    if not dataset_root or not Path(dataset_root).exists():
        print(Panel.fit(
            f"[red]Dataset root not found[/red]\nExpected at: [bold]{dataset_root}[/bold]\n\n"
            "Set TRAINING_SET_LOCATION in .env or config.dataset.root.",
            title="Setup error",
        ))
        raise typer.Exit(code=2)

    out_dir = Path(cfg.get("run", {}).get("output_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(Panel.fit(
        "\n".join([
            "[green]VidGen setup validated[/green]",
            f"Config: [bold]{config}[/bold]",
            f"Dataset root: [bold]{Path(dataset_root)}[/bold]",
            f"Output dir: [bold]{out_dir}[/bold]",
            f"Mode: {'dry-run' if dry_run else 'train'}",
        ]),
        title="VidGen Train",
    ))

    # Placeholder: integrate actual training here.
    if dry_run:
        return

    print("Starting training loop (not yet implemented)...")


@app.command('prepare-dataset')
def prepare_dataset_command(
    input_dir: Optional[Path] = typer.Option(
        None,
        dir_okay=True,
        file_okay=False,
        exists=False,
        help='Directory containing raw videos (defaults to TRAINING_SET_LOCATION).',
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        dir_okay=True,
        file_okay=False,
        exists=False,
        help='Directory to store processed clips (defaults to .env setting).',
    ),
    recursive: Optional[bool] = typer.Option(
        None,
        '--recursive/--no-recursive',
        help='Override recursive search setting (defaults to .env value).',
    ),
    overwrite: Optional[bool] = typer.Option(
        None,
        '--overwrite/--no-overwrite',
        help='Override overwrite behaviour (defaults to .env value).',
    ),
    dry_run: bool = typer.Option(False, help='Show planned conversions without running ffmpeg.'),
):
    """Prepare video dataset according to .env limits and formatting rules."""
    load_environment()
    console = Console()

    try:
        settings = DatasetPreparationSettings.from_env(
            input_root=input_dir,
            output_root=output_dir,
        )
    except ValueError as err:
        console.print(f'[red]{err}[/red]')
        raise typer.Exit(code=2)

    settings.console = console
    if recursive is not None:
        settings.recursive = recursive
    if overwrite is not None:
        settings.overwrite = overwrite

    summary_lines = [
        '[cyan]Preparing dataset[/cyan]',
        f"Input: [bold]{settings.input_root}[/bold]",
        f"Output: [bold]{settings.output_root}[/bold]",
        f"Duration <= [bold]{settings.max_duration_seconds:.2f}s[/bold]",
        f"Resolution: [bold]{settings.target_width}x{settings.target_height}[/bold]",
        f"Model output: [bold]{settings.model_output_width}x{settings.model_output_height}[/bold]",
        f"Format: [bold].{settings.target_format}[/bold]",
        f"Recursive: [bold]{settings.recursive}[/bold]",
        f"Overwrite: [bold]{settings.overwrite}[/bold]",
        f"Mode: {'dry-run' if dry_run else 'execute'}",
    ]

    panel = Panel.fit(
        "\n".join(summary_lines),
        title="Dataset Prep",
    )
    console.print(panel)

    summary = run_dataset_preparation(settings, dry_run=dry_run)

    if dry_run and summary.processed:
        console.print()
        console.print("[blue]Dry run would process the following files:[/blue]")
        for source, destination in summary.processed:
            console.print(f"- {source} -> {destination}")

    summary.log(console)

    if summary.failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
