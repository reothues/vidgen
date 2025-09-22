"""Dataset preparation utilities for VidGen."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console

SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
)


@dataclass
class DatasetPreparationSettings:
    """Configuration collected from environment or CLI for dataset prep."""

    input_root: Path
    output_root: Path
    max_duration_seconds: float
    target_width: int
    target_height: int
    target_format: str
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    recursive: bool = True
    overwrite: bool = False
    model_output_width: Optional[int] = None
    model_output_height: Optional[int] = None
    console: Console = field(default_factory=Console)

    @property
    def target_resolution(self) -> Tuple[int, int]:
        return self.target_width, self.target_height

    @classmethod
    def from_env(
        cls,
        *,
        input_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
    ) -> "DatasetPreparationSettings":
        from ..utils.env import get_training_root

        env = os.environ
        root = Path(input_root or env.get("VIDGEN_PREP_INPUT_DIR") or (get_training_root() or Path.cwd()))
        if not root.exists():
            raise ValueError(f"Input directory does not exist: {root}")

        max_duration_raw = env.get("VIDGEN_MAX_VIDEO_DURATION_SECONDS") or env.get(
            "VIDGEN_MAX_VIDEO_LENGTH_SECONDS"
        )
        if not max_duration_raw:
            raise ValueError("Set VIDGEN_MAX_VIDEO_DURATION_SECONDS in your .env")
        try:
            max_duration = float(max_duration_raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("VIDGEN_MAX_VIDEO_DURATION_SECONDS must be numeric") from exc

        model_resolution_raw = env.get("VIDGEN_MODEL_OUTPUT_RESOLUTION")
        resolution_raw = env.get("VIDGEN_TARGET_RESOLUTION") or model_resolution_raw
        if not resolution_raw:
            raise ValueError("Set VIDGEN_TARGET_RESOLUTION or VIDGEN_MODEL_OUTPUT_RESOLUTION in your .env (e.g. 512x512)")
        width, height = parse_resolution(resolution_raw)

        model_resolution = None
        if model_resolution_raw:
            model_resolution = parse_resolution(model_resolution_raw)
            if (width, height) != model_resolution:
                width, height = model_resolution

        fmt = env.get("VIDGEN_TARGET_FORMAT")
        if not fmt:
            raise ValueError("Set VIDGEN_TARGET_FORMAT in your .env (e.g. mp4)")
        fmt = fmt.lower().lstrip(".")

        overwrite = env.get("VIDGEN_PREP_OVERWRITE", "false").lower() in {"1", "true", "yes"}

        output_dir = output_root or env.get("VIDGEN_PREP_OUTPUT_DIR")
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                output_path = root / output_path
        else:
            subdir = env.get("VIDGEN_PREPARED_DATASET_SUBDIR", "processed")
            output_path = root / subdir

        recursive = env.get("VIDGEN_PREP_RECURSIVE", "true").lower() not in {"0", "false", "no"}

        model_width, model_height = (model_resolution or (width, height))

        return cls(
            input_root=root,
            output_root=output_path,
            max_duration_seconds=max_duration,
            target_width=width,
            target_height=height,
            target_format=fmt,
            recursive=recursive,
            overwrite=overwrite,
            model_output_width=model_width,
            model_output_height=model_height,
        )


def parse_resolution(value: str) -> Tuple[int, int]:
    parts = value.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution '{value}'. Use WIDTHxHEIGHT (e.g. 512x512).")
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError as exc:
        raise ValueError(f"Resolution components must be integers: '{value}'") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Resolution must be positive non-zero integers")
    return width, height


@dataclass
class VideoMetadata:
    duration: float
    width: Optional[int]
    height: Optional[int]


@dataclass
class ProcessingSummary:
    processed: List[Tuple[Path, Path]] = field(default_factory=list)
    skipped: List[Tuple[Path, str]] = field(default_factory=list)
    failed: List[Tuple[Path, str]] = field(default_factory=list)

    def log(self, console: Console) -> None:
        if self.processed:
            console.print(f"[green]Processed {len(self.processed)} videos[/green]")
        if self.skipped:
            console.print(f"[yellow]Skipped {len(self.skipped)} videos[/yellow]")
        if self.failed:
            console.print(f"[red]Failed to process {len(self.failed)} videos[/red]")


def gather_video_files(settings: DatasetPreparationSettings) -> List[Path]:
    candidates: List[Path] = []
    skip_root = settings.output_root.resolve()
    iterator: Iterable[Path]
    if settings.recursive:
        iterator = settings.input_root.rglob("*")
    else:
        iterator = settings.input_root.glob("*")

    for item in iterator:
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        try:
            if skip_root in item.resolve().parents:
                continue
        except FileNotFoundError:
            continue
        candidates.append(item)
    return candidates


def probe_video(path: Path, settings: DatasetPreparationSettings) -> Optional[VideoMetadata]:
    cmd = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install ffmpeg to continue.") from exc
    except subprocess.CalledProcessError as exc:
        settings.console.print(f"[yellow]Skipping {path}: unable to probe video ({exc.stderr.strip()})[/yellow]")
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        settings.console.print(f"[yellow]Skipping {path}: ffprobe returned invalid JSON[/yellow]")
        return None

    streams = payload.get("streams") or []
    if not streams:
        settings.console.print(f"[yellow]Skipping {path}: ffprobe returned no streams[/yellow]")
        return None

    stream = streams[0]
    duration = stream.get("duration")
    try:
        duration_f = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_f = None

    if duration_f is None:
        settings.console.print(f"[yellow]Skipping {path}: missing duration metadata[/yellow]")
        return None

    width = stream.get("width")
    height = stream.get("height")

    return VideoMetadata(duration=duration_f, width=width, height=height)


def build_ffmpeg_command(
    source: Path,
    destination: Path,
    settings: DatasetPreparationSettings,
    _metadata: VideoMetadata,
) -> List[str]:
    width, height = settings.target_resolution
    aspect_ratio = f'{width}/{height}'
    scale_filter = (
        f"scale='if(gt(a,{aspect_ratio}),{height}*a,{width})'"
        f":'if(gt(a,{aspect_ratio}),{height},{width}/a)'"
        f':flags=lanczos'
    )
    vf_steps = [scale_filter, f'crop={width}:{height}']
    video_args = ["-vf", ",".join(vf_steps)]

    fmt = settings.target_format
    if fmt in {"mp4", "m4v", "mov"}:
        codec_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
        container_args: List[str] = ["-movflags", "+faststart"]
    elif fmt == "webm":
        codec_args = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32"]
        audio_args = ["-c:a", "libopus", "-b:a", "128k"]
        container_args = []
    else:
        codec_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
        container_args = []

    base_cmd = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(source),
        *video_args,
        *codec_args,
        *audio_args,
        *container_args,
        str(destination),
    ]
    return base_cmd


def prepare_dataset(
    settings: DatasetPreparationSettings,
    *,
    dry_run: bool = False,
    probe_fn: Optional[Callable[[Path, DatasetPreparationSettings], Optional[VideoMetadata]]] = None,
    run_fn: Optional[Callable[[Sequence[str]], subprocess.CompletedProcess]] = None,
) -> ProcessingSummary:
    probe_fn = probe_fn or probe_video
    run_fn = run_fn or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))

    settings.output_root.mkdir(parents=True, exist_ok=True)
    summary = ProcessingSummary()

    videos = gather_video_files(settings)
    if not videos:
        settings.console.print("[yellow]No candidate videos found for processing[/yellow]")
        return summary

    for video in videos:
        metadata = probe_fn(video, settings)
        if metadata is None:
            summary.skipped.append((video, "metadata"))
            continue

        if metadata.duration > settings.max_duration_seconds:
            summary.skipped.append((video, f"duration {metadata.duration:.2f}s exceeds limit"))
            continue

        rel_path = video.relative_to(settings.input_root)
        destination = settings.output_root / rel_path
        destination = destination.with_suffix(f".{settings.target_format}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not settings.overwrite:
            summary.skipped.append((video, "already processed"))
            continue

        if dry_run:
            summary.processed.append((video, destination))
            continue

        cmd = build_ffmpeg_command(video, destination, settings, metadata)
        try:
            result = run_fn(cmd)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found. Install ffmpeg to continue.") from exc

        if isinstance(result, subprocess.CompletedProcess) and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            summary.failed.append((video, stderr or "ffmpeg returned non-zero status"))
            continue

        summary.processed.append((video, destination))

    return summary


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)
