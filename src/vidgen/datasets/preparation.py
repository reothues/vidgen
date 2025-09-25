"""Dataset preparation utilities for VidGen."""
from __future__ import annotations

import json
import math
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

# Keep processed clips at the lowest FPS the fine-tuning pipeline can handle.
FINE_TUNE_MIN_FRAME_RATE = 8
DEFAULT_MAX_SEGMENT_DURATION = 5.0


def _format_seconds(value: float) -> str:
    formatted = f"{value:.3f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


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
    max_raw_duration_seconds: Optional[float] = None
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

        max_duration_raw = (
            env.get("VIDGEN_MAX_VIDEO_DURATION_SECONDS")
            or env.get("VIDGEN_MAX_VIDEO_LENGTH_SECONDS")
            or str(DEFAULT_MAX_SEGMENT_DURATION)
        )
        try:
            max_duration = float(max_duration_raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("VIDGEN_MAX_VIDEO_DURATION_SECONDS must be numeric") from exc
        if max_duration <= 0:
            raise ValueError("VIDGEN_MAX_VIDEO_DURATION_SECONDS must be greater than zero")

        raw_duration_limit: Optional[float] = None
        raw_duration_raw = env.get("VIDGEN_MAX_RAW_DURATION_SECONDS")
        if raw_duration_raw:
            try:
                raw_duration_limit = float(raw_duration_raw)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError("VIDGEN_MAX_RAW_DURATION_SECONDS must be numeric") from exc
            if raw_duration_limit <= 0:
                raw_duration_limit = None

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
            max_raw_duration_seconds=raw_duration_limit,
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


def _parse_duration_value(raw: object) -> Optional[float]:
    """Convert ffprobe duration values (including timecode strings) into seconds."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if not value or value.upper() == "N/A":
            return None
        try:
            return float(value)
        except ValueError:
            parts = value.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = parts
            elif len(parts) == 2:
                hours = "0"
                minutes, seconds = parts
            else:
                return None
            try:
                h = float(hours)
                m = float(minutes)
                s = float(seconds)
            except ValueError:
                return None
            return h * 3600 + m * 60 + s
    return None


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
        if item.name.startswith("._"):
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
    def build_command(*, force_matroska: bool, probe_boost: bool) -> List[str]:
        cmd_parts = [settings.ffprobe_binary, "-v", "error"]
        if probe_boost:
            cmd_parts.extend(["-probesize", "200M", "-analyzeduration", "200M"])
        if force_matroska:
            cmd_parts.extend(["-f", "matroska,webm"])
        cmd_parts.extend(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration,width,height,tags:format=duration:format_tags=duration,DURATION:stream_tags=duration,DURATION",
                "-of",
                "json",
                str(path),
            ]
        )
        return cmd_parts

    suffix = path.suffix.lower()
    is_matroska_like = suffix in {".webm", ".mkv"}

    attempts: List[Tuple[bool, bool]] = []
    seen: set[Tuple[bool, bool]] = set()

    def register(force_matroska: bool, probe_boost: bool) -> None:
        key = (force_matroska, probe_boost)
        if key not in seen:
            attempts.append(key)
            seen.add(key)

    register(False, False)
    if is_matroska_like:
        register(True, False)
    register(False, True)
    if is_matroska_like:
        register(True, True)

    last_error: Optional[str] = None
    success_stdout: Optional[str] = None

    for force_matroska, probe_boost in attempts:
        cmd = build_command(force_matroska=force_matroska, probe_boost=probe_boost)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffprobe not found. Install ffmpeg to continue.") from exc

        if result.returncode == 0:
            success_stdout = result.stdout
            break

        stderr = (result.stderr or "").strip()
        last_error = stderr or None

        lower_err = stderr.lower() if stderr else ""
        if is_matroska_like and not force_matroska and "moov atom not found" in lower_err:
            continue
        if not probe_boost and "invalid data found" in lower_err:
            continue
    else:
        message = last_error or "unknown ffprobe error"
        settings.console.print(f"[yellow]Skipping {path}: unable to probe video ({message})[/yellow]")
        return None

    if success_stdout is None:
        settings.console.print(
            f"[yellow]Skipping {path}: unable to probe video (ffprobe produced no output)[/yellow]"
        )
        return None

    stdout_payload = success_stdout

    try:
        payload = json.loads(stdout_payload)
    except json.JSONDecodeError:
        settings.console.print(f"[yellow]Skipping {path}: ffprobe returned invalid JSON[/yellow]")
        return None

    streams = payload.get("streams") or []
    if not streams:
        settings.console.print(f"[yellow]Skipping {path}: ffprobe returned no streams[/yellow]")
        return None

    stream = streams[0]
    duration_f = _parse_duration_value(stream.get("duration"))
    if duration_f is None:
        duration_f = _parse_duration_value((stream.get("tags") or {}).get("duration"))
    if duration_f is None:
        duration_f = _parse_duration_value((stream.get("tags") or {}).get("DURATION"))

    if duration_f is None:
        format_block = payload.get("format") or {}
        duration_f = _parse_duration_value(format_block.get("duration"))
        if duration_f is None:
            format_tags = format_block.get("tags") or {}
            duration_f = _parse_duration_value(format_tags.get("duration"))
            if duration_f is None:
                duration_f = _parse_duration_value(format_tags.get("DURATION"))

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
    *,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> List[str]:
    width, height = settings.target_resolution
    aspect_ratio = f'{width}/{height}'
    scale_filter = (
        f"scale='if(gt(a,{aspect_ratio}),{height}*a,{width})'"
        f":'if(gt(a,{aspect_ratio}),{height},{width}/a)'"
        f':flags=lanczos'
    )
    vf_steps = [scale_filter, f'crop={width}:{height}', f'fps={FINE_TUNE_MIN_FRAME_RATE}']
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

    base_cmd: List[str] = [settings.ffmpeg_binary, "-y", "-i", str(source)]
    if start_time is not None and start_time > 0:
        base_cmd.extend(["-ss", _format_seconds(start_time)])
    if duration is not None and duration > 0:
        base_cmd.extend(["-t", _format_seconds(duration)])
    base_cmd.extend([
        *video_args,
        *codec_args,
        *audio_args,
        *container_args,
        str(destination),
    ])
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

        max_raw_duration = settings.max_raw_duration_seconds
        if max_raw_duration is not None and metadata.duration > max_raw_duration:
            summary.skipped.append(
                (
                    video,
                    f"duration {metadata.duration:.2f}s exceeds raw limit {max_raw_duration:.2f}s",
                )
            )
            continue

        rel_path = video.relative_to(settings.input_root)
        base_destination = settings.output_root / rel_path
        base_destination = base_destination.with_suffix(f".{settings.target_format}")

        segment_duration = settings.max_duration_seconds
        if segment_duration <= 0:
            segment_duration = metadata.duration

        needs_segmentation = metadata.duration > segment_duration and segment_duration > 0
        if not needs_segmentation:
            segment_count = 1
        else:
            segment_count = max(1, math.ceil(metadata.duration / segment_duration))

        segments: List[Tuple[Path, Optional[float], Optional[float], int]] = []

        for idx in range(segment_count):
            start_time = idx * segment_duration if needs_segmentation else 0.0
            if start_time >= metadata.duration:
                break

            clip_duration = None
            if needs_segmentation:
                remaining = max(0.0, metadata.duration - start_time)
                clip_duration = min(segment_duration, remaining)
                if clip_duration <= 0.01:
                    continue

            if segment_count == 1:
                destination = base_destination
            else:
                stem = f"{base_destination.stem}_part{idx + 1:02d}"
                destination = base_destination.with_name(f"{stem}{base_destination.suffix}")

            if destination.exists() and not settings.overwrite:
                summary.skipped.append((destination, "already processed"))
                continue

            segments.append((destination, start_time if needs_segmentation else None, clip_duration, idx))

        if not segments:
            continue

        if dry_run:
            for destination, _, _, _ in segments:
                summary.processed.append((video, destination))
            continue

        for destination, start_time, clip_duration, segment_index in segments:
            destination.parent.mkdir(parents=True, exist_ok=True)

            cmd = build_ffmpeg_command(
                video,
                destination,
                settings,
                metadata,
                start_time=start_time,
                duration=clip_duration,
            )
            try:
                result = run_fn(cmd)
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg not found. Install ffmpeg to continue.") from exc

            if isinstance(result, subprocess.CompletedProcess) and result.returncode != 0:
                stderr = (result.stderr or "").strip()
                reason = stderr or "ffmpeg returned non-zero status"
                summary.failed.append((destination, f"segment {segment_index + 1}: {reason}"))
                continue

            if not destination.exists():
                summary.failed.append((destination, "ffmpeg reported success but output is missing"))
                continue

            summary.processed.append((video, destination))

    return summary


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)
