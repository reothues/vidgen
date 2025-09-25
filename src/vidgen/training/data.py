from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import torch
from torch.utils.data import Dataset

SUPPORTED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".m4v"}


def _gather_video_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Training dataset root not found: {root}")
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No video files discovered under {root}")
    return sorted(files)


def _load_prompt(path: Path, default_prompt: str) -> str:
    prompt_path = path.with_suffix(".txt")
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8").strip() or default_prompt
        except OSError:
            return default_prompt
    return default_prompt


def _select_frame_indices(frame_count: int, target: int) -> List[int]:
    if frame_count <= 0:
        return [0] * target
    if frame_count >= target:
        step = frame_count / target
        return [min(frame_count - 1, int(math.floor(i * step))) for i in range(target)]
    # Duplicate last frame when not enough frames are available
    indices = list(range(frame_count))
    while len(indices) < target:
        indices.append(frame_count - 1)
    return indices[:target]


def _load_frames(path: Path, num_frames: int, size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = _select_frame_indices(frame_count, num_frames)
        frames = []
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            if not success or frame is None:
                if frames:
                    frames.append(frames[-1])
                    continue
                raise RuntimeError(f"Failed to decode frame {index} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if size:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
            frames.append(frame)
        video = torch.stack(frames, dim=0)
        video = video * 2.0 - 1.0  # scale to [-1, 1]
        return video
    finally:
        cap.release()


@dataclass(slots=True)
class VideoDatasetConfig:
    root: Path
    num_frames: int
    resolution: int
    prompt: str = "A photo"


class VideoSampleDataset(Dataset):
    def __init__(self, config: VideoDatasetConfig) -> None:
        self.config = config
        self.files = _gather_video_files(config.root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        video = _load_frames(path, self.config.num_frames, self.config.resolution)
        prompt = _load_prompt(path, self.config.prompt)
        return {
            "pixel_values": video,
            "prompt": prompt,
            "path": str(path),
        }


def video_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    paths = [example["path"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "paths": paths,
    }
