import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_environment(env_file: Optional[Path] = None) -> None:
    """Load environment variables from .env and normalize keys.

    - Loads variables from the provided file or ".env" in CWD.
    - Mirrors common lowercase keys to expected uppercase variants.
    """
    if env_file is None:
        env_file = Path.cwd() / ".env"

    if env_file.exists():
        load_dotenv(env_file)

    # Normalize keys we expect (accept more ergonomic lowercase variants)
    pairs = {
        "training_set_location": "TRAINING_SET_LOCATION",
        "vidgen_training_set_location": "TRAINING_SET_LOCATION",
        "max_video_duration_seconds": "VIDGEN_MAX_VIDEO_DURATION_SECONDS",
        "max_video_length_seconds": "VIDGEN_MAX_VIDEO_DURATION_SECONDS",
        "vidgen_max_video_duration_seconds": "VIDGEN_MAX_VIDEO_DURATION_SECONDS",
        "target_resolution": "VIDGEN_TARGET_RESOLUTION",
        "vidgen_target_resolution": "VIDGEN_TARGET_RESOLUTION",
        "target_format": "VIDGEN_TARGET_FORMAT",
        "vidgen_target_format": "VIDGEN_TARGET_FORMAT",
        "prepared_dataset_subdir": "VIDGEN_PREPARED_DATASET_SUBDIR",
        "vidgen_prepared_dataset_subdir": "VIDGEN_PREPARED_DATASET_SUBDIR",
        "model_output_resolution": "VIDGEN_MODEL_OUTPUT_RESOLUTION",
        "vidgen_model_output_resolution": "VIDGEN_MODEL_OUTPUT_RESOLUTION",
    }
    for lower, upper in pairs.items():
        if lower in os.environ and upper not in os.environ:
            os.environ[upper] = os.environ[lower]


def get_training_root() -> Optional[Path]:
    path = os.environ.get("TRAINING_SET_LOCATION")
    return Path(path) if path else None
