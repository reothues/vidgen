import subprocess
from pathlib import Path

import pytest

from vidgen.datasets.preparation import (
    DatasetPreparationSettings,
    VideoMetadata,
    gather_video_files,
    parse_resolution,
    prepare_dataset,
)


def test_parse_resolution_valid():
    assert parse_resolution('512x512') == (512, 512)
    assert parse_resolution('1280X720') == (1280, 720)


def test_parse_resolution_invalid():
    with pytest.raises(ValueError):
        parse_resolution('512')
    with pytest.raises(ValueError):
        parse_resolution('abcx200')


def test_settings_from_env(tmp_path, monkeypatch):
    raw_root = tmp_path / 'raw'
    raw_root.mkdir()

    monkeypatch.setenv('TRAINING_SET_LOCATION', str(raw_root))
    monkeypatch.setenv('VIDGEN_MAX_VIDEO_DURATION_SECONDS', '12.5')
    monkeypatch.setenv('VIDGEN_TARGET_RESOLUTION', '256x256')
    monkeypatch.setenv('VIDGEN_MODEL_OUTPUT_RESOLUTION', '512x288')
    monkeypatch.setenv('VIDGEN_TARGET_FORMAT', 'webm')
    monkeypatch.setenv('VIDGEN_PREPARED_DATASET_SUBDIR', 'prepared')
    monkeypatch.setenv('VIDGEN_PREP_RECURSIVE', 'false')
    monkeypatch.setenv('VIDGEN_PREP_OVERWRITE', 'true')

    settings = DatasetPreparationSettings.from_env()

    assert settings.input_root == raw_root
    assert settings.output_root == raw_root / 'prepared'
    assert settings.max_duration_seconds == pytest.approx(12.5)
    assert settings.target_resolution == (512, 288)
    assert settings.model_output_width == 512
    assert settings.model_output_height == 288
    assert settings.target_format == 'webm'
    assert settings.recursive is False
    assert settings.overwrite is True




def test_settings_from_env_defaults_to_model(monkeypatch, tmp_path):
    raw_root = tmp_path / 'root'
    raw_root.mkdir()

    monkeypatch.setenv('TRAINING_SET_LOCATION', str(raw_root))
    monkeypatch.setenv('VIDGEN_MODEL_OUTPUT_RESOLUTION', '640x360')
    monkeypatch.setenv('VIDGEN_MAX_VIDEO_DURATION_SECONDS', '9.5')
    monkeypatch.setenv('VIDGEN_TARGET_FORMAT', 'mp4')
    settings = DatasetPreparationSettings.from_env()

    assert settings.target_resolution == (640, 360)
    assert settings.model_output_width == 640
    assert settings.model_output_height == 360

def test_gather_video_files_skips_processed(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    processed = root / 'processed'
    processed.mkdir()

    keep = root / 'keep.mp4'
    keep.write_bytes(b'0')
    discard = processed / 'already.mp4'
    discard.write_bytes(b'0')

    settings = DatasetPreparationSettings(
        input_root=root,
        output_root=processed,
        max_duration_seconds=10,
        target_width=256,
        target_height=256,
        target_format='mp4',
    )

    files = gather_video_files(settings)

    assert keep in files
    assert all(processed not in path.parents for path in files)


def test_prepare_dataset_filters_and_runs(tmp_path):
    root = tmp_path / 'raw'
    root.mkdir()
    output = tmp_path / 'processed'

    short_clip = root / 'short.mov'
    short_clip.write_bytes(b'0')
    long_clip = root / 'long.mp4'
    long_clip.write_bytes(b'0')

    settings = DatasetPreparationSettings(
        input_root=root,
        output_root=output,
        max_duration_seconds=8,
        target_width=128,
        target_height=128,
        target_format='mp4',
    )

    metadata = {
        short_clip: VideoMetadata(duration=5.0, width=640, height=480),
        long_clip: VideoMetadata(duration=20.0, width=640, height=480),
    }

    commands = []

    def fake_probe(path: Path, _settings: DatasetPreparationSettings):
        return metadata.get(path)

    def fake_run(cmd):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    summary = prepare_dataset(
        settings,
        dry_run=False,
        probe_fn=fake_probe,
        run_fn=fake_run,
    )

    assert summary.processed == [(short_clip, output / 'short.mp4')]
    assert (long_clip, 'duration 20.00s exceeds limit') in summary.skipped
    assert commands, 'ffmpeg command should have been invoked'
    assert commands[0][0] == 'ffmpeg'

    vf_index = commands[0].index('-vf') + 1
    vf_filters = commands[0][vf_index]
    assert 'crop=128:128' in vf_filters
    assert 'pad=' not in vf_filters


def test_prepare_dataset_handles_failures(tmp_path):
    root = tmp_path / 'raw'
    root.mkdir()
    output = tmp_path / 'processed'

    clip = root / 'clip.mp4'
    clip.write_bytes(b'0')

    settings = DatasetPreparationSettings(
        input_root=root,
        output_root=output,
        max_duration_seconds=8,
        target_width=128,
        target_height=128,
        target_format='mp4',
    )

    def fake_probe(path: Path, _settings: DatasetPreparationSettings):
        return VideoMetadata(duration=5.0, width=640, height=480)

    def fake_run(cmd):
        return subprocess.CompletedProcess(cmd, 1, stderr='boom')

    summary = prepare_dataset(
        settings,
        dry_run=False,
        probe_fn=fake_probe,
        run_fn=fake_run,
    )

    assert summary.failed == [(clip, 'boom')]
    assert not summary.processed
