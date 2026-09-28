from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.fixtures.generate import ffmpeg_available, generate_all

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pytest_configure(config) -> None:
    if not hasattr(config, "workerinput") and ffmpeg_available() and config.getoption("numprocesses", None):
        generate_all()


@pytest.fixture(scope="session")
def media_files() -> dict[str, Path]:
    if not ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return generate_all()


@pytest.fixture(scope="session")
def encoders() -> set[str]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return {line.split()[1] for line in out.splitlines() if len(line.split()) > 1 and line.startswith(" ")}


@pytest.fixture(scope="session")
def transnet() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transnetv2_pytorch")

