"""Pytest configuration and shared fixtures"""
import sys
import tempfile
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def temp_video_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_video_files(temp_video_dir):
    """Create mock video files in temp directory"""
    files = [
        temp_video_dir / "video1.mp4",
        temp_video_dir / "video2.mkv",
        temp_video_dir / "video3.avi",
        temp_video_dir / "document.pdf",  # Non-video file
        temp_video_dir / "video1_conv.mp4",  # Already converted
    ]

    for file in files:
        file.touch()

    return temp_video_dir


@pytest.fixture
def sample_ffprobe_output():
    """Sample ffprobe JSON output for testing"""
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "bit_rate": "5000000",
                "r_frame_rate": "30/1",
            },
            {"index": 1, "codec_type": "audio", "bit_rate": "128000"},
        ],
        "format": {
            "duration": "3600",
            "bit_rate": "5128000",
        },
    }
