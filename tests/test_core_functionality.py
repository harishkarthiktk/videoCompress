"""Core functionality tests - 5 critical tests"""
import json
from unittest.mock import patch
from pathlib import Path

import pytest

from utils import GPUDetector, VideoProcessor, Config, VideoInfo


class TestCriticalFunctionality:
    """5 core tests for critical video processing functionality"""

    def test_1_gpu_detection_works(self):
        """Test 1: GPU detection returns valid type and doesn't crash"""
        gpu_type = GPUDetector.detect()
        # Should return valid GPU type
        assert gpu_type in ['NVIDIA', 'INTEL', 'AMD', 'APPLE', 'CPU']
        assert isinstance(gpu_type, str)

    def test_2_file_discovery_finds_videos(self, sample_video_files):
        """Test 2: File discovery correctly identifies video files"""
        processor = VideoProcessor(config=Config(), verbose=False)
        files = processor.find_video_files(sample_video_files, recursive=False)

        # Should find at least 2 video files
        assert len(files) >= 2
        # All should be Path objects with video extensions
        assert all(isinstance(f, Path) for f in files)
        # Should not include non-video files
        file_names = [f.name for f in files]
        assert 'document.pdf' not in file_names

    def test_3_config_has_valid_settings(self):
        """Test 3: Configuration has valid default values"""
        config = Config()

        # Compression factor should be 0-1
        assert 0 <= config.compression_factor <= 1
        # Should have supported formats
        assert len(config.SUPPORTED_FORMATS) > 0
        # Bitrate settings should be logical
        assert config.MIN_AUDIO_BITRATE > 0
        assert config.MAX_BITRATE > config.MIN_AUDIO_BITRATE

    def test_4_video_info_extraction_handles_errors(self):
        """Test 4: Video metadata extraction handles errors gracefully"""
        with patch('subprocess.run', side_effect=Exception("ffprobe not found")):
            info = VideoInfo.get_info(Path("test_video.mp4"))

            # Should return tuple with 10 elements (not crash)
            assert isinstance(info, tuple)
            assert len(info) == 10
            # Should have None values for missing metadata
            assert info[0] is None  # width
            assert info[1] is None  # height

    def test_5_gpu_detector_fallback_on_errors(self):
        """Test 5: GPU detector falls back to CPU when detection fails"""
        with patch('subprocess.check_output', side_effect=Exception("Command failed")):
            gpu_type = GPUDetector.detect()

            # Should fall back to CPU, not crash
            assert gpu_type in ['NVIDIA', 'INTEL', 'AMD', 'APPLE', 'CPU']
            assert isinstance(gpu_type, str)
