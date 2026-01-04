# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Python-based video processing tools suite** with three complementary tools for batch multimedia operations:

- **Video Compression** (`compressVid.py`, `compressVidGUI.py`) - Convert and compress videos to MP4 with GPU acceleration
- **Audio Extraction** (`extractAudio.py`) - Extract audio from videos in MP3, AAC, or FLAC formats
- **Conversation Detection** (`isConversation.py`) - Analyze video audio to detect human speech using OpenAI Whisper

The codebase is organized into ~2,500 lines of Python split across 5 main modules with shared utilities.

## Architecture

The project follows a modular, layered design:

- **CLI Layer** (`compressVid.py`, `extractAudio.py`, `isConversation.py`) - Command-line interfaces using argparse
- **GUI Layer** (`compressVidGUI.py`) - Tkinter-based graphical interface for video compression
- **Shared Utilities** (`utils.py`) - Core processing engine with FFmpeg/FFprobe wrappers, GPU detection, and configuration management
- **External Dependencies** - FFmpeg (system binary), Whisper, WebRTC VAD

Key classes in `utils.py`:
- `Config` - Centralized configuration (dataclass)
- `GPUDetector` - Platform-aware GPU detection (NVIDIA > Intel > AMD > Apple > CPU)
- `VideoInfo` - FFprobe wrapper for video metadata
- `VideoConverter` - FFmpeg execution with GPU optimization and adaptive compression
- `VideoProcessor` - High-level orchestrator for batch processing

## Common Development Tasks

**Install dependencies:**
```bash
pip install -r requirements.txt
# For development and testing: pip install -r requirements-dev.txt
# Includes pytest, numpy, Pillow, webrtcvad (for conversation detection), whisper
```

**Run video compression CLI:**
```bash
python3 compressVid.py -w /path/to/videos -c 70 --max-workers 4
```

**Run video compression GUI:**
```bash
python3 compressVidGUI.py
```

**Run audio extraction:**
```bash
python3 extractAudio.py -w /path/to/videos -o mp3 --max-workers 2
```

**Run conversation detection:**
```bash
python3 isConversation.py -w /path/to/videos --model base
```

## Key Design Patterns

- **Adaptive Compression** - `VideoConverter` analyzes video metadata (bitrate, resolution, duration, codec) to dynamically adjust compression parameters. Enabled by default in `Config.use_adaptive`; set to `False` for static 70% compression.
- **GPU Fallback** - Platform-aware GPU detection with automatic CPU fallback. GPU encoders used when available (h264_nvenc, h264_qsv, h264_amf, h264_videotoolbox).
- **Progress Tracking** - FFmpeg output parsing for real-time progress (frame-based or time-based). GUI uses callbacks; CLI uses tqdm progress bars.
- **Batch Processing** - `VideoProcessor` handles sequential or parallel processing with `ThreadPoolExecutor`. File discovery validates formats against `Config.SUPPORTED_FORMATS`.
- **Error Handling** - Failed conversions logged to timestamped files in `logs/` directory with full FFmpeg output. 3600s timeout with cleanup.

## Important Implementation Details

- **File Output Naming** - All scripts skip already-processed files:
  - `video.mp4` -> `video_conv.mp4` (compression)
  - `video.mp4` -> `video_audio.mp3/aac/flac` (extraction)
  - Implemented via filename checking in `VideoProcessor.get_files()`

- **FFmpeg Constraints**:
  - Requires system-level installation (not pip package)
  - Uses `-movflags +faststart` for web-optimized MP4
  - GPU encoder selection platform-specific; CPU fallback uses CRF for quality
  - Audio extraction uses stream copy when codecs match for zero quality loss

- **Adaptive Compression Logic** (in `VideoConverter.get_compression_params()`):
  - High bitrate (>15 Mbps) or 4K videos: aggressive compression (up to 60%)
  - Low bitrate (<2 Mbps) or SD videos: conservative settings to preserve quality
  - Inefficient codecs (MPEG-2, MPEG-4): leniency applied
  - CPU encoder uses CRF scaling instead of bitrate limits

- **Conversation Detection** (in `isConversation.py`):
  - Uses WebRTC VAD for initial voice detection (lightweight)
  - Falls back to OpenAI Whisper for transcription and dialogue analysis
  - Configurable Whisper models: tiny, base, small, medium, large (default: base)
  - Dry-run mode (`--dry-run`) performs VAD-only analysis without transcription

## Configuration and Customization

All configuration lives in `utils.py` `Config` dataclass:
- `compression_factor` - Percentage of original bitrate to target (default: 70)
- `max_workers` - Parallel processing threads (1-10)
- `use_adaptive` - Enable adaptive compression (default: True)
- `MAX_BITRATE`, `MIN_AUDIO_BITRATE` - Bitrate limits
- `SUPPORTED_FORMATS` - Tuple of file extensions to process

To customize, modify `Config` class instantiation or pass arguments via CLI flags.

## Testing and Debugging

- Use `-v` or `--verbose` flag for debug-level logging
- FFmpeg error logs saved to `logs/` on failures (timestamp-prefixed files)
- GPU detection can be tested via `GPUDetector().detect()` in Python REPL
- Progress bar output controlled by tqdm; suppress with `--no-progress` if needed (not all tools)

## System Requirements

- Python 3.7+
- FFmpeg and FFprobe in system PATH
- ~100 MB free space for Whisper models (base ~140 MB, medium ~1.4 GB for conversation detection)
- Tkinter (included with Python, required for GUI)
