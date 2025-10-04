# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Run Commands
- All scripts require FFmpeg in PATH; check with `ffmpeg -version` before running.
- No build/lint/test setup; run directly: `python3 compressVid.py -W /path/to/videos` (CLI compression), `python3 compressVidGUI.py` (GUI), `python3 extractAudio.py -W /path/to/videos --output-format aac` (audio extraction).
- Parallel processing via `--max-workers N` (default 1); resource-intensive, use cautiously as FFmpeg is CPU/GPU heavy.
- Scripts skip processing if output (_conv.mp4 or _audio.mp3/.m4a) exists; no force-overwrite flag.

## Project-Specific Conventions
- GPU acceleration auto-detected via [GPUDetector.detect()](compressVid.py:67) using platform commands (wmic on Windows, lspci on Linux, system_profiler on macOS); priority NVIDIA > Intel > AMD > Apple > CPU fallback.
- All FFmpeg calls use subprocess with 3600s timeout; errors logged to logs/ with full command/output; partial outputs deleted on failure.
- Compression targets 70% original bitrate (configurable), scales resolution down 50% max while preserving aspect ratio; audio min 64kbps.
- No requirements.txt; manual deps: tqdm (progress), Pillow (optional GUI icon); tkinter standard.
- Logs auto-created in logs/ with timestamps; GUI logs to text widget via [GUITextHandler](compressVidGUI.py:464).

## Gotchas
- No tests; changes risk breaking FFmpeg subprocess logic without validation.
- Scripts assume video files in supported formats; ffprobe used for info extraction, fails silently on invalid streams.
- Parallel mode (max_workers>1) lacks per-file progress bars to avoid threading issues.