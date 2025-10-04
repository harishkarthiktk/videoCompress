# Feature Parity Analysis: compressVid.py (CLI) vs. compressVidGUI.py (GUI)

## Overview
Both scripts implement video compression using FFmpeg with GPU acceleration support, targeting .mp4 output with bitrate reduction (default 70% of original), resolution downscaling (up to 50%), and audio bitrate minimization (min 64kbps). They share core logic for GPU detection, video info extraction, and FFmpeg command building. However, the GUI version duplicates much of the CLI's functionality but introduces deviations in configuration, progress tracking, error handling, and missing features. Parity is partial: core conversion works similarly, but GUI lacks CLI's flexibility, robustness, and some optimizations. No shared modules exist; code is duplicated, leading to maintenance risks.

## Gaps in GUI (Missing or Incomplete Relative to CLI)
The GUI version omits several CLI features, reducing flexibility and robustness:

- **Configuration Options**:
  - Compression factor: CLI configurable via --compression-factor (0-100%, default 70%); GUI hardcoded at 0.7 in [Config.COMPRESSION_FACTOR](compressVidGUI.py:58).
  - Move originals: CLI has --move-files (moves to original_files/ subdir after success); GUI lacks this entirely, leaving originals in place.
  - Verbose logging: Both supported, but GUI checkbox doesn't fully propagate to VideoProcessor (CLI uses arg for setup_logging).
  - Max workers: Supported in both (default 1), but GUI limited to 1-10 via spinbox; CLI unbounded (user-specified).

- **VideoInfo Extraction**:
  - CLI extracts nb_frames (from stream, fallback -1) and rotation (from side_data_list 'Display Matrix'); GUI extracts fps (via eval r_frame_rate) but omits nb_frames/rotation. This leads to inaccurate progress (GUI estimates frames as duration * fps, poor for VFR) and no rotation correction.

- **Progress Tracking**:
  - CLI: Advanced tqdm integration (outer file bar, inner FFmpeg bar via frame= or out_time_ms= from -progress pipe:1). Supports single/multi-file, sequential/parallel modes with fallbacks.
  - GUI: Basic ttk.Progressbar (file count total, per-file % via frame= from stderr only). No time-based fallback, no sub-progress, and estimation inaccuracies.

- **Timeout and Process Management**:
  - CLI: 3600s FFMPEG_TIMEOUT with proc.kill() and full logging to {stem}_timeout.log.
  - GUI: No timeout in [convert_video()](compressVidGUI.py:263) (Popen without timeout); vulnerable to indefinite hangs. No timeout-specific logs.

- **FFmpeg Command**:
  - No rotation handling in GUI's -vf (CLI prepends rotate=PI/2 etc. based on degrees), potentially causing incorrect orientation/scaling.
  - Parallel processing: GUI lacks CLI's overall progress bar nuance (tqdm vs. simple increment).

- **Error Logging**:
  - CLI: Per-file error logs ({stem}_ffmpeg_error.log with cmd + full output).
  - GUI: General logging to file/GUI, but no structured per-file error dumps; stderr read only after wait().

- **UI/UX**:
  - No recursion in file finding (both top-level only, but CLI warns on invalid dirs/files more explicitly).
  - No stop/cancel during processing (GUI has no button; CLI interruptible via KeyboardInterrupt).