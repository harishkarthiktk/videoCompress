# Project Coding Rules (Non-Obvious Only)
- GPU detection via [GPUDetector.detect()](compressVid.py:67) uses platform-specific commands (wmic Windows, lspci Linux, system_profiler macOS); fails silently to CPU fallback.
- Video info extraction with [VideoInfo.get_info()](compressVid.py:112) parses ffprobe JSON; rotation from side_data_list 'Display Matrix' (not standard rotation field).
- FFmpeg commands built dynamically in [VideoConverter._build_ffmpeg_command()](compressVid.py:246) with GPU-specific encoders (h264_nvenc NVIDIA, h264_qsv Intel, etc.); always include -movflags +faststart.
- Config uses dataclass with hardcoded SUPPORTED_FORMATS list; compression_factor applied to bitrates, but resolution scaled by fixed MAX_DOWNSCALE_PERCENT=0.5.
- Progress parsing in convert_video() relies on frame= or out_time_ms= from FFmpeg -progress pipe:1; falls back to simple bar if metrics unavailable.
- Parallel processing with ThreadPoolExecutor disables per-file tqdm bars to avoid threading conflicts; use pbar=None for worker calls.
- Error handling: Partial outputs deleted on FFmpeg failure/timeout; full command/output logged to logs/{stem}_{error}.log.
- No type hints beyond basic typing imports; error handling via broad except clauses with logging.