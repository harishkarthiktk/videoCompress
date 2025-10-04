# Project Debug Rules (Non-Obvious Only)
- FFmpeg errors logged to logs/{stem}_{error}.log with full command and output; partial outputs deleted on failure/timeout.
- GUI logs redirected to text widget via [GUITextHandler](compressVidGUI.py:464); file logs in logs/ with timestamps.
- GPU detection fails silently to CPU fallback; check platform-specific commands (wmic Windows, lspci Linux) for issues.
- No tests; debug FFmpeg subprocess logic by inspecting captured output in except blocks.
- Parallel mode (max_workers>1) lacks per-file progress; threading conflicts cause tqdm bar issues.