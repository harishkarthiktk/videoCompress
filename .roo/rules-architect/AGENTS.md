# Project Architecture Rules (Non-Obvious Only)
- All processing routed through FFmpeg subprocess calls; no abstraction layer, must preserve dynamic command building in [VideoConverter._build_ffmpeg_command()](compressVid.py:246) for GPU-specific encoders.
- GPU detection tightly coupled to platform (wmic/lspci/system_profiler); priority order (NVIDIA>Intel>AMD>Apple>CPU) hardcoded, altering requires updates in [GPUDetector.detect()](compressVid.py:67) across all scripts.
- Parallel processing with ThreadPoolExecutor lacks shared state or queue management; potential I/O races in logs/ dir, but no explicit locking.
- GUI (compressVidGUI.py) duplicates CLI logic in VideoProcessor/Converter classes; changes must sync both to prevent divergence in compression params or error handling.
- Fixed 3600s FFmpeg timeout per file; architecture assumes long-running processes, no adaptive scaling for large batches.
- Standalone scripts with no shared modules; Config dataclass duplicated, no central utils for formats or constants.