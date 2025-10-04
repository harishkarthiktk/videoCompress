# Issues and Recommendations: compressVid.py (CLI) vs. compressVidGUI.py (GUI)

## Potential Issues
- **Code Duplication**: ~80% overlap in classes (Config, GPUDetector, VideoInfo, VideoConverter) without shared utils.py. Changes (e.g., new encoder) must be manually synced, risking inconsistencies.
- **CPU Encoding Inconsistency**: CLI's libx264 uses -crf 23 -preset medium (quality-focused, no bitrate cap); GUI adds -b:v {target}kbps, which may cap quality unnecessarily or conflict with CRF.
- **Progress Inaccuracy**: GUI's frame estimation fails for variable frame rate (VFR), unknown fps, or non-standard streams; CLI's direct nb_frames/out_time_ms is more reliable. No handling for "no metrics" fallback in GUI.
- **Resource Overload**: Parallel FFmpeg (max_workers >1) in both can saturate GPU/CPU without limits; GUI's daemon threads may not terminate cleanly on window close.
- **Timeout Vulnerabilities**: GUI processes can hang indefinitely; CLI's kill() prevents this but logs/ dir may have race conditions in parallel (no file locking).
- **Platform/Dependency Edge Cases**: Both assume FFmpeg in PATH; GUI's optional PIL for icon warns but doesn't affect core. No validation for corrupted videos (CLI logs "No video stream"; GUI may crash on JSON parse).
- **Scalability/UI Freezes**: GUI threaded but heavy logging/progress updates could lag on many files; CLI's tqdm is non-blocking.
- **Security/Maintenance**: Hardcoded formats/encoders; no input sanitization (e.g., path injection in cmd). Duplication hinders updates (e.g., new GPU support).

## Recommendations for Achieving Parity
To close gaps and resolve issues while maintaining GUI usability:

1. **Refactor for Shared Code**:
   - Extract duplicated classes to utils.py (e.g., Config, GPUDetector, VideoInfo, VideoConverter).
   - Import in both scripts: from utils import *. This ensures single source of truth.

2. **Expose CLI Options in GUI**:
   - Add UI: Spinbox for compression_factor (0-100, default 70, update Config on start).
   - Checkbox for move_files: Implement in VideoConverter.convert_video (copy CLI logic: mkdir original_files/, rename on success).
   - Ensure verbose propagates: Pass self.verbose.get() to VideoProcessor.__init__.

3. **Align VideoInfo and Progress**:
   - Update GUI's [VideoInfo.get_info()](compressVidGUI.py:126) to match CLI: Add nb_frames (from 'nb_frames') and rotation (side_data_list parsing). Return tuple with these.
   - In convert_video: Prefer nb_frames for progress total; fallback to time (parse out_time_ms=). Use -progress pipe:1 (stdout) for consistent parsing like CLI.
   - For rotation: Modify _build_ffmpeg_command to prepend rotate filter (e.g., "rotate=PI/2," if 90°).

4. **Enforce Timeouts and Logging**:
   - In GUI's convert_video: Wrap Popen in try: ... except subprocess.TimeoutExpired: (3600s), proc.kill(), log to logs/{stem}_timeout.log (cmd + output_lines).
   - On failure: Always log to {stem}_ffmpeg_error.log (full cmd + stderr lines), matching CLI.
   - Create logs/ dir in VideoProcessor.__init__ (like CLI).

5. **Fix Command and Processing**:
   - CPU fallback: Remove -b:v from GUI's libx264 (use CRF-only like CLI for quality).
   - Add Stop button: Set self.stop_processing=True to break as_completed; cancel futures if possible.
   - Progress: For sequential, add indeterminate sub-bar or % label; for parallel, increment on completion like CLI.

6. **Enhance Validation and UX**:
   - File finding: Add recursion option (checkbox) using os.walk if needed, but start with CLI's iterdir.
   - Error popups: On init/failure, use messagebox for user-friendly alerts (e.g., "FFmpeg missing").
   - Thread safety: Use queue for log/progress updates to prevent UI freezes.

7. **Testing and Maintenance**:
   - Add tests/ dir with pytest: Test conversion on sample videos (different GPUs, rotations, VFR). Verify outputs match between CLI/GUI.
   - Version shared Config (e.g., dataclass with defaults).
   - Estimated effort: 4-6 hours for refactors; 2 hours for tests. Run both on same inputs post-fix to confirm parity.

Implementing these will achieve full feature parity, reduce duplication, and improve reliability.