# Priority Implementation Plan

Detailed phase-based implementation plan for 10 critical improvements. Total effort: ~10 hours across 3 phases.

---

## Phase 1: Foundation (2.5 hours)

Building blocks that enable other improvements.

### Item 1: Create Constants Module (30 min)

**Files affected:** `constants.py` (new), `utils.py`, `isConversation.py`, `compressVidGUI.py`

**Steps:**

1. Create `/Users/harish-5102/Downloads/Programs/videoCompress/constants.py`:
   ```python
   # Video bitrate thresholds (Mbps)
   BITRATE_THRESHOLDS = {
       'HIGH': 15,      # >15 Mbps: aggressive compression
       'MEDIUM': 8,     # 8-15 Mbps: moderate
       'LOW': 2,        # <2 Mbps: conservative
   }

   # Resolution thresholds
   RESOLUTION_THRESHOLDS = {
       '4K': 2160,
       'HD': 1080,
       'SD': 720,
   }

   # Audio settings
   AUDIO_BITRATES = {
       'MP3_HIGH': '320k',
       'AAC_HIGH': '256k',
       'AAC_DEFAULT': '128k',
       'MIN': '128k',
   }

   # Quality settings
   QUALITY = {
       'HIGH_CRF': 23,
       'MEDIUM_CRF': 25,
       'LOW_CRF': 28,
       'MAX_BITRATE_LIMIT': 15000,     # 15 Mbps
       'MIN_AUDIO_BITRATE': 128,        # kbps
   }

   # Processing
   TIMEOUTS = {
       'GPU_DETECTION': 10,    # seconds
       'FFMPEG_CONVERSION': 3600,  # 1 hour
   }

   # File formats
   SUPPORTED_VIDEO_FORMATS = ('mp4', 'mkv', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mpg', 'mpeg', 'm4v')
   SUPPORTED_AUDIO_FORMATS = ('mp3', 'aac', 'flac')
   ```

2. In `utils.py`, replace all magic numbers:
   - Line 258-260: Replace with `BITRATE_THRESHOLDS`, `RESOLUTION_THRESHOLDS`
   - Line 290+: Replace CRF values with `QUALITY['*_CRF']`
   - Line 94: Update `Config.SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS`
   - Add at top: `from constants import *`

3. In `isConversation.py`, update bitrate comparisons:
   - Replace hardcoded `256000` with `AUDIO_BITRATES['AAC_HIGH']` converted to int

4. In `compressVidGUI.py`, update timeout:
   - Replace hardcoded timeouts with `TIMEOUTS` values

**Validation:** All tests still pass, no behavior change, only structure change.

---

### Item 2: Standardize Error Logging (45 min)

**Files affected:** `utils.py` (refactor `setup_logging`), `compressVid.py`, `extractAudio.py`, `isConversation.py`

**Current state:** Each file calls `setup_logging()` differently, logger named inconsistently.

**Steps:**

1. In `utils.py`, enhance `setup_logging()` function:
   ```python
   import logging
   import sys

   def setup_logging(tool_name: str, verbose: bool = False) -> logging.Logger:
       """
       Configure logging for all tools consistently.

       Args:
           tool_name: Name of tool (e.g., 'compressVid', 'extractAudio')
           verbose: Enable debug-level logging

       Returns:
           Configured logger instance
       """
       logger = logging.getLogger(tool_name)
       level = logging.DEBUG if verbose else logging.INFO
       logger.setLevel(level)

       # Console handler
       console_handler = logging.StreamHandler(sys.stdout)
       console_handler.setLevel(level)
       formatter = logging.Formatter(
           fmt='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
           datefmt='%Y-%m-%d %H:%M:%S'
       )
       console_handler.setFormatter(formatter)
       logger.addHandler(console_handler)

       # File handler (optional, logs to file)
       log_file = Path('logs') / f'{tool_name}.log'
       log_file.parent.mkdir(exist_ok=True)
       file_handler = logging.FileHandler(log_file)
       file_handler.setLevel(logging.DEBUG)
       file_handler.setFormatter(formatter)
       logger.addHandler(file_handler)

       return logger
   ```

2. Update `compressVid.py` main():
   ```python
   logger = setup_logging('compressVid', args.verbose)
   logger.info(f"Starting compression with {args.max_workers} workers")
   ```

3. Update `extractAudio.py` main():
   ```python
   logger = setup_logging('extractAudio', args.verbose)
   ```

4. Update `isConversation.py` main():
   ```python
   logger = setup_logging('isConversation', args.verbose)
   ```

5. Replace all `print()` statements with `logger.info()` or `logger.error()`

**Validation:** Run each tool with `-v` flag, check both console and `logs/` directory for consistent output.

---

### Item 3: Input Validation Helper (1 hour)

**Files affected:** `utils.py` (add new functions), `compressVid.py`, `extractAudio.py`, `isConversation.py`

**Steps:**

1. In `utils.py`, add validation functions:
   ```python
   class ValidationError(ValueError):
       """Custom exception for validation failures"""
       pass

   def validate_compression_factor(value: float) -> float:
       """Ensure compression_factor is 0-1 range"""
       try:
           cf = float(value)
           if not (0 <= cf <= 1):
               raise ValidationError(
                   f"Compression factor must be 0-1, got {cf}"
               )
           return cf
       except (TypeError, ValueError):
           raise ValidationError(
               f"Compression factor must be numeric, got {value}"
           )

   def validate_workers(value: int) -> int:
       """Ensure worker count is 1-10"""
       try:
           workers = int(value)
           if not (1 <= workers <= 10):
               raise ValidationError(
                   f"Workers must be 1-10, got {workers}"
               )
           return workers
       except (TypeError, ValueError):
           raise ValidationError(
               f"Workers must be integer, got {value}"
           )

   def validate_video_metadata(width, height, video_bitrate, audio_bitrate) -> bool:
       """Validate that video metadata is reasonable"""
       if width is None or height is None:
           raise ValidationError("Video must have width and height")
       if width <= 0 or height <= 0:
           raise ValidationError(
               f"Invalid resolution: {width}x{height}"
           )
       return True
   ```

2. In `compressVid.py` main(), add validation:
   ```python
   try:
       args.compression_factor = validate_compression_factor(args.compression_factor)
       args.max_workers = validate_workers(args.max_workers)
   except ValidationError as e:
       logger.error(str(e))
       sys.exit(1)
   ```

3. Similar updates for `extractAudio.py` and `isConversation.py`

4. Replace `shell=True` GPU detection calls in `GPUDetector.detect()`:
   ```python
   # OLD (line 70):
   output = subprocess.check_output('lspci | grep -i nvidia', shell=True)

   # NEW:
   lspci = subprocess.check_output(['lspci'])
   if b'nvidia' in lspci.lower():
       return 'NVIDIA'
   ```

**Validation:** Test with invalid inputs (compression_factor=1.5, workers=15), verify clear error messages.

---

## Phase 2: Core Refactoring (3 hours)

Depends on Phase 1. Eliminates duplication and improves consistency.

### Item 4: Extract Log File Writing (45 min)

**Files affected:** `utils.py` (new function + updates), `extractAudio.py` (remove duplication)

**Current duplication:** Lines in utils.py (517-521, 536-540) and extractAudio.py (192-196, 207-211) all write error logs identically.

**Steps:**

1. In `utils.py`, add utility function:
   ```python
   def write_error_log(
       log_dir: Path,
       file_stem: str,
       error_type: str,
       cmd: List[str],
       output_lines: List[str]
   ) -> Path:
       """
       Write standardized error log file.

       Args:
           log_dir: Directory to save log (created if missing)
           file_stem: Filename stem (e.g., 'video_name')
           error_type: Type of error (e.g., 'TIMEOUT', 'FFmpeg Error')
           cmd: Command that failed (list)
           output_lines: Output/error lines from command

       Returns:
           Path to written log file
       """
       log_dir = Path(log_dir)
       log_dir.mkdir(parents=True, exist_ok=True)

       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       log_file = log_dir / f'{file_stem}_{error_type}_{timestamp}.log'

       with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
           f.write(f"Error Type: {error_type}\n")
           f.write(f"Command: {' '.join(cmd)}\n")
           f.write(f"Timestamp: {datetime.now().isoformat()}\n")
           f.write("=" * 80 + "\n")
           f.write(''.join(output_lines))

       return log_file
   ```

2. In `utils.py` VideoConverter, replace lines 517-521:
   ```python
   # OLD:
   log_file = log_dir / f'{video_stem}_timeout_{timestamp}.log'
   with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
       # ... manual writing

   # NEW:
   log_file = write_error_log(log_dir, video_stem, 'TIMEOUT', cmd, output_lines)
   ```

3. Same for lines 536-540 (FFmpeg error)

4. In `extractAudio.py` lines 192-196 and 207-211:
   ```python
   # Replace with:
   log_file = write_error_log(log_dir, audio_stem, 'TIMEOUT', cmd, output_lines)
   ```

**Validation:** Run conversion that times out, verify log file created with consistent format.

---

### Item 5: Config Validation on Startup (40 min)

**Files affected:** `utils.py` (add validation method), all three CLI tools

**Steps:**

1. In `utils.py` Config class, add method:
   ```python
   @dataclass
   class Config:
       compression_factor: float = 0.7
       max_workers: int = 1
       use_adaptive: bool = True
       # ... existing fields ...

       def validate(self) -> None:
           """Validate all config values"""
           if not (0 <= self.compression_factor <= 1):
               raise ValueError(
                   f"compression_factor must be 0-1, got {self.compression_factor}"
               )
           if not (1 <= self.max_workers <= 10):
               raise ValueError(
                   f"max_workers must be 1-10, got {self.max_workers}"
               )
           if self.MAX_BITRATE < self.MIN_AUDIO_BITRATE:
               raise ValueError(
                   f"MAX_BITRATE ({self.MAX_BITRATE}) < "
                   f"MIN_AUDIO_BITRATE ({self.MIN_AUDIO_BITRATE})"
               )
   ```

2. In `compressVid.py` main():
   ```python
   config = Config(
       compression_factor=args.compression_factor / 100,
       max_workers=args.max_workers
   )
   try:
       config.validate()
   except ValueError as e:
       logger.error(f"Invalid configuration: {e}")
       sys.exit(1)
   ```

3. Similar updates for `extractAudio.py` and `isConversation.py`

**Validation:** Modify Config to invalid state, verify clear error before processing starts.

---

### Item 6: Reusable File Discovery (50 min)

**Files affected:** `utils.py` (new function, update VideoProcessor), `extractAudio.py`, `isConversation.py`

**Current state:** File discovery logic in 3 places with slight variations.

**Steps:**

1. In `utils.py`, create unified function:
   ```python
   def find_media_files(
       directory: Path,
       supported_formats: Tuple[str, ...],
       recursive: bool = False,
       exclude_suffix: str = None
   ) -> List[Path]:
       """
       Find media files matching criteria.

       Args:
           directory: Root directory to search
           supported_formats: Tuple of extensions (e.g., ('mp4', 'mkv'))
           recursive: If True, search subdirectories
           exclude_suffix: Skip files ending with this (e.g., '_conv.mp4')

       Returns:
           List of matching file paths
       """
       directory = Path(directory)
       if not directory.is_dir():
           raise ValueError(f"{directory} is not a directory")

       pattern = '**/*.{*}' if recursive else '*.{*}'
       files = []

       for fmt in supported_formats:
           glob_pattern = f"**/*.{fmt}" if recursive else f"*.{fmt}"
           files.extend(directory.glob(glob_pattern))

       # Filter out already-processed files
       if exclude_suffix:
           files = [f for f in files if not f.stem.endswith(exclude_suffix)]

       return sorted(set(files))
   ```

2. In `utils.py` VideoProcessor, replace `get_files()`:
   ```python
   # OLD: Lines with complex Path.glob logic
   # NEW:
   def get_files(self, directory: Path) -> List[Path]:
       return find_media_files(
           directory,
           supported_formats=self.config.SUPPORTED_FORMATS,
           recursive=self.recursive,
           exclude_suffix='_conv'
       )
   ```

3. In `extractAudio.py`, add similar usage:
   ```python
   video_files = find_media_files(
       args.working_dir,
       SUPPORTED_VIDEO_FORMATS,
       recursive=True,
       exclude_suffix='_audio'
   )
   ```

4. In `isConversation.py`, similar update

**Validation:** Run each tool, verify files discovered are same as before.

---

## Phase 3: Polish (4 hours)

Improves robustness and user experience. Depends on earlier phases.

### Item 7: Progress Bar Context Manager (1 hour)

**Files affected:** `utils.py` (new class + refactor), potentially `compressVidGUI.py`

**Current state:** Progress bars manually closed in 9+ places, boilerplate code.

**Steps:**

1. In `utils.py`, add context manager:
   ```python
   from contextlib import contextmanager

   @contextmanager
   def progress_bar_context(total, desc: str = None, disable: bool = False):
       """Context manager for automatic progress bar cleanup"""
       pbar = tqdm(total=total, desc=desc, disable=disable)
       try:
           yield pbar
       finally:
           pbar.close()
   ```

2. In `utils.py` VideoConverter.convert_video(), refactor line 483+:
   ```python
   # OLD:
   if self.progress_callback:
       pbar = tqdm(...)
   # ... lines of code ...
   if pbar:
       pbar.close()

   # NEW:
   if self.progress_callback:
       with progress_bar_context(total_frames, desc='Converting') as pbar:
           # ... code using pbar ...
           # Auto-closes after block
   ```

3. Similar updates for `extract_audio()` method

**Validation:** Interrupt process (Ctrl+C) mid-conversion, verify progress bars properly close.

---

### Item 8: Progress Callback Error Handling (30 min)

**Files affected:** `utils.py` (new wrapper), `compressVidGUI.py` (optional usage)

**Steps:**

1. In `utils.py`, add callback wrapper:
   ```python
   def safe_callback_wrapper(callback, timeout: float = 1.0):
       """Wrap callback to prevent errors from crashing processing"""
       def wrapper(*args, **kwargs):
           try:
               # Use timeout to prevent callback from blocking
               signal.alarm(int(timeout))
               callback(*args, **kwargs)
               signal.alarm(0)  # Cancel alarm
           except Exception as e:
               logger.warning(f"Progress callback failed: {e}")
       return wrapper
   ```

2. In `VideoProcessor.process_files()` where callbacks are called:
   ```python
   # OLD:
   self.overall_update_callback(processed_count)

   # NEW:
   if self.overall_update_callback:
       safe_callback = safe_callback_wrapper(self.overall_update_callback)
       safe_callback(processed_count)
   ```

**Validation:** Create callback that raises exception, verify processing continues.

---

### Item 9: Graceful Dependency Degradation (1 hour)

**Files affected:** `utils.py`, `compressVidGUI.py`

**Current state:** Missing PIL causes icon to fail silently. Missing tqdm would crash.

**Steps:**

1. In `utils.py`, add optional dependency handler:
   ```python
   class OptionalDependency:
       def __init__(self, import_name: str, package_name: str):
           self.import_name = import_name
           self.package_name = package_name
           self.available = False
           self.module = None
           self._try_import()

       def _try_import(self):
           try:
               self.module = __import__(self.import_name)
               self.available = True
           except ImportError:
               logger.warning(
                   f"Optional dependency '{self.package_name}' not available. "
                   f"Install with: pip install {self.package_name}"
               )

   PIL = OptionalDependency('PIL', 'Pillow')
   TQDM = OptionalDependency('tqdm', 'tqdm')
   ```

2. In `compressVidGUI.py` icon loading (line 30):
   ```python
   # OLD:
   from PIL import Image
   img = Image.open('src/images/icon_corrected.jpg')

   # NEW:
   if PIL.available:
       img = PIL.module.Image.open('src/images/icon_corrected.jpg')
       photo = ImageTk.PhotoImage(img)
       root.iconphoto(False, photo)
   else:
       logger.info("GUI will run without custom icon (PIL not installed)")
   ```

3. For tqdm, update to fallback:
   ```python
   if TQDM.available:
       from tqdm import tqdm
   else:
       # No-op tqdm replacement
       def tqdm(iterable, **kwargs):
           return iterable
   ```

**Validation:** Uninstall Pillow, run GUI - should work without icon. Uninstall tqdm, run CLI - should work without progress bars.

---

### Item 10: GPU Detection Subprocess Safety (50 min)

**Files affected:** `utils.py` GPUDetector class

**Current state:** Uses `shell=True` for GPU detection (security risk, unreliable).

**Steps:**

1. In `utils.py` GPUDetector.detect(), refactor for each platform:
   ```python
   def detect(self) -> str:
       """Detect GPU type using platform-specific commands (no shell=True)"""
       if sys.platform == 'win32':
           return self._detect_windows()
       elif sys.platform == 'darwin':
           return self._detect_macos()
       else:
           return self._detect_linux()

   def _detect_windows(self) -> str:
       """Windows GPU detection via WMI"""
       try:
           output = subprocess.check_output(
               ['powershell', '-Command',
                'Get-WmiObject Win32_VideoController | Select Name'],
               timeout=self.timeout
           ).decode().lower()
           if 'nvidia' in output:
               return 'NVIDIA'
           elif 'intel' in output:
               return 'Intel'
           elif 'amd' in output or 'radeon' in output:
               return 'AMD'
       except Exception as e:
           logger.debug(f"Windows GPU detection failed: {e}")
       return 'CPU'

   def _detect_linux(self) -> str:
       """Linux GPU detection via lspci"""
       try:
           output = subprocess.check_output(
               ['lspci'],
               timeout=self.timeout
           ).decode().lower()
           if 'nvidia' in output:
               return 'NVIDIA'
           elif 'intel' in output:
               return 'Intel'
           elif 'amd' in output or 'radeon' in output:
               return 'AMD'
       except Exception as e:
           logger.debug(f"Linux GPU detection failed: {e}")
       return 'CPU'

   def _detect_macos(self) -> str:
       """macOS GPU detection (already safe)"""
       try:
           output = subprocess.check_output(
               ['system_profiler', 'SPDisplaysDataType'],
               timeout=self.timeout
           ).decode().lower()
           if 'gpu' in output and 'apple' in output:
               return 'Apple'
       except Exception as e:
           logger.debug(f"macOS GPU detection failed: {e}")
       return 'CPU'
   ```

2. Remove all old shell=True calls (lines 70, 75, 80, etc.)

3. Add logging for each detection step:
   ```python
   logger.debug(f"Detected GPU: {gpu_type}")
   ```

**Validation:** Test on each platform, verify same GPU is detected, no subprocess injection risk.

---

## Testing Checklist

After each phase, verify:

- [ ] All three CLI tools run without errors
- [ ] GUI launches and displays correctly
- [ ] File processing works (dummy test files)
- [ ] No new dependencies introduced
- [ ] Error messages are clear
- [ ] Logs created in appropriate locations

---

## Timeline Summary

| Phase | Items | Effort | Focus |
|-------|-------|--------|-------|
| Phase 1 | Items 1-3 | 2.5 hrs | Foundation: constants, logging, validation |
| Phase 2 | Items 4-6 | 3 hrs | Refactoring: DRY, consistency, reusability |
| Phase 3 | Items 7-10 | 4 hrs | Polish: robustness, UX, security |
| **Total** | **10 items** | **~9.5 hrs** | Complete improvement suite |

**Recommended order:** Follow phases sequentially. Items within each phase are independent and can be done in any order.

---

## Notes for Implementer

1. **Run tests frequently** - After each item, run the tools to ensure no regressions
2. **Keep diffs small** - Easier to review and debug if changes are focused
3. **Use git commits** - One commit per item with descriptive message
4. **Update CLAUDE.md** - After Phase 1, update constants/logging sections if behavior changes
5. **Document changes** - Add docstrings to new functions/classes
6. **Test edge cases** - Invalid inputs, missing files, timeouts, etc.
