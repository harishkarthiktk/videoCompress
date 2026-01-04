# Video Processing Tools

This repository contains three Python scripts for handling common video-processing tasks using **FFmpeg**:

1. **`compressVidGUI.py`** – Graphical user interface for video compression with GPU acceleration.
2. **`compressVid.py`** – Compress and convert videos with GPU acceleration.
3. **`extractAudio.py`** – Extract audio tracks from video files into MP3 or AAC.

---

## Requirements

* Python 3.7+
* [FFmpeg](https://ffmpeg.org/download.html) (must be installed and available in your system PATH)

**Installation:**
```bash
pip install -r requirements.txt
```

**For development and testing:**
```bash
pip install -r requirements-dev.txt
```

**Optional features:**
* For the GUI: tkinter (included in standard Python installation), Pillow (PIL) for window icon
* For conversation detection: numpy, webrtcvad, whisper (included in requirements-dev.txt)
* For testing: pytest (included in requirements-dev.txt)

---

## Scripts

### 1. `compressVidGUI.py`

Provides an intuitive graphical interface for video compression.

Features include:

* Directory or file selection (with recursive search option)
* GPU detection display
* Overall and per-file progress bars with real-time logging
* Configurable parallel processing (1-10 workers)
* Adjustable compression factor (0-100%)
* Option to move original files to 'original_files' folder
* Error logging to timestamped files in logs/

**Examples:**

```bash
# Launch the GUI application
python3 compressVidGUI.py
```
Or open the `compressVidGUI.pyw` by double-clicking it.

---

### 2. `compressVid.py`

Compresses video files into `.mp4` format with GPU acceleration where available.
Features include:

* GPU support (NVIDIA, Intel, AMD, Apple Silicon) with CPU fallback (uses CRF for quality on CPU)
* Automatic bitrate and resolution optimization
* Configurable compression factor (default 70% of original bitrate)
* Skips already converted files
* Sequential or parallel batch processing
* Option to move original files to 'original_files' folder after conversion
* Adaptive compression based on video metadata (bitrate, resolution, duration, codec) for optimal quality/size balance – enabled by default
* Detailed error logging to logs/ on failures/timeouts

**Examples:**

```bash
# Compress all videos in a directory
python3 compressVid.py -w /path/to/videos

# Compress specific files
python3 compressVid.py -f video1.mkv video2.avi

# Verbose mode
python3 compressVid.py -w . -v

# Custom compression factor (50%) and move originals
python3 compressVid.py -w . -c 50 -m

# Parallel processing
python3 compressVid.py -w . --max-workers 4

# Note: Adaptive compression is enabled by default. For static compression (original behavior), modify Config.use_adaptive=False in utils.py.
```

---

### 3. `extractAudio.py`

Extracts high-quality audio tracks from video files, prioritizing lossless stream copy when possible (e.g., AAC to AAC) or adaptive high-bitrate re-encoding (320kbps MP3, 256kbps AAC, lossless FLAC). CLI mimics `compressVid.py` for consistency.

Features include:

* Output formats: `mp3`, `aac` (default), `flac` (lossless)
* Automatic probing with ffprobe for original audio details (codec, bitrate, sample rate, channels)
* Stream copy for compatible formats (zero quality loss); adaptive bitrates otherwise (preserves or upscales to hi-fi levels)
* Sequential or parallel batch processing with optional tqdm progress bars (outer for files, inner for FFmpeg in sequential mode)
* Skips already extracted files (e.g., `_audio.mp3`)
* Option to move original videos to 'original_files' folder after extraction
* Error logging to `logs/` (full FFmpeg output on failures/timeouts, 3600s timeout)
* No per-file progress in parallel mode to avoid threading issues

**Examples:**

```bash
# Extract AAC (default) from directory
python3 extractAudio.py -w /path/to/videos

# Lossless FLAC for specific files
python3 extractAudio.py -f video1.mkv video2.avi -o flac

# Verbose parallel processing with move originals
python3 extractAudio.py -w . -v --max-workers 2 -m

# High-quality MP3 re-encode
python3 extractAudio.py -w . -o mp3
```

---

## Notes
* All scripts automatically skip files that have already been processed (outputs named with suffix like `_conv.mp4` or `_audio.mp3`).
* Parallel processing can be enabled with `--max-workers` (CLI) or via GUI settings, but may be resource-intensive due to FFmpeg's CPU/GPU demands.
* Error logs for timeouts (1 hour default) or FFmpeg failures are saved to `logs/` directory with file stem prefixes.
* **New Feature (v2.0+)**: Adaptive compression analyzes video metadata (bitrate, resolution, duration, codec) to dynamically adjust bitrate and resolution scaling. High-bitrate/4K videos get more aggressive compression (up to 60% reduction), inefficient codecs get leniency, while low-bitrate/SD videos use conservative settings to preserve quality. CPU fallback uses CRF for better quality control. Falls back to static 70% compression if disabled. Logs show applied factors for transparency.
* For custom control, edit `utils.py` Config (e.g., `use_adaptive=False` for legacy behavior, adjust `MAX_BITRATE`, `MIN_AUDIO_BITRATE`, etc.).
* GUI supports recursive directory search and displays detected GPU type.
* All FFmpeg operations include `-movflags +faststart` for web-optimized MP4 output.