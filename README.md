# Video Processing Tools

This repository contains three Python scripts for handling common video-processing tasks using **FFmpeg**:

1. **`compressVidGUI.py`** – Graphical user interface for video compression with GPU acceleration.
2. **`compressVid.py`** – Compress and convert videos with GPU acceleration.
3. **`extractAudio.py`** – Extract audio tracks from video files into MP3 or AAC.

---

## Requirements

* Python 3.7+
* [FFmpeg](https://ffmpeg.org/download.html) (must be installed and available in your system PATH)
* For the GUI: tkinter (included in standard Python installation), optional Pillow (PIL) for window icon

---

## Scripts

### 1. `compressVidGUI.py`

Provides an intuitive graphical interface for video compression.

Features include:

* Directory or file selection
* GPU detection display
* Progress bar and real-time logging
* Configurable parallel processing

**Examples:**

```bash
# Launch the GUI application
python3 compressVidGUI.py
```

---

### 2. `compressVid.py`

Compresses video files into `.mp4` format with GPU acceleration where available.
Features include:

* GPU support (NVIDIA, Intel, AMD, Apple Silicon) with CPU fallback
* Automatic bitrate and resolution optimization
* Skips already converted files
* Sequential or parallel batch processing

**Examples:**

```bash
# Compress all videos in a directory
python3 compressVid.py -W /path/to/videos

# Compress specific files
python3 compressVid.py -F video1.mkv video2.avi

# Verbose mode
python3 compressVid.py -W . -v
```

---

### 3. `extractAudio.py`

Extracts audio tracks from video files and saves them as MP3 or AAC.
Features include:

* Configurable output format (`mp3` or `aac`)
* Default bitrates: 192 kbps for MP3, 128 kbps for AAC
* Ability to exclude files by name
* Sequential or parallel batch processing

**Examples:**

```bash
# Extract AAC audio from all videos in a directory
python3 extractAudio.py -W /path/to/videos --output-format aac

# Extract MP3 audio from specific files
python3 extractAudio.py -F video1.mkv video2.avi --output-format mp3

# Exclude files containing "sample" in the name
python3 extractAudio.py -W . --exclude sample --output-format mp3
```

---

## Notes
* All scripts automatically skip files that have already been processed.
* Parallel processing can be enabled with `--max-workers` (CLI) or via GUI settings, but may be resource-intensive.