# Video Processing Tools

This repository contains two Python scripts for handling common video-processing tasks using **FFmpeg**:

1. **`compressVid.py`** – Compress and convert videos with GPU acceleration.
2. **`extractAudio.py`** – Extract audio tracks from video files into MP3 or AAC.

---

## Requirements

* Python 3.7+
* [FFmpeg](https://ffmpeg.org/download.html) (must be installed and available in your system PATH)

---

## Scripts

### 1. `compressVid.py`

Compresses video files into `.mp4` format with GPU acceleration where available.
Features include:

* GPU support (NVIDIA, Intel, AMD, Apple Silicon) with CPU fallback
* Automatic bitrate and resolution optimization
* Skips already converted files
* Sequential or parallel batch processing

**Examples:**

```bash
# Compress all videos in a directory
python3 compressVid.py -w /path/to/videos

# Compress specific files
python3 compressVid.py -f video1.mkv video2.avi

# Verbose mode
python3 compressVid.py -w . -v
```

---

### 2. `extractAudio.py`

Extracts audio tracks from video files and saves them as MP3 or AAC.
Features include:

* Configurable output format (`mp3` or `aac`)
* Default bitrates: 192 kbps for MP3, 128 kbps for AAC
* Ability to exclude files by name
* Sequential or parallel batch processing

**Examples:**

```bash
# Extract AAC audio from all videos in a directory
python3 extractAudio.py -w /path/to/videos --output-format aac

# Extract MP3 audio from specific files
python3 extractAudio.py -f video1.mkv video2.avi --output-format mp3

# Exclude files containing "sample" in the name
python3 extractAudio.py -w . --exclude sample --output-format mp3
```

---

## Notes
* Both scripts automatically skip files that have already been processed.
* Parallel processing can be enabled with `--max-workers`, but may be resource-intensive.