#!/usr/bin/env python3
"""
Video Compression Script with GPU Acceleration

This script converts videos to .mp4 format with optimal compression using
hardware acceleration when available.

Features:
- GPU acceleration support (NVIDIA, Intel, AMD, Apple Silicon)
- Configurable working directory or specific file processing
- Skips already converted files
- Intelligent bitrate and resolution optimization
- Progress tracking with tqdm (overall files and per-file FFmpeg progress) and error handling
"""

import os
import sys
import subprocess
import platform
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


# Configuration constants
import tqdm
from tqdm import tqdm

@dataclass
class Config:
    """Configuration settings for video compression"""
    SUPPORTED_FORMATS = (
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    )
    MAX_BITRATE = 10000  # Max total bitrate in kbps (10 Mbps)
    COMPRESSION_FACTOR = 0.7  # Reduce bitrate by 30%
    MAX_DOWNSCALE_PERCENT = 0.5  # 20% max resolution reduction
    OUTPUT_SUFFIX = "_conv"
    MIN_AUDIO_BITRATE = 64 # Minimum audio bitrate in kbps


# Setup logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class GPUDetector:
    """Handles GPU detection across different platforms"""

    @staticmethod
    def detect() -> str:
        """Detect available GPU type with priority: NVIDIA > Intel > AMD > Apple > CPU"""
        system = platform.system()
        gpu_type = "CPU"

        try:
            if system == "Windows":
                output = subprocess.check_output(
                    "wmic path win32_VideoController get Name",
                    shell=True, text=True, timeout=10
                )
            elif system == "Linux":
                output = subprocess.check_output(
                    "lspci | grep -i vga",
                    shell=True, text=True, timeout=10
                )
            elif system == "Darwin":  # macOS
                output = subprocess.check_output(
                    "system_profiler SPDisplaysDataType",
                    shell=True, text=True, timeout=10
                )
            else:
                return gpu_type

            output = output.lower()

            # Priority order detection
            if "nvidia" in output:
                gpu_type = "NVIDIA"
            elif "intel" in output:
                gpu_type = "INTEL"
            elif "amd" in output or "radeon" in output:
                gpu_type = "AMD"
            elif "apple" in output or "m1" in output or "m2" in output or "m3" in output:
                gpu_type = "APPLE"

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
            logging.warning(f"GPU detection failed: {e}")

        return gpu_type


class VideoInfo:
    """Handles video information extraction"""

    @staticmethod
    def get_info(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], bool, Optional[int], Optional[float], Optional[int]]:
        """
        Extract video resolution, bitrates, audio presence, duration, and frame count using ffprobe

        Returns:
            Tuple of (width, height, video_bitrate_kbps, has_audio, audio_bitrate_kbps, duration_seconds, nb_frames)
            or (None, None, None, False, None, None, None) on error
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_streams",
                "-of", "json",
                str(file_path)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True
            )

            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            format_info = data.get('format', {})

            video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)

            if not video_stream:
                logging.error(f"No video stream found in {file_path}")
                return None, None, None, False, None

            width_str = video_stream.get('width')
            width = int(width_str) if width_str is not None else None
            height_str = video_stream.get('height')
            height = int(height_str) if height_str is not None else None
            video_bitrate_str = video_stream.get('bit_rate')
            video_bitrate = None
            if video_bitrate_str is not None:
                try:
                    video_bitrate = int(video_bitrate_str) // 1000
                except (ValueError, TypeError):
                    video_bitrate = None

            has_audio = audio_stream is not None
            audio_bitrate = None
            if has_audio and 'bit_rate' in audio_stream:
                audio_bitrate_str = audio_stream['bit_rate']
                try:
                    audio_bitrate = int(audio_bitrate_str) // 1000
                except (ValueError, TypeError):
                    audio_bitrate = None

            # Extract duration from format
            duration_str = format_info.get('duration')
            duration = float(duration_str) if duration_str else None

            # Extract nb_frames from video stream (may be -1 if unknown)
            nb_frames_str = video_stream.get('nb_frames') if video_stream else None
            nb_frames = -1
            if nb_frames_str is not None:
                try:
                    nb_frames = int(nb_frames_str)
                except (ValueError, TypeError):
                    nb_frames = -1

            return width, height, video_bitrate, has_audio, audio_bitrate, duration, nb_frames

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get video info for {file_path}: {e}")
            return None, None, None, False, None


class VideoConverter:
    """Handles video conversion with GPU acceleration"""

    def __init__(self, gpu_type: str, config: Config):
        self.gpu_type = gpu_type
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _calculate_target_params(self, width: int, height: int, video_bitrate: Optional[int], audio_bitrate: Optional[int]) -> Tuple[int, int, int, int]:
        """Calculate target resolution and bitrates"""
        # Handle missing bitrates
        if video_bitrate is None or video_bitrate <= 0:
            target_video_bitrate = self.config.MAX_BITRATE
        else:
            target_video_bitrate = int(video_bitrate * self.config.COMPRESSION_FACTOR)

        if audio_bitrate is None or audio_bitrate <= 0:
            target_audio_bitrate = 128
        else:
            target_audio_bitrate = max(int(audio_bitrate * self.config.COMPRESSION_FACTOR), self.config.MIN_AUDIO_BITRATE)

        # Print warnings for low bitrates
        if target_video_bitrate < 500:
            self.logger.warning(
                f"Video bitrate will be compressed to {target_video_bitrate}kbps, which is below the "
                "suggested 500kbps threshold and may result in noticeable quality loss."
            )
        if target_audio_bitrate < 128:
            self.logger.warning(
                f"Audio bitrate will be compressed to {target_audio_bitrate}kbps, which is below the "
                "suggested 128kbps threshold and may result in noticeable quality loss."
            )

        # Calculate new resolution (maximum downscale)
        scale_factor = 1 - self.config.MAX_DOWNSCALE_PERCENT
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure even dimensions for better encoding compatibility
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        return new_width, new_height, target_video_bitrate, target_audio_bitrate

    def _build_ffmpeg_command(self, input_file: Path, output_file: Path,
                             new_width: int, new_height: int, target_video_bitrate: int, target_audio_bitrate: int) -> List[str]:
        """Build FFmpeg command based on GPU type"""
        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vf", f"scale='min(iw,{new_width})':'min(ih,{new_height})':force_original_aspect_ratio=decrease",
            "-progress", "pipe:1"
        ]

        # Select encoder based on GPU type
        if self.gpu_type == "NVIDIA":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "INTEL":
            cmd.extend(["-c:v", "h264_qsv", "-preset", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "AMD":
            cmd.extend(["-c:v", "h264_amf", "-quality", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "APPLE":
            cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", f"{target_video_bitrate}k"])
        else:  # CPU fallback
            cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-b:v", f"{target_video_bitrate}k"])

        # Audio settings and output
        cmd.extend([
            "-c:a", "aac", "-b:a", f"{target_audio_bitrate}k",
            "-movflags", "+faststart",  # Web optimization
        ])
        cmd.append(str(output_file))

        return cmd

    def convert_video(self, input_file: Path, pbar: tqdm = None) -> bool:
        """
        Convert a single video file with optional progress bar using FFmpeg.

        Progress is tracked via frame count or time elapsed, parsed from FFmpeg's
        -progress output. Falls back to no bar if metrics unavailable.

        Args:
            input_file: Path to input video file
            pbar: Optional tqdm progress bar for this file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate output filename
            output_file = input_file.parent / f"{input_file.stem}{self.config.OUTPUT_SUFFIX}.mp4"

            # Skip if already exists
            if output_file.exists():
                self.logger.info(f"Skipping {input_file.name} - already converted")
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return True

            # Get video info
            width, height, video_bitrate, has_audio, audio_bitrate, duration, nb_frames = VideoInfo.get_info(input_file)
            if width is None or height is None:
                self.logger.error(f"Could not determine video dimensions for {input_file}")
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return False

            # Determine progress mode (prefer frames, fallback to time)
            use_frame = nb_frames is not None and nb_frames > 0
            use_time = duration is not None and duration > 0
            if not use_frame and not use_time:
                self.logger.warning(f"No reliable progress metrics available for {input_file.name} (duration: {duration}, frames: {nb_frames}). Proceeding with simple completion bar.")

            # Setup progress bar if provided
            if pbar:
                if use_frame:
                    pbar.total = nb_frames
                    pbar.desc = f"Processing {input_file.name} (frames)"
                elif use_time:
                    pbar.total = 100  # Percentage
                    pbar.desc = f"Processing {input_file.name} (time)"
                pbar.refresh()

            # Calculate target parameters
            new_width, new_height, target_video_bitrate, target_audio_bitrate = self._calculate_target_params(width, height, video_bitrate, audio_bitrate)

            self.logger.info(
                f"Converting {input_file.name} -> {output_file.name} "
                f"({width}x{height} -> {new_width}x{new_height}, "
                f"Video: {video_bitrate or 'unknown'}kbps -> {target_video_bitrate}kbps, "
                f"Audio: {audio_bitrate or 'unknown'}kbps -> {target_audio_bitrate}kbps, "
                f"Duration: {duration}s, Frames: {nb_frames}) "
                f"using {self.gpu_type}"
            )

            # Build and execute FFmpeg command
            cmd = self._build_ffmpeg_command(input_file, output_file, new_width, new_height, target_video_bitrate, target_audio_bitrate)

            # Use Popen for streaming progress output
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            current_frame = 0
            current_time_ms = 0
            try:
                for line in iter(proc.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        # Parse FFmpeg progress output (key=value format)
                        if pbar:
                            if use_frame and '=' in line and line.startswith('frame='):
                                try:
                                    frame_str = line.split('=')[1].split(' ')[0]
                                    frame = int(frame_str)
                                    if frame > current_frame:
                                        pbar.update(frame - current_frame)
                                        current_frame = frame
                                except (ValueError, IndexError):
                                    pass  # Ignore parse errors
                            elif use_time and '=' in line and line.startswith('out_time_ms='):
                                try:
                                    time_str = line.split('=')[1].split(' ')[0]
                                    time_ms = int(time_str)
                                    if time_ms > current_time_ms:
                                        percentage = min((time_ms / (duration * 1000)) * 100, 100)
                                        pbar.update(percentage - pbar.n)
                                        current_time_ms = time_ms
                                except (ValueError, IndexError):
                                    pass  # Ignore parse errors
                proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                self.logger.error(f"Conversion timeout for {input_file}")
                if pbar:
                    pbar.n = pbar.total  # Mark as complete/failed
                    pbar.close()
                if output_file.exists():
                    output_file.unlink()
                return False

            if proc.returncode != 0:
                error_msg = ' '.join(output_lines[-10:]) if output_lines else "Unknown error"
                self.logger.error(f"FFmpeg failed for {input_file}: {error_msg}")
                if pbar:
                    pbar.n = pbar.total  # Mark as failed
                    pbar.close()
                # Clean up partial file
                if output_file.exists():
                    output_file.unlink()
                return False

            if pbar:
                pbar.n = pbar.total
                pbar.close()

            self.logger.info(f"Successfully converted {input_file.name}")
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error converting {input_file}: {e}")
            if pbar:
                pbar.n = pbar.total
                pbar.close()
            return False


class VideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.logger = setup_logging(verbose)
        self.gpu_type = GPUDetector.detect()
        self.converter = VideoConverter(self.gpu_type, config)

        self.logger.info(f"Detected GPU: {self.gpu_type}")

    def find_video_files(self, directory: Path) -> List[Path]:
        """Find all supported video files in directory"""
        video_files = []

        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory}")
            return video_files

        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix in self.config.SUPPORTED_FORMATS:
                    video_files.append(file_path)

            self.logger.info(f"Found {len(video_files)} video files in {directory}")
            return sorted(video_files)

        except PermissionError:
            self.logger.error(f"Permission denied accessing directory: {directory}")
            return video_files

    def validate_files(self, file_paths: List[str]) -> List[Path]:
        """Validate and convert file paths to Path objects"""
        valid_files = []

        for file_str in file_paths:
            file_path = Path(file_str)

            if not file_path.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                continue

            if not file_path.is_file():
                self.logger.warning(f"Not a file: {file_path}")
                continue

            if file_path.suffix not in self.config.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {file_path}")
                continue

            valid_files.append(file_path)

        return valid_files

    def process_files(self, files: List[Path], max_workers: int = 1) -> None:
        """
        Process video files sequentially or in parallel with tqdm progress tracking.

        For sequential (max_workers=1):
        - Single file: One progress bar for FFmpeg processing.
        - Multiple files: Outer bar for overall files, nested inner bar for each file's FFmpeg progress.

        For parallel (max_workers>1):
        - Overall bar for file completion (no per-file bars to avoid threading issues).
        - Individual files process without bars (pbar=None).

        Note: Requires 'tqdm' library: pip install tqdm
        """
        if not files:
            self.logger.warning("No files to process")
            return

        successful = 0
        failed = 0

        if max_workers == 1:
            # Sequential processing with progress bars
            if len(files) == 1:
                # Single file: simple progress bar
                pbar = tqdm(total=1, position=0, desc=f"Processing {files[0].name}")
                success = self.converter.convert_video(files[0], pbar=pbar)
                if success:
                    successful += 1
                else:
                    failed += 1
            else:
                # Multiple files: outer bar for files, inner for each file
                outer_pbar = tqdm(total=len(files), position=0, desc="Processing files")
                for i, file_path in enumerate(files):
                    inner_desc = f"File {i+1}/{len(files)}: {file_path.name}"
                    inner_pbar = tqdm(total=1, position=1, desc=inner_desc, leave=False)
                    success = self.converter.convert_video(file_path, pbar=inner_pbar)
                    if success:
                        successful += 1
                    else:
                        failed += 1
                    outer_pbar.update(1)
                outer_pbar.close()
        else:
            # Parallel processing with overall progress bar (no inner bars to avoid interference)
            self.logger.info(f"Parallel processing {len(files)} files with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.converter.convert_video, file_path, pbar=None): file_path
                    for file_path in files
                }

                parallel_pbar = tqdm(total=len(files), position=0, desc="Processing files (parallel)")
                for future in as_completed(future_to_file):
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                    parallel_pbar.update(1)
                parallel_pbar.close()

        self.logger.info(f"Processing complete: {successful} successful, {failed} failed")


def check_dependencies() -> bool:
    """Check if required dependencies are available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert videos to .mp4 with optimal compression and GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -W /path/to/videos          # Process all videos in directory
  %(prog)s -F video1.mkv video2.avi   # Process specific files
  %(prog)s -W . -v                     # Process current directory with verbose output
        """
    )

    # Mutually exclusive group for working directory or specific files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-W", "--working-dir",
        type=str,
        help="Directory containing videos to process"
    )
    group.add_argument(
        "-F", "--files",
        nargs="+",
        help="Specific video files to process (space-separated)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers (default: 1, sequential)"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        print("Error: FFmpeg and FFprobe are required but not found in PATH")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Initialize processor
    config = Config()
    processor = VideoProcessor(config, args.verbose)

    try:
        if args.working_dir:
            # Process directory
            directory = Path(args.working_dir).resolve()
            files = processor.find_video_files(directory)
        else:
            # Process specific files
            files = processor.validate_files(args.files)

        # Process the files
        processor.process_files(files, args.max_workers)

    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
