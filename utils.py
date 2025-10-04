#!/usr/bin/env python3
"""
Video Compression Utils

Shared utilities for CLI and GUI video compression.
"""

import os
import subprocess
import platform
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

# Constants
GPU_DETECT_TIMEOUT = 10
FFPROBE_TIMEOUT = 30
FFMPEG_TIMEOUT = 3600

@dataclass
class Config:
    """Configuration settings for video compression"""
    SUPPORTED_FORMATS: Tuple[str, ...] = (
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    )
    MAX_BITRATE: int = 10000  # Max total bitrate in kbps (10 Mbps)
    compression_factor: float = 0.7  # Compression factor (0.0-1.0)
    MAX_DOWNSCALE_PERCENT: float = 0.5  # Max resolution reduction
    OUTPUT_SUFFIX: str = "_conv"
    MIN_AUDIO_BITRATE: int = 64  # Minimum audio bitrate in kbps
    use_adaptive: bool = True  # Use adaptive compression factors based on metadata

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def check_dependencies() -> bool:
    """Check if required dependencies are available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=FFPROBE_TIMEOUT)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=FFPROBE_TIMEOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

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
                    shell=True, text=True, timeout=GPU_DETECT_TIMEOUT
                )
            elif system == "Linux":
                output = subprocess.check_output(
                    "lspci | grep -i vga",
                    shell=True, text=True, timeout=GPU_DETECT_TIMEOUT
                )
            elif system == "Darwin":  # macOS
                output = subprocess.check_output(
                    "system_profiler SPDisplaysDataType",
                    shell=True, text=True, timeout=GPU_DETECT_TIMEOUT
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
    def get_info(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], bool, Optional[int], Optional[float], int, int, Optional[float], Optional[str]]:
        """
        Extract video resolution, bitrates, audio presence, duration, nb_frames, rotation, fps, codec using ffprobe.
        Supports adaptive compression by including codec for inefficiency detection.

        Returns:
            Tuple of (width, height, video_bitrate_kbps, has_audio, audio_bitrate_kbps, duration_seconds, nb_frames, rotation_degrees, fps, codec_name)
            or (None, None, None, False, None, None, -1, 0, None, None) on error
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_streams",
                "-of", "json",
                str(file_path)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT, check=True
            )

            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            format_info = data.get('format', {})

            video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)

            if not video_stream:
                logging.error(f"No video stream found in {file_path}")
                return None, None, None, False, None, None, -1, 0, None, None

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

            # Detect rotation from side data
            rotation = 0
            side_data_list = video_stream.get('side_data_list', [])
            for side in side_data_list:
                if side.get('side_data_type') == 'Display Matrix' and 'rotation' in side:
                    try:
                        rotation = int(side['rotation'])
                    except (ValueError, TypeError):
                        rotation = 0
                    break

            # Extract FPS from r_frame_rate (GUI addition)
            r_frame_rate = video_stream.get('r_frame_rate')
            fps = None
            if r_frame_rate:
                try:
                    fps = eval(r_frame_rate, {"__builtins__": {}})
                except (ValueError, TypeError, NameError, ZeroDivisionError):
                    fps = None

            # Extract codec name
            codec = video_stream.get('codec_name')

            return width, height, video_bitrate, has_audio, audio_bitrate, duration, nb_frames, rotation, fps, codec

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get video info for {file_path}: {e}")
            return None, None, None, False, None, None, -1, 0, None, None

class VideoConverter:
    """Handles video conversion with GPU acceleration"""

    def __init__(self, gpu_type: str, config: Config, logs_dir: Optional[Path] = None, move_files: bool = False, per_file_callback: Optional[Callable[[float], None]] = None):
        self.gpu_type = gpu_type
        self.config = config
        self.logs_dir = logs_dir
        self.move_files = move_files
        self.per_file_callback = per_file_callback
        self.logger = logging.getLogger(__name__)

    def _calculate_target_params(self, width: int, height: int, video_bitrate: Optional[int], audio_bitrate: Optional[int], duration: Optional[float] = None, codec: Optional[str] = None) -> Tuple[int, int, int, int]:
        """
        Calculate target resolution and bitrates with adaptive logic if enabled.
        Adaptive mode adjusts factors based on bitrate, resolution, duration, and codec for better efficiency.
        Falls back to static compression_factor if use_adaptive=False in Config.
        """
        if not self.config.use_adaptive:
            # Fallback to static calculation for backward compatibility
            # Handle missing bitrates
            if video_bitrate is None or video_bitrate <= 0:
                target_video_bitrate = self.config.MAX_BITRATE
            else:
                target_video_bitrate = int(video_bitrate * self.config.compression_factor)

            if audio_bitrate is None or audio_bitrate <= 0:
                target_audio_bitrate = 128
            else:
                target_audio_bitrate = max(int(audio_bitrate * self.config.compression_factor), self.config.MIN_AUDIO_BITRATE)

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

            self.logger.info(f"Using static compression: factor={self.config.compression_factor}, downscale={self.config.MAX_DOWNSCALE_PERCENT}")
            return new_width, new_height, target_video_bitrate, target_audio_bitrate

        # Adaptive logic
        # Thresholds
        HIGH_BITRATE_THRESHOLD = 8000  # kbps
        LOW_BITRATE_THRESHOLD = 2000   # kbps
        HD_HEIGHT = 1080
        FOURK_HEIGHT = 2160
        LONG_DURATION = 3600  # seconds
        INEFFICIENT_CODECS = {'mpeg4', 'wmv2', 'msmpeg4'}

        # Base factors
        base_factor = self.config.compression_factor  # 0.7 default
        video_factor = base_factor
        audio_factor = base_factor
        downscale_factor = self.config.MAX_DOWNSCALE_PERCENT  # 0.5 max

        # Bitrate-based adjustment for video
        if video_bitrate is not None and video_bitrate > 0:
            if video_bitrate > HIGH_BITRATE_THRESHOLD:
                video_factor = max(0.4, base_factor - 0.2)
            elif video_bitrate < LOW_BITRATE_THRESHOLD:
                video_factor = min(0.9, base_factor + 0.2)
                if video_bitrate < 1000:
                    self.logger.info("Low bitrate video detected; skipping compression.")
                    video_factor = 1.0  # No compression

        # Resolution-based downscale
        if height > FOURK_HEIGHT:
            downscale_factor = 0.5
        elif height > HD_HEIGHT:
            downscale_factor = 0.3
        else:
            downscale_factor = 0.0  # No downscale for SD/HD

        # Duration adjustment (slight leniency for long videos to prioritize speed)
        if duration is not None and duration > LONG_DURATION:
            video_factor = min(0.95, video_factor + 0.05)

        # Codec adjustment for inefficient codecs
        if codec in INEFFICIENT_CODECS:
            video_factor = min(0.9, video_factor + 0.1)
            self.logger.info(f"Inefficient codec '{codec}' detected; adjusting factor to {video_factor}")

        # Audio factor (more conservative)
        if audio_bitrate is not None and audio_bitrate > 128:
            audio_factor = video_factor
        else:
            audio_factor = 0.8  # Fixed for low/no audio

        # Calculate targets
        if video_bitrate is None or video_bitrate <= 0:
            target_video_bitrate = self.config.MAX_BITRATE
        else:
            target_video_bitrate = int(video_bitrate * video_factor)

        if audio_bitrate is None or audio_bitrate <= 0:
            target_audio_bitrate = 128
        else:
            target_audio_bitrate = max(int(audio_bitrate * audio_factor), self.config.MIN_AUDIO_BITRATE)

        # Clamp factors
        target_video_bitrate = max(500, min(target_video_bitrate, self.config.MAX_BITRATE))  # Avoid too low/high
        target_audio_bitrate = max(self.config.MIN_AUDIO_BITRATE, min(target_audio_bitrate, 320))  # Reasonable audio range

        # Warnings
        if target_video_bitrate < 500:
            self.logger.warning(
                f"Adaptive video bitrate {target_video_bitrate}kbps below 500kbps threshold; quality loss possible."
            )
        if target_audio_bitrate < 128:
            self.logger.warning(
                f"Adaptive audio bitrate {target_audio_bitrate}kbps below 128kbps threshold; quality loss possible."
            )

        # Resolution
        scale_factor = 1 - downscale_factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure even dimensions
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        self.logger.info(
            f"Adaptive compression: video_factor={video_factor:.2f}, audio_factor={audio_factor:.2f}, "
            f"downscale={downscale_factor:.2f} (bitrate={video_bitrate}kbps, height={height}, duration={duration}s, codec={codec})"
        )

        return new_width, new_height, target_video_bitrate, target_audio_bitrate

    def _build_ffmpeg_command(self, input_file: Path, output_file: Path,
                              new_width: int, new_height: int, target_video_bitrate: int, target_audio_bitrate: int, rotation: int = 0) -> List[str]:
        """Build FFmpeg command based on GPU type"""
        rotate_str = ""
        if rotation == 90:
            rotate_str = "rotate=PI/2,"
        elif rotation == 180:
            rotate_str = "rotate=PI,"
        elif rotation == 270:
            rotate_str = "rotate=3*PI/2,"

        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vf", rotate_str + f"scale='min(iw,{new_width})':'min(ih,{new_height})':force_original_aspect_ratio=decrease",
            "-progress", "pipe:1"
        ]

        # Select encoder based on GPU type (unified to use -b:v for all)
        if self.gpu_type == "NVIDIA":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "INTEL":
            cmd.extend(["-c:v", "h264_qsv", "-preset", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "AMD":
            cmd.extend(["-c:v", "h264_amf", "-quality", "slow", "-b:v", f"{target_video_bitrate}k"])
        elif self.gpu_type == "APPLE":
            cmd.extend(["-c:v", "h264_videotoolbox", "-b:v", f"{target_video_bitrate}k"])
        else:  # CPU fallback - use CRF like original CLI for quality focus
            # Adaptive CRF for CPU: base 23, adjust based on factor (lower CRF = higher quality/less compression)
            if self.config.use_adaptive:
                crf = max(18, min(28, 23 + int((1 - self.config.compression_factor) * 10)))
                self.logger.info(f"Adaptive CRF for CPU: {crf}")
                cmd.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"])
            else:
                cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium"])

        # Audio settings and output
        cmd.extend([
            "-c:a", "aac", "-b:a", f"{target_audio_bitrate}k",
            "-movflags", "+faststart",  # Web optimization
        ])
        cmd.append(str(output_file))

        return cmd

    def convert_video(self, input_file: Path, pbar: Optional[Any] = None) -> bool:
        """
        Convert a single video file with optional progress bar or callback.

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
                    if hasattr(pbar, 'total') and pbar.total is not None:
                        pbar.n = pbar.total
                    pbar.close()
                if self.per_file_callback:
                    self.per_file_callback(100.0)
                return True

            # Get video info (unified return)
            info = VideoInfo.get_info(input_file)
            width, height, video_bitrate, has_audio, audio_bitrate, duration, nb_frames, rotation, fps, codec = info
            if width is None or height is None:
                self.logger.error(f"Could not determine video dimensions for {input_file}")
                if pbar:
                    if hasattr(pbar, 'total') and pbar.total is not None:
                        pbar.n = pbar.total
                    pbar.close()
                if self.per_file_callback:
                    self.per_file_callback(0.0)
                return False

            # Determine progress mode (prefer frames, fallback to time)
            use_frame = nb_frames > 0
            use_time = duration is not None and duration > 0
            total_frames = nb_frames if use_frame else None
            if not use_frame and not use_time:
                self.logger.warning(f"No reliable progress metrics for {input_file.name} (duration: {duration}, frames: {nb_frames}). Using simple completion.")

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
            new_width, new_height, target_video_bitrate, target_audio_bitrate = self._calculate_target_params(
                width, height, video_bitrate, audio_bitrate, duration=duration, codec=codec
            )

            self.logger.info(
                f"Converting {input_file.name} -> {output_file.name} "
                f"({width}x{height} -> {new_width}x{new_height}, "
                f"Video: {video_bitrate or 'unknown'}kbps -> {target_video_bitrate}kbps, "
                f"Audio: {audio_bitrate or 'unknown'}kbps -> {target_audio_bitrate}kbps, "
                f"Duration: {duration}s, Frames: {nb_frames}, Rotation: {rotation}°, Codec: {codec}) "
                f"using {self.gpu_type}"
            )

            # Build and execute FFmpeg command
            cmd = self._build_ffmpeg_command(input_file, output_file, new_width, new_height, target_video_bitrate, target_audio_bitrate, rotation)

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
            last_percent = 0.0
            try:
                for line in iter(proc.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        # Parse FFmpeg progress output
                        if pbar or self.per_file_callback:
                            if use_frame and '=' in line and line.startswith('frame='):
                                try:
                                    frame_str = line.split('=')[1].split(' ')[0]
                                    frame = int(frame_str)
                                    if frame > current_frame:
                                        delta = frame - current_frame
                                        if pbar:
                                            pbar.update(delta)
                                        if total_frames and total_frames > 0:
                                            percent = (frame / total_frames) * 100
                                            if self.per_file_callback and percent > last_percent:
                                                self.per_file_callback(min(percent, 100.0))
                                                last_percent = percent
                                        current_frame = frame
                                except (ValueError, IndexError):
                                    pass  # Ignore parse errors
                            elif use_time and '=' in line and line.startswith('out_time_ms='):
                                try:
                                    time_str = line.split('=')[1].split(' ')[0]
                                    time_ms = int(time_str)
                                    if time_ms > current_time_ms:
                                        if pbar:
                                            percent = min((time_ms / (duration * 1000)) * 100, 100)
                                            pbar.update(percent - (pbar.n if hasattr(pbar, 'n') else 0))
                                        if self.per_file_callback:
                                            percent = min((time_ms / (duration * 1000)) * 100, 100)
                                            if percent > last_percent:
                                                self.per_file_callback(percent)
                                                last_percent = percent
                                        current_time_ms = time_ms
                                except (ValueError, IndexError):
                                    pass  # Ignore parse errors
                proc.wait(timeout=FFMPEG_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                if self.logs_dir:
                    log_path = self.logs_dir / f"{input_file.stem}_timeout.log"
                    with open(log_path, 'w') as f:
                        f.write(f"FFmpeg Command: {' '.join(cmd)}\n\n")
                        f.write("Output:\n")
                        for line in output_lines:
                            f.write(line + '\n')
                self.logger.error(f"Conversion timeout for {input_file}; full log saved to {log_path}")
                if pbar:
                    if hasattr(pbar, 'total') and pbar.total is not None:
                        pbar.n = pbar.total
                    pbar.close()
                if self.per_file_callback:
                    self.per_file_callback(0.0)
                if output_file.exists():
                    output_file.unlink()
                return False

            if proc.returncode != 0:
                if self.logs_dir:
                    log_path = self.logs_dir / f"{input_file.stem}_ffmpeg_error.log"
                    with open(log_path, 'w') as f:
                        f.write(f"FFmpeg Command: {' '.join(cmd)}\n\n")
                        f.write("Output:\n")
                        for line in output_lines:
                            f.write(line + '\n')
                error_msg = ' '.join(output_lines[-10:]) if output_lines else "Unknown error"
                self.logger.error(f"FFmpeg failed for {input_file}: {error_msg} (full log: {log_path})")
                if pbar:
                    if hasattr(pbar, 'total') and pbar.total is not None:
                        pbar.n = pbar.total
                    pbar.close()
                if self.per_file_callback:
                    self.per_file_callback(0.0)
                # Clean up partial file
                if output_file.exists():
                    output_file.unlink()
                return False

            if pbar:
                pbar.n = pbar.total
                pbar.close()

            if self.per_file_callback:
                self.per_file_callback(100.0)

            self.logger.info(f"Successfully converted {input_file.name}")

            # Move original file if requested
            if self.move_files:
                original_dir = input_file.parent / "original_files"
                original_dir.mkdir(exist_ok=True)
                new_path = original_dir / input_file.name
                try:
                    input_file.rename(new_path)
                    self.logger.info(f"Moved original {input_file.name} to {original_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to move {input_file.name} to {original_dir}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Unexpected error converting {input_file}: {e}")
            if pbar:
                if hasattr(pbar, 'total') and pbar.total is not None:
                    pbar.n = pbar.total
                pbar.close()
            if self.per_file_callback:
                self.per_file_callback(0.0)
            return False

class VideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, config: Config, verbose: bool = False, move_files: bool = False, per_file_callback: Optional[Callable[[float], None]] = None, logs_dir: Optional[Path] = None):
        self.config = config
        self.move_files = move_files
        self.logger = setup_logging(verbose)
        self.logs_dir = logs_dir or Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.gpu_type = GPUDetector.detect()
        self.converter = VideoConverter(self.gpu_type, config, self.logs_dir, move_files, per_file_callback)
        self.logger.info(f"Detected GPU: {self.gpu_type}")

    def find_video_files(self, directory: Path, recursive: bool = False) -> List[Path]:
        """Find all supported video files in directory, optionally recursive"""
        video_files = []

        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory}")
            return video_files

        try:
            if recursive:
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix in self.config.SUPPORTED_FORMATS:
                            video_files.append(file_path)
                self.logger.info(f"Found {len(video_files)} video files in {directory} (recursive)")
            else:
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

    def process_files(self, files: List[Path], max_workers: int = 1, use_tqdm: bool = True, overall_update_callback: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
        """
        Process video files sequentially or in parallel.

        Args:
            files: List of video files to process
            max_workers: Number of parallel workers
            use_tqdm: Whether to use tqdm progress bars (CLI)
            overall_update_callback: Optional callback for overall progress updates (e.g., GUI)

        Returns:
            Tuple of (successful, failed)
        """
        if not files:
            self.logger.warning("No files to process")
            return 0, 0

        successful = 0
        failed = 0

        # Handle tqdm import
        tqdm_available = False
        if use_tqdm:
            try:
                from tqdm import tqdm
                tqdm_available = True
            except ImportError:
                self.logger.warning("tqdm not available, disabling progress bars")
                use_tqdm = False

        if max_workers == 1:
            # Sequential processing
            outer_pbar = None
            if use_tqdm and len(files) > 1:
                outer_pbar = tqdm(total=len(files), position=0, desc="Processing files")

            for i, file_path in enumerate(files):
                inner_pbar = None
                if use_tqdm:
                    inner_desc = f"File {i+1}/{len(files)}: {file_path.name}"
                    inner_pbar = tqdm(total=1, position=1, desc=inner_desc, leave=False)

                success = self.converter.convert_video(file_path, pbar=inner_pbar)
                if inner_pbar:
                    inner_pbar.close()

                if success:
                    successful += 1
                else:
                    failed += 1

                completed = successful + failed
                if outer_pbar:
                    outer_pbar.update(1)
                if overall_update_callback:
                    overall_update_callback(completed)

            if outer_pbar:
                outer_pbar.close()
        else:
            # Parallel processing
            self.logger.info(f"Parallel processing {len(files)} files with {max_workers} workers")
            outer_pbar = None
            if use_tqdm:
                outer_pbar = tqdm(total=len(files), position=0, desc="Processing files (parallel)")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.converter.convert_video, file_path, pbar=None): file_path
                    for file_path in files
                }

                for future in as_completed(future_to_file):
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1

                    completed = successful + failed
                    if outer_pbar:
                        outer_pbar.update(1)
                    if overall_update_callback:
                        overall_update_callback(completed)

            if outer_pbar:
                outer_pbar.close()

        self.logger.info(f"Processing complete: {successful} successful, {failed} failed")
        return successful, failed