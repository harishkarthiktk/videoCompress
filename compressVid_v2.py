#!/usr/bin/env python3
"""
Video Compression Script v2 with Adaptive GPU Acceleration

Enhanced version with automatic quality-based compression.
Uses content analysis to determine optimal CRF, resolution scaling,
and other parameters for better size/quality balance.

Features:
- Adaptive compression based on video complexity
- Optional VMAF quality metrics
- Two-pass encoding for CPU (libx264)
- GPU acceleration with quality modes (NVIDIA, Intel, AMD, Apple Silicon)
- Configurable working directory or specific file processing
- Skips already converted files
- Retry logic to ensure output < input size
- Backward compatible with original fixed method
"""

import os
import sys
import subprocess
import platform
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import tempfile


# Configuration constants
@dataclass
class Config:
    """Configuration settings for video compression"""
    SUPPORTED_FORMATS = (
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    )
    MAX_BITRATE = 10000  # Max total bitrate in kbps (10 Mbps) for fallback
    COMPRESSION_FACTOR = 0.7  # Legacy fixed reduction
    MAX_DOWNSCALE_PERCENT = 0.2  # 20% max resolution reduction (fixed: 0.2)
    OUTPUT_SUFFIX = "_conv"
    MIN_AUDIO_BITRATE = 64  # Minimum audio bitrate in kbps
    # New adaptive config
    BASE_CRF = 23
    CRF_RANGE = (18, 32)  # Min/max CRF
    ENABLE_VMAF = False  # Requires FFmpeg with libvmaf
    MAX_RETRIES = 2
    COMPLEXITY_SAMPLES = 10  # Frames to sample for complexity
    TARGET_SIZE_REDUCTION = 0.7  # Aim for 70% of original size initially
    VMAF_MODEL = "model_vmaf_v0.6.1.json"  # Default VMAF model


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
    def get_info(file_path: Path, config: Config) -> Tuple[Optional[int], Optional[int], Optional[int], bool, Optional[int], Optional[float], int, str, str]:
        """
        Extract video info including complexity score using ffprobe

        Returns:
            Tuple of (width, height, video_bitrate_kbps, has_audio, audio_bitrate_kbps, duration, input_size, complexity, pix_fmt)
            or (None, None, None, False, None, None, 0, 'low', 'yuv420p') on error
        """
        try:
            input_size = file_path.stat().st_size

            # Basic stream info
            cmd = [
                "ffprobe", "-v", "error",
                "-show_streams", "-show_format",
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
                return None, None, None, False, None, None, input_size, 'low', 'yuv420p'

            width = video_stream.get('width')
            height = video_stream.get('height')
            video_bitrate = video_stream.get('bit_rate')
            if video_bitrate is not None:
                video_bitrate = int(video_bitrate) // 1000

            duration = float(format_info.get('duration', 0))

            has_audio = audio_stream is not None
            audio_bitrate = None
            if has_audio and 'bit_rate' in audio_stream:
                audio_bitrate = int(audio_stream['bit_rate']) // 1000

            # Compute complexity: Sample frames for pict_type (I/P/B ratio as motion proxy)
            complexity = VideoInfo._compute_complexity(file_path, config.COMPLEXITY_SAMPLES)

            pix_fmt = video_stream.get('pix_fmt', 'yuv420p')

            return width, height, video_bitrate, has_audio, audio_bitrate, duration, input_size, complexity, pix_fmt

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get video info for {file_path}: {e}")
            return None, None, None, False, None, None, file_path.stat().st_size if file_path.exists() else 0, 'low', 'yuv420p'

    @staticmethod
    def _compute_complexity(file_path: Path, samples: int) -> str:
        """Compute complexity category based on frame type sampling"""
        try:
            # Sample up to 'samples' key frames or total frames
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "frame=pict_type", "-of", "csv=p=0",
                "-read_intervals", f"%+{samples}", str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if result.returncode != 0:
                return 'medium'  # Default

            frame_types = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not frame_types:
                return 'medium'

            # Count I-frames (keyframes) vs others (P/B indicate motion)
            i_count = frame_types.count('I')
            total = len(frame_types)
            i_ratio = i_count / total if total > 0 else 0

            # Low motion: high I-ratio (many keyframes, static scenes)
            # High motion: low I-ratio (few keyframes, lots of P/B)
            if i_ratio > 0.3:
                return 'low'  # Static/low motion
            elif i_ratio > 0.1:
                return 'medium'
            else:
                return 'high'  # High motion

        except Exception as e:
            logging.warning(f"Complexity computation failed for {file_path}: {e}")
            return 'medium'


def check_vmaf_support() -> bool:
    """Check if FFmpeg supports libvmaf"""
    try:
        result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        return 'libvmaf' in result.stdout
    except:
        return False


class VideoConverter:
    """Handles video conversion with adaptive GPU acceleration"""

    def __init__(self, gpu_type: str, config: Config, use_auto: bool = False):
        self.gpu_type = gpu_type
        self.config = config
        self.use_auto = use_auto
        self.logger = logging.getLogger(__name__)
        self.vmaf_supported = check_vmaf_support() if config.ENABLE_VMAF else False
        if config.ENABLE_VMAF and not self.vmaf_supported:
            self.logger.warning("VMAF enabled but not supported by FFmpeg. Disabling.")
            config.ENABLE_VMAF = False

    def _calculate_target_params(self, width: int, height: int, video_bitrate: Optional[int],
                                 audio_bitrate: Optional[int], duration: float, input_size: int,
                                 complexity: str) -> Tuple[int, int, int, int, float, int]:
        """
        Calculate adaptive target parameters (CRF mode) or legacy params

        Returns: (new_width, new_height, crf_or_bitrate, audio_param, scale_factor, passes)
        """
        if not self.use_auto:
            # Legacy fixed method
            if video_bitrate is None or video_bitrate <= 0:
                target_video_bitrate = self.config.MAX_BITRATE
            else:
                target_video_bitrate = int(video_bitrate * self.config.COMPRESSION_FACTOR)

            if audio_bitrate is None or audio_bitrate <= 0:
                target_audio_bitrate = 128
            else:
                target_audio_bitrate = max(int(audio_bitrate * self.config.COMPRESSION_FACTOR), self.config.MIN_AUDIO_BITRATE)

            scale_factor = 1 - self.config.MAX_DOWNSCALE_PERCENT
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)

            if target_video_bitrate < 500:
                self.logger.warning(f"Video bitrate {target_video_bitrate}kbps below 500kbps threshold.")
            if target_audio_bitrate < 128:
                self.logger.warning(f"Audio bitrate {target_audio_bitrate}kbps below 128kbps threshold.")

            return new_width, new_height, target_video_bitrate, target_audio_bitrate, scale_factor, 1

        # Adaptive method
        # CRF based on complexity
        complexity_map = {'low': 28, 'medium': 23, 'high': 18}
        base_crf = complexity_map.get(complexity, self.config.BASE_CRF)
        crf = max(self.config.CRF_RANGE[0], min(self.config.CRF_RANGE[1], base_crf))

        # Audio quality (VBR scale 0-9, higher better)
        audio_q_map = {'low': 3, 'medium': 4, 'high': 5}
        audio_q = audio_q_map.get(complexity, 4)

        # Target size reduction based on complexity
        reduction_map = {'low': 0.8, 'medium': 0.7, 'high': 0.6}
        target_reduction = reduction_map.get(complexity, self.config.TARGET_SIZE_REDUCTION)
        target_size = int(input_size * target_reduction)

        # Resolution: Start with no downscale, adjust if needed
        scale_factor = 1.0
        new_width, new_height = width, height

        # Project size roughly: bitrate estimate from CRF (empirical: ~CRF inverse)
        est_bitrate_kbps = 10000 / (crf - 10)  # Rough H.264 estimate
        projected_size = (est_bitrate_kbps * 1000 * duration) / 8  # bytes
        if projected_size > target_size * 1.2:  # If over 20% above target, downscale
            scale_factor = 0.9
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)

        # Passes: 2 for CPU, 1 for GPU (limited support)
        passes = 2 if self.gpu_type == "CPU" else 1

        self.logger.info(f"Adaptive params: CRF={crf}, Scale={scale_factor}, Audio q={audio_q}, Passes={passes}, Target size={target_size/1024/1024:.1f}MB")

        return new_width, new_height, crf, audio_q, scale_factor, passes

    def _build_ffmpeg_command(self, input_file: Path, output_file: Path,
                              new_width: int, new_height: int, video_param: int,
                              audio_param: int, scale_factor: float, passes: int,
                              pass_num: int = 1, vmaf: bool = False, pix_fmt: str = 'yuv420p') -> List[str]:
        """Build FFmpeg command (single or multi-pass, with optional VMAF)"""
        vf_filters = []

        if scale_factor < 1.0:
            vf_filters.append(f"scale={new_width}:{new_height}:flags=lanczos")
            vf_filters.append("unsharp=5:5:0.8:3:3:0.4")  # Minor sharpening

        # Add format only if needed
        if pix_fmt != 'yuv420p':
            vf_filters.append("format=yuv420p")

        vf = ",".join(vf_filters) if vf_filters else None

        cmd = ["ffmpeg", "-y", "-i", str(input_file)]
        if vf:
            cmd.extend(["-vf", vf])

        if vmaf:
            # VMAF requires two inputs: original and encoded (post-encode comparison)
            # For simplicity, we'll compute VMAF separately after encoding
            pass  # Placeholder: implement post-encode VMAF if enabled

        is_crf_mode = self.use_auto  # Assume video_param is CRF in auto mode

        if passes > 1 and pass_num == 1:
            # First pass: analysis
            if self.gpu_type == "CPU":
                cmd.extend(["-c:v", "libx264", "-pass", "1", "-an", "-f", "null", "NUL"])
            else:
                # GPU two-pass limited; fallback to single
                cmd = self._build_ffmpeg_command(input_file, output_file, new_width, new_height,
                                                 video_param, audio_param, scale_factor, 1, pix_fmt=pix_fmt)
                return cmd
        elif passes > 1 and pass_num == 2:
            # Second pass
            if self.gpu_type == "CPU":
                cmd.extend(["-c:v", "libx264", "-pass", "2"])
            else:
                return cmd  # Should not reach here
        else:
            # Single pass
            if self.gpu_type == "NVIDIA":
                if is_crf_mode:
                    cmd.extend(["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", str(video_param), "-preset", "slow"])
                else:
                    cmd.extend(["-c:v", "h264_nvenc", "-preset", "slow", "-b:v", f"{video_param}k"])
            elif self.gpu_type == "INTEL":
                if is_crf_mode:
                    cmd.extend(["-c:v", "h264_qsv", "-global_quality", str(video_param), "-preset", "slow"])
                else:
                    cmd.extend(["-c:v", "h264_qsv", "-preset", "slow", "-b:v", f"{video_param}k"])
            elif self.gpu_type == "AMD":
                if is_crf_mode:
                    cmd.extend(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", str(video_param), "-qp_p", str(video_param), "-qp_b", str(video_param)])
                else:
                    cmd.extend(["-c:v", "h264_amf", "-quality", "speed", "-b:v", f"{video_param}k"])
            elif self.gpu_type == "APPLE":
                cmd.extend(["-c:v", "h264_videotoolbox", "-q:v", str(video_param) if is_crf_mode else f"{video_param}k"])
            else:  # CPU
                if is_crf_mode:
                    cmd.extend(["-c:v", "libx264", "-crf", str(video_param), "-preset", "slow"])
                else:
                    cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "medium", "-b:v", f"{video_param}k"])

        # Audio: VBR in auto, fixed in legacy
        if self.use_auto:
            cmd.extend(["-c:a", "aac", "-q:a", str(audio_param)])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", f"{audio_param}k"])

        if passes == 1 or pass_num == 2:
            cmd.extend(["-movflags", "+faststart", str(output_file)])
        # For pass 1, no output file

        return cmd

    def _compute_vmaf(self, input_file: Path, output_file: Path, model_path: str) -> Optional[float]:
        """Compute VMAF score between input and output"""
        if not self.config.ENABLE_VMAF or not self.vmaf_supported:
            return None
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_file), "-i", str(output_file),
                "-lavfi", f"[0:v]scale=iw:ih[v];[v][1:v]libvmaf=model_path={model_path}:log_path=vmaf.log",
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # Parse VMAF from log (simplified: assume average from log)
                with open("vmaf.log", "r") as f:
                    for line in f:
                        if "VMAF score" in line:
                            return float(line.split()[-1])
                os.unlink("vmaf.log")
            return None
        except Exception as e:
            self.logger.warning(f"VMAF computation failed: {e}")
            return None

    def convert_video(self, input_file: Path) -> bool:
        """
        Convert a single video file with retry logic

        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = input_file.parent / f"{input_file.stem}{self.config.OUTPUT_SUFFIX}.mp4"

            if output_file.exists():
                self.logger.info(f"Skipping {input_file.name} - already converted")
                return True

            # Get video info
            width, height, video_bitrate, has_audio, audio_bitrate, duration, input_size, complexity, pix_fmt = VideoInfo.get_info(input_file, self.config)
            if width is None or height is None:
                self.logger.error(f"Could not determine video dimensions for {input_file}")
                return False

            # Calculate target parameters
            new_width, new_height, video_param, audio_param, scale_factor, passes = self._calculate_target_params(
                width, height, video_bitrate, audio_bitrate, duration, input_size, complexity
            )

            self.logger.info(
                f"Converting {input_file.name} -> {output_file.name} "
                f"({width}x{height} -> {new_width}x{new_height}, "
                f"Complexity: {complexity}, "
                f"{'CRF=' if self.use_auto else 'Video: '} {video_param}{'kbps' if not self.use_auto else ''}, "
                f"{'Audio q=' if self.use_auto else 'Audio: '}{audio_param}{'kbps' if not self.use_auto else ''}) "
                f"using {self.gpu_type}"
            )

            current_crf = video_param if self.use_auto else 23
            retry = 0
            success = False

            while retry <= self.config.MAX_RETRIES:
                temp_output = None
                if passes > 1:
                    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                    output_file = Path(temp_output)

                # Multi-pass handling
                for pass_num in range(1, passes + 1):
                    cmd = self._build_ffmpeg_command(input_file, output_file, new_width, new_height,
                                                     video_param, audio_param, scale_factor, passes, pass_num, pix_fmt)
                    self.logger.info(f"Running pass {pass_num}/{passes}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    if result.returncode != 0:
                        self.logger.error(f"Pass {pass_num} failed: {result.stderr}")
                        if temp_output:
                            os.unlink(temp_output)
                        return False

                # Check size after encoding
                if output_file.stat().st_size < input_size:
                    success = True
                    break
                else:
                    self.logger.warning(f"Output size {output_file.stat().st_size} >= input {input_size}, retrying...")
                    if self.use_auto:
                        current_crf += 2  # Increase CRF for smaller size
                        video_param = min(current_crf, self.config.CRF_RANGE[1])
                    retry += 1
                    if temp_output:
                        os.unlink(temp_output)

            if not success:
                self.logger.error(f"Failed to compress {input_file.name} after {self.config.MAX_RETRIES} retries")
                return False

            # Move temp to final if needed
            if temp_output:
                shutil.move(temp_output, output_file)

            # Optional VMAF
            vmaf_score = self._compute_vmaf(input_file, output_file, self.config.VMAF_MODEL)
            size_savings = ((input_size - output_file.stat().st_size) / input_size) * 100
            self.logger.info(f"Successfully converted {input_file.name}: {size_savings:.1f}% smaller, VMAF: {vmaf_score if vmaf_score else 'N/A'}")

            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"Conversion timeout for {input_file}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error converting {input_file}: {e}")
            return False


class VideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, config: Config, verbose: bool = False, use_auto: bool = False):
        self.config = config
        self.logger = setup_logging(verbose)
        self.gpu_type = GPUDetector.detect()
        self.use_auto = use_auto
        self.converter = VideoConverter(self.gpu_type, config, use_auto)

        self.logger.info(f"Detected GPU: {self.gpu_type}, Auto mode: {use_auto}")

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
        """Process video files (sequential or parallel)"""
        if not files:
            self.logger.warning("No files to process")
            return

        successful = 0
        failed = 0

        if max_workers == 1:
            # Sequential processing
            for file_path in files:
                if self.converter.convert_video(file_path):
                    successful += 1
                else:
                    failed += 1
        else:
            # Parallel processing (use with caution - resource intensive)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.converter.convert_video, file_path): file_path
                    for file_path in files
                }

                for future in as_completed(future_to_file):
                    if future.result():
                        successful += 1
                    else:
                        failed += 1

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
        description="Convert videos to .mp4 with adaptive compression and GPU acceleration (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -w /path/to/videos          # Process all videos in directory (legacy mode)
  %(prog)s -w /path/to/videos --auto   # Adaptive mode
  %(prog)s -f video1.mkv video2.avi   # Process specific files
  %(prog)s -w . -v --auto --vmaf      # Verbose adaptive with VMAF
        """
    )

    # Mutually exclusive group for working directory or specific files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-w", "--working-dir",
        type=str,
        help="Directory containing videos to process"
    )
    group.add_argument(
        "-f", "--files",
        nargs="+",
        help="Specific video files to process (space-separated)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable adaptive compression mode (default: legacy fixed)"
    )

    parser.add_argument(
        "--vmaf",
        action="store_true",
        help="Enable VMAF quality metrics (requires FFmpeg with libvmaf)"
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

    # Initialize config
    config = Config()
    if args.vmaf:
        config.ENABLE_VMAF = True

    processor = VideoProcessor(config, args.verbose, args.auto)

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
