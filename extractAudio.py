#!/usr/bin/env python3
"""
Video Processing Script: Extract Audio Only

This script processes video files by extracting their audio tracks
and saving them as MP3 or AAC files (controlled via --output-format).
"""

import sys
import subprocess
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import Config, setup_logging, check_dependencies, VideoInfo, validate_workers, ValidationError, write_error_log


class AudioExtractor:
    """Handles audio extraction"""

    def __init__(self, config: Config, output_format: str, move_files: bool = False, logs_dir: Optional[Path] = None):
        self.config = config
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)
        self.move_files = move_files
        self.logs_dir = logs_dir if logs_dir else Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # Audio-specific config (to avoid modifying shared Config)
        self.OUTPUT_SUFFIX = "_audio"
        self.DEFAULT_FORMAT = "aac"  # Updated default for better quality
        self.DEFAULT_BITRATES = {
            "mp3": "320k",  # High quality
            "aac": "256k",  # High quality
            "flac": None    # Lossless, no bitrate
        }
        self.EXTENSIONS = {
            "mp3": ".mp3",
            "aac": ".m4a",
            "flac": ".flac"
        }

    def _get_audio_info(self, input_file: Path) -> Optional[Tuple[bool, Optional[str], Optional[int], Optional[int], Optional[int], Optional[float]]]:
        """Extract audio stream info using ffprobe (has_audio, codec, bitrate_kbps, sample_rate, channels, duration)"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_streams",
                "-show_format",
                "-select_streams", "a:0",  # First audio stream
                "-of", "json",
                str(input_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            format_info = data.get('format', {})

            if not streams:
                return None, None, None, None, None, None

            audio_stream = streams[0]
            has_audio = True
            audio_codec = audio_stream.get('codec_name')
            audio_bitrate_str = audio_stream.get('bit_rate')
            audio_bitrate = int(audio_bitrate_str) // 1000 if audio_bitrate_str else None
            sample_rate = int(audio_stream.get('sample_rate', 44100)) if audio_stream.get('sample_rate') else 44100
            channels = int(audio_stream.get('channels', 2)) if audio_stream.get('channels') else 2
            duration_str = format_info.get('duration')
            duration = float(duration_str) if duration_str else None

            return has_audio, audio_codec, audio_bitrate, sample_rate, channels, duration
        except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Failed to get audio info for {input_file}: {e}")
            return None, None, None, None, None, None

    def _build_ffmpeg_command(self, input_file: Path, output_file: Path, has_audio: bool, audio_codec: Optional[str], audio_bitrate: Optional[int], sample_rate: int, channels: int) -> List[str]:
        """Build FFmpeg command to extract audio only"""
        if not has_audio:
            raise ValueError("No audio stream available")

        cmd = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vn", "-af", "aresample=async=1"
        ]

        # Add sample rate and channels preservation
        cmd.extend(["-ar", str(sample_rate), "-ac", str(channels)])

        copy_possible = False
        if self.output_format == "flac":
            cmd.extend(["-c:a", "flac"])
        elif self.output_format == "mp3":
            if audio_codec == "mp3" or audio_codec == "libmp3lame":
                cmd.extend(["-c:a", "copy"])
                copy_possible = True
            else:
                bitrate = str(audio_bitrate) + "k" if audio_bitrate and audio_bitrate >= 320 else self.DEFAULT_BITRATES["mp3"]
                cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate])
        elif self.output_format == "aac":
            aac_codecs = ["aac", "mp4a"]
            if audio_codec in aac_codecs:
                cmd.extend(["-c:a", "copy"])
                copy_possible = True
            else:
                bitrate = str(audio_bitrate) + "k" if audio_bitrate and audio_bitrate >= 256 else self.DEFAULT_BITRATES["aac"]
                cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        if copy_possible:
            self.logger.info(f"Using stream copy for {self.output_format} (original codec: {audio_codec})")
        else:
            self.logger.info(f"Re-encoding to {self.output_format} (original codec: {audio_codec}, bitrate: {audio_bitrate}kbps)")

        cmd.append(str(output_file))
        return cmd

    def extract_audio(self, input_file: Path, inner_pbar: Optional[Any] = None) -> bool:
        """Extract audio from a single video file"""
        try:
            ext = self.EXTENSIONS[self.output_format]
            output_file_name = f"{input_file.stem}{self.OUTPUT_SUFFIX}{ext}"
            output_file = input_file.parent / output_file_name

            if output_file.exists():
                self.logger.info(f"Skipping {input_file.name} - already extracted")
                if inner_pbar:
                    inner_pbar.n = 100
                    inner_pbar.close()
                return True

            # Probe for audio info
            audio_info = self._get_audio_info(input_file)
            has_audio, audio_codec, audio_bitrate, sample_rate, channels, duration = audio_info or (False, None, None, 44100, 2, None)

            if not has_audio:
                self.logger.warning(f"No audio stream found in {input_file.name}, skipping")
                if inner_pbar:
                    inner_pbar.n = 100
                    inner_pbar.close()
                return True

            self.logger.info(f"Processing {input_file.name} -> {output_file.name} (Extracting {self.output_format.upper()}, codec: {audio_codec}, bitrate: {audio_bitrate}kbps)")

            cmd = self._build_ffmpeg_command(input_file, output_file, has_audio, audio_codec, audio_bitrate, sample_rate, channels)

            # Use Popen for progress parsing and output collection
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines = []
            current_time_ms = 0
            last_percent = 0.0

            try:
                for line in iter(proc.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        # Parse progress (out_time_ms for time-based)
                        if inner_pbar and line.startswith('out_time_ms='):
                            try:
                                time_str = line.split('=')[1].split(' ')[0]
                                time_ms = int(time_str)
                                if time_ms > current_time_ms:
                                    if duration and duration > 0:
                                        percent = min((time_ms / (duration * 1000)) * 100, 100)
                                    else:
                                        percent = min((time_ms / 100000) * 100, 100)  # Rough estimate
                                    if percent > last_percent:
                                        inner_pbar.update(percent - last_percent)
                                        last_percent = percent
                                    current_time_ms = time_ms
                            except (ValueError, IndexError):
                                pass

                proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                log_path = write_error_log(self.logs_dir, input_file.stem, "TIMEOUT", cmd, output_lines)
                self.logger.error(f"Timeout extracting audio from {input_file}; log saved to {log_path}")
                if inner_pbar:
                    inner_pbar.n = 0
                    inner_pbar.close()
                if output_file.exists():
                    output_file.unlink()
                return False

            if proc.returncode != 0:
                log_path = write_error_log(self.logs_dir, input_file.stem, "ERROR", cmd, output_lines)
                self.logger.error(f"FFmpeg failed for {input_file}; log saved to {log_path}")
                if inner_pbar:
                    inner_pbar.n = 0
                    inner_pbar.close()
                if output_file.exists():
                    output_file.unlink()
                return False

            if inner_pbar:
                inner_pbar.n = 100
                inner_pbar.close()

            self.logger.info(f"Successfully processed {input_file.name}")

            # Move original if requested
            if self.move_files:
                original_dir = input_file.parent / "original_files"
                original_dir.mkdir(exist_ok=True)
                new_path = original_dir / input_file.name
                try:
                    input_file.rename(new_path)
                    self.logger.info(f"Moved original {input_file.name} to {original_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to move {input_file.name}: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Unexpected error processing {input_file}: {e}")
            if inner_pbar:
                inner_pbar.n = 0
                inner_pbar.close()
            return False


class VideoProcessor:
    """Main processing orchestrator"""

    def __init__(self, config: Config, output_format: str, verbose: bool = False, move_files: bool = False):
        self.config = config
        self.logger = setup_logging(verbose)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.extractor = AudioExtractor(config, output_format, move_files, self.logs_dir)

    def find_video_files(self, directory: Path) -> List[Path]:
        """Find all supported video files in directory"""
        video_files = []

        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory}")
            return video_files

        for file_path in directory.iterdir():
            if (file_path.is_file()
                    and file_path.suffix in self.config.SUPPORTED_FORMATS):
                video_files.append(file_path)

        self.logger.info(f"Found {len(video_files)} video files in {directory}")
        return sorted(video_files)

    def validate_files(self, file_paths: List[str]) -> List[Path]:
        """Validate file paths"""
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

        # Handle tqdm import
        tqdm_available = False
        if max_workers == 1:
            try:
                from tqdm import tqdm
                tqdm_available = True
            except ImportError:
                self.logger.warning("tqdm not available, disabling progress bars")
                tqdm_available = False

        if max_workers == 1:
            outer_pbar = None
            if tqdm_available and len(files) > 1:
                outer_pbar = tqdm(total=len(files), desc="Processing files")

            for i, file_path in enumerate(files):
                inner_pbar = None
                if tqdm_available:
                    inner_desc = f"File {i+1}/{len(files)}: {file_path.name}"
                    inner_pbar = tqdm(total=100, position=1, desc=inner_desc, leave=False)

                success = self.extractor.extract_audio(file_path, inner_pbar=inner_pbar)
                if inner_pbar:
                    inner_pbar.close()

                if success:
                    successful += 1
                else:
                    failed += 1

                if outer_pbar:
                    outer_pbar.update(1)

            if outer_pbar:
                outer_pbar.close()
        else:
            self.logger.info(f"Parallel processing {len(files)} files with {max_workers} workers")
            outer_pbar = None
            if tqdm_available:
                outer_pbar = tqdm(total=len(files), desc="Processing files (parallel)")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.extractor.extract_audio, file_path): file_path
                    for file_path in files
                }

                for future in as_completed(future_to_file):
                    success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1

                    if outer_pbar:
                        outer_pbar.update(1)

            if outer_pbar:
                outer_pbar.close()

        self.logger.info(f"Processing complete: {successful} successful, {failed} failed")




def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Extract high-quality audio from videos (lossless copy or adaptive re-encode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extractAudio.py -w /path/to/videos          # Extract AAC from directory
  python extractAudio.py -f video1.mkv video2.avi -o flac   # Lossless FLAC for files
  python extractAudio.py -w . -v --max-workers 2 -m  # Parallel with move
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--working-dir", type=str, help="Directory containing videos to process")
    group.add_argument("-f", "--files", nargs="+", help="Specific video files to process (space-separated)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel workers (default: 1)")
    parser.add_argument("-o", "--output-format", type=str, choices=["mp3", "aac", "flac"], default="aac",
                        help="Output audio format: mp3, aac (default), or flac (lossless)")
    parser.add_argument("-m", "--move-files", action="store_true", help="Move original files to 'original_files' after extraction")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(tool_name="extractAudio", verbose=args.verbose)

    if not check_dependencies():
        logger.error("FFmpeg is required but not found in PATH")
        sys.exit(1)

    # Validate inputs
    try:
        max_workers = validate_workers(args.max_workers)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    config = Config()

    # Validate config before processing
    try:
        config.validate()
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    processor = VideoProcessor(config, args.output_format, args.verbose, args.move_files)

    try:
        if args.working_dir:
            directory = Path(args.working_dir).resolve()
            files = processor.find_video_files(directory)
        else:
            files = processor.validate_files(args.files)

        processor.process_files(files, args.max_workers)

    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
