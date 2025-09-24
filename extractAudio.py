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
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings"""
    SUPPORTED_FORMATS = (
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    )
    OUTPUT_SUFFIX = "_audio"   # Suffix for the output file
    DEFAULT_FORMAT = "mp3"     # Default format if not specified
    DEFAULT_BITRATES = {
        "mp3": "192k",
        "aac": "128k"
    }
    EXTENSIONS = {
        "mp3": ".mp3",
        "aac": ".m4a"
    }


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class AudioExtractor:
    """Handles audio extraction"""

    def __init__(self, config: Config, output_format: str):
        self.config = config
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)

    def _build_ffmpeg_command(self, input_file: Path, output_file: Path) -> List[str]:
        """Build FFmpeg command to extract audio only"""
        if self.output_format == "mp3":
            return [
                "ffmpeg", "-y", "-i", str(input_file),
                "-vn", "-c:a", "libmp3lame",
                "-b:a", self.config.DEFAULT_BITRATES["mp3"],
                str(output_file)
            ]
        elif self.output_format == "aac":
            return [
                "ffmpeg", "-y", "-i", str(input_file),
                "-vn", "-c:a", "aac",
                "-b:a", self.config.DEFAULT_BITRATES["aac"],
                str(output_file)
            ]
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def extract_audio(self, input_file: Path) -> bool:
        """Extract audio from a single video file"""
        try:
            ext = self.config.EXTENSIONS[self.output_format]
            output_file_name = f"{input_file.stem}{self.config.OUTPUT_SUFFIX}{ext}"
            output_file = input_file.parent / output_file_name

            if output_file.exists():
                self.logger.info(f"Skipping {input_file.name} - already extracted")
                return True

            self.logger.info(f"Processing {input_file.name} -> {output_file.name} (Extracting {self.output_format.upper()})")

            cmd = self._build_ffmpeg_command(input_file, output_file)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                self.logger.error(f"FFmpeg failed for {input_file}: {result.stderr}")
                if output_file.exists():
                    output_file.unlink()
                return False

            self.logger.info(f"Successfully processed {input_file.name}")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout extracting audio from {input_file}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error processing {input_file}: {e}")
            return False


class VideoProcessor:
    """Main processing orchestrator"""

    def __init__(self, config: Config, output_format: str, verbose: bool = False):
        self.config = config
        self.logger = setup_logging(verbose)
        self.extractor = AudioExtractor(config, output_format)

    def find_video_files(self, directory: Path, exclude: str) -> List[Path]:
        """Find all supported video files in directory (ignores excluded ones)"""
        video_files = []

        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory}")
            return video_files

        for file_path in directory.iterdir():
            if (file_path.is_file()
                    and file_path.suffix in self.config.SUPPORTED_FORMATS
                    and (exclude not in file_path.name)):
                video_files.append(file_path)

        self.logger.info(f"Found {len(video_files)} video files in {directory}")
        return sorted(video_files)

    def validate_files(self, file_paths: List[str], exclude: str) -> List[Path]:
        """Validate file paths, filter excluded ones"""
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
            if exclude in file_path.name:
                self.logger.info(f"Excluding file: {file_path}")
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
            for file_path in files:
                if self.extractor.extract_audio(file_path):
                    successful += 1
                else:
                    failed += 1
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.extractor.extract_audio, file_path): file_path
                    for file_path in files
                }
                for future in as_completed(future_to_file):
                    if future.result():
                        successful += 1
                    else:
                        failed += 1

        self.logger.info(f"Processing complete: {successful} successful, {failed} failed")


def check_dependencies() -> bool:
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Extract audio from videos and save as MP3 or AAC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -W /path/to/videos --output-format aac   # Extract AAC audio
  %(prog)s -F video1.mkv video2.avi --output-format mp3
  %(prog)s -W . --exclude sample --output-format mp3
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-W", "--working-dir", type=str, help="Directory containing videos to process")
    group.add_argument("-F", "--files", nargs="+", help="Specific video files to process (space-separated)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel workers (default: 1)")
    parser.add_argument("--exclude", type=str, default="", help="Exclude files containing this string")
    parser.add_argument("--output-format", type=str, choices=["mp3", "aac"], default=Config.DEFAULT_FORMAT,
                        help="Choose output audio format: mp3 (default) or aac")

    args = parser.parse_args()

    if not check_dependencies():
        print("Error: FFmpeg is required but not found in PATH")
        sys.exit(1)

    config = Config()
    processor = VideoProcessor(config, args.output_format, args.verbose)

    try:
        if args.working_dir:
            directory = Path(args.working_dir).resolve()
            files = processor.find_video_files(directory, args.exclude)
        else:
            files = processor.validate_files(args.files, args.exclude)

        processor.process_files(files, args.max_workers)

    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
