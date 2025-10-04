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
- Option to move original files to 'original_files' folder after conversion
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

from utils import Config, setup_logging, check_dependencies, VideoProcessor

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert videos to .mp4 with optimal compression and GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compressVid.py -w /path/to/videos          # Process all videos in directory
  python compressVid.py -f video1.mkv video2.avi   # Process specific files
  python compressVid.py -w . -v                     # Process current directory with verbose output
  python compressVid.py -w . -c 50                  # Process with 50% compression factor
  python compressVid.py -w . -m                     # Process and move originals to 'original_files'
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
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers (default: 1, sequential)"
    )

    parser.add_argument(
        "-c", "--compression-factor",
        type=float,
        default=70.0,
        help="Compression factor as percentage of original bitrate (0-100, default: 70)"
    )

    parser.add_argument(
        "-m", "--move-files",
        action="store_true",
        help="Move original files to 'original_files' folder after conversion"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        print("Error: FFmpeg and FFprobe are required but not found in PATH")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Initialize config
    compression_factor = args.compression_factor / 100.0
    if not 0 <= compression_factor <= 1:
        print("Error: Compression factor must be between 0 and 100")
        sys.exit(1)
    config = Config(compression_factor=compression_factor)

    # Initialize processor
    processor = VideoProcessor(config, args.verbose, args.move_files)

    try:
        if args.working_dir:
            # Process directory
            directory = Path(args.working_dir).resolve()
            files = processor.find_video_files(directory)
        else:
            # Process specific files
            files = processor.validate_files(args.files)

        if not files:
            processor.logger.warning("No valid video files found to process")
            sys.exit(0)

        # Process the files
        successful, failed = processor.process_files(files, args.max_workers, use_tqdm=True)

        processor.logger.info(f"Processing complete: {successful} successful, {failed} failed")

    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()