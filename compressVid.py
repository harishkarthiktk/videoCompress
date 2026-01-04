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

from utils import Config, setup_logging, check_dependencies, VideoProcessor, validate_compression_factor, validate_workers, ValidationError
from presets import PresetManager, Preset, DEFAULT_PRESETS
from queue import QueueManager, ProcessingStatus
from workflows import WorkflowExecutor, WORKFLOWS

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
    group = parser.add_mutually_exclusive_group(required=False)
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

    parser.add_argument(
        "--preset",
        type=str,
        choices=['fast', 'balanced', 'quality', 'archive', 'streaming'],
        help="Use preset compression profile"
    )

    parser.add_argument(
        "--save-preset",
        type=str,
        metavar="NAME",
        help="Save current compression settings as preset NAME"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze files and estimate compression without encoding"
    )

    parser.add_argument(
        "--create-queue",
        action="store_true",
        help="Create processing queue (save state for resuming)"
    )

    parser.add_argument(
        "--resume-queue",
        type=str,
        metavar="QUEUE_ID",
        help="Resume incomplete processing queue"
    )

    parser.add_argument(
        "--list-queues",
        action="store_true",
        help="List incomplete queues available to resume"
    )

    parser.add_argument(
        "--workflow",
        type=str,
        choices=list(WORKFLOWS.keys()),
        help="Run multi-step workflow"
    )

    parser.add_argument(
        "--list-workflows",
        action="store_true",
        help="List available workflows"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(tool_name="compressVid", verbose=args.verbose)

    # Initialize preset manager
    preset_mgr = PresetManager()

    # Initialize queue manager
    queue_mgr = QueueManager()

    # Handle --list-workflows
    if args.list_workflows:
        print("Available workflows:")
        for name, workflow in WORKFLOWS.items():
            print(f"  {name}: {workflow.description}")
        sys.exit(0)

    # Handle --list-queues
    if args.list_queues:
        incomplete = queue_mgr.list_incomplete_queues()
        if not incomplete:
            print("No incomplete queues")
        else:
            print("Incomplete queues:")
            for queue_id in incomplete:
                queue_obj = queue_mgr.load_queue(queue_id)
                print(f"  {queue_id}: {queue_obj.get_summary()}")
        sys.exit(0)

    # Handle --list-presets
    if args.list_presets:
        print("Available presets:")
        presets = preset_mgr.list_presets()
        if not presets:
            print("  (no custom presets - using built-in defaults)")
        for name, desc in presets.items():
            print(f"  {name}: {desc}")
        for name, preset in DEFAULT_PRESETS.items():
            print(f"  {name}: {preset.description}")
        sys.exit(0)

    # Require either -w or -f for processing
    if not args.working_dir and not args.files:
        parser.error("Either -w/--working-dir or -f/--files is required")

    # Check dependencies
    if not check_dependencies():
        logger.error("FFmpeg and FFprobe are required but not found in PATH")
        logger.error("Please install FFmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Validate inputs
    try:
        compression_factor = validate_compression_factor(args.compression_factor / 100.0)
        max_workers = validate_workers(args.max_workers)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)

    # Handle --preset
    if args.preset:
        # First check custom presets, then built-in defaults
        preset = preset_mgr.load_preset(args.preset)
        if not preset:
            preset = DEFAULT_PRESETS.get(args.preset)
        if not preset:
            logger.error(f"Preset '{args.preset}' not found")
            sys.exit(1)
        compression_factor = preset.compression_factor
        max_workers = preset.max_workers
        logger.info(f"Using preset '{args.preset}': {preset.description}")
        config = Config(compression_factor=compression_factor, use_adaptive=preset.use_adaptive, dry_run=args.dry_run)
    else:
        # Initialize config
        config = Config(compression_factor=compression_factor, dry_run=args.dry_run)

    # Log dry-run mode if enabled
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no files will be modified")

    # Validate config before processing
    try:
        config.validate()
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize processor
    processor = VideoProcessor(config, args.verbose, args.move_files)

    # Handle queue operations
    queue_obj = None
    if args.resume_queue:
        queue_obj = queue_mgr.load_queue(args.resume_queue)
        if not queue_obj:
            logger.error(f"Queue '{args.resume_queue}' not found")
            sys.exit(1)
        logger.info(f"Resuming queue: {queue_obj.get_summary()}")
        processor.queue = queue_obj
    elif args.create_queue:
        queue_obj = queue_mgr.create_queue(preset=args.preset or 'balanced', dry_run=args.dry_run)
        logger.info(f"Created processing queue: {queue_obj.id}")
        processor.queue = queue_obj

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

        # Populate queue with files if it was created
        if queue_obj and args.create_queue:
            for file in files:
                output_file = file.parent / f"{file.stem}{config.OUTPUT_SUFFIX}.mp4"
                queue_obj.add_file(file, output_file)
            queue_mgr.save_queue(queue_obj)

        # Handle workflow execution
        if args.workflow:
            workflow = WORKFLOWS[args.workflow]
            executor = WorkflowExecutor(logger=logger)

            for file in files:
                logger.info(f"Running workflow '{workflow.name}' on {file.name}")
                try:
                    results = executor.execute(workflow, file)
                    logger.info(f"Workflow completed: {results}")
                except Exception as e:
                    logger.error(f"Workflow failed: {e}")
            # Exit after workflow execution
            sys.exit(0)

        # Process the files
        successful, failed = processor.process_files(files, args.max_workers, use_tqdm=True)

        processor.logger.info(f"Processing complete: {successful} successful, {failed} failed")

        # Handle --save-preset
        if args.save_preset:
            new_preset = Preset(
                name=args.save_preset,
                compression_factor=compression_factor,
                max_workers=max_workers,
                use_adaptive=config.use_adaptive,
                description=f"Custom preset: {compression_factor*100:.0f}% factor, {max_workers} workers",
                tags=['custom']
            )
            preset_mgr.save_preset(new_preset)
            logger.info(f"Preset '{args.save_preset}' saved")

    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()