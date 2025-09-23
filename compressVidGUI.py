#!/usr/bin/env python3
"""
Video Compression GUI with GPU Acceleration

Graphical user interface for video compression script.
Provides intuitive file selection, progress tracking, and logging.

Features:
- Directory or file selection
- GPU detection display
- Progress bar and real-time logging
- Configurable parallel processing
"""

import datetime
import json
import logging
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Application constants
DEFAULT_MAX_WORKERS = 1
DEFAULT_WINDOW_SIZE = "800x600"
DEFAULT_LOG_HEIGHT = 15
DEFAULT_WORKER_RANGE = (1, 10)
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_DIR_NAME = "logs"
FFMPEG_TIMEOUT = 3600  # 1 hour timeout for FFmpeg operations
DEPENDENCY_CHECK_TIMEOUT = 10
GPU_DETECTION_TIMEOUT = 10
VIDEO_INFO_TIMEOUT = 30


@dataclass
class Config:
    """Configuration settings for video compression"""
    SUPPORTED_FORMATS: Tuple[str, ...] = (
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    )
    MAX_BITRATE: int = 10000  # Max total bitrate in kbps (10 Mbps)
    COMPRESSION_FACTOR: float = 0.7  # Reduce bitrate by 30%
    MAX_DOWNSCALE_PERCENT: float = 0.5  # 20% max resolution reduction
    OUTPUT_SUFFIX: str = "_conv"
    MIN_AUDIO_BITRATE: int = 64  # Minimum audio bitrate in kbps


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
    def get_info(file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], bool, Optional[int], Optional[float], Optional[float]]:
        """
        Extract video resolution, bitrates, audio presence, duration, and fps using ffprobe

        Returns:
            Tuple of (width, height, video_bitrate_kbps, has_audio, audio_bitrate_kbps, duration_sec, fps)
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

            video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)

            if not video_stream:
                logging.error(f"No video stream found in {file_path}")
                return None, None, None, False, None, None, None

            width = video_stream.get('width')
            height = video_stream.get('height')
            video_bitrate = video_stream.get('bit_rate')

            if video_bitrate is not None:
                video_bitrate = int(video_bitrate) // 1000

            has_audio = audio_stream is not None
            audio_bitrate = None
            if has_audio and 'bit_rate' in audio_stream:
                audio_bitrate = int(audio_stream['bit_rate']) // 1000

            duration = video_stream.get('duration')
            if duration is not None:
                duration = float(duration)

            r_frame_rate = video_stream.get('r_frame_rate')
            fps = None
            if r_frame_rate:
                try:
                    # Safely evaluate frame rate expression like "30/1" or "29970/1000"
                    fps = eval(r_frame_rate, {"__builtins__": {}})
                except (ValueError, TypeError, NameError, ZeroDivisionError):
                    fps = None

            return width, height, video_bitrate, has_audio, audio_bitrate, duration, fps

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get video info for {file_path}: {e}")
            return None, None, None, False, None, None, None


class VideoConverter:
    """Handles video conversion with GPU acceleration"""

    def __init__(self, gpu_type: str, config: Config, progress_callback: Optional[callable] = None) -> None:
        self.gpu_type = gpu_type
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.progress_callback = progress_callback

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
            "-vf", f"scale='min(iw,{new_width})':'min(ih,{new_height})':force_original_aspect_ratio=decrease"
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
            str(output_file)
        ])

        return cmd

    def convert_video(self, input_file: Path) -> bool:
        """
        Convert a single video file.

        Args:
            input_file: Path to the input video file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate output filename
            output_file = input_file.parent / f"{input_file.stem}{self.config.OUTPUT_SUFFIX}.mp4"

            # Skip if already exists
            if output_file.exists():
                self.logger.info(f"Skipping {input_file.name} - already converted")
                return True

            # Get video info
            video_info = VideoInfo.get_info(input_file)
            if video_info[0] is None or video_info[1] is None:
                self.logger.error(f"Could not determine video dimensions for {input_file}")
                return False

            width, height, video_bitrate, has_audio, audio_bitrate, duration, fps = video_info

            # Calculate total frames for progress
            total_frames = None
            if duration and fps and duration > 0 and fps > 0:
                total_frames = int(duration * fps)

            # Calculate target parameters
            new_width, new_height, target_video_bitrate, target_audio_bitrate = self._calculate_target_params(
                width, height, video_bitrate, audio_bitrate
            )

            self.logger.info(
                f"Converting {input_file.name} -> {output_file.name} "
                f"({width}x{height} -> {new_width}x{new_height}, "
                f"Video: {video_bitrate or 'unknown'}kbps -> {target_video_bitrate}kbps, "
                f"Audio: {audio_bitrate or 'unknown'}kbps -> {target_audio_bitrate}kbps) "
                f"using {self.gpu_type}"
            )

            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(
                input_file, output_file, new_width, new_height,
                target_video_bitrate, target_audio_bitrate
            )

            # Run FFmpeg with progress parsing
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            frame = 0
            try:
                for line in iter(process.stderr.readline, ''):
                    if not line:  # Process finished
                        break
                    if 'frame=' in line and total_frames:
                        try:
                            frame_str = line.split('frame=')[1].split()[0]
                            frame = int(frame_str)
                            if self.progress_callback and total_frames > 0:
                                percent = min((frame / total_frames) * 100, 100.0)
                                self.progress_callback(percent)
                        except (ValueError, IndexError, ZeroDivisionError):
                            pass
            except Exception as e:
                self.logger.warning(f"Error parsing FFmpeg progress for {input_file}: {e}")

            retcode = process.wait()
            if retcode != 0:
                stderr_output = process.stderr.read() if process.stderr else ""
                self.logger.error(f"FFmpeg failed for {input_file}: {stderr_output}")
                # Clean up partial file
                if output_file.exists():
                    try:
                        output_file.unlink()
                    except OSError as e:
                        self.logger.warning(f"Could not remove partial file {output_file}: {e}")
                return False

            # Final progress update
            if self.progress_callback and total_frames:
                self.progress_callback(100.0)

            self.logger.info(f"Successfully converted {input_file.name}")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"Conversion timeout for {input_file}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error converting {input_file}: {e}")
            return False


class VideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, config: Config, verbose: bool = False, progress_callback: Optional[callable] = None) -> None:
        self.config = config
        self.logger = setup_logging(verbose)
        self.gpu_type = GPUDetector.detect()
        self.converter = VideoConverter(self.gpu_type, config, progress_callback)

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


class GUITextHandler(logging.Handler):
    """Custom logging handler to redirect logs to GUI text widget"""

    def __init__(self, text_widget: scrolledtext.ScrolledText) -> None:
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the GUI text widget."""
        try:
            msg = self.format(record)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
        except Exception:
            # Avoid infinite recursion if logging fails
            pass


class VideoCompressorGUI:
    """Main GUI application class"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Compressor GUI")
        self.root.geometry(DEFAULT_WINDOW_SIZE)

        # Set window icon
        if HAS_PIL:
            try:
                icon_image = Image.open("src/images/icon_corrected.jpg")
                icon_photo = ImageTk.PhotoImage(icon_image)
                self.root.iconphoto(True, icon_photo)
            except Exception as e:
                print(f"Warning: Failed to load icon: {e}")

        # Configuration
        self.config = Config()
        self.verbose = tk.BooleanVar()
        self.max_workers = tk.IntVar(value=DEFAULT_MAX_WORKERS)

        # State variables
        self.files: List[Path] = []
        self.processor: Optional[VideoProcessor] = None
        self.processing_running = False
        self.stop_processing = False
        self.current_file = ""

        # UI elements (will be initialized in setup_ui)
        self.dir_entry: Optional[tk.Entry] = None
        self.gpu_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.progress: Optional[ttk.Progressbar] = None
        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.start_button: Optional[tk.Button] = None

        # Progress callback
        self.progress_callback = self._create_progress_callback()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            self.check_dependencies()
            self.detect_gpu()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
            self.root.quit()

    def _create_progress_callback(self) -> callable:
        """Create a progress callback function."""
        def progress_callback(percent: float) -> None:
            if self.status_label:
                self.status_label.config(text=f"Processing: {self.current_file} ({percent:.1f}%)")
        return progress_callback

    def setup_ui(self) -> None:
        """Setup the user interface components"""
        self._setup_file_selection()
        self._setup_options()
        self._setup_status_display()
        self._setup_log_area()
        self._setup_buttons()
        self._configure_grid_weights()

    def _setup_file_selection(self) -> None:
        """Setup directory and file selection components"""
        # Directory selection
        tk.Label(self.root, text="Select Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.dir_entry = tk.Entry(self.root, width=50)
        self.dir_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.select_directory).grid(row=0, column=2, padx=5, pady=5)

        # File selection
        tk.Label(self.root, text="Or Select Files:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Button(self.root, text="Browse Files", command=self.select_files).grid(row=1, column=1, sticky='w', padx=5, pady=5)

    def _setup_options(self) -> None:
        """Setup options frame with configuration controls"""
        options_frame = ttk.LabelFrame(self.root, text="Options", padding="5")
        options_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        tk.Checkbutton(options_frame, text="Verbose Logging", variable=self.verbose).grid(row=0, column=0, sticky='w')
        tk.Label(options_frame, text="Max Workers:").grid(row=1, column=0, sticky='w')
        tk.Spinbox(
            options_frame,
            from_=DEFAULT_WORKER_RANGE[0],
            to=DEFAULT_WORKER_RANGE[1],
            textvariable=self.max_workers,
            width=5
        ).grid(row=1, column=1, sticky='w')

    def _setup_status_display(self) -> None:
        """Setup GPU info, status label, and progress bar"""
        # GPU info
        self.gpu_label = tk.Label(self.root, text="Detected GPU: Detecting...")
        self.gpu_label.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    def _setup_log_area(self) -> None:
        """Setup the log output area"""
        log_frame = ttk.LabelFrame(self.root, text="Log Output", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=DEFAULT_LOG_HEIGHT, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _setup_buttons(self) -> None:
        """Setup control buttons"""
        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=7, column=0, columnspan=3, pady=5)

        self.start_button = tk.Button(buttons_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

    def _configure_grid_weights(self) -> None:
        """Configure grid weights for proper resizing"""
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def select_directory(self) -> None:
        """Open directory selection dialog"""
        dir_path = filedialog.askdirectory(title="Select Video Directory")
        if dir_path:
            if self.dir_entry:
                self.dir_entry.delete(0, tk.END)
                self.dir_entry.insert(0, dir_path)
            self.files = []  # Clear selected files

    def select_files(self) -> None:
        """Open file selection dialog"""
        filetypes = [
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.mpg *.3gp"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=filetypes)
        if files:
            try:
                self.files = [Path(f) for f in files]
                if self.dir_entry:
                    self.dir_entry.delete(0, tk.END)  # Clear directory selection
                if self.log_text:
                    self.log_text.insert(tk.END, f"Selected {len(self.files)} files\n")
                    self.log_text.see(tk.END)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process selected files: {e}")

    def check_dependencies(self) -> None:
        """Check for FFmpeg dependencies"""
        if not check_dependencies():
            messagebox.showerror(
                "Missing Dependencies",
                "FFmpeg and FFprobe are required but not found in PATH.\n\n"
                "Please install FFmpeg from: https://ffmpeg.org/download.html"
            )
            self.root.quit()

    def detect_gpu(self) -> None:
        """Detect and display GPU information"""
        gpu = GPUDetector.detect()
        if self.gpu_label:
            self.gpu_label.config(text=f"Detected GPU: {gpu}")

    def start_processing(self) -> None:
        """Start the video processing"""
        try:
            if not self._validate_selection():
                return

            self._setup_logging()
            self._reset_ui_for_processing()

            # Create processor
            self.processor = VideoProcessor(self.config, self.verbose.get(), progress_callback=self.progress_callback)

            # Get files to process
            files = self._get_files_to_process()
            if not files:
                messagebox.showinfo("No Files", "No supported video files found in the selected location.")
                return

            self._start_processing_thread(files)

        except Exception as e:
            logging.error(f"Error in start_processing: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
            self._reset_processing_state()

    def _validate_selection(self) -> bool:
        """Validate that files or directory are selected."""
        if not self.files and (not self.dir_entry or not self.dir_entry.get().strip()):
            messagebox.showwarning("No Selection", "Please select a directory or specific files to process.")
            return False
        return True

    def _setup_logging(self) -> None:
        """Setup logging configuration for processing."""
        # Create logs directory
        log_dir = Path(LOG_DIR_NAME)
        log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
        log_file = log_dir / f"log_{timestamp}.txt"

        # Setup logging to GUI and file
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.verbose.get() else logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Add GUI handler
        if self.log_text:
            gui_handler = GUITextHandler(self.log_text)
            gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
            logger.addHandler(gui_handler)

    def _reset_ui_for_processing(self) -> None:
        """Reset UI elements for processing start."""
        if self.progress:
            self.progress['value'] = 0
        if self.status_label:
            self.status_label.config(text="Initializing...")
        self.clear_log()

    def _get_files_to_process(self) -> List[Path]:
        """Get the list of files to process."""
        if self.files and self.processor:
            return self.processor.validate_files([str(f) for f in self.files])
        elif self.dir_entry and self.processor:
            directory = Path(self.dir_entry.get().strip())
            return self.processor.find_video_files(directory)
        return []

    def _start_processing_thread(self, files: List[Path]) -> None:
        """Start processing in a separate thread."""
        if self.progress:
            self.progress['maximum'] = len(files)
        if self.status_label:
            self.status_label.config(text=f"Starting processing of {len(files)} files...")
        if self.log_text:
            self.log_text.insert(tk.END, f"Starting processing of {len(files)} files...\n")
            self.log_text.see(tk.END)

        # Start processing in separate thread to avoid freezing GUI
        self.processing_running = True
        if self.start_button:
            self.start_button.config(state='disabled')
        processing_thread = threading.Thread(target=self.process_files, args=(files,))
        processing_thread.daemon = True
        processing_thread.start()

    def _reset_processing_state(self) -> None:
        """Reset processing state after error."""
        if self.status_label:
            self.status_label.config(text="Error occurred")
        self.processing_running = False
        if self.start_button:
            self.start_button.config(state='normal')

    def process_files(self, files: List[Path]) -> None:
        """Process files in background thread using ThreadPoolExecutor"""
        try:
            successful = 0
            failed = 0

            max_workers = self.max_workers.get()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.processor.converter.convert_video, file_path): file_path
                    for file_path in files
                }

                for future in as_completed(future_to_file):
                    if self.stop_processing:
                        if self.status_label:
                            self.status_label.config(text="Processing stopped")
                        break

                    file_path = future_to_file[future]
                    self.current_file = file_path.name

                    if self.status_label:
                        self.status_label.config(text=f"Processing: {file_path.name}")

                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logging.error(f"Error processing {file_path.name}: {e}")
                        if self.log_text:
                            self.log_text.insert(tk.END, f"Error processing {file_path.name}: {e}\n")
                        failed += 1

                    # Update progress
                    if self.progress:
                        self.progress['value'] = successful + failed
                    self.root.update_idletasks()

            if not self.stop_processing:
                self._show_processing_results(successful, failed)

        except Exception as e:
            logging.error(f"Unexpected error in process_files: {e}")
            if self.log_text:
                self.log_text.insert(tk.END, f"Unexpected error: {e}\n")
            if self.status_label:
                self.status_label.config(text="Error occurred")

        self.processing_running = False

    def _show_processing_results(self, successful: int, failed: int) -> None:
        """Display processing completion results."""
        if self.status_label:
            self.status_label.config(text="Processing complete")

        summary = f"\nProcessing complete: {successful} successful, {failed} failed\n"
        if self.log_text:
            self.log_text.insert(tk.END, summary)
            self.log_text.see(tk.END)

        # Show completion message
        if failed == 0:
            messagebox.showinfo("Success", f"All {successful} files processed successfully!")
        else:
            messagebox.showwarning("Completed with Errors", f"Processed {successful} files successfully, {failed} failed.")
    def clear_log(self) -> None:
        """Clear the log output"""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)

    def on_closing(self) -> None:
        """Handle window close event"""
        if self.processing_running:
            if messagebox.askyesno("Confirm Exit", "Processing is running. Stop processing and exit?"):
                self.stop_processing = True
                if self.status_label:
                    self.status_label.config(text="Stopping...")
                # Give a moment for threads to stop
                self.root.after(1000, self.root.quit)
            # If no, do nothing - user cancelled exit
        else:
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCompressorGUI(root)
    root.mainloop()