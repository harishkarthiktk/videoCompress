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
- Compression factor adjustment
- Option to move original files
"""

import datetime
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from utils import Config, check_dependencies, VideoProcessor

# Application constants
DEFAULT_MAX_WORKERS = 1
DEFAULT_WINDOW_SIZE = "800x600"
DEFAULT_LOG_HEIGHT = 15
DEFAULT_WORKER_RANGE = (1, 10)
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_DIR_NAME = "logs"

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

        # Configuration variables
        self.verbose_var = tk.BooleanVar()
        self.max_workers_var = tk.IntVar(value=DEFAULT_MAX_WORKERS)
        self.compression_factor_var = tk.DoubleVar(value=70.0)
        self.move_files_var = tk.BooleanVar()
        self.recursive_var = tk.BooleanVar()

        # State variables
        self.files: List[Path] = []
        self.processor: Optional[VideoProcessor] = None
        self.processing_running = False
        self.stop_processing = False
        self.current_file = ""

        # UI elements
        self.dir_entry: Optional[tk.Entry] = None
        self.gpu_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.progress: Optional[ttk.Progressbar] = None
        self.per_file_progress: Optional[ttk.Progressbar] = None
        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.start_button: Optional[tk.Button] = None
        self.stop_button: Optional[tk.Button] = None
        self.stop_button: Optional[tk.Button] = None

        # Progress callback
        self.per_file_callback = self._create_per_file_callback()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            self.check_dependencies()
            self.detect_gpu()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
            self.root.quit()

    def _create_per_file_callback(self) -> Callable[[float], None]:
        """Create a per-file progress callback function."""
        def callback(percent: float) -> None:
            self.root.after(0, lambda p=percent: (
                self.status_label.config(text=f"Processing: {self.current_file} ({p:.1f}%)" if self.current_file else f"Processing ({p:.1f}%)"),
                self.per_file_progress.config(value=p) if self.per_file_progress else None
            ))
        return callback

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

        # Verbose logging
        tk.Checkbutton(options_frame, text="Verbose Logging", variable=self.verbose_var).grid(row=0, column=0, sticky='w')

        # Max workers
        tk.Label(options_frame, text="Max Workers:").grid(row=1, column=0, sticky='w')
        tk.Spinbox(
            options_frame,
            from_=DEFAULT_WORKER_RANGE[0],
            to=DEFAULT_WORKER_RANGE[1],
            textvariable=self.max_workers_var,
            width=5
        ).grid(row=1, column=1, sticky='w')

        # Compression factor
        tk.Label(options_frame, text="Compression Factor (%):").grid(row=0, column=2, sticky='w', padx=(10,0))
        self.compression_scale = tk.Scale(
            options_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.compression_factor_var,
            length=200
        )
        self.compression_scale.grid(row=0, column=3, sticky='w', padx=(5,0))

        # Move files
        tk.Checkbutton(options_frame, text="Move Original Files to Folder", variable=self.move_files_var).grid(row=2, column=0, columnspan=2, sticky='w', pady=5)

        # Recursive search
        tk.Checkbutton(options_frame, text="Recursive Directory Search", variable=self.recursive_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=5)

    def _setup_status_display(self) -> None:
        """Setup GPU info, status label, and progress bar"""
        # GPU info
        self.gpu_label = tk.Label(self.root, text="Detected GPU: Detecting...")
        self.gpu_label.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        # Overall progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        # Per-file progress bar (for sequential)
        self.per_file_progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.per_file_progress.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=(5,0))

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
        self.stop_button = tk.Button(buttons_frame, text="Stop Processing", command=self.stop_processing, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
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
        # Since VideoProcessor detects GPU, but to show early, import and use
        from utils import GPUDetector
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

            # Create config
            compression_factor = self.compression_factor_var.get() / 100.0
            config = Config(compression_factor=compression_factor)

            # Create processor with callbacks
            self.processor = VideoProcessor(
                config,
                self.verbose_var.get(),
                self.move_files_var.get(),
                per_file_callback=self.per_file_callback
            )

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

        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.verbose_var.get() else logging.INFO)

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
            self.progress['maximum'] = 0  # Will set later
        if self.per_file_progress:
            self.per_file_progress['value'] = 0
            self.per_file_progress['maximum'] = 100
        if self.status_label:
            self.status_label.config(text="Initializing...")
        self.clear_log()

    def _get_files_to_process(self) -> List[Path]:
        """Get the list of files to process."""
        if self.files and self.processor:
            return self.processor.validate_files([str(f) for f in self.files])
        elif self.dir_entry and self.dir_entry.get().strip() and self.processor:
            directory = Path(self.dir_entry.get().strip())
            return self.processor.find_video_files(directory, recursive=self.recursive_var.get())
        return []

    def _start_processing_thread(self, files: List[Path]) -> None:
        """Start processing in a separate thread."""
        if self.progress:
            self.progress['maximum'] = len(files)
        if self.per_file_progress:
            self.per_file_progress['maximum'] = 100
        if self.status_label:
            self.status_label.config(text=f"Starting processing of {len(files)} files...")
        if self.log_text:
            self.log_text.insert(tk.END, f"Starting processing of {len(files)} files...\n")
            self.log_text.see(tk.END)

        # Start processing in separate thread to avoid freezing GUI
        self.processing_running = True
        self.stop_processing = False
        if self.start_button:
            self.start_button.config(state='disabled')
        if self.stop_button:
            self.stop_button.config(state='normal')
        processing_thread = threading.Thread(target=self._process_files_thread, args=(files,))
        processing_thread.daemon = True
        processing_thread.start()

    def _reset_processing_state(self) -> None:
        """Reset processing state after error."""
        if self.status_label:
            self.status_label.config(text="Error occurred")
        self.processing_running = False
        if self.start_button:
            self.start_button.config(state='normal')

    def _process_files_thread(self, files: List[Path]) -> None:
        """Process files in background thread using ThreadPoolExecutor"""
        try:
            successful = 0
            failed = 0

            max_workers = self.max_workers_var.get()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.processor.converter.convert_video, file_path): file_path
                    for file_path in files
                }

                for future in as_completed(future_to_file):
                    if self.stop_processing:
                        if self.status_label:
                            self.root.after(0, lambda: self.status_label.config(text="Processing stopped"))
                        break

                    file_path = future_to_file[future]
                    self.current_file = file_path.name
                    self.root.after(0, lambda fn=file_path.name: self.status_label.config(text=f"Processing: {fn}"))

                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logging.error(f"Error processing {file_path.name}: {e}")
                        if self.log_text:
                            self.root.after(0, lambda msg=f"Error processing {file_path.name}: {e}\n": self.log_text.insert(tk.END, msg))
                        failed += 1

                    # Update progress
                    completed = successful + failed
                    self.root.after(0, lambda c=completed: self.progress.config(value=c))

            if not self.stop_processing:
                self.root.after(0, lambda s=successful, f=failed: self._show_processing_results(s, f))

        except Exception as e:
            logging.error(f"Unexpected error in process_files: {e}")
            if self.log_text:
                self.root.after(0, lambda msg=f"Unexpected error: {e}\n": self.log_text.insert(tk.END, msg))
            if self.status_label:
                self.root.after(0, lambda: self.status_label.config(text="Error occurred"))

        self.processing_running = False
        self.root.after(0, lambda: (self.start_button.config(state='normal'), self.stop_button.config(state='disabled')) if self.start_button and self.stop_button else None)

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

    def stop_processing(self) -> None:
        """Stop the current processing."""
        self.stop_processing = True
        if self.status_label:
            self.status_label.config(text="Stopping processing...")

    def stop_processing(self) -> None:
        """Stop the current processing."""
        self.stop_processing = True
        if self.status_label:
            self.status_label.config(text="Stopping processing...")

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