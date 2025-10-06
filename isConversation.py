#!/usr/bin/env python3
"""
Offline Video Conversation Detector
===================================

This script analyzes video files to detect human conversation in the audio track.
It supports multilingual speech (e.g., English, Hindi, Tamil) using local, offline tools:
- FFmpeg for audio extraction
- WebRTC VAD for voice activity detection
- OpenAI Whisper for transcription and conversation heuristics

Prerequisites (install once):
- FFmpeg: https://ffmpeg.org/download.html (add to PATH)
- pip install openai-whisper webrtcvad tqdm soundfile numpy

Features:
- Processes directories or specific files
- Configurable thresholds and Whisper model
- Progress tracking with tqdm
- JSON reports (optional)
- Dry-run mode for testing (VAD only)
- No file modifications; analysis-only

Configuration is at the top as a JSON-like dict for easy editing.
"""

import os
import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
import numpy as np
import tqdm
from tqdm import tqdm

# External libraries (pip install required)
import webrtcvad
import whisper

# Global constants (reused/adapted from compressVid.py)
FFPROBE_TIMEOUT = 30
FFMPEG_TIMEOUT = 3600

# Custom exception
class ConversationAnalyzerError(Exception):
    """Custom exception for analyzer errors."""
    pass

# Top-level configuration (JSON-like dict)
CONFIG = {
    "supported_formats": [
        ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
        ".mpg", ".3gp", ".MP4", ".MKV", ".AVI", ".MOV", ".FLV", ".WMV",
        ".WEBM", ".M4V", ".MPG", ".3GP"
    ],
    "analysis": {
        # Whisper model options (from openai-whisper):
        # - "tiny": ~39M params, fastest (~0.5x realtime on CPU), lowest accuracy (best for quick English tests; poor multilingual).
        # - "base": ~74M params, fast (~1x realtime), good accuracy for English (~85% WER); decent for common languages like Hindi/Tamil.
        # - "small": ~244M params, balanced (~2x realtime), improved accuracy/multilingual support (recommended for mixed languages).
        # - "medium": ~769M params, slower (~4x realtime), high accuracy for accents/noise/non-English (e.g., better Tamil/Hindi).
        # - "large-v2" or "large-v3": ~1.55B params, slowest (~8x realtime), state-of-the-art accuracy across 99+ languages; use for production but requires more RAM/GPU.
        # Tradeoffs: Larger models = better accuracy (esp. multilingual) but higher download size (100MB-3GB), memory (1-10GB), and time. CLI supports "base/small/medium"; edit here or use --model for others (add to choices if needed).
        "whisper_model": "base",
        "vad_aggressiveness": 2,
        "min_speech_duration": 5.0,
        "min_speech_ratio": 0.1,
        "min_segments": 2,
        "min_transcript_length": 50,
        "has_dialogue_heuristic": True
    },
    "output": {
        "json_report": True,
        "output_suffix": "_analysis",
        "temp_audio_suffix": ".wav"
    },
    "ffmpeg": {
        "timeout": 3600,
        "audio_extract_cmd_base": [
            "ffmpeg", "-y", "-i", "{input}", "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "{output}"
        ]
    },
    "dry_run": False
}

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

class VideoInfo:
    """Handles video information extraction (simplified for audio/duration)."""

    @staticmethod
    def get_info(file_path: Path) -> Tuple[bool, Optional[float]]:
        """
        Extract has_audio and duration using ffprobe.

        Returns:
            (has_audio: bool, duration_seconds: Optional[float])
            or (False, None) on error
        """
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_streams", "-show_format",
                "-of", "json",
                str(file_path)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT, check=True
            )

            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            format_info = data.get('format', {})

            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
            has_audio = audio_stream is not None

            duration_str = format_info.get('duration')
            duration = float(duration_str) if duration_str else None

            return has_audio, duration

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to get video info for {file_path}: {e}")
            return False, None

def extract_audio(input_path: Path, output_path: Path, timeout: int = FFMPEG_TIMEOUT) -> bool:
    """Extract audio using FFmpeg."""
    try:
        cmd = CONFIG["ffmpeg"]["audio_extract_cmd_base"][:]
        cmd[cmd.index("{input}")] = str(input_path)
        cmd[cmd.index("{output}")] = str(output_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        )
        logging.info(f"Audio extracted to {output_path}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        logging.error(f"Audio extraction failed for {input_path}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False

def detect_speech_vad(audio_path: Path, config: Dict[str, Any]) -> Tuple[float, float, bool]:
    """Detect speech using WebRTC VAD."""
    vad = webrtcvad.Vad(config["analysis"]["vad_aggressiveness"])
    speech_duration = 0.0
    total_duration = 0.0

    try:
        with wave.open(str(audio_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            if sample_rate != 16000:
                raise ValueError("Audio must be 16kHz")
            frames = wf.readframes(wf.getnframes())
            total_duration = len(frames) / (sample_rate * 2)  # Seconds

            # Process in 30ms frames (480 bytes for 16kHz mono 16-bit)
            frame_duration_ms = 30
            bytes_per_frame = int(sample_rate * frame_duration_ms / 1000 * 2)
            num_frames = len(frames) // bytes_per_frame

            for i in range(num_frames):
                frame_start = i * bytes_per_frame
                frame_end = frame_start + bytes_per_frame
                frame = frames[frame_start:frame_end]
                if len(frame) == bytes_per_frame and vad.is_speech(frame, sample_rate):
                    speech_duration += frame_duration_ms / 1000.0

        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
        has_speech = speech_duration >= config["analysis"]["min_speech_duration"] and speech_ratio >= config["analysis"]["min_speech_ratio"]
        return speech_duration, total_duration, has_speech

    except Exception as e:
        logging.error(f"VAD failed for {audio_path}: {e}")
        return 0.0, 0.0, False

def transcribe_and_analyze(analyzer, audio_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe with Whisper and analyze for conversation."""
    try:
        model = analyzer.model
        result = model.transcribe(str(audio_path), language=None)  # Auto-detect

        transcript = result["text"].strip()
        segments = result["segments"]
        num_segments = len(segments)

        # Heuristic for dialogue
        has_dialogue = False
        if config["analysis"]["has_dialogue_heuristic"]:
            has_dialogue = any(
                "?" in seg["text"] or len(seg["text"].split()) > 3
                for seg in segments
            )

        has_conversation = (
            num_segments >= config["analysis"]["min_segments"]
            and len(transcript) >= config["analysis"]["min_transcript_length"]
            and (has_dialogue or num_segments > 3)  # Fallback for multi-turn
        )
        analyzer.logger.debug(f"Transcription analysis: num_segments={num_segments}, transcript_len={len(transcript)}, has_dialogue={has_dialogue}, has_conversation={has_conversation}")

        reason = "Multiple segments with meaningful transcript and dialogue patterns" if has_conversation else "Insufficient segments, short transcript, or no clear dialogue"

        return {
            "has_conversation": has_conversation,
            "transcript": transcript,
            "num_segments": num_segments,
            "reason": reason,
            "has_dialogue": has_dialogue
        }
    except Exception as e:
        analyzer.logger.error(f"Transcription failed for {audio_path}: {e}")
        return {"has_conversation": False, "transcript": "", "num_segments": 0, "reason": f"Transcription error: {str(e)}", "has_dialogue": False}

def safe_write_json(output_dir: Path, filename: str, data: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Write JSON report with safety (no overwrite, numeric suffixes)."""
    try:
        output_path = output_dir / f"{filename}{config['output']['output_suffix']}.json"
        counter = 1
        original_path = output_path

        while output_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            output_path = original_path.parent / f"{stem}_{counter}{suffix}"
            counter += 1
            if counter > 1000:
                # Fallback to timestamp
                timestamp = str(int(np.datetime64('now').astype('datetime64[s]').astype(int)))
                output_path = original_path.parent / f"{stem}_{timestamp}{suffix}"
                break

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Report saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write report: {e}")
        return False



class ConversationAnalyzer:
    """Main analyzer orchestrator."""

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        self.config = config
        self.logger = setup_logging(verbose)
        self.logs_dir = Path("analysis_logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.stats = {
            "analyzed": 0,
            "no_audio": 0,
            "failed": 0,
            "skipped": 0,
            "renamed": 0,
            "failed_renames": 0
        }
        self.model = None
        if not config["dry_run"]:
            self.model = whisper.load_model(config["analysis"]["whisper_model"])

    def safe_rename_file(self, file_path: Path) -> Optional[Path]:
        """Safely rename file by appending '_aud' suffix, handling conflicts."""
        try:
            parent = file_path.parent
            stem = file_path.stem
            suffix = file_path.suffix
            new_path = parent / f"{stem}_aud{suffix}"
            counter = 1
            original_path = new_path

            while new_path.exists():
                new_stem = f"{stem}_aud_{counter}"
                new_path = parent / f"{new_stem}{suffix}"
                counter += 1
                if counter > 1000:
                    # Fallback to timestamp
                    timestamp = str(int(np.datetime64('now').astype('datetime64[s]').astype(int)))
                    new_path = parent / f"{stem}_aud_{timestamp}{suffix}"
                    break

            os.rename(str(file_path), str(new_path))
            self.logger.info(f"Renamed {file_path} to {new_path}")
            return new_path
        except (OSError, Exception) as e:
            self.logger.error(f"Failed to rename {file_path}: {e}")
            return None

    def _extract_audio_task(self, file_path: Path) -> Optional[Path]:
        """Internal task for parallel audio extraction."""
        audio_path = file_path.parent / f"{file_path.stem}_temp{self.config['output']['temp_audio_suffix']}"
        success = extract_audio(file_path, audio_path)
        if success:
            self.logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        else:
            if audio_path.exists():
                audio_path.unlink()
            return None

    def _analyze_audio_task(self, file_path: Path, audio_path: Path) -> Dict[str, Any]:
        """Internal task for parallel audio analysis (VAD + transcription)."""
        try:
            # VAD
            speech_duration, total_duration, has_speech = detect_speech_vad(audio_path, self.config)
            self.logger.info(f"{file_path.name}: Speech {speech_duration:.1f}s / {total_duration:.1f}s (ratio: {speech_duration / total_duration if total_duration > 0 else 0:.2f})")
            if not has_speech:
                self.logger.info(f"No significant speech in {file_path}")
                result = {
                    "file": str(file_path),
                    "has_conversation": False,
                    "speech_duration": speech_duration,
                    "total_duration": total_duration,
                    "reason": "No significant speech detected"
                }
                self.stats["analyzed"] += 1
            else:
                # Transcription/analysis
                if self.config["dry_run"]:
                    trans_result = {"has_conversation": None, "transcript": "", "num_segments": 0, "reason": "Dry run - transcription skipped", "has_dialogue": False}
                else:
                    trans_result = transcribe_and_analyze(self, audio_path, self.config)

                output_data = {
                    "file": str(file_path),
                    "has_conversation": trans_result["has_conversation"],
                    "transcript": trans_result["transcript"],
                    "num_segments": trans_result["num_segments"],
                    "speech_duration": speech_duration,
                    "total_duration": total_duration,
                    "reason": trans_result["reason"],
                    "has_dialogue": trans_result.get("has_dialogue", False)
                }
                self.stats["analyzed"] += 1
                decision = "Yes" if trans_result["has_conversation"] else "No"
                self.logger.info(f"{file_path.name}: {decision} - {trans_result['reason']}")
                self.logger.debug(f"{file_path.name}: num_segments={trans_result['num_segments']}, transcript_len={len(trans_result.get('transcript', ''))}, has_dialogue={trans_result.get('has_dialogue', 'N/A')}, computed has_conversation={trans_result['has_conversation']}")
                result = output_data

            # Cleanup temp audio
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)

            return result

        except Exception as e:
            self.logger.error(f"Unexpected error analyzing {file_path}: {e}")
            self.stats["failed"] += 1
            result = {"file": str(file_path), "has_conversation": False, "reason": f"Analysis error: {str(e)}"}
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)
            return result

    def find_video_files(self, directory: Path) -> List[Path]:
        """Find all supported video files in directory."""
        video_files = []
        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Directory does not exist: {directory}")
            return video_files

        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.config["supported_formats"]:
                    video_files.append(file_path)
            self.logger.info(f"Found {len(video_files)} video files in {directory}")
            return sorted(video_files)
        except PermissionError:
            self.logger.error(f"Permission denied accessing {directory}")
            return video_files

    def validate_files(self, file_paths: List[str]) -> List[Path]:
        """Validate and convert file paths to Path objects."""
        valid_files = []
        for file_str in file_paths:
            file_path = Path(file_str)
            if not file_path.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                self.stats["skipped"] += 1
                continue
            if not file_path.is_file():
                self.logger.warning(f"Not a file: {file_path}")
                self.stats["skipped"] += 1
                continue
            if file_path.suffix.lower() not in self.config["supported_formats"]:
                self.logger.warning(f"Unsupported format: {file_path}")
                self.stats["skipped"] += 1
                continue
            valid_files.append(file_path)
        return valid_files

    def analyze_video(self, file_path: Path, pbar: Optional[tqdm] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Analyze a single video file."""
        try:
            # Check audio presence
            has_audio, duration = VideoInfo.get_info(file_path)
            if not has_audio:
                self.logger.info(f"No audio in {file_path}")
                self.stats["no_audio"] += 1
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return {"file": str(file_path), "has_conversation": False, "reason": "No audio track"}

            if duration is None or duration == 0:
                self.logger.warning(f"Invalid duration for {file_path}")
                self.stats["failed"] += 1
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return {"file": str(file_path), "has_conversation": False, "reason": "Invalid duration"}

            # Setup progress bar
            if pbar:
                pbar.total = 100  # Percentage for simplicity
                pbar.desc = f"Analyzing {file_path.name}"
                pbar.refresh()

            # Extract audio
            audio_path = file_path.parent / f"{file_path.stem}_temp{self.config['output']['temp_audio_suffix']}"
            extract_success = extract_audio(file_path, audio_path)
            if not extract_success:
                self.stats["failed"] += 1
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return {"file": str(file_path), "has_conversation": False, "reason": "Audio extraction failed"}

            if pbar:
                pbar.update(20)  # 20% for extraction

            # VAD
            speech_duration, total_duration, has_speech = detect_speech_vad(audio_path, self.config)
            self.logger.info(f"{file_path.name}: Speech {speech_duration:.1f}s / {total_duration:.1f}s (ratio: {speech_duration/total_duration:.2f})")
            if not has_speech:
                self.logger.info(f"No significant speech in {file_path}")
                # Cleanup
                audio_path.unlink(missing_ok=True)
                self.stats["analyzed"] += 1  # Count as analyzed even if no speech
                if pbar:
                    pbar.n = pbar.total
                    pbar.close()
                return {
                    "file": str(file_path),
                    "has_conversation": False,
                    "speech_duration": speech_duration,
                    "total_duration": total_duration,
                    "reason": "No significant speech detected"
                }

            if pbar:
                pbar.update(40)  # 40% for VAD

            # Transcription (skip in dry-run)
            if dry_run:
                result = {"has_conversation": None, "transcript": "", "num_segments": 0, "reason": "Dry run - transcription skipped"}
            else:
                result = transcribe_and_analyze(self, audio_path, self.config)
                if pbar:
                    pbar.update(40)  # 40% for transcription

            # Prepare output
            output_data = {
                "file": str(file_path),
                "has_conversation": result["has_conversation"],
                "transcript": result["transcript"],
                "num_segments": result["num_segments"],
                "speech_duration": speech_duration,
                "total_duration": total_duration,
                "reason": result["reason"]
            }

            # JSON report if enabled
            if self.config["output"]["json_report"]:
                output_dir = Path(self.config.get("output_dir", "."))
                output_dir.mkdir(exist_ok=True)
                safe_write_json(output_dir, file_path.stem, output_data, self.config)

            # Cleanup
            audio_path.unlink(missing_ok=True)

            if pbar:
                pbar.n = pbar.total
                pbar.close()

            self.stats["analyzed"] += 1
            decision = "Yes" if result["has_conversation"] else "No"
            self.logger.info(f"{file_path.name}: {decision} - {result['reason']}")
            self.logger.debug(f"{file_path.name}: num_segments={result['num_segments']}, transcript_len={len(result.get('transcript', ''))}, has_dialogue={result.get('has_dialogue', 'N/A')}, computed has_conversation={result['has_conversation']}")
            return output_data

        except Exception as e:
            self.logger.error(f"Unexpected error analyzing {file_path}: {e}")
            self.stats["failed"] += 1
            if pbar:
                pbar.n = pbar.total
                pbar.close()
            if 'audio_path' in locals() and audio_path.exists():
                audio_path.unlink(missing_ok=True)
            return {"file": str(file_path), "has_conversation": False, "reason": f"Analysis error: {str(e)}"}

    def process_files(self, files: List[Path], max_workers: int = 1, dry_run: bool = False, extract_parallel: bool = True, analyze_serial: bool = True) -> None:
        """Process video files in 3 phases: parallel extraction, serial analysis, serial output."""
        if not files:
            self.logger.warning("No files to process")
            return

        results = []
        output_dir = Path(self.config.get("output_dir", "."))
        output_dir.mkdir(exist_ok=True)

        # Phase 0: Pre-check audio and duration for all files
        valid_files = []
        precheck_pbar = tqdm(total=len(files), desc="Pre-checking files") if len(files) > 1 else None
        for file_path in files:
            if precheck_pbar:
                precheck_pbar.update(1)

            has_audio, duration = VideoInfo.get_info(file_path)
            if not has_audio:
                self.logger.info(f"No audio in {file_path}")
                self.stats["no_audio"] += 1
                results.append({"file": str(file_path), "has_conversation": False, "reason": "No audio track"})
                continue

            if duration is None or duration == 0:
                self.logger.warning(f"Invalid duration for {file_path}")
                self.stats["failed"] += 1
                results.append({"file": str(file_path), "has_conversation": False, "reason": "Invalid duration"})
                continue

            valid_files.append(file_path)

        if precheck_pbar:
            precheck_pbar.close()

        if not valid_files:
            self._log_stats_and_print(results, [])
            return

        # Phase 1: Parallel or serial audio extraction
        self.logger.info(f"Extracting audio for {len(valid_files)} files " + ("serially" if not extract_parallel else f"with {max_workers} workers"))
        extracted_audios = {}  # original_path -> temp_audio_path
        if extract_parallel:
            extraction_pbar = tqdm(total=len(valid_files), desc="Extracting audios (parallel)")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self._extract_audio_task, file_path): file_path
                    for file_path in valid_files
                }
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        audio_path = future.result()
                        if audio_path:
                            extracted_audios[file_path] = audio_path
                        else:
                            self.stats["failed"] += 1
                            results.append({"file": str(file_path), "has_conversation": False, "reason": "Audio extraction failed"})
                    except Exception as e:
                        self.stats["failed"] += 1
                        results.append({"file": str(file_path), "has_conversation": False, "reason": f"Extraction error: {str(e)}"})
                        self.logger.error(f"Error in extraction for {file_path}: {e}")
                    extraction_pbar.update(1)
            extraction_pbar.close()
        else:
            extraction_pbar = tqdm(total=len(valid_files), desc="Extracting audios (serial)")
            for file_path in valid_files:
                try:
                    audio_path = self._extract_audio_task(file_path)
                    if audio_path:
                        extracted_audios[file_path] = audio_path
                    else:
                        self.stats["failed"] += 1
                        results.append({"file": str(file_path), "has_conversation": False, "reason": "Audio extraction failed"})
                except Exception as e:
                    self.stats["failed"] += 1
                    results.append({"file": str(file_path), "has_conversation": False, "reason": f"Extraction error: {str(e)}"})
                    self.logger.error(f"Error in extraction for {file_path}: {e}")
                extraction_pbar.update(1)
            extraction_pbar.close()

        if not extracted_audios:
            self._log_stats_and_print(results, [])
            return

        # Phase 2: Serial or parallel analysis (VAD + transcription if not dry-run)
        self.logger.info(f"Analyzing {len(extracted_audios)} extracted audios " + ("serially" if analyze_serial else f"in parallel with {max_workers} workers"))
        analysis_pbar = tqdm(total=len(extracted_audios), desc="Analyzing audios" + (" (serial)" if analyze_serial else " (parallel)"))
        analysis_results = []
        if analyze_serial:
            for file_path, audio_path in extracted_audios.items():
                result = self._analyze_audio_task(file_path, audio_path)
                analysis_results.append(result)
                analysis_pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(self._analyze_audio_task, file_path, audio_path): file_path
                    for file_path, audio_path in extracted_audios.items()
                }
                for future in as_completed(future_to_pair):
                    file_path = future_to_pair[future]
                    result = future.result()
                    analysis_results.append(result)
                    analysis_pbar.update(1)
        analysis_pbar.close()

        # Append analysis results to overall results
        for result in analysis_results:
            results.append(result)

        # Write JSON reports for files with speech analysis
        if self.config["output"]["json_report"]:
            for result in analysis_results:
                if "transcript" in result:
                    file_path = Path(result["file"])
                    safe_write_json(output_dir, file_path.stem, result, self.config)

        # Phase 3: Serial rename for positive results (JSON already handled in phase 2)
        new_names = []
        failed_renames = []
        if not dry_run:
            positive_paths = [Path(r["file"]) for r in results if r.get("has_conversation") == True]
            for path in positive_paths:
                new_path = self.safe_rename_file(path)
                if new_path:
                    new_names.append(new_path)
                    self.stats["renamed"] += 1
                else:
                    failed_renames.append(path)
                    self.stats["failed_renames"] += 1
                    self.logger.error(f"Failed to rename {path}")

        # Aggregate failed analyses
        failed_analyses = [Path(r["file"]) for r in results if "error" in r.get("reason", "").lower()]

        # Log stats and print
        self._log_stats_and_print(results, new_names, failed_analyses, failed_renames)

    def _log_stats_and_print(self, results: List[Dict[str, Any]], new_names: List[Path], failed_analyses: List[Path], failed_renames: List[Path]) -> None:
        """Internal method to log stats and print final results."""
        self.logger.info(
            f"Analysis complete: {self.stats['analyzed']} analyzed, "
            f"{self.stats['no_audio']} no audio, {self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped, {self.stats['renamed']} renamed, "
            f"{self.stats['failed_renames']} failed renames"
        )

        print("\n")
        print("List of files that tested positive for conversations:")
        for name in new_names:
            print(str(name))

        if failed_analyses:
            print("\nFailed analyses:")
            for f in failed_analyses:
                print(str(f))

        if failed_renames:
            print("\nFailed renames:")
            for f in failed_renames:
                print(str(f))


def check_dependencies() -> bool:
    """Check if FFmpeg and ffprobe are available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=FFPROBE_TIMEOUT)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=FFPROBE_TIMEOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline Video Conversation Detector: Analyze videos for human speech/conversation using local tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python isConversation.py -w /path/to/videos          # Analyze all videos in directory
  python isConversation.py -f video1.mp4 video2.mkv    # Analyze specific files
  python isConversation.py -w . -v                     # Current dir, verbose
  python isConversation.py -w . --model small --dry-run # Use small Whisper model, simulate
  python isConversation.py -w . --output-dir reports   # Save JSON reports to 'reports/'
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--working-dir", type=str, help="Directory containing videos")
    group.add_argument("-f", "--files", nargs="+", help="Specific video files")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--model", type=str, default="base", choices=["base", "small", "medium"], help="Whisper model size")
    parser.add_argument("--dry-run", action="store_true", help="Simulate: VAD/extract only, no transcription")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for JSON reports (default: current)")
    parser.add_argument("--thresholds", type=str, help="JSON string to override analysis thresholds (e.g., '{\"min_segments\": 3}')")
    parser.add_argument("--extract-parallel", action="store_true", default=True, help="Enable parallel audio extraction (default: True)")
    parser.add_argument("--analyze-serial", action="store_true", default=True, help="Enable serial analysis (default: True; set False for parallel)")

    args = parser.parse_args()

    # Update config from args
    config = CONFIG.copy()
    config["analysis"]["whisper_model"] = args.model
    config["dry_run"] = args.dry_run
    if args.working_dir:
        config["output_dir"] = os.path.join(args.working_dir, "logs")
    else:
        config["output_dir"] = args.output_dir

    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
            config["analysis"].update(thresholds)
        except json.JSONDecodeError as e:
            print(f"Invalid thresholds JSON: {e}")
            sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("Error: FFmpeg and ffprobe are required but not found in PATH.")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        print("\nFor Python libs: pip install openai-whisper webrtcvad tqdm soundfile numpy")
        sys.exit(1)

    # Initialize analyzer
    analyzer = ConversationAnalyzer(config, args.verbose)

    try:
        if args.working_dir:
            directory = Path(args.working_dir).resolve()
            files = analyzer.find_video_files(directory)
        else:
            files = analyzer.validate_files(args.files)

        analyzer.process_files(files, args.max_workers, args.dry_run, args.extract_parallel, args.analyze_serial)

    except KeyboardInterrupt:
        analyzer.logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        analyzer.logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()