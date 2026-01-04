#!/usr/bin/env python3
"""
Processing Queue Management

Allows resuming interrupted batch jobs and tracking file processing status.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from enum import Enum


class ProcessingStatus(Enum):
    """File processing status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedFile:
    """Single file in processing queue"""
    input_path: str
    output_path: str
    status: str = ProcessingStatus.PENDING.value
    error_message: str = None
    started_at: str = None
    completed_at: str = None


@dataclass
class ProcessingQueue:
    """Persistent processing queue"""
    id: str  # Unique identifier (timestamp)
    created_at: str
    input_files: List[QueuedFile] = field(default_factory=list)
    preset: str = 'balanced'
    dry_run: bool = False
    completed: int = 0
    failed: int = 0

    def add_file(self, input_path: Path, output_path: Path) -> None:
        """Add file to queue"""
        self.input_files.append(QueuedFile(
            input_path=str(input_path),
            output_path=str(output_path)
        ))

    def mark_completed(self, input_path: Path) -> None:
        """Mark file as completed"""
        for f in self.input_files:
            if f.input_path == str(input_path):
                f.status = ProcessingStatus.COMPLETED.value
                f.completed_at = datetime.now().isoformat()
                self.completed += 1

    def mark_failed(self, input_path: Path, error: str) -> None:
        """Mark file as failed"""
        for f in self.input_files:
            if f.input_path == str(input_path):
                f.status = ProcessingStatus.FAILED.value
                f.error_message = error
                self.failed += 1

    def get_pending_files(self) -> List[QueuedFile]:
        """Get files not yet processed"""
        return [f for f in self.input_files
                if f.status in [ProcessingStatus.PENDING.value, ProcessingStatus.FAILED.value]]

    def get_summary(self) -> str:
        """Get progress summary"""
        total = len(self.input_files)
        pending = len(self.get_pending_files())
        return f"Progress: {self.completed}/{total} completed, {self.failed} failed, {pending} pending"


class QueueManager:
    """Manage persistent processing queues"""

    def __init__(self, queue_dir: Path = None):
        self.queue_dir = queue_dir or Path.home() / '.videocompress' / 'queues'
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def create_queue(self, preset: str = 'balanced', dry_run: bool = False) -> ProcessingQueue:
        """Create new processing queue"""
        queue_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        return ProcessingQueue(
            id=queue_id,
            created_at=datetime.now().isoformat(),
            preset=preset,
            dry_run=dry_run
        )

    def save_queue(self, queue: ProcessingQueue) -> Path:
        """Save queue to disk"""
        queue_file = self.queue_dir / f'{queue.id}.json'
        with open(queue_file, 'w') as f:
            json.dump({
                'id': queue.id,
                'created_at': queue.created_at,
                'preset': queue.preset,
                'dry_run': queue.dry_run,
                'completed': queue.completed,
                'failed': queue.failed,
                'files': [
                    {
                        'input_path': f.input_path,
                        'output_path': f.output_path,
                        'status': f.status,
                        'error_message': f.error_message,
                        'started_at': f.started_at,
                        'completed_at': f.completed_at,
                    }
                    for f in queue.input_files
                ]
            }, f, indent=2)
        return queue_file

    def load_queue(self, queue_id: str) -> 'ProcessingQueue':
        """Load queue from disk"""
        queue_file = self.queue_dir / f'{queue_id}.json'
        if not queue_file.exists():
            return None

        with open(queue_file, 'r') as f:
            data = json.load(f)

        queue = ProcessingQueue(
            id=data['id'],
            created_at=data['created_at'],
            preset=data['preset'],
            dry_run=data['dry_run'],
            completed=data['completed'],
            failed=data['failed']
        )

        for file_data in data['files']:
            queued_file = QueuedFile(
                input_path=file_data['input_path'],
                output_path=file_data['output_path'],
                status=file_data['status'],
                error_message=file_data.get('error_message'),
                started_at=file_data.get('started_at'),
                completed_at=file_data.get('completed_at')
            )
            queue.input_files.append(queued_file)

        return queue

    def list_queues(self) -> List[str]:
        """List all available queues"""
        return [f.stem for f in self.queue_dir.glob('*.json')]

    def list_incomplete_queues(self) -> List[str]:
        """List queues with pending work"""
        incomplete = []
        for queue_id in self.list_queues():
            queue = self.load_queue(queue_id)
            if queue and queue.get_pending_files():
                incomplete.append(queue_id)
        return incomplete
