#!/usr/bin/env python3
"""
Batch Workflows

Multi-step media processing workflows that orchestrate compression, audio extraction,
and conversation detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from enum import Enum
import logging


class WorkflowTask(Enum):
    """Supported workflow tasks"""
    COMPRESS = "compress"
    EXTRACT_AUDIO = "extract_audio"
    DETECT_CONVERSATION = "detect_conversation"


@dataclass
class WorkflowStep:
    """Single step in workflow"""
    task: WorkflowTask
    params: dict  # Task-specific parameters
    skip_on_error: bool = False  # Continue to next step if this fails


@dataclass
class Workflow:
    """Multi-step media processing workflow"""
    name: str
    steps: List[WorkflowStep]
    description: str = ""


class WorkflowExecutor:
    """Execute multi-step workflows"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, workflow: Workflow, input_file: Path,
                progress_callback: Optional[Callable] = None) -> dict:
        """
        Execute workflow on input file.

        Returns dict with results from each step.
        """
        results = {
            'file': str(input_file),
            'workflow': workflow.name,
            'steps': {}
        }

        current_input = input_file

        for i, step in enumerate(workflow.steps):
            try:
                self.logger.info(f"Step {i+1}/{len(workflow.steps)}: {step.task.value}")

                if step.task == WorkflowTask.COMPRESS:
                    result = self._compress(current_input, step.params)
                    current_input = result['output_file']
                    results['steps'][step.task.value] = result

                elif step.task == WorkflowTask.EXTRACT_AUDIO:
                    result = self._extract_audio(current_input, step.params)
                    results['steps'][step.task.value] = result

                elif step.task == WorkflowTask.DETECT_CONVERSATION:
                    result = self._detect_conversation(current_input, step.params)
                    results['steps'][step.task.value] = result

            except Exception as e:
                self.logger.error(f"Step {step.task.value} failed: {e}")
                if not step.skip_on_error:
                    raise
                results['steps'][step.task.value] = {'error': str(e)}

        return results

    def _compress(self, input_file: Path, params: dict) -> dict:
        """Compress video"""
        # Import here to avoid circular dependency
        from utils import VideoProcessor, Config

        config = Config(
            compression_factor=params.get('compression_factor', 0.7),
            use_adaptive=params.get('use_adaptive', True)
        )
        processor = VideoProcessor(config, verbose=params.get('verbose', False))

        output_file = input_file.parent / f"{input_file.stem}_conv.mp4"
        success = processor.converter.convert_video(input_file, pbar=None)

        return {
            'success': success,
            'output_file': output_file,
            'input_size': input_file.stat().st_size,
            'output_size': output_file.stat().st_size if output_file.exists() else 0
        }

    def _extract_audio(self, input_file: Path, params: dict) -> dict:
        """Extract audio"""
        # Placeholder for audio extraction integration
        format_choice = params.get('format', 'aac')
        output_file = input_file.parent / f"{input_file.stem}_audio.{format_choice}"

        return {
            'success': True,
            'output_file': output_file,
            'format': format_choice
        }

    def _detect_conversation(self, input_file: Path, params: dict) -> dict:
        """Detect conversation in video"""
        # Placeholder for conversation detection integration
        model = params.get('model', 'base')
        dry_run = params.get('dry_run', False)

        return {
            'success': True,
            'has_conversation': True,
            'model': model,
            'dry_run': dry_run
        }


# Pre-built workflow templates
WORKFLOWS = {
    'archive': Workflow(
        name='archive',
        description='Compress video aggressively for long-term storage',
        steps=[
            WorkflowStep(
                task=WorkflowTask.COMPRESS,
                params={'compression_factor': 0.5, 'use_adaptive': True}
            ),
            WorkflowStep(
                task=WorkflowTask.EXTRACT_AUDIO,
                params={'format': 'aac'},
                skip_on_error=True
            ),
        ]
    ),

    'streaming': Workflow(
        name='streaming',
        description='Optimize for streaming platforms (YouTube, etc)',
        steps=[
            WorkflowStep(
                task=WorkflowTask.COMPRESS,
                params={'compression_factor': 0.6, 'use_adaptive': True}
            ),
            WorkflowStep(
                task=WorkflowTask.EXTRACT_AUDIO,
                params={'format': 'aac'},
                skip_on_error=True
            ),
        ]
    ),

    'content_analysis': Workflow(
        name='content_analysis',
        description='Extract audio and detect conversations',
        steps=[
            WorkflowStep(
                task=WorkflowTask.EXTRACT_AUDIO,
                params={'format': 'aac'}
            ),
            WorkflowStep(
                task=WorkflowTask.DETECT_CONVERSATION,
                params={'model': 'base', 'dry_run': False},
                skip_on_error=True
            ),
        ]
    ),

    'full_optimization': Workflow(
        name='full_optimization',
        description='Compress, extract audio, and analyze content',
        steps=[
            WorkflowStep(
                task=WorkflowTask.COMPRESS,
                params={'compression_factor': 0.65, 'use_adaptive': True}
            ),
            WorkflowStep(
                task=WorkflowTask.EXTRACT_AUDIO,
                params={'format': 'aac'},
                skip_on_error=True
            ),
            WorkflowStep(
                task=WorkflowTask.DETECT_CONVERSATION,
                params={'model': 'base'},
                skip_on_error=True
            ),
        ]
    ),
}
