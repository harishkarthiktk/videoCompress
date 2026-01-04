# Implementation Plan: Phase 4 - Core Workflow Features

4 critical features that will 3x program value with ~15 hours total effort.

**Total Estimated Effort:** 15 hours
**Expected Impact:** 3x more useful for power users
**Difficulty:** Medium
**Status:** Planning

---

## Feature 1: Smart Presets (3 hours)

Allow users to save/load compression profiles without manual tuning.

### Files Affected
- `presets.py` (new)
- `utils.py` (Config class)
- `compressVid.py` (main)
- `CLAUDE.md` (documentation)

### Implementation Steps

**1. Create `/presets.py` (new file):**
```python
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, Optional

@dataclass
class Preset:
    """Named compression configuration"""
    name: str
    compression_factor: float  # 0-1
    max_workers: int           # 1-10
    use_adaptive: bool
    description: str
    tags: list  # e.g., ['streaming', 'youtube']

class PresetManager:
    """Manage compression presets"""

    def __init__(self, presets_dir: Path = None):
        self.presets_dir = presets_dir or Path.home() / '.videocompress' / 'presets'
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.presets = self._load_all_presets()

    def save_preset(self, preset: Preset) -> Path:
        """Save preset to JSON file"""
        preset_file = self.presets_dir / f'{preset.name}.json'
        with open(preset_file, 'w') as f:
            json.dump({
                'name': preset.name,
                'compression_factor': preset.compression_factor,
                'max_workers': preset.max_workers,
                'use_adaptive': preset.use_adaptive,
                'description': preset.description,
                'tags': preset.tags
            }, f, indent=2)
        return preset_file

    def load_preset(self, name: str) -> Optional[Preset]:
        """Load preset by name"""
        preset_file = self.presets_dir / f'{name}.json'
        if not preset_file.exists():
            return None
        with open(preset_file, 'r') as f:
            data = json.load(f)
        return Preset(**data)

    def list_presets(self) -> Dict[str, str]:
        """List all available presets with descriptions"""
        presets = {}
        for preset_file in self.presets_dir.glob('*.json'):
            with open(preset_file, 'r') as f:
                data = json.load(f)
            presets[data['name']] = data.get('description', '')
        return presets

    def _load_all_presets(self) -> Dict[str, Preset]:
        """Load all presets from disk"""
        presets = {}
        for preset_file in self.presets_dir.glob('*.json'):
            with open(preset_file, 'r') as f:
                data = json.load(f)
            presets[data['name']] = Preset(**data)
        return presets

# Built-in default presets
DEFAULT_PRESETS = {
    'fast': Preset(
        name='fast',
        compression_factor=0.9,
        max_workers=4,
        use_adaptive=False,
        description='Minimal compression, maximum speed (90% of original bitrate)',
        tags=['speed']
    ),
    'balanced': Preset(
        name='balanced',
        compression_factor=0.7,
        max_workers=2,
        use_adaptive=True,
        description='Default balanced compression (70% of original bitrate)',
        tags=['default']
    ),
    'quality': Preset(
        name='quality',
        compression_factor=0.85,
        max_workers=1,
        use_adaptive=True,
        description='High quality, conservative compression (85% of original)',
        tags=['quality']
    ),
    'archive': Preset(
        name='archive',
        compression_factor=0.5,
        max_workers=2,
        use_adaptive=True,
        description='Aggressive compression for long-term storage (50% of original)',
        tags=['storage']
    ),
    'streaming': Preset(
        name='streaming',
        compression_factor=0.6,
        max_workers=4,
        use_adaptive=True,
        description='Optimized for streaming platforms (60% of original)',
        tags=['youtube', 'streaming']
    ),
}
```

**2. Update `compressVid.py` main():**
```python
# Add imports
from presets import PresetManager

# Add arguments
parser.add_argument(
    '--preset',
    type=str,
    choices=['fast', 'balanced', 'quality', 'archive', 'streaming'],
    help='Use preset compression profile'
)
parser.add_argument(
    '--save-preset',
    type=str,
    metavar='NAME',
    help='Save current compression settings as preset NAME'
)
parser.add_argument(
    '--list-presets',
    action='store_true',
    help='List all available presets'
)

# Add to main()
preset_mgr = PresetManager()

if args.list_presets:
    presets = preset_mgr.list_presets()
    for name, desc in presets.items():
        print(f"  {name}: {desc}")
    sys.exit(0)

if args.preset:
    preset = preset_mgr.load_preset(args.preset)
    if not preset:
        logger.error(f"Preset '{args.preset}' not found")
        sys.exit(1)
    compression_factor = preset.compression_factor
    args.max_workers = preset.max_workers
    config = Config(
        compression_factor=compression_factor,
        use_adaptive=preset.use_adaptive
    )

if args.save_preset:
    preset = Preset(
        name=args.save_preset,
        compression_factor=compression_factor,
        max_workers=args.max_workers,
        use_adaptive=True,
        description=f"Custom preset: {compression_factor*100:.0f}% factor",
        tags=['custom']
    )
    preset_mgr.save_preset(preset)
    logger.info(f"Preset '{args.save_preset}' saved")
```

### Usage Examples
```bash
# List all presets
python compressVid.py --list-presets

# Use a preset
python compressVid.py -w /videos --preset streaming

# Save current settings as preset
python compressVid.py -w /videos -c 60 --max-workers 4 --save-preset my-custom

# Use custom preset
python compressVid.py -w /videos --preset my-custom
```

---

## Feature 2: Dry Run Mode (2 hours)

Preview what compression will achieve without actually encoding.

### Files Affected
- `utils.py` (VideoProcessor, VideoConverter)
- `compressVid.py` (main)

### Implementation Steps

**1. Add dry_run flag to Config:**
```python
@dataclass
class Config:
    # ... existing fields ...
    dry_run: bool = False  # If True, only analyze, don't compress
```

**2. Add to VideoConverter:**
```python
def convert_video(self, input_file: Path, output_file: Path, dry_run: bool = False) -> bool:
    """
    Convert video with optional dry-run mode.

    In dry-run mode: analyzes file, estimates output size, reports time,
    but doesn't actually encode.
    """
    if dry_run:
        width, height, video_bitrate, has_audio, audio_bitrate, duration, nb_frames, rotation, fps, codec = VideoInfo.get_info(input_file)

        if video_bitrate is None:
            self.logger.warning(f"Cannot estimate - no bitrate info for {input_file}")
            return False

        target_bitrate, target_audio_bitrate, new_width, new_height = self._calculate_target_params(
            width, height, video_bitrate, audio_bitrate, duration, codec
        )

        # Estimate output size
        if duration:
            estimated_video_bytes = (target_bitrate * 1000 * duration) // 8
            estimated_audio_bytes = (target_audio_bitrate * 1000 * duration) // 8 if audio_bitrate else 0
            estimated_total = estimated_video_bytes + estimated_audio_bytes

            # Estimate encoding time (rough: 100-500 fps depending on GPU)
            fps_processing = 300 if self.gpu_type != 'CPU' else 100
            estimated_time = duration / fps_processing

            input_size = input_file.stat().st_size
            compression_ratio = estimated_total / input_size if input_size > 0 else 0

            self.logger.info(f"DRY RUN: {input_file.name}")
            self.logger.info(f"  Original: {input_size / 1e9:.2f} GB ({video_bitrate} kbps video, {audio_bitrate or 0} kbps audio)")
            self.logger.info(f"  Estimated: {estimated_total / 1e9:.2f} GB ({target_bitrate} kbps video, {target_audio_bitrate} kbps audio)")
            self.logger.info(f"  Compression: {compression_ratio*100:.1f}%")
            self.logger.info(f"  Est. time: {estimated_time:.1f}s with {self.gpu_type}")
            self.logger.info(f"  Resolution: {width}x{height} → {new_width}x{new_height}")

        return True

    # ... existing conversion code ...
```

**3. Update VideoProcessor:**
```python
def process_files(self, files: List[Path], max_workers: int = 1, use_tqdm: bool = True, overall_update_callback: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
    """Add dry_run parameter"""
    # ... existing code ...
    for file_path in files:
        success = self.converter.convert_video(
            file_path,
            output_file,
            dry_run=self.config.dry_run
        )
```

**4. Update compressVid.py:**
```python
parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Analyze files and estimate compression without encoding'
)

# In main()
if args.dry_run:
    config.dry_run = True
    logger.info("Running in DRY RUN mode - no files will be modified")
```

### Usage Examples
```bash
# Dry run on directory
python compressVid.py -w /videos --dry-run

# Dry run with specific preset
python compressVid.py -w /videos --preset streaming --dry-run

# Output shows estimates for each file
# Original: 2.50 GB (5000 kbps video, 128 kbps audio)
# Estimated: 1.75 GB (3500 kbps video, 128 kbps audio)
# Compression: 70.0%
# Est. time: 120.5s with NVIDIA
```

---

## Feature 3: Processing Resume (3 hours)

Resume interrupted batch jobs, skip already-processed files.

### Files Affected
- `queue.py` (new)
- `compressVid.py` (main)
- `utils.py` (VideoProcessor)

### Implementation Steps

**1. Create `queue.py` (new file):**
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from enum import Enum

class ProcessingStatus(Enum):
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
    id: str  # Unique identifier (timestamp + hash)
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

    def load_queue(self, queue_id: str) -> ProcessingQueue:
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
            if queue.get_pending_files():
                incomplete.append(queue_id)
        return incomplete
```

**2. Update VideoProcessor:**
```python
def __init__(self, config: Config, verbose: bool = False, move_files: bool = False,
             per_file_callback: Optional[Callable[[float], None]] = None,
             logs_dir: Optional[Path] = None, queue: Optional[ProcessingQueue] = None):
    # ... existing init code ...
    self.queue = queue

def process_files(self, files: List[Path], max_workers: int = 1, use_tqdm: bool = True,
                  overall_update_callback: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
    # ... existing code ...

    # If queue exists, filter out completed files
    if self.queue:
        pending_files = [f for f in files
                        if f.name not in [qf.input_path for qf in self.queue.input_files
                                         if qf.status == ProcessingStatus.COMPLETED.value]]
        files = pending_files
        self.logger.info(f"Resuming queue: {self.queue.get_summary()}")

    # ... rest of processing ...

    # Update queue after each file
    if self.queue:
        self.queue.mark_completed(file_path)
        self.queue.save_queue(self.queue)
```

**3. Update compressVid.py:**
```python
from queue import QueueManager, ProcessingStatus

parser.add_argument(
    '--create-queue',
    action='store_true',
    help='Create processing queue (save state for resuming)'
)
parser.add_argument(
    '--resume-queue',
    type=str,
    metavar='QUEUE_ID',
    help='Resume incomplete processing queue'
)
parser.add_argument(
    '--list-queues',
    action='store_true',
    help='List incomplete queues available to resume'
)

# In main()
queue_mgr = QueueManager()

if args.list_queues:
    incomplete = queue_mgr.list_incomplete_queues()
    if not incomplete:
        print("No incomplete queues")
    else:
        print("Incomplete queues:")
        for queue_id in incomplete:
            queue = queue_mgr.load_queue(queue_id)
            print(f"  {queue_id}: {queue.get_summary()}")
    sys.exit(0)

queue = None
if args.resume_queue:
    queue = queue_mgr.load_queue(args.resume_queue)
    if not queue:
        logger.error(f"Queue '{args.resume_queue}' not found")
        sys.exit(1)
    logger.info(f"Resuming queue: {queue.get_summary()}")
    compression_factor = queue.preset  # Load preset settings
elif args.create_queue:
    queue = queue_mgr.create_queue(preset=args.preset or 'balanced', dry_run=args.dry_run)
    logger.info(f"Created processing queue: {queue.id}")

# Populate queue with files
if queue:
    for file in files:
        queue.add_file(file, file.parent / f"{file.stem}_conv{file.suffix}")
    queue_mgr.save_queue(queue)

# Pass queue to processor
processor.queue = queue
```

### Usage Examples
```bash
# Create queue for batch processing
python compressVid.py -w /videos --create-queue --preset streaming

# Shows: "Created processing queue: 20250104_114000"

# Later, if interrupted, resume the queue
python compressVid.py --resume-queue 20250104_114000

# List incomplete queues
python compressVid.py --list-queues

# Output:
# Incomplete queues:
#   20250104_114000: Progress: 45/100 completed, 2 failed, 53 pending
```

---

## Feature 4: Batch Workflows (3 hours)

Orchestrate multi-tool pipelines (compress + extract audio + detect conversations).

### Files Affected
- `workflows.py` (new)
- `compressVid.py`, `extractAudio.py`, `isConversation.py` (integrate)
- `CLAUDE.md` (documentation)

### Implementation Steps

**1. Create `workflows.py` (new file):**
```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional
from enum import Enum
import logging

class WorkflowTask(Enum):
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

        output_file = input_file.parent / f"{input_file.stem}_conv{input_file.suffix}"
        success = processor.converter.convert_video(input_file, output_file)

        return {
            'success': success,
            'output_file': output_file,
            'input_size': input_file.stat().st_size,
            'output_size': output_file.stat().st_size if output_file.exists() else 0
        }

    def _extract_audio(self, input_file: Path, params: dict) -> dict:
        """Extract audio"""
        # Would integrate with extractAudio.py AudioExtractor class
        format = params.get('format', 'aac')
        output_file = input_file.parent / f"{input_file.stem}_audio.{format}"

        # Placeholder - actual implementation calls AudioExtractor
        return {
            'success': True,
            'output_file': output_file,
            'format': format
        }

    def _detect_conversation(self, input_file: Path, params: dict) -> dict:
        """Detect conversation in video"""
        # Would integrate with isConversation.py ConversationAnalyzer
        model = params.get('model', 'base')
        dry_run = params.get('dry_run', False)

        # Placeholder - actual implementation calls ConversationAnalyzer
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
```

**2. Update compressVid.py:**
```python
from workflows import WorkflowExecutor, WORKFLOWS

parser.add_argument(
    '--workflow',
    type=str,
    choices=list(WORKFLOWS.keys()),
    help='Run multi-step workflow'
)
parser.add_argument(
    '--list-workflows',
    action='store_true',
    help='List available workflows'
)

# In main()
if args.list_workflows:
    print("Available workflows:")
    for name, workflow in WORKFLOWS.items():
        print(f"  {name}: {workflow.description}")
    sys.exit(0)

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
```

### Usage Examples
```bash
# List available workflows
python compressVid.py --list-workflows

# Output:
# Available workflows:
#   archive: Compress video aggressively for long-term storage
#   streaming: Optimize for streaming platforms (YouTube, etc)
#   content_analysis: Extract audio and detect conversations
#   full_optimization: Compress, extract audio, and analyze content

# Run workflow on single file
python compressVid.py -f video.mp4 --workflow streaming

# Run workflow on directory
python compressVid.py -w /videos --workflow full_optimization

# Combine with queue and preset
python compressVid.py -w /videos --workflow archive --create-queue --preset archive
```

---

## Summary & Timeline

| Feature | Effort | Impact | Implementation Order |
|---------|--------|--------|----------------------|
| Smart Presets | 3h | Essential | 1st (foundation) |
| Dry Run | 2h | High-value UX | 2nd (quick win) |
| Processing Resume | 3h | Production-critical | 3rd (power users) |
| Batch Workflows | 3h | Highest value | 4th (complete suite) |
| **TOTAL** | **11h** | **3X improvement** | **Sequential** |

### Implementation Sequence
1. **Presets** → Enables users to save favorite settings
2. **Dry Run** → Users can preview before committing
3. **Queue Resume** → Critical for large batches
4. **Workflows** → Transforms from tool to complete suite

All 4 features work together:
- Use **presets** for workflow steps
- **Dry run** validates workflow before queuing
- **Queue** persists workflow jobs
- **Workflows** orchestrate everything

---

## Testing Checklist

### Presets
- [ ] Create preset and verify JSON saved
- [ ] Load preset and verify values applied
- [ ] List presets shows all available
- [ ] Save/load custom preset works

### Dry Run
- [ ] Dry run analyzes file without encoding
- [ ] Output shows size estimates correctly
- [ ] Time estimates reasonable
- [ ] Works with all presets

### Resume
- [ ] Create queue saves state
- [ ] Resume loads pending files correctly
- [ ] Completed files skipped on resume
- [ ] Failed files retry-able

### Workflows
- [ ] Each step runs independently
- [ ] Workflow completes all steps
- [ ] Error in one step doesn't stop others (skip_on_error)
- [ ] Results properly returned

---

## Documentation Updates

Add to `CLAUDE.md`:
```markdown
### Batch Workflows & Advanced Features

**Smart Presets:** Save/load compression profiles
```bash
python compressVid.py --list-presets
python compressVid.py -w /videos --preset streaming
python compressVid.py -w /videos -c 60 --save-preset my-custom
```

**Dry Run:** Preview compression without encoding
```bash
python compressVid.py -w /videos --dry-run --preset streaming
```

**Processing Queues:** Resume interrupted batch jobs
```bash
python compressVid.py -w /videos --create-queue
# Later:
python compressVid.py --resume-queue 20250104_114000
python compressVid.py --list-queues
```

**Batch Workflows:** Multi-step media processing
```bash
python compressVid.py --list-workflows
python compressVid.py -w /videos --workflow full_optimization
```
```

---

## Dependencies

- No new external dependencies required
- Uses standard library: `json`, `dataclasses`, `enum`, `datetime`
- Integrates with existing: `VideoProcessor`, `VideoConverter`, `VideoInfo`

---

## Success Criteria

✓ All 4 features implemented and tested
✓ 100% backward compatible (no breaking changes)
✓ All existing tests passing
✓ New tests for each feature
✓ Documentation complete
✓ Ready for production use
