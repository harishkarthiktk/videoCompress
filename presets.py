#!/usr/bin/env python3
"""
Smart Presets System

Allows users to save and load compression profiles.
"""

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
