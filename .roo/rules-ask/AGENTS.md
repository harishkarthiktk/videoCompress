# Project Documentation Rules (Non-Obvious Only)
- src/ directory contains only images (icon_corrected.jpg); all main Python code in project root (counterintuitive for "src").
- No formal docs beyond README.md; script docstrings provide usage, but FFmpeg integration details hidden in code patterns.
- .kilocode/ and .specify/ are VSCode extension artifacts, not project code; ignore for codebase analysis.
- Supported formats hardcoded in Config dataclass across scripts; no central config file.