# Tests

Minimal test suite for critical functionalities. Exactly 5 core tests.

## Running Tests

**Install pytest (if needed):**
```bash
pip install pytest
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Run only core tests:**
```bash
pytest tests/test_core_functionality.py -v
```

## Core Tests (5 Total)

| # | Test | What It Validates |
|---|------|-------------------|
| 1 | GPU Detection | GPU detection works without crashing, returns valid type |
| 2 | File Discovery | Correctly finds video files, ignores non-video files |
| 3 | Configuration | Config has valid defaults and logical thresholds |
| 4 | Video Info Extraction | Handles ffprobe errors gracefully without crashing |
| 5 | GPU Detector Fallback | Falls back to CPU when detection fails |

## Test Coverage

**Critical Functionality Tested:**
- GPU detection (NVIDIA, Intel, AMD, Apple, CPU fallback)
- File discovery and filtering by video format
- Configuration validation and defaults
- Video metadata extraction error handling
- Error resilience and graceful degradation

## Notes

- All tests use mocking to avoid requiring actual video files or FFmpeg installation
- Tests are focused on critical paths only (5 tests maximum)
- Each test is simple and self-contained
- Tests use pytest fixtures for setup (conftest.py)
