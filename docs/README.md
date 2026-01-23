# Documentation Index

## Getting Started

- **[START_HERE.md](../START_HERE.md)** - Quick start (installation + first run)
- **[README.md](../README.md)** - Project overview and features

## Core Documentation

- **[ARCHITECTURE_PLAN.md](../ARCHITECTURE_PLAN.md)** - Technical architecture, v2.0 design
- **[CLAUDE.md](../CLAUDE.md)** - Development guidelines for Claude AI

## Guides

- **[MANUAL_TESTING_GUIDE.md](MANUAL_TESTING_GUIDE.md)** - Manual testing procedures
- **[TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)** - QA testing checklist
- **[GROUND_TRUTH_ANNOTATION_GUIDE.md](GROUND_TRUTH_ANNOTATION_GUIDE.md)** - Data labeling guide
- **[STORAGE_MANAGEMENT.md](STORAGE_MANAGEMENT.md)** - Cache and storage management

## Archived Documentation

Historical documentation from development phases. Reference only.

- **[archived_phases/](archived_phases/)** - Phase 1-3 implementation notes
- **[archived_features/](archived_features/)** - Feature-specific documentation
- **[archived_qa/](archived_qa/)** - QA and testing documentation
- **[archived_legacy/](archived_legacy/)** - Legacy/superseded documentation

## Quick Reference

### Installation
```bash
pip install -r requirements.txt
brew install ffmpeg  # or apt install ffmpeg
python scripts/test_installation.py
```

### Basic Usage
```bash
python -m diveanalyzer process video.mp4 --enable-server
```

### Testing
```bash
python -m pytest tests/
python scripts/run_fixture_tests.py
```

## Directory Structure

```
diveanalyzer/
├── detection/          # Detection algorithms
├── extraction/         # Video extraction
├── storage/            # Storage management
├── utils/              # Utility functions
└── server/             # Web server for gallery review

tests/
├── fixtures/           # Test video data
└── integration/        # Integration tests

docs/                   # This directory
├── archived_*/         # Historical documentation
└── *.md                # Current guides

scripts/                # Development scripts
├── test_installation.py
├── quick_test.py
└── batch_test.py
```

---

**For issues or questions**, check the appropriate documentation above or open an issue on GitHub.
