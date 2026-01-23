# DiveAnalyzer

**Automated diving video analysis using multi-modal AI detection (audio, motion, person detection).**

DiveAnalyzer detects and extracts individual dives from swimming pool videos using a fusion of:
- 🔊 Audio peak detection (librosa)
- 🎬 Motion burst detection (frame differencing)
- 👤 YOLO person detection
- ⚡ FFmpeg stream-copy extraction (instant clips with audio)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python -m diveanalyzer process video.mp4

# With live gallery review
python -m diveanalyzer process video.mp4 --enable-server --server-port 8765
```

See [START_HERE.md](START_HERE.md) for detailed setup instructions.

## Features

- **Multi-modal Detection**: Audio + motion + person detection fusion for robust dive detection
- **Real-time Extraction**: Instant dive clip generation using FFmpeg stream copy
- **Live Gallery Review**: Interactive web interface for reviewing and accepting dives
- **Audio Preservation**: Original audio maintained in all extracted clips
- **Performance Optimized**: Multi-GPU support, FP16 quantization, frame batching
- **Comprehensive Analytics**: Detailed metrics and processing statistics

## Documentation

- **[START_HERE.md](START_HERE.md)** - Quick start guide for first-time users
- **[ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md)** - v2.0 system design and modules
- **[docs/](docs/)** - Additional documentation and guides
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines for Claude AI

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- FFmpeg installed (for audio/video processing)

### Installation by Platform

**macOS:**
```bash
brew install ffmpeg
pip install -r requirements.txt
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

**Windows:**
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH, then:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Processing
```bash
# Extract all dives
python -m diveanalyzer process video.mp4

# Specify output directory
python -m diveanalyzer process video.mp4 --output dives/

# Enable live gallery review
python -m diveanalyzer process video.mp4 --enable-server
```

### Advanced Options
```bash
# Multi-GPU processing
python -m diveanalyzer process video.mp4 --gpus 0 1

# FP16 quantization (faster, lower memory)
python -m diveanalyzer process video.mp4 --fp16

# Batch frames for better GPU utilization
python -m diveanalyzer process video.mp4 --batch-size 8

# Debug mode with visualization
python -m diveanalyzer process video.mp4 --debug
```

See `python -m diveanalyzer --help` for complete options.

## Architecture

The v2.0 system uses modular signal fusion:

```
Video Input
    ↓
┌───────────────────────────────────┐
│ Detection Algorithms              │
├─────────────────────────────────┤
│ • Audio peak detection (librosa)  │
│ • Motion burst detection          │
│ • YOLO person detection           │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Signal Fusion                     │
│ (Multi-modal consensus)           │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Extraction (FFmpeg)               │
│ (Stream copy for instant output)  │
└───────────────────────────────────┘
    ↓
Extracted Dives + Gallery
```

See [ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md) for technical details.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No dives detected** | Check audio quality, try `--debug` for visualization |
| **Missing audio in output** | Verify FFmpeg installed, check source has audio |
| **Slow processing** | Enable `--fp16` and `--batch-size`, use `--gpus` if available |
| **Server won't start** | Check port availability, try different `--server-port` |

## Testing

```bash
# Run test suite
python -m pytest tests/

# Run specific test
python -m pytest tests/test_audio_detection.py

# Integration tests
python scripts/run_fixture_tests.py
```

## Performance

- **Detection**: 1-2x real-time on CPU, 5-10x on GPU
- **Extraction**: Near-instant (stream copy via FFmpeg)
- **Memory**: ~2GB per GPU with FP16, ~4GB with full precision

## Contributing

See [docs/MANUAL_TESTING_GUIDE.md](docs/MANUAL_TESTING_GUIDE.md) for development guidelines.

## License

MIT License - See LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/maxime-c16/DiveAnalizer/issues)
- **Email**: [macauchy@student.42.fr](mailto:macauchy@student.42.fr)

---

**DiveAnalyzer** - AI-powered diving video analysis at scale.
