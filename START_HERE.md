# Getting Started with DiveAnalyzer

Get up and running in 10 minutes.

## Installation (5 min)

```bash
# Clone repository
git clone https://github.com/maxime-c16/DiveAnalizer.git
cd DiveAnalizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from ffmpeg.org and add to PATH

# Verify installation
python scripts/test_installation.py
```

## First Run (5 min)

```bash
# Process a video to extract dives
python -m diveanalyzer process your_video.mp4

# Output appears in ./dives/ by default
ls -lh dives/
```

## With Live Gallery Review

```bash
# Process with interactive gallery for reviewing/accepting dives
python -m diveanalyzer process your_video.mp4 --enable-server

# Opens browser at http://localhost:8765
# Select dives to keep, click "Accept & Close" to finish
```

## Common Commands

```bash
# Just detect dives (no extraction)
python -m diveanalyzer process video.mp4 --detect-only

# Specify output directory
python -m diveanalyzer process video.mp4 --output ./my_dives/

# Multi-GPU processing (faster)
python -m diveanalyzer process video.mp4 --gpus 0 1

# FP16 quantization (lower memory)
python -m diveanalyzer process video.mp4 --fp16

# Debug mode (shows detection)
python -m diveanalyzer process video.mp4 --debug
```

See `python -m diveanalyzer --help` for all options.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **FFmpeg not found** | Install: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux) |
| **No dives detected** | Check audio quality; try `--debug` flag |
| **Slow processing** | Use `--fp16` and `--gpus` flags if GPU available |
| **Server won't start** | Try different port: `--server-port 8766` |
| **Missing audio in clips** | FFmpeg must be installed and in PATH |

## Documentation

- **[README.md](README.md)** - Full feature overview
- **[ARCHITECTURE_PLAN.md](ARCHITECTURE_PLAN.md)** - Technical architecture
- **[docs/](docs/)** - Additional guides
- **[CLAUDE.md](CLAUDE.md)** - Development setup

## Next Steps

✅ Extract your videos with the CLI
✅ Use `--enable-server` for interactive review
✅ Tune parameters with `--debug` mode
✅ Process multiple videos with `--batch`

---

**Questions?** See [docs/MANUAL_TESTING_GUIDE.md](docs/MANUAL_TESTING_GUIDE.md) or check troubleshooting above.
