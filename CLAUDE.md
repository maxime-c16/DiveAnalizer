# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overhaul in Progress

**See `ARCHITECTURE_PLAN.md` for the complete v2.0 redesign plan.**

The project is transitioning from a MediaPipe-based visual splash detection system to a multi-modal approach using:
- Audio peak detection (librosa) - primary signal
- Motion burst detection (Decord + frame diff) - validation
- YOLO-nano person detection - validation
- FFmpeg stream copy extraction - instant clip export

## Project Overview

DiveAnalyzer is an automated diving video analysis tool that uses computer vision and AI to detect and extract individual dives from swimming pool diving videos. It combines MediaPipe pose detection, OpenCV computer vision, and multiple splash detection algorithms in a state machine architecture.

## Development Commands

### Running the Main Application

```bash
# Basic usage with default settings (motion_intensity splash detection)
python3 slAIcer.py path/to/video.mp4

# With custom output directory
python3 slAIcer.py video.mp4 --output_dir extracted_dives

# Enable debug visualization window (helpful for tuning detection parameters)
python3 slAIcer.py video.mp4 --debug

# Try different splash detection methods
python3 slAIcer.py video.mp4 --splash_method combined
python3 slAIcer.py video.mp4 --splash_method optical_flow
python3 slAIcer.py video.mp4 --splash_method frame_diff
python3 slAIcer.py video.mp4 --splash_method contour

# Enable pose overlay in output videos (disabled by default for performance)
python3 slAIcer.py video.mp4 --show-pose-overlay

# Disable audio preservation (audio enabled by default via FFmpeg)
python3 slAIcer.py video.mp4 --no-audio

# Disable threading for debugging performance issues
python3 slAIcer.py video.mp4 --no-threading
```

### Running Tests

```bash
# Test splash-only detection system
python3 test_splash_only_clean.py

# Test improved splash detection with debugging
python3 improved_splash_test.py

# Run the modular detection tool with evaluation metrics
python3 detection_tool.py path/to/video.mp4
```

### Code Quality

```bash
# Run linter (mentioned in README)
ruff check slAIcer.py

# Format code with black (mentioned in README)
black slAIcer.py
```

### Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (for testing/quality)
pip install ruff black pytest

# Optional: Install scipy and matplotlib for advanced detection features
pip install scipy matplotlib tqdm pandas scikit-learn
```

## Architecture Overview

### Core Files

- **slAIcer.py** (2923 lines): Main application with state machine-based dive detection
  - Entry point: `main()` at line 2786
  - Core detection: `detect_and_extract_dives_realtime()` at line 1873
  - Extraction: `extract_and_save_dive()` at line 2208
  - Frame generation: `frame_generator()` around line 428

- **splash_only_detector.py** (2553 lines): Alternative pure splash-based detection system
  - Uses Gaussian filtering and peak detection instead of state machine
  - Implements temporal consistency validation
  - Extracts 10s before + 2s after splash

- **detection_tool.py** (776 lines): Modular signal aggregation and evaluation tool
  - Provides CSV export of detection signals
  - GUI-based frame tagging for ground truth
  - Evaluation metrics (precision, recall, F1)

### State Machine Architecture

The main detection system (`slAIcer.py`) uses a 3-state machine:

1. **WAITING** (0): No diver detected, scanning for activity
2. **DIVER_ON_PLATFORM** (1): Diver detected on diving board/platform
3. **DIVING** (2): Diver has left platform, in flight or entering water

State transitions triggered by:
- Pose detection (MediaPipe) for diver presence on platform
- Splash detection for water entry
- Timeout/cooldown periods to prevent false positives

### Splash Detection Methods

Five different splash detection algorithms available via `--splash_method`:

1. **motion_intensity** (default): Gradient-based motion analysis with temporal consistency
   - Function: `detect_splash_motion_intensity()` at line 881
   - Uses Sobel gradients and adaptive thresholding
   - Best for general-purpose detection

2. **combined**: Voting system using multiple methods (≥2 of 3 must agree)
   - Function: `detect_splash_combined()` at line 833
   - Combines frame_diff, optical_flow, and contour detection
   - Most robust, reduces false positives

3. **frame_diff**: Consecutive frame difference with area thresholding
   - Function: `detect_splash_frame_diff()` at line 772
   - Simple and computationally efficient
   - Best for stable camera setups

4. **optical_flow**: Lucas-Kanade optical flow tracking
   - Function: `detect_splash_optical_flow()` at line 794
   - Very precise for high-resolution videos
   - Tracks pixel movement patterns

5. **contour**: Canny edge detection with contour analysis
   - Function: `detect_splash_contours()` at line 815
   - Good for high-contrast water surfaces
   - Detects shape changes in water surface

### Pose Detection Integration

MediaPipe pose detection is used to identify divers on the platform:

- Import at line 418: `import mediapipe as mp`
- Detection occurs within user-defined "diver detection zone"
- Adaptive pose optimization skips detection during certain states to improve performance
- Full-frame landmarks are preserved for optional pose overlay in output videos

### Performance Optimization

- **Multi-threading**: Enabled by default, extracts dives in parallel (`--no-threading` to disable)
- **Performance caching**: `PerformanceCache` class (line 32) stores statistics in `dive_performance_cache.pkl`
- **Adaptive pose detection**: Skips pose detection during safe states when diver known to be absent
- **Real-time extraction**: Dives extracted immediately upon detection (4.46x speedup)

### Audio Preservation

FFmpeg integration preserves audio in extracted videos:
- Audio enabled by default (`--no-audio` to disable)
- Audio offset configurable to sync with video extraction timing
- Requires FFmpeg installed on system

### Interactive Zone Configuration

On first run, user must configure three detection zones interactively:

1. **Diving board line**: Horizontal reference line on diving board/platform
2. **Diver detection zone**: Rectangle covering area where diver stands before diving
3. **Splash detection zone**: Horizontal band at water surface for splash monitoring

Coordinates normalized to [0,1] range for resolution independence.

## Key Implementation Details

### Frame Generation
- Custom `frame_generator()` uses VideoCapture with optional target FPS
- Handles frame buffering and cleanup automatically
- Supports stderr suppression for MediaPipe noise reduction

### Debug Mode
When `--debug` is enabled:
- Real-time visualization window shows detection zones
- State transitions displayed with color-coded indicators
- Pose detection overlay on live feed
- Splash detection scores visualized

### Output Structure
Extracted dives saved as:
```
output_directory/
├── dive_1.mp4
├── dive_2.mp4
├── dive_3_low_conf.mp4  # Low confidence detections marked
└── ...
```

Filenames include dive number and confidence indicators.

### Metrics and Logging
After processing, comprehensive metrics written to:
- Console output with emoji-rich formatting
- `dive_analysis.log` via `write_compact_log()` function (line 242)
- Performance cache for future run optimization

Metrics include:
- Video information (resolution, FPS, duration)
- Processing performance (detection time, extraction time, realtime ratio)
- Dive statistics (count, durations, confidence levels)
- Anomaly detection (very long dives, stuck states)

## Common Development Patterns

### Adding New Splash Detection Method
1. Implement detection function following signature: `def detect_splash_NAME(splash_band, prev_band, **params) -> (bool, float)`
2. Add to choices in argparse at line 2792
3. Add conditional in `find_next_dive_threaded()` around line 1429
4. Update combined method if appropriate

### Modifying State Machine Behavior
State transitions occur in `find_next_dive_threaded()`:
- WAITING → DIVER_ON_PLATFORM: When pose detected in diver zone
- DIVER_ON_PLATFORM → DIVING: When diver leaves zone
- DIVING → END: When splash detected

Thresholds and timeouts configurable via constants near line 1400.

### Testing Splash Detection
Use `detection_tool.py` for systematic evaluation:
1. Run on test video to generate CSV of detection signals
2. Use GUI to tag ground truth splash frames
3. Evaluate precision/recall/F1 for different methods and thresholds

## Dependencies Notes

Core requirements (requirements.txt):
- opencv-python: Computer vision processing
- mediapipe: Pose detection
- numpy: Numerical computations

Optional but recommended:
- matplotlib: Debug visualizations
- scipy: Advanced filtering (Gaussian, peak detection)
- tqdm: Progress bars
- FFmpeg: Audio preservation (system dependency, not pip)

The codebase gracefully degrades when optional dependencies unavailable.
