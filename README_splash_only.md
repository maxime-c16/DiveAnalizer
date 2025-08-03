# Splash-Only Dive Detection System

A revolutionary approach to dive detection that completely abandons state-machine based methods in favor of pure splash detection with advanced Gaussian filtering and peak detection algorithms.

## 🌊 Overview

This system represents a **complete paradigm shift** from the traditional state-machine approach used in `slAIcer.py`. Instead of tracking diver movements and states, it focuses exclusively on detecting water splashes to identify dive endings, then extracts 10 seconds before + 2 seconds after each splash to capture complete dives.

## 🔧 Key Features

### Advanced Splash Detection
- **Multiple Detection Methods**: motion_intensity, combined, frame_diff, optical_flow, contour
- **Gaussian Spatial Filtering**: Reduces image noise before analysis
- **Temporal Gaussian Smoothing**: Eliminates false positives in score sequences
- **Adaptive Thresholding**: Dynamic threshold adjustment based on video statistics

### Robust Event Identification
- **Peak Detection Algorithm**: Uses scipy-like peak detection with prominence and distance constraints
- **Temporal Consistency**: Validates sustained splash signals
- **False Positive Reduction**: Multi-stage filtering pipeline
- **Cooldown Mechanism**: Prevents duplicate detections

### Comprehensive Debugging
- **Real-time Score Visualization**: Track raw vs filtered scores
- **Threshold Adaptation Plots**: Monitor adaptive threshold behavior
- **Statistical Analysis**: Mean, std deviation, background estimation over time
- **Peak Detection Visualization**: Show detected peaks and final events
- **Debug Data Export**: JSON export for further analysis

## 🚀 Quick Start

### Basic Usage
```bash
# Simple detection with debug plots
python3 splash_only_detector.py your_video.mp4 --debug

# Interactive zone selection (recommended for new videos)
python3 splash_only_detector.py your_video.mp4 --interactive-zone --debug

# Detection only (no video extraction)
python3 splash_only_detector.py your_video.mp4 --no-extract --debug
```

### Advanced Configuration
```bash
# Custom parameters with interactive zone selection
python3 splash_only_detector.py video.mp4 \
    --interactive-zone \
    --method combined \
    --threshold 15.0 \
    --temporal-sigma 2.0 \
    --peak-prominence 5.0

# Manual zone specification for batch processing
python3 splash_only_detector.py video.mp4 \
    --zone 0.75 0.9 \
    --method motion_intensity \
    --threshold 12.0

# High sensitivity detection
python3 splash_only_detector.py video.mp4 \
    --threshold 8.0 \
    --adaptive-factor 1.5 \
    --peak-distance 20 \
    --temporal-window 20
```

## 📊 Detection Methods

### `motion_intensity` (Recommended)
- Analyzes gradient magnitude of frame differences
- Most sophisticated and reliable method
- Best for typical diving scenarios

### `combined`
- Voting system using multiple detection algorithms
- More robust but computationally expensive
- Good for challenging conditions

### `frame_diff`
- Simple frame difference analysis
- Fast but basic
- Good for clear, high-contrast scenarios

### `optical_flow`
- Motion pattern analysis
- Detects radial outward splash patterns
- Sensitive to camera movement

### `contour`
- Foam and bubble detection using HSV analysis
- Good for detecting white water/foam
- May be sensitive to lighting conditions

## ⚙️ Configuration Parameters

### Zone Definition
- `--zone TOP BOTTOM`: Splash detection zone (normalized coordinates 0-1)
- `--interactive-zone`: Interactive zone selection using mouse click and drag
- Default: `0.7 0.95` (bottom 25% of frame)

**Interactive Zone Selection**:
- Click and drag to visually select the splash detection area
- Preview shows selected zone with overlay
- ESC key uses default zone
- Displays zone statistics (height, coordinates)

### Thresholding
- `--threshold VALUE`: Base detection threshold (default: 12.0)
- `--adaptive-factor FACTOR`: Adaptive threshold multiplier (default: 1.2)

### Gaussian Filtering
- `--temporal-sigma SIGMA`: Temporal smoothing strength (default: 1.5)
- `--temporal-window SIZE`: Smoothing window size in frames (default: 15)

### Peak Detection
- `--peak-prominence VALUE`: Minimum peak prominence (default: 3.0)
- `--peak-distance FRAMES`: Minimum frames between peaks (default: 30)

### Video Extraction
- `--pre-duration SECONDS`: Time before splash to include (default: 10.0)
- `--post-duration SECONDS`: Time after splash to include (default: 2.0)

## 🧪 Testing

Run the test suite to verify installation:
```bash
python3 test_splash_only_clean.py
```

## 📈 Debug Output

When `--debug` is enabled, the system generates:

1. **Comprehensive Plots** (`debug_splash_detection/splash_detection_analysis.png`):
   - Raw vs filtered scores over time
   - Peak detection visualization
   - Threshold adaptation behavior
   - Statistical analysis

2. **Debug Data** (`debug_splash_detection/debug_data.json`):
   - Complete detection metadata
   - Performance statistics
   - Configuration parameters
   - Event details

## 🎯 Technical Approach

### Problem Analysis
The main challenges in pure splash detection are:
1. **False Positives**: Water movement, camera shake, lighting changes
2. **Temporal Consistency**: Distinguishing real events from noise spikes
3. **Adaptive Thresholding**: Handling varying video conditions
4. **Peak Detection**: Robust event identification in noisy signals

### Solution Architecture

1. **Multi-Stage Filtering Pipeline**:
   ```
   Raw Video → Spatial Gaussian → Splash Detection → Temporal Gaussian → Peak Detection → Event Validation
   ```

2. **Adaptive Threshold System**:
   - Rolling median background estimation
   - Statistical threshold adjustment
   - Bounded threshold ranges

3. **Gaussian Temporal Smoothing**:
   - Scipy-based gaussian_filter1d when available
   - Fallback weighted moving average
   - Configurable smoothing strength

4. **Peak Detection Algorithm**:
   - Prominence-based peak identification
   - Minimum distance constraints
   - Width validation

## 🔬 Advantages over State-Machine Approach

### Simplicity
- No complex state transitions
- Fewer parameters to tune
- More predictable behavior

### Robustness
- Less sensitive to diver pose variations
- Works with different diving styles
- Handles partial occlusions better

### Performance
- Focus on splash zones only
- No pose detection overhead during detection
- Faster processing pipeline

### Debugging
- Clear visualization of detection process
- Easy threshold tuning
- Quantitative performance metrics

## 📋 Output

### Detection Results
```
🎯 DETECTION RESULTS
==================
📊 Total splash events detected: 3
⏱️  Detection time: 12.3 seconds

🌊 Detected Events:
  1. Frame   1205 | t=  40.2s | Score=  18.7 | Confidence=high
  2. Frame   1856 | t=  61.9s | Score=  15.3 | Confidence=high
  3. Frame   2447 | t=  81.6s | Score=  12.8 | Confidence=medium
```

### Extracted Videos
- `dive_splash_1_t40.2s.mp4`: 12s video around first splash
- `dive_splash_2_t61.9s.mp4`: 12s video around second splash
- `dive_splash_3_t81.6s_medium.mp4`: 12s video around third splash (medium confidence)

## 🛠️ Dependencies

- OpenCV (`cv2`)
- NumPy
- Matplotlib (for debug plots)
- SciPy (optional, for enhanced filtering)
- Original `slAIcer.py` (for splash detection methods)

## 📝 Notes

- The system requires the original `slAIcer.py` file for splash detection algorithms
- Debug plots require matplotlib
- Enhanced Gaussian filtering requires scipy
- Videos are extracted as MP4 files with detection zone overlays
- Splash zones are marked on extracted videos for validation

## 🎪 Future Enhancements

- Real-time processing mode
- Multiple splash zone support
- Machine learning-based event classification
- Audio-based splash detection
- Integration with pose estimation for validation
