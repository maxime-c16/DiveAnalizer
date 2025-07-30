# DiveAnalyzer 🏊‍♂️

**Automated diving video analysis tool for swimming pool diving videos using computer vision and AI.**

DiveAnalyzer automatically detects and extracts individual dives from swimming pool diving videos, providing clean, professional-quality clips with preserved audio and optional pose visualization overlay.

## 🚀 Features

### Core Functionality
- **🔍 Automatic Dive Detection**: Advanced computer vision algorithms detect takeoff and water entry points
- **🎬 Real-time Extraction**: Background processing for maximum efficiency
- **🔊 Audio Preservation**: FFmpeg integration maintains original audio quality
- **🎯 Pose Analysis**: Optional MediaPipe pose detection and visualization
- **⚡ Performance Optimized**: Multi-threaded processing with 4.46x speedup over sequential processing
- **📊 Comprehensive Metrics**: Detailed performance analytics and timing information

### Video Processing
- **📹 Multiple Formats**: Supports MP4, AVI, MOV, and other common video formats
- **🎨 Customizable Output**: Optional pose overlay, audio preservation controls
- **📏 Flexible Resolution**: Works with various video resolutions and aspect ratios
- **⏱️ Frame-accurate**: Precise frame-level dive detection and extraction

### Advanced Analytics
- **🏊 Dive Statistics**: Duration, confidence levels, and detailed metrics
- **📈 Performance Tracking**: Processing speed, extraction times, and efficiency metrics
- **💾 Intelligent Caching**: Performance data caching for improved subsequent runs
- **🎯 Zone-based Detection**: Customizable detection zones for optimal accuracy

## 📋 Requirements

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for dependencies and cache

### Dependencies
- **OpenCV**: Computer vision processing
- **MediaPipe**: Pose detection and analysis
- **NumPy**: Numerical computations
- **FFmpeg**: Audio processing (optional, for audio preservation)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/maxime-c16/DiveAnalizer.git
cd DiveAnalizer
```

### 2. Set Up Python Environment
```bash
# Create virtual environment (recommended)
python3 -m venv dive_env
source dive_env/bin/activate  # On Windows: dive_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install FFmpeg (Optional, for Audio Preservation)
**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH.

### 4. Verify Installation
```bash
python3 slAIcer.py --help
```

## 🎯 Quick Start

### Basic Usage
```bash
# Extract dives with default settings (clean videos + audio)
python3 slAIcer.py path/to/your/video.mp4

# Extract to specific directory
python3 slAIcer.py video.mp4 --output_dir extracted_dives

# Enable pose overlay
python3 slAIcer.py video.mp4 --show-pose-overlay

# Disable audio preservation
python3 slAIcer.py video.mp4 --no-audio
```

### Advanced Options
```bash
# Debug mode with visual feedback
python3 slAIcer.py video.mp4 --debug

# Different splash detection method
python3 slAIcer.py video.mp4 --splash_method optical_flow

# Disable threading (for debugging)
python3 slAIcer.py video.mp4 --no-threading
```

## 📐 Zone Configuration Guide

### Understanding Detection Zones

DiveAnalyzer uses three critical zones for optimal detection accuracy:

#### 1. **Diving Board Line** 🏗️
- **Purpose**: Establishes the diving platform reference point
- **Selection**: Click on the diving board edge or platform line
- **Tips**:
  - Choose a clear, horizontal line on the diving board
  - Avoid shadows or reflections
  - Use the auto-detect feature ('a' key) for automatic detection

#### 2. **Diver Detection Zone** 👤
- **Purpose**: Focuses pose detection on the relevant area
- **Selection**: Click and drag to create a rectangle covering the diving area
- **Best Practices**:
  - Include the entire diving board/platform area
  - Extend vertically to capture full diver height during approach
  - Include some margin around the expected diver positions
  - Avoid including spectators or background distractions

#### 3. **Splash Detection Zone** 💦
- **Purpose**: Monitors water surface for entry splash detection
- **Selection**: Click and drag to define the water surface monitoring area
- **Optimization Tips**:
  - Focus on the water surface where splashes occur
  - Include 20-30cm above and below the waterline
  - Avoid pool edges, lane markers, or lighting reflections
  - Consider water depth and typical dive entry points

### Zone Selection Best Practices

#### Video Quality Considerations
- **Lighting**: Ensure consistent lighting across detection zones
- **Camera Angle**: Side-view angles work best for dive detection
- **Stability**: Use tripod-mounted cameras for consistent zone boundaries
- **Resolution**: Higher resolution videos provide more accurate zone detection

#### Common Setup Mistakes to Avoid
- ❌ **Too Narrow Diver Zone**: May miss parts of the diving motion
- ❌ **Splash Zone Too High**: May not capture water entry effectively
- ❌ **Including Background Motion**: Trees, people, or vehicles in zones
- ❌ **Poor Lighting Conditions**: Shadows or glare affecting detection

#### Optimal Zone Configuration
```
┌─────────────────────────────────────┐
│                 SKY                 │
├─────────────────────────────────────┤
│   🏗️ DIVING BOARD LINE             │
│   👤 DIVER DETECTION ZONE          │
│     ┌─────────────────┐             │
│     │   Diving Area   │             │
│     │                 │             │
│     └─────────────────┘             │
├─────────────────────────────────────┤
│  💦 SPLASH DETECTION ZONE          │
│  ══════════════════════════════════ │ ← Water Surface
│                                     │
│            POOL WATER               │
└─────────────────────────────────────┘
```

## 🎬 Supported Video Types

### Recommended Formats
- **MP4** (H.264): Best compatibility and performance
- **MOV**: Good for high-quality recordings
- **AVI**: Widely supported format

### Video Specifications
- **Resolution**: 720p minimum, 1080p or 4K recommended
- **Frame Rate**: 30fps minimum, 60fps for detailed analysis
- **Duration**: No specific limits, but longer videos require more processing time
- **Codec**: H.264/H.265 for best compatibility

### Recording Guidelines
- **Camera Position**: Side view of the diving area
- **Stability**: Use tripod or stable mounting
- **Lighting**: Consistent, bright lighting without harsh shadows
- **Background**: Minimal background motion or distractions
- **Audio**: Include audio for complete dive analysis experience

## ⚙️ Configuration Options

### Command Line Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `video_path` | Required | Path to input video file |
| `--output_dir` | `data/pose_analysis/extracted_dives` | Output directory for extracted dives |
| `--splash_method` | `motion_intensity` | Splash detection algorithm |
| `--show-pose-overlay` | `False` | Enable pose visualization overlay |
| `--no-audio` | `False` | Disable audio preservation |
| `--debug` | `False` | Enable debug visualization |
| `--no-threading` | `False` | Disable multi-threaded processing |

### Splash Detection Methods

#### 1. **Motion Intensity** (Recommended)
- **Description**: Analyzes motion magnitude in splash zone
- **Best For**: General-purpose dive detection
- **Advantages**: Robust, fast, works in various lighting
- **Usage**: `--splash_method motion_intensity`

#### 2. **Optical Flow**
- **Description**: Tracks pixel movement patterns
- **Best For**: High-resolution videos with clear water surface
- **Advantages**: Very precise, good for detailed analysis
- **Usage**: `--splash_method optical_flow`

#### 3. **Frame Difference**
- **Description**: Compares consecutive frames for changes
- **Best For**: Stable camera setups with minimal background motion
- **Advantages**: Simple, computationally efficient
- **Usage**: `--splash_method frame_diff`

#### 4. **Contour Analysis**
- **Description**: Detects shape changes in water surface
- **Best For**: High-contrast water surface scenarios
- **Advantages**: Good for clear water entry detection
- **Usage**: `--splash_method contour`

#### 5. **Combined Method**
- **Description**: Uses multiple methods with voting system
- **Best For**: Challenging videos or maximum accuracy
- **Advantages**: Most robust, reduces false positives
- **Usage**: `--splash_method combined`

## 📊 Performance & Analytics

### Processing Performance
- **Real-time Processing**: 0.89x to 1.2x real-time speed
- **Multi-threading**: 4.46x speedup over sequential processing
- **Memory Usage**: Optimized for efficient memory utilization
- **Caching**: Intelligent performance caching for repeated runs

### Output Metrics
After processing, DiveAnalyzer provides comprehensive analytics:

```
📊 DIVE ANALYSIS METRICS
🎥 Video Information:
    📁 File: dive_video.mp4
    📏 Resolution: 1920x1080
    🎬 FPS: 30.0
    ⏱️  Duration: 45.2s (1356 frames)

⚡ Processing Performance:
    🔍 Detection Time: 12.34s
    💾 Extraction Time: 3.21s
    ⚡ Realtime Ratio: 1.15x

🏊 Dive Statistics:
    📊 Total Dives Found: 5
    ⏱️  Average Duration: 3.2s
    🏆 Longest Dive: #3 (4.1s)
    ⚡ Shortest Dive: #1 (2.8s)
```

### File Output Structure
```
output_directory/
├── dive_1.mp4              # First detected dive
├── dive_2.mp4              # Second detected dive
├── dive_3_low_conf.mp4     # Low confidence detection
└── ...
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **No Dives Detected**
**Possible Causes:**
- Detection zones poorly configured
- Insufficient lighting or contrast
- Camera angle not optimal

**Solutions:**
- Reconfigure detection zones more carefully
- Try different splash detection methods
- Use `--debug` mode for visual feedback
- Ensure side-view camera angle

#### 2. **Audio Missing in Output**
**Possible Causes:**
- FFmpeg not installed
- Audio disabled with `--no-audio`
- Source video has no audio track

**Solutions:**
- Install FFmpeg (see installation section)
- Remove `--no-audio` flag
- Verify source video has audio

#### 3. **Poor Performance**
**Possible Causes:**
- Large video files
- High-resolution processing
- Insufficient system resources

**Solutions:**
- Use `--no-threading` to debug threading issues
- Process shorter video segments
- Reduce video resolution before processing
- Close unnecessary applications

#### 4. **Inaccurate Dive Detection**
**Possible Causes:**
- Incorrect zone configuration
- Background motion interference
- Suboptimal splash detection method

**Solutions:**
- Reconfigure detection zones
- Try different `--splash_method` options
- Use `--debug` mode to analyze detection process
- Ensure stable camera setup

### Debug Mode

Enable debug mode for detailed analysis:
```bash
python3 slAIcer.py video.mp4 --debug
```

Debug mode provides:
- Real-time visualization of detection zones
- Frame-by-frame processing feedback
- Pose detection overlay
- Splash detection indicators
- State transition visualization

## 🎯 Advanced Usage

### Batch Processing
For processing multiple videos:
```bash
# Process all videos in a directory
for video in *.mp4; do
    python3 slAIcer.py "$video" --output_dir "processed/${video%.*}"
done
```

### Custom Output Organization
```bash
# Organize by date and video name
python3 slAIcer.py video.mp4 --output_dir "dives/$(date +%Y-%m-%d)/$(basename video.mp4 .mp4)"
```

### Performance Benchmarking
Use the included benchmark tool:
```bash
python3 threading_benchmark.py video.mp4 --frames 500
```

## 📚 API Reference

### Core Functions

#### `detect_and_extract_dives_realtime()`
Main processing function with real-time extraction.

**Parameters:**
- `video_path`: Path to input video
- `board_y_norm`: Normalized diving board Y position
- `water_y_norm`: Normalized water surface Y position
- `splash_zone_top_norm`: Top of splash detection zone
- `splash_zone_bottom_norm`: Bottom of splash detection zone
- `diver_zone_norm`: Diver detection zone coordinates
- `debug`: Enable debug visualization
- `splash_method`: Splash detection algorithm
- `use_threading`: Enable multi-threaded processing
- `output_dir`: Output directory for extracted videos
- `show_pose_overlay`: Enable pose visualization
- `preserve_audio`: Enable audio preservation

**Returns:**
Dictionary containing detected dives and comprehensive metrics.

#### `extract_and_save_dive()`
Extracts and saves individual dive videos.

**Parameters:**
- `video_path`: Source video path
- `dive_number`: Sequential dive number
- `start_idx`: Start frame index
- `end_idx`: End frame index
- `confidence`: Detection confidence level
- `output_dir`: Output directory
- `diver_zone_norm`: Diver detection zone (optional)
- `video_fps`: Video frame rate (optional)
- `show_pose_overlay`: Enable pose overlay (default: False)
- `preserve_audio`: Enable audio preservation (default: True)

## 🤝 Contributing

We welcome contributions to improve DiveAnalyzer! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow coding standards and add tests
4. **Run quality checks**: `ruff check slAIcer.py`
5. **Commit changes**: Use descriptive commit messages
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes and their benefits

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install ruff black pytest

# Run code quality checks
ruff check slAIcer.py
black slAIcer.py

# Run tests
python3 -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV**: Computer vision foundation
- **MediaPipe**: Advanced pose detection capabilities
- **FFmpeg**: Professional audio/video processing
- **NumPy**: Efficient numerical computations
- **Python Community**: Excellent ecosystem and libraries

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/maxime-c16/DiveAnalizer/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/maxime-c16/DiveAnalizer/discussions)
- **Email**: [macauchy@student.42.fr](mailto:macauchy@student.42.fr)

---

**DiveAnalyzer** - Transforming diving video analysis with AI-powered precision. 🏊‍♂️🚀
