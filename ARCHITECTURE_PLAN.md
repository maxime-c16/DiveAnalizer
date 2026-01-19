# DiveAnalyzer v2.0 - Architecture Overhaul Plan

## Executive Summary

Complete rework of DiveAnalyzer to achieve:
- **10x faster detection** through multi-modal signals (audio + motion + simple person detection)
- **Instant video extraction** via FFmpeg stream copy (no re-encoding)
- **Minimal storage footprint** through proxy workflow and smart caching
- **Seamless iCloud integration** for iPhone filming workflow

---

## Part 1: Storage & Cloud Strategy

### The Problem

| Recording Format | 4K 30fps Size | 1-Hour Session |
|------------------|---------------|----------------|
| HEVC (default)   | ~1.8 GB/10min | **~10.8 GB**   |
| ProRes 422       | ~12 GB/10min  | **~72 GB**     |

A typical 1-hour diving session in HEVC = 10-11GB. Processing this repeatedly wastes time and storage.

### Solution: Three-Tier Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STORAGE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TIER 1: iPhone/iCloud (Source of Truth)                                │
│  ════════════════════════════════════════                               │
│  • Original 4K HEVC recordings                                          │
│  • Never modified, never deleted by tool                                │
│  • Auto-synced via iCloud Drive                                         │
│  • Location: ~/Library/Mobile Documents/com~apple~CloudDocs/            │
│                                                                          │
│  TIER 2: Local Processing Cache (~/.diveanalyzer/)                      │
│  ════════════════════════════════════════════════                       │
│  • 480p proxy videos (10x smaller)                                      │
│  • Detection metadata (JSON)                                            │
│  • Audio waveforms (extracted once)                                     │
│  • Auto-cleanup after 7 days                                            │
│                                                                          │
│  TIER 3: Output (User-specified)                                        │
│  ════════════════════════════════                                       │
│  • Final extracted dive clips                                           │
│  • Full resolution from original                                        │
│  • User manages retention                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Storage Savings Calculation

| Asset | Old Approach | New Approach | Savings |
|-------|--------------|--------------|---------|
| Detection processing | Full 4K (10.8GB) | 480p proxy (0.5GB) | **95%** |
| Intermediate files | Multiple copies | Stream copy (0 extra) | **100%** |
| Cached analysis | None | JSON metadata (1MB) | N/A |
| Audio extraction | Per-run | Once, cached | **90%** |

**For a 1-hour session**: Old = ~30GB temp files, New = ~0.6GB temp files

### iCloud Integration Options

#### Option A: Native macOS Path (Recommended for simplicity)
```python
import os
from pathlib import Path

# iCloud Drive is automatically synced here on macOS
ICLOUD_PATH = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"

def find_diving_videos(folder_name="Diving"):
    """Find all videos in iCloud Diving folder."""
    diving_folder = ICLOUD_PATH / folder_name
    return list(diving_folder.glob("**/*.mov")) + list(diving_folder.glob("**/*.mp4"))
```

**Pros**: Zero setup, automatic sync, works offline
**Cons**: macOS only, requires iCloud sync enabled

#### Option B: pyicloud (For cross-platform or server deployment)
```python
from pyicloud import PyiCloudService

api = PyiCloudService('email@example.com')
# Handle 2FA if needed
if api.requires_2fa:
    code = input("Enter 2FA code: ")
    api.validate_2fa_code(code)

# Access Drive
drive = api.drive
diving_folder = drive['Diving']
for item in diving_folder.dir():
    if item.name.endswith(('.mov', '.mp4')):
        # Download to local cache
        with open(f"cache/{item.name}", 'wb') as f:
            f.write(item.open(stream=True).content)
```

**Pros**: Works on Linux/Windows, API access
**Cons**: Requires auth, 2FA handling, slower

#### Option C: rclone mount (For power users)
```bash
# One-time setup
rclone config  # Configure iCloud Drive

# Mount as virtual drive
rclone mount iclouddrive: ~/icloud-mount --vfs-cache-mode full

# Then access like local files
python diveanalyzer.py ~/icloud-mount/Diving/session.mov
```

**Pros**: Universal, can mount any cloud
**Cons**: Setup complexity, token refresh every 30 days

### Recommended Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   iPhone     │────▶│ iCloud Drive │────▶│   macOS      │
│  (Record)    │     │   (Sync)     │     │  (Process)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                     ┌─────────────────────────────────────┐
                     │        DiveAnalyzer v2.0            │
                     │                                     │
                     │  1. Detect iCloud folder            │
                     │  2. Generate 480p proxy (cached)    │
                     │  3. Run detection on proxy          │
                     │  4. Extract from ORIGINAL 4K        │
                     │  5. Output to user folder           │
                     └─────────────────────────────────────┘
```

---

## Part 2: New Detection Architecture

### Core Philosophy Change

| Old Approach | New Approach |
|--------------|--------------|
| Detect splash (visual) | Detect splash (audio) + motion + person |
| Process every frame | Sample at 5 FPS |
| Full resolution | 480p proxy |
| MediaPipe 33 landmarks | YOLO binary person detection |
| State machine (5 states) | Signal fusion (3 signals) |
| Re-encode extracted videos | FFmpeg stream copy |

### Multi-Modal Signal Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SIGNAL FUSION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SIGNAL 1: Audio Peaks (Primary - Most Reliable)                        │
│  ═══════════════════════════════════════════════                        │
│  • Extract audio track once (librosa/ffmpeg)                            │
│  • Compute RMS energy in 50ms windows                                   │
│  • Detect peaks above -20dB threshold                                   │
│  • Splash sound = sharp transient (< 500ms)                             │
│  • Processing: ~5 seconds for 1-hour video                              │
│                                                                          │
│  SIGNAL 2: Motion Activity (Secondary)                                  │
│  ═════════════════════════════════════                                  │
│  • Frame differencing at 5 FPS on 480p proxy                            │
│  • Detect sudden activity bursts                                        │
│  • Zone-restricted (diving area only)                                   │
│  • Processing: ~30 seconds for 1-hour video                             │
│                                                                          │
│  SIGNAL 3: Person Presence (Tertiary - Validation)                      │
│  ════════════════════════════════════════════════                       │
│  • YOLO-nano at 5 FPS on 480p proxy                                     │
│  • Binary: person in zone? yes/no                                       │
│  • Used to validate dive start (person → no person)                     │
│  • Processing: ~60 seconds for 1-hour video                             │
│                                                                          │
│  FUSION LOGIC                                                           │
│  ════════════                                                           │
│  A dive is detected when:                                               │
│    1. Audio peak detected (splash sound)                                │
│    2. Motion activity 2-12s before audio peak                           │
│    3. Person was present, then absent before splash                     │
│                                                                          │
│  Confidence scoring:                                                    │
│    - All 3 signals = HIGH confidence                                    │
│    - Audio + Motion = MEDIUM confidence                                 │
│    - Audio only = LOW confidence (still extract, flag for review)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Audio is the Best Signal

1. **Splash sound is distinctive**: Sharp transient, specific frequency range (100-2000 Hz)
2. **Works regardless of camera angle**: No visual occlusion issues
3. **Olympic divers minimize visual splash**: But audio signature remains
4. **Processing is instant**: Audio analysis is 100x faster than video
5. **Already captured**: iPhone records high-quality audio

### Detection Pipeline

```python
# Pseudocode for new detection flow

def detect_dives(video_path: str) -> List[DiveEvent]:
    """
    Multi-modal dive detection.
    Total processing time: ~2 minutes for 1-hour video.
    """

    # Phase 1: Extract/cache audio (5 seconds)
    audio_path = cache.get_or_create_audio(video_path)

    # Phase 2: Detect audio peaks (2 seconds)
    splash_candidates = detect_audio_peaks(audio_path)
    # Returns: [(timestamp, amplitude, confidence), ...]

    # Phase 3: Generate/cache proxy if needed (60 seconds, one-time)
    proxy_path = cache.get_or_create_proxy(video_path, resolution=480)

    # Phase 4: Validate with motion analysis (30 seconds)
    motion_events = detect_motion_bursts(proxy_path, sample_fps=5)

    # Phase 5: Validate with person detection (60 seconds)
    person_events = detect_person_transitions(proxy_path, sample_fps=5)

    # Phase 6: Fuse signals
    dives = []
    for splash_time, amplitude, _ in splash_candidates:
        # Look for motion 2-12s before splash
        motion_match = find_motion_before(motion_events, splash_time, window=(2, 12))

        # Look for person→no_person transition
        person_match = find_person_departure(person_events, splash_time, window=(2, 15))

        confidence = calculate_confidence(amplitude, motion_match, person_match)

        if confidence > 0.3:  # Low threshold - better to over-detect
            dives.append(DiveEvent(
                start_time=splash_time - 10,  # 10s before splash
                end_time=splash_time + 3,      # 3s after splash
                splash_time=splash_time,
                confidence=confidence
            ))

    return merge_overlapping_dives(dives)
```

---

## Part 3: Technology Stack

### Core Dependencies

```
# requirements.txt for DiveAnalyzer v2.0

# === Video Processing ===
decord>=0.6.0              # Fast video loading with seeking (2x faster than OpenCV)
# OR
av>=10.0.0                 # PyAV alternative (good FFmpeg bindings)

# === Audio Processing ===
librosa>=0.10.0            # Audio analysis, peak detection
soundfile>=0.12.0          # Audio file I/O

# === Detection ===
ultralytics>=8.0.0         # YOLO-nano for person detection
# Note: NOT mediapipe - too heavy for our use case

# === Scene Detection (Optional) ===
scenedetect>=0.6.0         # PySceneDetect for motion analysis

# === Utilities ===
numpy>=1.24.0              # Array operations
scipy>=1.10.0              # Signal processing (peak detection)
tqdm>=4.65.0               # Progress bars

# === Optional: GPU Acceleration ===
# torch>=2.0.0             # If using YOLO with GPU
# onnxruntime-gpu>=1.15.0  # Alternative GPU inference
```

### Why This Stack?

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Video Loading | **Decord** | 2x faster than OpenCV, proper O(1) seeking |
| Audio Analysis | **librosa** | Industry standard, excellent peak detection |
| Person Detection | **YOLO-nano** | 3MB model, 100+ FPS on CPU, accurate |
| Scene Detection | **PySceneDetect** | Proven, optimized, handles edge cases |
| Video Extraction | **FFmpeg CLI** | Stream copy = instant, preserves quality |

### What We're Removing

| Removed | Replacement | Reason |
|---------|-------------|--------|
| MediaPipe | YOLO-nano | 100x faster, we only need person yes/no |
| OpenCV VideoCapture | Decord | Proper seeking, no frame-by-frame skip |
| OpenCV VideoWriter | FFmpeg subprocess | Stream copy, no re-encoding |
| Custom splash detection | Audio peak detection | More reliable, faster |
| Complex state machine | Signal fusion | Simpler, more maintainable |

### System Dependencies

```bash
# macOS (Homebrew)
brew install ffmpeg           # Video/audio extraction
brew install libsndfile       # Audio file support

# Ubuntu/Debian
sudo apt install ffmpeg libsndfile1-dev

# Optional: For GPU acceleration
# NVIDIA: Install CUDA toolkit
# Apple Silicon: Metal acceleration is automatic
```

---

## Part 4: Project Structure

```
DiveAnalyzer/
├── diveanalyzer/                 # Main package
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   │
│   ├── detection/                # Detection modules
│   │   ├── __init__.py
│   │   ├── audio.py              # Audio peak detection
│   │   ├── motion.py             # Motion burst detection
│   │   ├── person.py             # YOLO person detection
│   │   └── fusion.py             # Signal fusion logic
│   │
│   ├── extraction/               # Video extraction
│   │   ├── __init__.py
│   │   ├── ffmpeg.py             # FFmpeg wrapper
│   │   └── proxy.py              # Proxy generation
│   │
│   ├── storage/                  # Storage management
│   │   ├── __init__.py
│   │   ├── cache.py              # Local cache management
│   │   ├── icloud.py             # iCloud integration
│   │   └── cleanup.py            # Auto-cleanup
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── video.py              # Video info helpers
│       └── progress.py           # Progress reporting
│
├── tests/                        # Test suite
│   ├── test_audio_detection.py
│   ├── test_motion_detection.py
│   ├── test_person_detection.py
│   ├── test_fusion.py
│   └── fixtures/                 # Test video clips
│
├── scripts/                      # Utility scripts
│   ├── benchmark.py              # Performance benchmarking
│   └── calibrate.py              # Zone calibration tool
│
├── CLAUDE.md                     # AI assistant guidance
├── ARCHITECTURE_PLAN.md          # This document
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package configuration
└── README.md                     # User documentation
```

---

## Part 5: Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Get basic audio detection working

```
Tasks:
├── [ ] Set up new project structure
├── [ ] Implement audio extraction (FFmpeg)
├── [ ] Implement audio peak detection (librosa)
├── [ ] Implement FFmpeg video extraction (stream copy)
├── [ ] Create basic CLI
└── [ ] Test on sample video

Deliverable: Can detect dives using audio only, extract clips
```

**Key Code: Audio Detection**
```python
# diveanalyzer/detection/audio.py

import librosa
import numpy as np
from scipy.signal import find_peaks

def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio track from video using FFmpeg."""
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '22050',  # Sample rate
        '-ac', '1',  # Mono
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path

def detect_splash_peaks(audio_path: str,
                        threshold_db: float = -25,
                        min_distance_sec: float = 5.0) -> list:
    """
    Detect splash sounds in audio.

    Returns list of (timestamp, amplitude) tuples.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Compute RMS energy in short windows
    hop_length = 512  # ~23ms windows
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Find peaks
    min_distance_frames = int(min_distance_sec * sr / hop_length)
    peaks, properties = find_peaks(
        rms_db,
        height=threshold_db,
        distance=min_distance_frames,
        prominence=5  # Must stand out from surroundings
    )

    # Convert to timestamps
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    peak_amplitudes = rms_db[peaks]

    return list(zip(peak_times, peak_amplitudes))
```

### Phase 2: Proxy Workflow (Week 2)
**Goal**: Add proxy generation and caching

```
Tasks:
├── [ ] Implement proxy video generation (FFmpeg)
├── [ ] Implement cache management
├── [ ] Add cache directory structure
├── [ ] Implement auto-cleanup (7 days)
├── [ ] Add iCloud path detection
└── [ ] Test proxy workflow end-to-end

Deliverable: Detection runs on 480p proxy, extraction from original
```

**Key Code: Proxy Generation**
```python
# diveanalyzer/extraction/proxy.py

import subprocess
import hashlib
from pathlib import Path

CACHE_DIR = Path.home() / '.diveanalyzer' / 'cache'

def get_video_hash(video_path: str) -> str:
    """Generate hash for video file (based on path + size + mtime)."""
    p = Path(video_path)
    stat = p.stat()
    key = f"{p.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:12]

def get_or_create_proxy(video_path: str, height: int = 480) -> str:
    """
    Get cached proxy or create new one.

    Proxy is ~10x smaller than original, sufficient for detection.
    """
    video_hash = get_video_hash(video_path)
    proxy_path = CACHE_DIR / 'proxies' / f"{video_hash}_{height}p.mp4"

    if proxy_path.exists():
        return str(proxy_path)

    proxy_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate proxy with FFmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'scale=-2:{height}',  # Scale to height, maintain aspect
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Fast encoding
        '-crf', '28',  # Lower quality OK for detection
        '-an',  # No audio in proxy (we extract separately)
        str(proxy_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    return str(proxy_path)
```

### Phase 3: Motion Detection (Week 3)
**Goal**: Add motion-based validation

```
Tasks:
├── [ ] Implement Decord video loader
├── [ ] Implement frame differencing at 5 FPS
├── [ ] Add zone restriction (user-defined area)
├── [ ] Implement motion burst detection
├── [ ] Integrate with audio peaks
└── [ ] Test combined detection

Deliverable: Audio + Motion detection working together
```

**Key Code: Motion Detection**
```python
# diveanalyzer/detection/motion.py

import numpy as np
from decord import VideoReader, cpu

def detect_motion_bursts(video_path: str,
                         sample_fps: float = 5.0,
                         zone: tuple = None) -> list:
    """
    Detect bursts of motion activity.

    Args:
        video_path: Path to video (use proxy for speed)
        sample_fps: Frames per second to sample
        zone: (x1, y1, x2, y2) normalized coordinates, or None for full frame

    Returns:
        List of (start_time, end_time, intensity) tuples
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    frame_skip = max(1, int(video_fps / sample_fps))

    motion_scores = []
    prev_gray = None

    for i in range(0, len(vr), frame_skip):
        frame = vr[i].asnumpy()

        # Apply zone if specified
        if zone:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = zone
            frame = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]

        # Convert to grayscale
        gray = np.mean(frame, axis=2).astype(np.uint8)

        if prev_gray is not None:
            # Simple frame difference
            diff = np.abs(gray.astype(float) - prev_gray.astype(float))
            score = np.mean(diff)
            timestamp = i / video_fps
            motion_scores.append((timestamp, score))

        prev_gray = gray

    # Find bursts (consecutive high-motion frames)
    return find_motion_bursts(motion_scores, threshold_percentile=80)

def find_motion_bursts(scores: list, threshold_percentile: float = 80) -> list:
    """Group consecutive high-motion frames into bursts."""
    if not scores:
        return []

    timestamps, values = zip(*scores)
    threshold = np.percentile(values, threshold_percentile)

    bursts = []
    burst_start = None
    burst_max = 0

    for t, v in scores:
        if v > threshold:
            if burst_start is None:
                burst_start = t
            burst_max = max(burst_max, v)
        else:
            if burst_start is not None:
                bursts.append((burst_start, t, burst_max))
                burst_start = None
                burst_max = 0

    return bursts
```

### Phase 4: Person Detection (Week 4)
**Goal**: Add YOLO-based person validation

```
Tasks:
├── [ ] Integrate YOLO-nano model
├── [ ] Implement zone-restricted detection
├── [ ] Implement person transition detection (present → absent)
├── [ ] Add confidence scoring
├── [ ] Optimize batch inference
└── [ ] Test full signal fusion

Deliverable: All 3 signals working together
```

**Key Code: Person Detection**
```python
# diveanalyzer/detection/person.py

from ultralytics import YOLO
from decord import VideoReader, cpu
import numpy as np

# Load model once at module level
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8n.pt')  # Nano model, ~6MB
    return _model

def detect_person_transitions(video_path: str,
                              sample_fps: float = 5.0,
                              zone: tuple = None,
                              confidence_threshold: float = 0.5) -> list:
    """
    Detect when person leaves the frame/zone.

    Returns list of (timestamp, transition_type) where transition_type is:
        - 'enter': person appeared
        - 'leave': person disappeared
    """
    model = get_model()
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    frame_skip = max(1, int(video_fps / sample_fps))

    transitions = []
    prev_person_present = False

    for i in range(0, len(vr), frame_skip):
        frame = vr[i].asnumpy()
        timestamp = i / video_fps

        # Run detection
        results = model(frame, classes=[0], verbose=False)  # class 0 = person

        # Check if person in zone
        person_present = False
        for box in results[0].boxes:
            if box.conf[0] < confidence_threshold:
                continue

            if zone:
                # Check if box overlaps with zone
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Normalized coords
                zx1, zy1, zx2, zy2 = zone

                # Simple overlap check
                if x2 > zx1 and x1 < zx2 and y2 > zy1 and y1 < zy2:
                    person_present = True
                    break
            else:
                person_present = True
                break

        # Detect transitions
        if person_present and not prev_person_present:
            transitions.append((timestamp, 'enter'))
        elif not person_present and prev_person_present:
            transitions.append((timestamp, 'leave'))

        prev_person_present = person_present

    return transitions
```

### Phase 5: Signal Fusion (Week 5)
**Goal**: Combine all signals into final detection

```
Tasks:
├── [ ] Implement signal fusion logic
├── [ ] Add confidence scoring system
├── [ ] Handle edge cases (overlapping dives, false positives)
├── [ ] Add configurable parameters
├── [ ] Implement dive merging
└── [ ] Full integration testing

Deliverable: Complete detection pipeline
```

**Key Code: Signal Fusion**
```python
# diveanalyzer/detection/fusion.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DiveEvent:
    start_time: float
    end_time: float
    splash_time: float
    confidence: float
    audio_amplitude: float
    motion_intensity: Optional[float] = None
    had_person: bool = False

def fuse_signals(audio_peaks: list,
                 motion_bursts: list,
                 person_transitions: list,
                 pre_splash_buffer: float = 10.0,
                 post_splash_buffer: float = 3.0) -> List[DiveEvent]:
    """
    Fuse multi-modal signals into dive events.

    Logic:
    1. Audio peak = potential splash (primary signal)
    2. Motion burst 2-12s before = dive activity (validation)
    3. Person leave 2-15s before = diver jumped (validation)

    Confidence:
    - Audio + Motion + Person = 1.0
    - Audio + Motion = 0.8
    - Audio + Person = 0.7
    - Audio only = 0.5
    """
    dives = []

    for splash_time, amplitude in audio_peaks:
        # Look for motion burst before splash
        motion_match = None
        for burst_start, burst_end, intensity in motion_bursts:
            time_before = splash_time - burst_end
            if 1.0 < time_before < 12.0:  # Motion ended 1-12s before splash
                motion_match = (burst_start, intensity)
                break

        # Look for person departure before splash
        person_match = None
        for transition_time, transition_type in person_transitions:
            if transition_type == 'leave':
                time_before = splash_time - transition_time
                if 1.0 < time_before < 15.0:  # Person left 1-15s before splash
                    person_match = transition_time
                    break

        # Calculate confidence
        confidence = 0.5  # Base: audio only
        if motion_match:
            confidence += 0.3
        if person_match:
            confidence += 0.2

        # Determine dive start
        if motion_match:
            dive_start = max(0, motion_match[0] - 2)  # 2s before motion
        elif person_match:
            dive_start = max(0, person_match - 2)  # 2s before person left
        else:
            dive_start = max(0, splash_time - pre_splash_buffer)

        dives.append(DiveEvent(
            start_time=dive_start,
            end_time=splash_time + post_splash_buffer,
            splash_time=splash_time,
            confidence=confidence,
            audio_amplitude=amplitude,
            motion_intensity=motion_match[1] if motion_match else None,
            had_person=person_match is not None
        ))

    return merge_overlapping_dives(dives)

def merge_overlapping_dives(dives: List[DiveEvent],
                            min_gap: float = 5.0) -> List[DiveEvent]:
    """Merge dives that are too close together."""
    if not dives:
        return []

    dives = sorted(dives, key=lambda d: d.start_time)
    merged = [dives[0]]

    for dive in dives[1:]:
        prev = merged[-1]
        if dive.start_time < prev.end_time + min_gap:
            # Merge: extend previous dive
            prev.end_time = max(prev.end_time, dive.end_time)
            prev.confidence = max(prev.confidence, dive.confidence)
        else:
            merged.append(dive)

    return merged
```

### Phase 6: CLI & Polish (Week 6)
**Goal**: User-friendly command-line interface

```
Tasks:
├── [ ] Implement CLI with argparse/click
├── [ ] Add progress reporting
├── [ ] Add configuration file support
├── [ ] Implement zone calibration tool
├── [ ] Add batch processing mode
├── [ ] Write documentation
└── [ ] Performance benchmarking

Deliverable: Production-ready tool
```

**Key Code: CLI**
```python
# diveanalyzer/cli.py

import argparse
from pathlib import Path
from .detection import detect_dives
from .extraction import extract_dives
from .storage import CacheManager

def main():
    parser = argparse.ArgumentParser(
        description='DiveAnalyzer - Automatic diving video clip extraction'
    )
    parser.add_argument('video', help='Path to video file or iCloud folder')
    parser.add_argument('-o', '--output', default='./dives',
                        help='Output directory for extracted clips')
    parser.add_argument('--zone', type=str,
                        help='Detection zone as x1,y1,x2,y2 (normalized 0-1)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum confidence threshold (0-1)')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Skip proxy generation, process original')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run zone calibration tool')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.video)
        return

    # Initialize cache
    cache = CacheManager()

    # Parse zone if provided
    zone = None
    if args.zone:
        zone = tuple(map(float, args.zone.split(',')))

    # Detect dives
    print(f"Analyzing: {args.video}")
    dives = detect_dives(
        args.video,
        cache=cache,
        zone=zone,
        use_proxy=not args.no_proxy,
        verbose=args.verbose
    )

    # Filter by confidence
    dives = [d for d in dives if d.confidence >= args.min_confidence]
    print(f"Found {len(dives)} dives")

    # Extract clips
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_dives(args.video, dives, output_dir, verbose=args.verbose)

    print(f"Extracted {len(dives)} clips to {output_dir}")

if __name__ == '__main__':
    main()
```

---

## Part 6: Performance Targets

### Processing Speed Goals

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| 1-hour video detection | ~60 min | **< 3 min** | 20x |
| Single dive extraction | ~10 sec | **< 1 sec** | 10x |
| Memory usage (1hr video) | ~4 GB | **< 500 MB** | 8x |
| Storage (temp files) | ~30 GB | **< 1 GB** | 30x |

### Benchmark Comparison

```
OLD PIPELINE (current):
├── Open video: 2 sec
├── Process 108,000 frames @ 30fps
│   ├── MediaPipe pose per frame: 100ms × 108,000 = 3 hours
│   ├── (With frame skip): 100ms × 10,800 = 18 min
│   └── Splash detection per frame: 50ms × 108,000 = 90 min
├── Extract each dive (re-encode): 10 sec × 10 dives = 100 sec
└── TOTAL: ~60-90 minutes

NEW PIPELINE (proposed):
├── Extract audio: 5 sec (one-time)
├── Analyze audio peaks: 2 sec
├── Generate proxy: 60 sec (one-time, cached)
├── Motion detection @ 5fps on 480p: 30 sec
├── YOLO person detection @ 5fps: 60 sec
├── Signal fusion: < 1 sec
├── FFmpeg stream-copy extraction: 1 sec × 10 dives = 10 sec
└── TOTAL: ~3 minutes (first run), ~2 minutes (cached)
```

---

## Part 7: Migration Path

### Keeping Backward Compatibility

```python
# Legacy wrapper for existing users
# slAIcer.py (deprecated, redirects to new system)

import warnings
from diveanalyzer import detect_dives, extract_dives

def main():
    warnings.warn(
        "slAIcer.py is deprecated. Use 'diveanalyzer' command instead.",
        DeprecationWarning
    )
    # Parse old arguments, map to new system
    # ... (compatibility shim)
```

### Files to Archive

```
# Move to archive/ folder (keep for reference)
├── slAIcer.py                    # Old main script
├── splash_only_detector.py       # Old splash detection
├── detection_tool.py             # Old detection tool
├── test_splash_only.py           # Old tests
├── test_splash_only_clean.py     # Old tests
└── improved_splash_test.py       # Old tests
```

---

## Part 8: Testing Strategy

### Test Data

```
tests/fixtures/
├── short_dive.mp4          # 30-second clip with 1 clear dive
├── multi_dive.mp4          # 5-minute clip with 3 dives
├── noisy_audio.mp4         # Background noise, crowd sounds
├── no_splash_audio.mp4     # Silent video (test motion-only)
├── edge_cases/
│   ├── back_to_back.mp4    # Two dives < 5 sec apart
│   ├── failed_dive.mp4     # Aborted dive (person returns)
│   └── false_positive.mp4  # Loud noise but no dive
└── ground_truth.json       # Manual annotations for testing
```

### Test Cases

```python
# tests/test_detection.py

def test_audio_detection_finds_splash():
    """Audio peak detection should find splash in clean recording."""
    peaks = detect_audio_peaks('fixtures/short_dive.mp4')
    assert len(peaks) == 1
    assert 8.0 < peaks[0][0] < 12.0  # Splash around 10 seconds

def test_motion_detection_finds_dive():
    """Motion detection should find activity burst before splash."""
    bursts = detect_motion_bursts('fixtures/short_dive.mp4')
    assert len(bursts) >= 1

def test_fusion_high_confidence():
    """All three signals should produce high confidence."""
    dives = detect_dives('fixtures/short_dive.mp4')
    assert len(dives) == 1
    assert dives[0].confidence >= 0.9

def test_extraction_preserves_quality():
    """Extracted clip should match original quality."""
    # Extract and compare resolution/bitrate
    ...

def test_cache_reuse():
    """Second run should use cached proxy/audio."""
    import time
    t1 = time.time()
    detect_dives('fixtures/multi_dive.mp4')
    first_run = time.time() - t1

    t2 = time.time()
    detect_dives('fixtures/multi_dive.mp4')
    second_run = time.time() - t2

    assert second_run < first_run * 0.5  # At least 2x faster
```

---

## Summary

### What Changes

1. **Detection**: Visual splash → Audio splash + Motion + Person (3 signals)
2. **Video Loading**: OpenCV → Decord (2x faster, proper seeking)
3. **Person Detection**: MediaPipe 33 landmarks → YOLO binary (100x faster)
4. **Extraction**: OpenCV re-encode → FFmpeg stream copy (instant)
5. **Storage**: Process original → Proxy workflow (10x smaller)
6. **Architecture**: Monolithic → Modular packages

### What Stays

1. **Zone configuration**: Still user-defined detection areas
2. **Output format**: Same extracted MP4 clips
3. **CLI interface**: Similar command structure
4. **Audio preservation**: Still keeps original audio

### Expected Outcomes

- **Speed**: 60 min → 3 min for 1-hour session
- **Storage**: 30 GB temp → 1 GB temp
- **Accuracy**: Better (multi-modal validation)
- **Maintainability**: Much better (modular, testable)
