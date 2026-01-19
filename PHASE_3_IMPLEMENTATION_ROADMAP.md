# Phase 3 Implementation Roadmap

**Status**: Ready to implement
**Timeline**: 2-3 weeks
**Complexity**: High (multi-modal sensor fusion)

---

## Phase 3 Vision

**Goal**: Add person detection as the third validation signal for maximum dive detection confidence.

**Current State** (Phase 2):
- Audio + Motion detection
- Confidence: 0.92 average
- False positives: 5-10%
- Speed: ~5 minutes per session

**Phase 3 Target**:
- Audio + Motion + Person detection
- Confidence: 0.96 average
- False positives: 1-3%
- Speed: ~6-7 minutes per session (only +1-2min for person inference)

---

## Architecture: Three-Signal Detection

```
PHASE 3 COMPLETE PIPELINE
═══════════════════════════════════════════════════════

INPUT VIDEO (4K, 8-10 min, 500MB)
  │
  ├─ SIGNAL 1: Audio (Primary)
  │  ├─ Extract audio (1.2s)
  │  ├─ RMS energy analysis
  │  ├─ Detect splash peaks
  │  └─ Base confidence: 0.5-0.9
  │
  ├─ SIGNAL 2: Motion (Secondary)
  │  ├─ Generate 480p proxy (60s, first time)
  │  ├─ Frame differencing at 5 FPS
  │  ├─ Detect motion bursts
  │  └─ Boost: +15% confidence
  │
  ├─ SIGNAL 3: Person (Tertiary) ← NEW
  │  ├─ YOLO-nano inference on proxy
  │  ├─ Track person in dive zone
  │  ├─ Detect person zone departures
  │  └─ Boost: +10% confidence
  │
  ├─ Signal Fusion
  │  ├─ Audio confidence base
  │  ├─ Motion boost if detected
  │  ├─ Person boost if departed zone
  │  └─ Final confidence: min(base + boosts, 1.0)
  │
  └─ Clip Extraction (4K quality)
     └─ Extract 30 dive clips from original

OUTPUT: High-confidence dive clips with three-signal validation
═══════════════════════════════════════════════════════
```

---

## Implementation Phases

### Phase 3a: Person Detection (Week 1)

**Objective**: Integrate YOLO-nano for frame-by-frame person detection

#### What Gets Created

```
diveanalyzer/detection/person.py (NEW - 300+ lines)
├─ Person detection functions
├─ Frame sampling at 5 FPS
├─ YOLO-nano model loading
├─ Confidence thresholding
└─ Person presence timeline

Key Functions:
├─ detect_person_frames()      - Get person presence per frame
├─ find_person_transitions()   - Detect appearance/disappearance
├─ estimate_person_confidence()- Confidence scoring
└─ load_yolo_model()           - Model management
```

#### Implementation Steps

1. **Install YOLO** (if not present)
   ```bash
   pip install ultralytics>=8.0.0
   ```

2. **Create person.py module**
   ```python
   # diveanalyzer/detection/person.py
   from ultralytics import YOLO
   import cv2

   def load_yolo_model(model_name: str = "yolov8n.pt"):
       """Load YOLO-nano model (fastest for real-time)."""
       model = YOLO(model_name)
       return model

   def detect_person_frames(
       video_path: str,
       sample_fps: float = 5.0,
       confidence_threshold: float = 0.5,
   ) -> List[Tuple[float, bool, float]]:
       """
       Detect frames where person is present.
       Returns: List of (timestamp, person_present, confidence)
       """
       # Implementation

   def find_person_zone_departures(
       person_timeline: List[Tuple[float, bool, float]],
       min_departure_duration: float = 0.5,
   ) -> List[float]:
       """
       Find moments when person leaves diving zone.
       Returns: List of departure timestamps
       """
       # Implementation
   ```

3. **Test on sample video**
   ```bash
   python3 -c "
   from diveanalyzer.detection.person import detect_person_frames
   frames = detect_person_frames('IMG_6496.MOV')
   print(f'Found {len(frames)} frames with person data')
   "
   ```

#### Expected Results

- YOLO model download: ~40MB (cached)
- Inference speed: 1-2 FPS on CPU, 5-10 FPS on GPU
- Detection: Person visible in ~70% of frames
- Departures: ~30 person-zone exits (one per dive)

---

### Phase 3b: Zone-Based Validation (Week 2)

**Objective**: Track person presence in a specific diving zone

#### What Gets Added

```
diveanalyzer/detection/person.py (EXPANDED)
├─ Zone calibration tools
├─ Zone validation functions
├─ Person tracking in zone
└─ Departure detection

Key Functions:
├─ calibrate_dive_zone_interactive() - User selects zone
├─ track_person_in_zone()            - Per-frame zone presence
├─ smooth_zone_timeline()            - Remove jitter
└─ find_zone_transitions()           - Detect departures
```

#### Implementation Steps

1. **Interactive Zone Calibration**
   ```python
   def calibrate_dive_zone_interactive(video_path: str):
       """
       Shows first frame of video.
       User clicks to define diving zone (pool area).
       Saves normalized coordinates for reuse.
       """
       # OpenCV mouse click handler
       # Rectangle selection
       # Zone validation on next 10 frames
   ```

2. **Zone Tracking**
   ```python
   def track_person_in_zone(
       video_path: str,
       zone: Tuple[float, float, float, float],  # (x1, y1, x2, y2) normalized
       sample_fps: float = 5.0,
   ) -> List[Tuple[float, bool, float]]:
       """
       For each sampled frame:
       1. Run YOLO person detection
       2. Check if bounding box overlaps zone
       3. Record presence in zone
       4. Smooth over time (reduce jitter)

       Returns: List of (timestamp, in_zone, confidence)
       """
   ```

3. **Zone Management**
   ```python
   # Cache zone coordinates per video
   def save_zone_config(video_path: str, zone: Tuple):
       cache = CacheManager()
       cache.put_metadata(
           video_path,
           {"dive_zone": zone},
           category="zone"
       )

   def load_zone_config(video_path: str):
       cache = CacheManager()
       metadata = cache.get_metadata(video_path, category="zone")
       return metadata.get("dive_zone") if metadata else None
   ```

#### Expected Results

- Zone selection: 30 seconds per video (first time)
- Zone tracking: ~2-3 seconds (on cached proxy)
- Accuracy: 95%+ of dives have person departure signal
- Cache: Zone coordinates reused for subsequent runs

---

### Phase 3c: Signal Fusion Integration (Week 2)

**Objective**: Update fusion algorithm for three-signal confidence scoring

#### What Gets Updated

```
diveanalyzer/detection/fusion.py (EXPANDED)
├─ Three-signal fusion function
├─ Adaptive confidence scoring
├─ Edge case handling
└─ Validation statistics

Key Changes:
├─ Add fuse_signals_audio_motion_person()
├─ Update confidence calculation
├─ Add validation flags to DiveEvent
└─ Statistics: 3-signal, 2-signal, audio-only splits
```

#### Implementation Steps

1. **Three-Signal Fusion**
   ```python
   def fuse_signals_audio_motion_person(
       audio_peaks: List[Tuple[float, float]],
       motion_events: List[Tuple[float, float, float]],
       person_departures: List[float],
       window_before: float = 5.0,
       motion_boost: float = 0.15,
       person_boost: float = 0.10,
   ) -> List[DiveEvent]:
       """
       For each audio peak:
       1. Start with audio confidence
       2. Check for motion in window
       3. Check for person departure in window
       4. Apply boosts (non-cumulative, capped at 1.0)
       5. Determine validation level
       """
       for splash_time, amplitude in audio_peaks:
           # Base confidence from audio amplitude
           audio_conf = 0.5 + (normalize(amplitude) * 0.5)

           # Check motion
           has_motion = find_motion_before(motion_events, splash_time, window_before)

           # Check person
           has_person = find_person_departure_before(person_departures, splash_time, window_before)

           # Calculate final confidence
           confidence = audio_conf
           if has_motion:
               confidence = min(1.0, confidence + motion_boost)
           if has_person:
               confidence = min(1.0, confidence + person_boost)

           # Create dive event with validation info
           yield DiveEvent(
               splash_time=splash_time,
               confidence=confidence,
               audio_amplitude=amplitude,
               motion_validated=has_motion,
               person_validated=has_person,
               validation_level=get_validation_level(has_motion, has_person),
           )
   ```

2. **Update DiveEvent Dataclass**
   ```python
   @dataclass
   class DiveEvent:
       splash_time: float
       confidence: float
       audio_amplitude: float
       motion_intensity: float = 0.0
       motion_validated: bool = False
       person_validated: bool = False
       notes: str = "audio"

       @property
       def validation_level(self) -> str:
           """Three-signal, two-signal, or audio-only."""
           if self.motion_validated and self.person_validated:
               return "3-signal"
           elif self.motion_validated or self.person_validated:
               return "2-signal"
           else:
               return "audio-only"
   ```

3. **Update CLI**
   ```python
   # New options in process command
   @click.option("--enable-person", is_flag=True, default=False)
   @click.option("--calibrate-zone", is_flag=True, default=False)

   # New commands
   @cli.command()
   def calibrate_zone(video_path: str):
       """Interactive zone selection for person detection."""

   @cli.command()
   def analyze_person(video_path: str):
       """Debug: Show person detection results."""
   ```

#### Expected Results

- Fusion algorithm: 50 lines of clean code
- Confidence calculation: Accurate 3-signal weighting
- CLI integration: Seamless --enable-person flag
- Statistics: Breakdown of 3-signal, 2-signal, audio-only

---

### Phase 3d: Performance Optimization (Week 3)

**Objective**: Ensure <6s total for person detection on proxy

#### Optimization Strategies

1. **GPU Acceleration** (if NVIDIA GPU available)
   ```python
   # YOLO GPU support (built-in)
   model = YOLO("yolov8n.pt")
   model.to("cuda")  # Automatic GPU detection

   # Expected speedup:
   # CPU: 2-3s for inference
   # GPU: 0.5-1s for inference
   ```

2. **Model Quantization**
   ```python
   # Use FP16 (half-precision) for 2x speedup
   model = YOLO("yolov8n.pt")
   results = model.predict(frame, half=True)  # FP16 inference

   # Trade-off: 1-2% accuracy loss for 2x speed
   ```

3. **Batch Processing**
   ```python
   # Process multiple frames at once
   frames_batch = load_batch_frames(video_path, batch_size=8)
   results = model.predict(frames_batch)  # 8 frames at once

   # Speedup: 1.5-2x with minimal accuracy loss
   ```

4. **Model Caching**
   ```python
   # Load model once, reuse across runs
   _model_cache = None

   def get_yolo_model():
       global _model_cache
       if _model_cache is None:
           _model_cache = YOLO("yolov8n.pt")
       return _model_cache
   ```

#### Performance Targets

```
CPU-Only (Default):
├─ Model load: 2s (first run)
├─ Inference (480p): 2-3s
├─ Zone tracking: 1s
└─ Total: 5-6s ✓

GPU (NVIDIA):
├─ Model load: 2s
├─ Inference (480p): 0.5-1s
├─ Zone tracking: 0.5s
└─ Total: 3-3.5s ✓

Expected: <6s for full three-signal detection
```

---

## What Gets Updated

### Files Modified

| File | Change | Lines | Impact |
|------|--------|-------|--------|
| `diveanalyzer/detection/person.py` | NEW (300+) | 300+ | Person detection |
| `diveanalyzer/detection/fusion.py` | Add 3-signal fusion | +50 | Confidence scoring |
| `diveanalyzer/config.py` | Add PersonDetectionConfig | +20 | Configuration |
| `diveanalyzer/cli.py` | Add person options | +50 | User interface |
| `diveanalyzer/storage/cache.py` | Add zone caching | +15 | Zone persistence |
| `tests/test_person_detection.py` | NEW (200+) | 200+ | Unit tests |
| `tests/integration_test_phase3.py` | NEW (150+) | 150+ | Integration tests |

### New Commands

```bash
# Enable person detection
diveanalyzer process video.mov --enable-person

# Calibrate zone (interactive)
diveanalyzer process video.mov --calibrate-zone

# Debug person detection
diveanalyzer analyze-person video.mov --verbose

# Full three-signal pipeline
diveanalyzer process video.mov --enable-motion --enable-person --verbose
```

### Updated Configuration

```python
# config.py additions
@dataclass
class PersonDetectionConfig:
    enabled: bool = False
    model_name: str = "yolov8n.pt"          # Nano model (fastest)
    confidence_threshold: float = 0.5        # Person detection threshold
    sample_fps: float = 5.0                  # Frame sampling
    zone: Optional[Tuple[float, float, float, float]] = None
    use_gpu: bool = False                    # Auto-detect NVIDIA GPU
    quantization: str = "fp32"               # fp32, fp16, int8
    person_validation_boost: float = 0.10   # Confidence boost
    auto_zone_calibration: bool = False      # Future: auto-detect zone
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_person_detection.py

def test_yolo_loads():
    """YOLO model downloads and loads."""
    model = load_yolo_model()
    assert model is not None

def test_person_detection_on_sample_frame():
    """YOLO detects person in sample image."""
    results = detect_person_frames("test_data/person.jpg")
    assert len(results) > 0

def test_zone_departure_detection():
    """Zone departure finder works correctly."""
    timeline = [
        (1.0, True),   # In zone
        (2.0, True),   # In zone
        (3.0, False),  # LEFT zone (departure)
        (4.0, False),  # Out of zone
        (5.0, True),   # Back in zone (re-entry)
    ]
    departures = find_person_zone_departures(timeline)
    assert len(departures) == 1
    assert departures[0] == 3.0

def test_three_signal_fusion():
    """Three-signal fusion increases confidence."""
    # Create test data
    audio_peaks = [(5.0, -20.0)]  # Splice at 5s
    motion_events = [(2.0, 6.0, 0.8)]  # Motion 2-6s
    person_departures = [(4.8, 0.9)]  # Person departure at 4.8s

    # Fuse signals
    dives = fuse_signals_audio_motion_person(
        audio_peaks, motion_events, person_departures
    )

    # Verify
    dive = dives[0]
    assert dive.confidence > 0.8
    assert dive.motion_validated == True
    assert dive.person_validated == True
    assert dive.validation_level == "3-signal"
```

### Integration Tests

```python
# tests/integration_test_phase3.py

def test_full_pipeline_three_signals():
    """End-to-end test with all three signals."""
    video = "test_videos/diving_with_person.mov"

    # Run full pipeline
    audio_peaks = detect_splash_peaks(...)
    motion_events = detect_motion_bursts(...)
    person_departures = detect_person_frames(...)
    dives = fuse_signals_audio_motion_person(...)

    # Verify results
    assert len(dives) == expected_count
    assert all(d.confidence >= 0.75 for d in dives)

    # Check validation levels
    three_signal = sum(1 for d in dives if d.validation_level == "3-signal")
    two_signal = sum(1 for d in dives if d.validation_level == "2-signal")
    audio_only = sum(1 for d in dives if d.validation_level == "audio-only")

    assert three_signal > 0  # At least some 3-signal
    assert audio_only < expected_count * 0.2  # <20% audio-only

def test_person_detection_performance():
    """Motion+person detection < 6s on 480p proxy."""
    video = "test_videos/480p_proxy.mp4"

    start = time.time()
    motion_events = detect_motion_bursts(video)
    person_departures = detect_person_frames(video)
    elapsed = time.time() - start

    assert elapsed < 6.0, f"Too slow: {elapsed}s"

def test_edge_case_no_visible_person():
    """Graceful degradation when person not visible."""
    video = "test_videos/pool_with_obstacles.mov"

    dives = full_pipeline(video, enable_person=True)

    # Should still detect dives (fall back to audio+motion)
    assert len(dives) > 0
    audio_only_count = sum(1 for d in dives if d.validation_level == "audio-only")
    # Most should be audio-only if person not visible
    assert audio_only_count > len(dives) * 0.7

def test_edge_case_multiple_people():
    """Handles multiple people in frame."""
    video = "test_videos/crowded_pool.mov"

    dives = full_pipeline(video, enable_person=True)

    # Should still work (ANY person leaving zone = dive)
    assert len(dives) > 0
```

---

## Phase 3 Deliverables

### Code
- ✅ `diveanalyzer/detection/person.py` (300+ lines)
- ✅ Updated `diveanalyzer/detection/fusion.py` (+50 lines)
- ✅ Updated `diveanalyzer/config.py` (+20 lines)
- ✅ Updated `diveanalyzer/cli.py` (+50 lines)
- ✅ `tests/test_person_detection.py` (200+ lines)
- ✅ `tests/integration_test_phase3.py` (150+ lines)

### Documentation
- ✅ `PHASE_3_IMPLEMENTATION_ROADMAP.md` (this file)
- ✅ `PHASE_3_USER_GUIDE.md` (how to use)
- ✅ `YOLO_INTEGRATION_GUIDE.md` (technical details)
- ✅ `ZONE_CALIBRATION_GUIDE.md` (interactive setup)
- ✅ Updated `README_V2.md` (all features)

### Testing
- ✅ Unit tests (8+ tests)
- ✅ Integration tests (4+ tests)
- ✅ Real-world validation (multiple videos)
- ✅ Performance benchmarks (GPU vs CPU)
- ✅ Edge case coverage

---

## Success Criteria for Phase 3

✅ **Implementation Complete When:**

1. Person detection module fully implemented and tested
2. Zone-based validation working on real videos
3. Three-signal fusion improving accuracy by 5-10%
4. CLI updated with --enable-person flag
5. Performance < 6s for full pipeline on 480p proxy
6. Documentation complete and clear
7. Real-world tested on user's diving videos
8. Graceful fallback when person not visible
9. All edge cases handled
10. Test coverage > 80%

---

## Timeline Estimate

```
WEEK 1: Person Detection Module
├─ YOLO integration: 6-8 hours
├─ Frame sampling: 2-3 hours
├─ Unit tests: 4-5 hours
└─ Integration testing: 3-4 hours

WEEK 2: Zone Validation + Fusion
├─ Zone calibration UI: 6-8 hours
├─ Zone tracking: 4-5 hours
├─ Three-signal fusion: 4-5 hours
├─ CLI updates: 3-4 hours
└─ Edge case testing: 4-5 hours

WEEK 3: Optimization + Docs
├─ GPU acceleration: 4-5 hours
├─ Performance tuning: 4-5 hours
├─ User documentation: 6-8 hours
├─ Real-world testing: 6-8 hours
└─ Polish & refinement: 4-5 hours

TOTAL: 60-75 hours (~3 weeks part-time)
```

---

## How to Start Phase 3

### Step 1: Install Dependencies

```bash
pip install ultralytics>=8.0.0
pip install opencv-contrib-python  # For interactive zone selection
```

### Step 2: Create Person Detection Module

```bash
touch diveanalyzer/detection/person.py
# Copy structure from motion.py as template
```

### Step 3: Implement Basic Detection

```python
# Start with simplest function
def detect_person_frames(video_path: str, sample_fps: float = 5.0):
    """Detect frames with person present."""
    # Implementation
```

### Step 4: Test on Sample

```bash
python3 -c "
from diveanalyzer.detection.person import detect_person_frames
results = detect_person_frames('IMG_6496.MOV')
print(f'Found person in {len(results)} frames')
"
```

### Step 5: Integrate Gradually

1. Person frames → Timeline
2. Zone calibration → Zone tracking
3. Fusion → CLI integration
4. Optimization → Real-world testing

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **YOLO model (40MB)** | Large download | Lazy load, cache locally |
| **Person privacy** | Recording people | Document opt-in, add warning |
| **Zone calibration** | Manual per-video | Auto-detect heuristic fallback |
| **Multiple divers** | Can't distinguish | ANY person leaving = valid signal |
| **No person visible** | Can't validate | Graceful fallback to audio+motion |
| **Performance regression** | Slower than Phase 2 | Use proxy, GPU acceleration |
| **Model accuracy** | Different lighting | Test diverse videos, adjust threshold |

---

## After Phase 3

### Phase 3 Complete = Production Ready

```
✓ Audio detection: 0.3s, 0.82 confidence
✓ Motion validation: 13.3s, +0.15 boost
✓ Person validation: 3-5s, +0.10 boost
✓ Final confidence: 0.96 average
✓ False positives: 1-3%
✓ Total: ~6-7 minutes per session

Ready for:
- Batch processing
- Automated pipelines
- Cloud deployment
- Production use
```

### Phase 4 Possibilities

- ML-based confidence calibration
- Web UI dashboard
- Mobile app (iOS)
- Advanced features (splash intensity, diver tracking)

---

## Key Insight

**Phase 3 = Confidence, not Detection Count**

All three phases detect roughly the same dives (100% sensitivity). But:

- **Phase 1**: "I found 30 potential dives" (confidence: 0.82)
- **Phase 2**: "I'm 92% confident these 30 are real dives" (motion validates)
- **Phase 3**: "I'm 96% confident these 30 are real dives" (person validates)

This difference is critical for production systems that don't need human review!

---

**Status**: Ready to implement - all planning complete! 🚀
