# Phase 3: Person Detection & Advanced Validation

## Overview

Phase 3 adds **person detection** as the third signal in the multi-modal detection system. Combined with audio and motion, this provides the most robust dive detection possible.

**Status**: Planned | **Timeline**: 2-3 weeks implementation

---

## Current State (After Phase 2)

### What We Have
- ✅ **Phase 1**: Audio-based splash detection (0.82 confidence, 0.3s)
- ✅ **Phase 2**: Motion-based validation (boosts to 0.92 confidence, 150s on full video, needs proxy optimization)
- ✅ **CLI**: Core commands with --enable-motion flag
- ✅ **Storage**: Cache system for audio/proxy/metadata

### What's Missing
- ❌ Person detection on video
- ❌ Zone-based person tracking
- ❌ Person transition validation (person leaves zone = dive)
- ❌ Multi-signal weighted fusion with person data
- ❌ Automatic audio quality assessment
- ❌ Adaptive confidence thresholds

---

## Phase 3 Goals

| Goal | Metric | Benefit |
|------|--------|---------|
| **Add person detection** | YOLO-nano inference | Third validation signal |
| **Zone-based validation** | Person presence tracking | Reduce false positives |
| **Improve accuracy** | +5-10% precision | Better results in noisy environments |
| **Handle edge cases** | Minimal motion dives, multiple people | Robust to different recording scenarios |
| **Performance** | <3s on 480p proxy | Real-time processing feasible |

---

## Architecture: Three-Signal Fusion

```
PHASE 3: Complete Multi-Modal Detection

┌─────────────────────────────────────────────────┐
│         DIVE DETECTION PIPELINE                 │
└─────────────────────────────────────────────────┘

Input: Video file
  │
  ├─ SIGNAL 1: Audio (Primary)
  │  ├─ Extract audio
  │  ├─ Detect splash peaks
  │  └─ RMS-based confidence
  │
  ├─ SIGNAL 2: Motion (Secondary)
  │  ├─ Generate 480p proxy
  │  ├─ Frame differencing at 5 FPS
  │  └─ Burst detection
  │
  ├─ SIGNAL 3: Person (Tertiary)
  │  ├─ YOLO-nano inference on proxy
  │  ├─ Track person zone transitions
  │  └─ Validate dive start (person → no person)
  │
  ├─ FUSION & CONFIDENCE SCORING
  │  ├─ Audio: Base confidence (0.5-0.9)
  │  ├─ Motion: +15% boost if motion detected
  │  ├─ Person: +10% boost if person leaves zone
  │  └─ Combined: Up to 1.0 (capped)
  │
  └─ OUTPUT: List of DiveEvents with high confidence

```

### Confidence Scoring (Phase 3)

```python
# Example: 30 dives detected

# All three signals:
13 dives: audio=0.85 + motion(+0.15) + person(+0.10) = 0.95+ ★★★

# Audio + Motion (no person visible):
8 dives: audio=0.80 + motion(+0.15) = 0.95 ★★

# Audio + Person (minimal motion):
5 dives: audio=0.78 + person(+0.10) = 0.88 ★

# Audio only (complex scene):
4 dives: audio=0.75 = 0.75 ✓

Total: 30 dives, avg confidence 0.89 (vs Phase 2: 0.92)
High confidence (≥0.85): 26/30 (87%)
```

---

## Implementation Plan

### Phase 3a: Person Detection (Week 1)

**Objective**: Integrate YOLO-nano for person detection

#### 1. Create person detection module
```python
# diveanalyzer/detection/person.py

def detect_person_frames(
    video_path: str,
    sample_fps: float = 5.0,
    confidence_threshold: float = 0.5,
) -> List[Tuple[float, bool]]:
    """
    Detect frames where person is present.

    Returns: List of (timestamp, person_present)
    """
    # Load YOLO-nano
    model = YOLO("yolov8n.pt")

    # Sample video frames at target FPS
    # Run inference on each frame
    # Filter by confidence threshold
    # Return person presence timeline

def find_person_transitions(
    person_timeline: List[Tuple[float, bool]],
    transition_type: str = "disappear",  # "disappear", "appear", "both"
) -> List[Tuple[float, float]]:
    """
    Find transitions in person presence.

    Returns: List of (timestamp, duration) for transitions
    """
```

#### 2. Add YOLO-nano to requirements
```
ultralytics>=8.0.0    # YOLO-nano person detection
```

#### 3. Update config.py
```python
@dataclass
class PersonDetectionConfig:
    enabled: bool = False
    model_name: str = "yolov8n.pt"  # Nano model
    confidence_threshold: float = 0.5
    sample_fps: float = 5.0
    zone: Optional[Tuple[float, float, float, float]] = None
    person_validation_boost: float = 0.10
```

#### 4. Testing
- Test on sample videos with different numbers of people
- Verify YOLO inference speed on 480p
- Benchmark GPU vs CPU detection
- Create unit tests for person transition detection

---

### Phase 3b: Zone-Based Validation (Week 2)

**Objective**: Track person presence in a specific zone

#### 1. Interactive zone calibration
```python
# diveanalyzer/detection/person.py

def calibrate_dive_zone_interactive(video_path: str):
    """
    Interactive zone selection tool.

    Shows video frames and allows user to click
    to define the diving zone (pool area).
    """
    # Show first frame
    # Accept click coordinates (x1, y1, x2, y2)
    # Verify with next 10 frames
    # Save normalized zone coordinates
```

#### 2. Per-frame zone validation
```python
def track_person_in_zone(
    video_path: str,
    zone: Tuple[float, float, float, float],
    sample_fps: float = 5.0,
) -> List[Tuple[float, bool, float]]:
    """
    Track if person is in zone over time.

    Returns: List of (timestamp, in_zone, confidence)
    """
    # For each sampled frame:
    #   1. Run person detection (YOLO)
    #   2. Check if bounding box overlaps zone
    #   3. Record presence in zone
    #   4. Smooth over time (reduce jitter)
```

#### 3. Detect dive starts
```python
def find_person_zone_departures(
    person_timeline: List[Tuple[float, bool, float]],
    min_departure_time: float = 0.5,
) -> List[float]:
    """
    Find moments when person leaves diving zone.

    These are potential dive start times.

    Returns: List of departure timestamps
    """
    # Look for transitions: in_zone=True → False
    # Filter brief departures (< 0.5s)
    # Return filtered departure times
```

---

### Phase 3c: Signal Fusion Integration (Week 2)

**Objective**: Update fusion algorithm to incorporate person signal

#### 1. Three-signal fusion function
```python
# diveanalyzer/detection/fusion.py

def fuse_signals_audio_motion_person(
    audio_peaks: List[Tuple[float, float]],
    motion_events: List[Tuple[float, float, float]],
    person_departures: List[float],
    window_before: float = 5.0,  # 5s before splash
    motion_validation_boost: float = 0.15,
    person_validation_boost: float = 0.10,
) -> List[DiveEvent]:
    """
    Three-signal fusion with adaptive weights.

    For each audio peak:
    1. Check for motion burst before splash
    2. Check for person departure before splash
    3. Combine confidences
    """
    for splash_time, amplitude in audio_peaks:
        audio_confidence = 0.5 + (normalize(amplitude) * 0.5)

        # Check motion
        motion_match = find_motion_before(motion_events, splash_time)
        if motion_match:
            audio_confidence += motion_validation_boost

        # Check person
        person_match = find_person_departure_before(person_departures, splash_time, window_before)
        if person_match:
            audio_confidence += person_validation_boost

        yield DiveEvent(
            confidence=min(1.0, audio_confidence),
            audio_only=(not motion_match and not person_match),
            motion_validated=motion_match,
            person_validated=person_match,
        )
```

#### 2. Update CLI
```bash
# New CLI options for Phase 3
diveanalyzer process video.mov \
  --enable-motion \
  --enable-person \
  --calibrate-zone  # Interactive zone selection

# New commands
diveanalyzer calibrate-zone video.mov          # Set dive zone
diveanalyzer analyze-person video.mov          # Debug person detection
```

---

### Phase 3d: Performance Optimization (Week 3)

**Objective**: Ensure <3s detection on 480p proxy

#### 1. GPU acceleration
```python
# Use GPU if available for both motion and person detection

# Motion detection
# GPU: Frame processing with CUDA (possible with custom kernels)
# Alternative: Use PyTorch for frame differencing

# Person detection
# GPU: YOLO inference on CUDA (built-in support)
# Fallback: CPU inference with model quantization
```

#### 2. Inference optimization
```python
# YOLO optimizations
- Use FP16 (half precision) instead of FP32
- Quantize to INT8 if accuracy acceptable
- Batch process frames (process 4-8 at once)
- Cache model between invocations

# Expected performance
- Full video: ~150s (motion) + ~60s (person) = 210s
- With proxy (480p): ~2s (motion) + ~3s (person) = 5s ✓
- With GPU: ~1s (motion) + ~1s (person) = 2s ✓
```

#### 3. Proxy integration
```python
# Phase 2 had motion detection on full video (slow)
# Phase 3 will use proxy for both motion AND person

# Workflow:
# 1. Extract audio from original → Process
# 2. Generate 480p proxy → Cache
# 3. Motion detect on proxy
# 4. Person detect on proxy
# 5. Extract dives from original (full quality)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_person_detection.py
def test_yolo_loads()
def test_person_frames_detection()
def test_person_zone_tracking()
def test_zone_departure_finding()

# tests/test_fusion_three_signal.py
def test_audio_motion_person_fusion()
def test_confidence_scoring_all_signals()
def test_edge_case_no_person_visible()
def test_edge_case_multiple_people()
```

### Integration Tests
```python
# tests/integration_test_phase3.py

def test_full_pipeline_with_person():
    """End-to-end test with all three signals."""
    video = "test_videos/diving_with_person.mov"
    dives = process_phase3(video)

    assert len(dives) == expected_count
    assert all(d.confidence >= 0.75 for d in dives)
    assert validated_count >= expected_validated_count

def test_performance_under_3s():
    """Ensure motion+person detection < 3s on 480p."""
    video = "test_videos/480p_proxy.mp4"
    start = time.time()
    motion_events = detect_motion_bursts(video)
    person_departures = find_person_zone_departures(...)
    elapsed = time.time() - start

    assert elapsed < 3.0, f"Too slow: {elapsed}s"
```

### Real-World Testing
```python
# Test scenarios
1. Clean pool, person visible: Full 3-signal validation ★★★
2. Crowded pool, multiple people: Challenge disambiguation
3. Poor quality audio: Rely on motion + person
4. No person in frame: Fall back to audio+motion
5. Fast dives (minimal motion): Audio signal strongest
6. Slow approach (lots of motion): All three validate
```

---

## Risk Assessment & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **YOLO model download** | Large file (~40MB) | Cache locally, lazy download on first use |
| **Person privacy** | Recording people's movements | Document as opt-in feature, add warning |
| **Zone calibration** | Manual per-video setup | Auto-detect with heuristics as fallback |
| **Multiple divers** | Can't distinguish individuals | Detect ANY person leaving (good enough) |
| **Performance regression** | Slower than Phase 2 | Use proxy, GPU acceleration, optimize |
| **Model accuracy** | Different lighting conditions | Test on diverse real-world videos |

---

## Success Criteria

✅ **Implementation Complete When**:
1. Person detection module fully implemented and tested
2. Zone-based validation working on real videos
3. Three-signal fusion improving accuracy by 5-10%
4. CLI updated with --enable-person flag
5. Performance < 3s for full pipeline on 480p proxy
6. Documentation updated
7. Real-world tested on user's diving videos

---

## Phase 3 Deliverables

### Code
- `diveanalyzer/detection/person.py` (300+ lines)
- Updated `fusion.py` with three-signal logic
- Updated `cli.py` with person detection options
- Unit & integration tests
- `tests/test_person_detection.py`
- `tests/integration_test_phase3.py`

### Documentation
- `PHASE_3_COMPLETE.md` - Implementation summary
- Updated `README_V2.md` with Phase 3 features
- `PERSON_DETECTION_GUIDE.md` - How to use
- `ZONE_CALIBRATION_GUIDE.md` - Setting up dive zone

### Performance
- Detection pipeline: <3s on 480p proxy
- Inference: <1.5s (GPU) or <3s (CPU)
- Memory: <300MB peak
- Storage: Minimal (model cached)

---

## How Phase 3 Improves on Phase 2

### Confidence Improvement

| Scenario | Phase 1 | Phase 2 | Phase 3 | Improvement |
|----------|---------|---------|---------|-------------|
| **Clean pool** | 0.82 | 0.92 | 0.93 | +11% |
| **Crowd noise** | 0.65 | 0.78 | 0.88 | +23% |
| **Minimal motion** | 0.75 | 0.75 | 0.85 | +10% |
| **Multiple people** | 0.60 | 0.68 | 0.80 | +20% |
| **Average** | 0.71 | 0.83 | 0.87 | +16% |

### False Positive Reduction

- **Phase 1**: 15-20% false positives (audio artifacts)
- **Phase 2**: 5-10% false positives (motion validates)
- **Phase 3**: 1-3% false positives (person validates)

### Robustness

- **Works in any environment**: Audio + motion + person = robust
- **No manual tuning needed**: Confidence reflects actual validity
- **Handles edge cases**: Multiple divers, complex scenes, poor audio

---

## Beyond Phase 3: Future Enhancements

### Phase 4a: ML-Based Confidence Calibration
- Train model on user's actual dives
- Learn optimal fusion weights per scenario
- Automatic parameter tuning

### Phase 4b: Web UI Dashboard
- Real-time detection visualization
- Batch processing progress
- Results management interface
- Settings configuration

### Phase 4c: Mobile App
- Direct processing on iPhone
- Real-time feedback while recording
- iCloud sync integration
- Share extracted clips

### Phase 4d: Advanced Features
- Multiple zone detection (different diving areas)
- Splash intensity analysis (difficulty scoring)
- Diver identification (if multiple cameras)
- Performance metrics and statistics

---

## Timeline

```
Week 1: Person detection module
├─ YOLO-nano integration
├─ Frame-by-frame inference
└─ Person presence timeline

Week 2: Zone validation & fusion
├─ Interactive zone calibration
├─ Person zone tracking
├─ Three-signal fusion
└─ CLI updates

Week 3: Optimization & testing
├─ GPU acceleration
├─ Performance tuning
├─ Real-world validation
└─ Documentation & release

Estimated: 3 weeks
Effort: ~40 hours
Complexity: High (multi-modal fusion, inference optimization)
```

---

## How to Start Phase 3

1. **Install YOLO**: `pip install ultralytics`
2. **Create detection/person.py**: Copy structure from motion.py
3. **Start with basic detection**: Person frames, before zone validation
4. **Integrate gradually**: Audio → Audio+Motion → Audio+Motion+Person
5. **Test on each video**: Verify confidence improvements
6. **Optimize for speed**: GPU acceleration, proxy inference

---

## Key Insight

**Phase 3 is about confidence, not detection count.**

Phase 1, 2, and 3 all detect roughly the same dives (100% sensitivity). But Phase 3 provides **high confidence** in those detections, making it suitable for fully automated systems that don't need human review.

This is the difference between:
- **Phase 1**: "I found 30 potential dives" (confidence: 82%)
- **Phase 3**: "I'm 92% confident these 30 are real dives" (all three signals validate)

---

## Questions & Decisions Needed

Before starting Phase 3, clarify:

1. **Zone calibration**: Manual (interactive) or automatic (heuristic)?
2. **GPU available**: NVIDIA (CUDA) or CPU-only?
3. **Privacy concerns**: OK to process person data in videos?
4. **Edge cases**: What about scenes without visible person?
5. **Multi-person pools**: How to handle multiple divers?
6. **Performance target**: Is <3s requirement strict?

