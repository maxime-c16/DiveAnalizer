# Phase 1 Quick Test Guide

Test DiveAnalyzer v2.0 on your actual diving videos in 10 minutes.

---

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Navigate to project
cd DiveAnalyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install CLI
pip install -e .
```

### 2. Verify Installation

```bash
python scripts/test_installation.py
```

Expected output:
```
✓ NumPy
✓ librosa
✓ scipy
✓ Click
✓ tqdm
✓ FFmpeg
✓ ffprobe
✓ diveanalyzer modules
✓ diveanalyzer command available

✅ All tests passed!
```

---

## Test Strategy

### Phase 1: Test on 8-Minute Session Video

**Why first?**
- Contains multiple dives in sequence
- Will validate detection on real session
- Can tune parameters before testing clips

### Phase 2: Test on Individual 10-Second Clips

**Why second?**
- Verify extraction quality on individual dives
- Baseline accuracy testing
- Edge case handling

---

## Step 1: Test on 8-Minute Session Video (3 minutes)

### Prepare Video

```bash
# Copy your 8-minute session video to project folder
cp /path/to/your/session_8min.mp4 ./test_video.mp4

# Verify it's readable
ls -lh test_video.mp4
```

### Run Detection (Dry Run - No Extraction)

```bash
# First, just detect without extracting
diveanalyzer detect test_video.mp4 --verbose
```

**Expected output:**
```
🔍 Detecting dives in: test_video.mp4
Extracting audio...
✓ Audio extracted to: /tmp/...

Detecting splashes (threshold: -25dB)...

Found 3 potential splashes:

    1. Time:    5.23s  |  Amplitude: -12.1dB  |  Confidence:  69.8%
    2. Time:   18.45s  |  Amplitude: -15.3dB  |  Confidence:  63.2%
    3. Time:   32.67s  |  Amplitude: -14.8dB  |  Confidence:  64.0%

Total: 3 splashes detected
```

### Analyze Results

Count detected dives in output:
- **Too few?** Lower threshold: `--threshold -30`
- **Too many?** Raise threshold: `--threshold -20`

### Example: Tuning Detection

```bash
# Too few detected? Try lower threshold
diveanalyzer detect test_video.mp4 --threshold -30

# Too many false positives? Try higher threshold
diveanalyzer detect test_video.mp4 --threshold -20

# Find sweet spot (usually -22 to -28)
diveanalyzer detect test_video.mp4 --threshold -24
```

### Once Tuned, Run Full Process

```bash
# Create output directory
mkdir -p test_output

# Process with tuned parameters
diveanalyzer process test_video.mp4 \
  --threshold -25 \
  --confidence 0.5 \
  -o test_output \
  -v

# Check results
ls -lh test_output/
```

**Expected output:**
```
🎬 DiveAnalyzer v2.0
📹 Input: test_video.mp4
📁 Output: test_output

⏱️  Video duration: 0:08:00
🔊 Extracting audio track...
✓ Audio extracted to: /tmp/...

🌊 Detecting splashes...
✓ Found 3 potential splashes

🔗 Fusing detection signals...
✓ Created 3 dive events
✓ Final dive count: 3

✂️  Extracting 3 dive clips...
✓ Successfully extracted 3/3 clips

📊 Summary:
  Total dives: 3
  Extracted: 3
  Output folder: test_output
  ✓ dive_001.mp4 (125.3MB)
  ✓ dive_002.mp4 (128.1MB)
  ✓ dive_003.mp4 (126.8MB)

✅ Done!
```

### Verify Extracted Clips

```bash
# Check file sizes and metadata
ffprobe test_output/dive_001.mp4

# Play in VLC or QuickTime to verify quality
open test_output/dive_001.mp4
```

---

## Step 2: Test on Individual 10-Second Clips (2 minutes)

### Prepare Test Clips

```bash
# Copy your 10-second clips
cp /path/to/clip1.mp4 test_clips/
cp /path/to/clip2.mp4 test_clips/
cp /path/to/clip3.mp4 test_clips/

mkdir -p test_clips_output
```

### Test Each Clip

```bash
# Process all clips at once
for clip in test_clips/*.mp4; do
  echo "Processing: $clip"
  diveanalyzer process "$clip" -o test_clips_output -v
done
```

Or process individually:

```bash
# Test first clip with verbose output
diveanalyzer process test_clips/clip1.mp4 \
  --threshold -25 \
  -o test_clips_output \
  -v
```

### Verify Results

```bash
# Check extracted clips
ls -lh test_clips_output/

# Verify audio is preserved
ffprobe -v quiet -show_format test_clips_output/dive_001.mp4 | grep audio
```

---

## Advanced: Batch Testing Script

Create automated test script:

```bash
# Create test script
cat > test_batch.sh << 'EOF'
#!/bin/bash

echo "🧪 DiveAnalyzer Phase 1 Batch Test"
echo "=================================="

TEST_DIR="./test_videos"
OUTPUT_DIR="./test_results"

# Create directories
mkdir -p "$TEST_DIR" "$OUTPUT_DIR"

# Process all videos
for video in "$TEST_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        echo ""
        echo "📹 Testing: $(basename "$video")"

        # Get video duration
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video")
        echo "   Duration: ${duration}s"

        # Run detection
        echo "   Detecting dives..."
        diveanalyzer process "$video" \
            --threshold -25 \
            --confidence 0.5 \
            -o "$OUTPUT_DIR/$(basename "$video" .mp4)" \
            -v

        echo "   ✓ Complete"
    fi
done

echo ""
echo "📊 Results in: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name "*.mp4" | wc -l
echo "total clips extracted"
EOF

chmod +x test_batch.sh
./test_batch.sh
```

---

## Debugging: Audio Analysis

If detection isn't working, analyze the audio directly:

### 1. Extract and Save Audio

```bash
# Extract audio to file
python3 << 'EOF'
from diveanalyzer.detection.audio import extract_audio, get_audio_properties
import os

video = "test_video.mp4"
audio = extract_audio(video)
props = get_audio_properties(audio)

print(f"Audio extracted to: {audio}")
print(f"Duration: {props['duration']:.1f}s")
print(f"Sample rate: {props['sample_rate']} Hz")
print(f"Channels: {props['channels']}")
EOF
```

### 2. Analyze Audio Peaks

```bash
# Analyze audio with different thresholds
diveanalyzer analyze-audio /tmp/audio_*.wav --threshold -30
diveanalyzer analyze-audio /tmp/audio_*.wav --threshold -25
diveanalyzer analyze-audio /tmp/audio_*.wav --threshold -20
```

### 3. Check Audio Quality

```bash
# Use ffmpeg to check audio levels
ffmpeg -i test_video.mp4 -af "volumedetect" -f null - 2>&1 | grep -E "mean_volume|max_volume"
```

If audio levels are very low:
- Increase microphone gain when recording
- Use `--threshold -30` or lower

---

## Parameter Tuning Guide

### Audio Threshold (`--threshold`)

**Range:** -40 to 0 dB

| Threshold | Effect |
|-----------|--------|
| -30 | Very sensitive, may catch false positives |
| -25 | Default, balanced |
| -20 | Conservative, may miss quiet dives |

```bash
# Try thresholds in range
for t in -30 -28 -26 -24 -22 -20; do
  echo "Testing threshold: $t"
  diveanalyzer detect test_video.mp4 --threshold $t | grep "Found"
done
```

### Confidence Filter (`--confidence`)

**Range:** 0.0 to 1.0

| Confidence | Effect |
|------------|--------|
| 0.3 | Include all detections (may have false positives) |
| 0.5 | Default, balanced |
| 0.7 | Conservative, only high-confidence dives |

```bash
# Compare different confidence levels
diveanalyzer process test_video.mp4 --confidence 0.3 -o out_low
diveanalyzer process test_video.mp4 --confidence 0.5 -o out_mid
diveanalyzer process test_video.mp4 --confidence 0.7 -o out_high
```

---

## Quick Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] FFmpeg installed (`brew install ffmpeg` or `apt install ffmpeg`)
- [ ] Installation test passes (`python scripts/test_installation.py`)
- [ ] 8-minute video detects dives (`diveanalyzer detect test_video.mp4`)
- [ ] Clips extract successfully (`diveanalyzer process test_video.mp4 -o output`)
- [ ] Audio is preserved in output clips
- [ ] Output folder contains `.mp4` files
- [ ] Extracted clips are playable

---

## Expected Results

### For 8-Minute Session Video

```
Input: 8-minute session (480MB 4K video)
Processing time: ~20 seconds
Output: 8-12 individual dive clips
Storage: 1.2 GB of extracted clips
```

### For 10-Second Clips

```
Input: Single 10-second clip (60MB)
Processing time: ~5 seconds
Output: 1 extracted dive clip
Storage: 60MB (preserved from original)
```

---

## Common Issues & Fixes

### Issue: "No dives detected"

**Solution 1: Check audio quality**
```bash
diveanalyzer analyze-audio extracted_audio.wav
# Should show multiple peaks > -25dB
```

**Solution 2: Lower threshold**
```bash
diveanalyzer detect test_video.mp4 --threshold -30
```

**Solution 3: Check video has audio**
```bash
ffprobe test_video.mp4 | grep Audio
```

---

### Issue: "FFmpeg not found"

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg libsndfile1-dev

# Verify installation
ffmpeg -version
ffprobe -version
```

---

### Issue: "Extracted clips have no audio"

```bash
# Audio is included by default, verify:
ffprobe test_output/dive_001.mp4 | grep Audio

# If missing, re-run without --no-audio flag:
diveanalyzer process test_video.mp4 --no-audio  # This REMOVES audio
diveanalyzer process test_video.mp4              # This KEEPS audio (default)
```

---

### Issue: "False positives (non-dive sounds detected)"

```bash
# Raise threshold and confidence
diveanalyzer process test_video.mp4 \
  --threshold -20 \
  --confidence 0.7 \
  -o output
```

---

### Issue: "Memory error or slow processing"

Phase 1 uses minimal memory (~200MB), but if issues occur:

```bash
# Process in smaller chunks manually
# Extract 2-minute chunks and process separately
ffmpeg -i test_video.mp4 -ss 0 -t 120 chunk_1.mp4
ffmpeg -i test_video.mp4 -ss 120 -t 120 chunk_2.mp4

diveanalyzer process chunk_1.mp4 -o output
diveanalyzer process chunk_2.mp4 -o output
```

---

## Performance Benchmarks

### On Your Hardware (macOS)

Expected processing time:

| Video Length | Processing Time | Expected Dives |
|--------------|-----------------|----------------|
| 8 minutes | ~20 seconds | 6-12 |
| 10 seconds | ~5 seconds | 1 |
| 1 hour | ~3 minutes | 60-120 |

### Storage Usage

| Video Type | Input Size | Audio Extract | Output Clips |
|------------|------------|---------------|-------------|
| 8-min 4K | 480MB | 5MB | 1.2GB |
| 10-sec HD | 60MB | 1MB | 60MB |

---

## Next Steps After Testing

### If Detection Works Well
✅ Move to Phase 2 (motion validation + proxy workflow)

### If Detection Needs Tuning
1. Create config file with optimal thresholds
2. Document best parameters for your camera setup
3. Build config system (see ARCHITECTURE_PLAN.md)

### If You Want to Improve Accuracy
1. Phase 2: Add motion detection (PySceneDetect)
2. Phase 3: Add person detection (YOLO-nano)
3. Phase 4: Train custom model on your videos

---

## Python Test Script

```python
# test_phase1.py
#!/usr/bin/env python3

from pathlib import Path
from diveanalyzer import (
    extract_audio,
    detect_audio_peaks,
    fuse_signals_audio_only,
    extract_multiple_dives,
)

def test_video(video_path, output_dir="./test_output"):
    """Test Phase 1 on a single video."""

    print(f"\n🧪 Testing: {video_path}")
    print("=" * 60)

    # 1. Extract audio
    print("1️⃣  Extracting audio...")
    audio_path = extract_audio(video_path)
    print(f"   ✓ Audio extracted: {audio_path}")

    # 2. Detect peaks
    print("2️⃣  Detecting splash peaks...")
    peaks = detect_audio_peaks(audio_path, threshold_db=-25.0)
    print(f"   ✓ Found {len(peaks)} potential splashes")
    for i, (time, amp) in enumerate(peaks[:3], 1):
        print(f"      {i}. {time:6.2f}s @ {amp:6.1f}dB")

    # 3. Fuse signals
    print("3️⃣  Creating dive events...")
    dives = fuse_signals_audio_only(peaks)
    print(f"   ✓ Created {len(dives)} dive events")
    for dive in dives:
        print(f"      Dive #{dive.dive_number}: {dive.splash_time:6.2f}s (confidence {dive.confidence:.1%})")

    # 4. Extract clips
    print("4️⃣  Extracting clips...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = extract_multiple_dives(video_path, dives, output_dir, verbose=True)

    success_count = sum(1 for s, _, _ in results.values() if s)
    print(f"   ✓ Extracted {success_count}/{len(dives)} clips")

    print("\n✅ Test complete!")
    print(f"📁 Results in: {output_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_phase1.py <video.mp4> [output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "./test_output"

    test_video(video, output)
```

### Run the test script:

```bash
python test_phase1.py test_video.mp4
python test_phase1.py test_clips/clip1.mp4 test_output/clip1
```

---

## Summary

**Total testing time: ~10 minutes**

1. **Setup** (5 min) - Install dependencies
2. **Test on 8-min video** (3 min) - Verify detection works
3. **Test on 10-sec clips** (2 min) - Verify extraction quality

**Result**: Fully functional Phase 1 with your actual diving videos! 🎉

Next: Move to Phase 2 if you want motion validation and proxy workflow.

See `ARCHITECTURE_PLAN.md` for Phase 2 details.
