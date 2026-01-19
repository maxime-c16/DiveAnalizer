# Test Scripts for DiveAnalyzer

Utility scripts for testing and validating Phase 1.

---

## Available Scripts

### 1. `test_installation.py` - Verify Setup

Checks that all dependencies and tools are installed.

```bash
python scripts/test_installation.py
```

**What it checks:**
- Python modules (numpy, librosa, scipy, etc.)
- External tools (FFmpeg, ffprobe)
- DiveAnalyzer modules
- CLI command

**Expected output:**
```
✓ NumPy
✓ librosa
✓ FFmpeg
...
✅ All tests passed!
```

**Use case**: Run first to verify installation is complete.

---

### 2. `quick_test.py` - Test Single Video

Test detection and extraction on one video file.

```bash
# Basic usage
python scripts/quick_test.py test_video.mp4

# With options
python scripts/quick_test.py session.mp4 \
  --threshold -24 \
  --confidence 0.6 \
  --output ./my_output \
  --verbose
```

**Options:**
- `-o, --output`: Output directory (default: `./test_output`)
- `-t, --threshold`: Audio threshold in dB (default: `-25`)
- `-c, --confidence`: Minimum confidence 0-1 (default: `0.5`)
- `-v, --verbose`: Detailed output

**Example output:**
```
🧪 DiveAnalyzer Phase 1 Quick Test
================================================================================
  Testing: session.mp4
================================================================================

📹 Video: session.mp4
⏱️  Duration: 00:08:00

[1/4] 🔊 Extracting audio...
      ✓ Audio extracted (5.2s)

[2/4] 🌊 Detecting splashes (threshold: -25dB)...
      ✓ Detection complete (2.1s)
      Found 8 potential splashes

      Splash details:
        1.    5.23s @ -12.1dB (confidence  69.8%)
        2.   18.45s @ -15.3dB (confidence  63.2%)
        ...

[3/4] 🔗 Creating dive events...
      ✓ Created 8 dive events
      Filtered by confidence: 8 → 8

[4/4] ✂️  Extracting 8 dive clips...
      ✓ Successfully extracted 8/8 clips

      Extracted files:
        ✓ dive_001.mp4 (125.3MB)
        ✓ dive_002.mp4 (128.1MB)
        ...

📊 Test Results Summary
================================================================================

✅ Test PASSED

📹 Video: session.mp4
⏱️  Processing time: 18.5s
🌊 Splashes detected: 8
✂️  Clips extracted: 8
📁 Output: ./test_output
💾 Total output size: 1003.2MB
```

**Use case**: Quickly test a single video and verify detection parameters.

---

### 3. `batch_test.py` - Test Multiple Videos

Test detection on multiple videos and generate a report.

```bash
# Test all videos in a folder
python scripts/batch_test.py --folder ./test_videos

# Test specific pattern
python scripts/batch_test.py \
  --folder ./clips \
  --pattern "dive_*.mp4" \
  --threshold -24 \
  --confidence 0.7

# Save results to JSON
python scripts/batch_test.py \
  --folder ./videos \
  --report results.json \
  --output ./extracted_clips
```

**Options:**
- `--folder`: Folder with videos (can use multiple times)
- `--pattern`: File pattern to match (default: `*.mp4`)
- `-o, --output`: Output directory (default: `./batch_output`)
- `-t, --threshold`: Audio threshold in dB (default: `-25`)
- `-c, --confidence`: Minimum confidence 0-1 (default: `0.5`)
- `-r, --report`: Save results to JSON file

**Example output:**
```
📁 Found 5 video(s)

[1/5] Testing: session_1.mp4
   ✓ 6 detected, 6 extracted (18.2s)

[2/5] Testing: session_2.mp4
   ✓ 8 detected, 8 extracted (20.1s)

[3/5] Testing: clip_001.mp4
   ✓ 1 detected, 1 extracted (5.3s)

[4/5] Testing: clip_002.mp4
   ✓ 1 detected, 1 extracted (5.1s)

[5/5] Testing: clip_003.mp4
   ✓ 1 detected, 1 extracted (5.2s)

📊 Batch Test Summary
================================================================================

Videos tested: 5
  ✓ Successful: 5
  ✗ Failed: 0

🌊 Total splashes detected: 22
✂️  Total dives extracted: 22
⏱️  Total processing time: 53.9s
   Average per video: 10.8s
💾 Total output size: 2203.4MB

📝 Details:
   ✓ session_1.mp4: 6 detected, 6 extracted
   ✓ session_2.mp4: 8 detected, 8 extracted
   ✓ clip_001.mp4: 1 detected, 1 extracted
   ✓ clip_002.mp4: 1 detected, 1 extracted
   ✓ clip_003.mp4: 1 detected, 1 extracted

📄 Results saved to: results.json
```

**JSON Report Format:**
```json
{
  "timestamp": "2026-01-19T16:30:45.123456",
  "parameters": {
    "threshold": -25.0,
    "confidence": 0.5,
    "pattern": "*.mp4"
  },
  "summary": {
    "total_videos": 5,
    "successful": 5,
    "failed": 0,
    "total_dives_detected": 22,
    "total_dives_extracted": 22,
    "total_processing_time": 53.9,
    "total_output_size_mb": 2203.4
  },
  "videos": [
    {
      "filename": "session_1.mp4",
      "success": true,
      "dives_detected": 6,
      "dives_extracted": 6,
      "processing_time": 18.2,
      "output_size_mb": 440.2
    },
    ...
  ]
}
```

**Use case**: Test multiple videos at once, compare detection parameters, generate reports.

---

## Quick Testing Workflow

### Step 1: Verify Installation
```bash
python scripts/test_installation.py
```

### Step 2: Test on Single Video
```bash
# Find optimal threshold
for t in -30 -28 -26 -24 -22 -20; do
  echo "Testing threshold $t..."
  python scripts/quick_test.py test_video.mp4 --threshold $t
done
```

### Step 3: Test on All Videos
```bash
python scripts/batch_test.py \
  --folder ./test_videos \
  --threshold -25 \
  --confidence 0.5 \
  --report test_results.json
```

### Step 4: Analyze Results
```bash
cat test_results.json | python -m json.tool
```

---

## Parameter Tuning

### Find Optimal Threshold

```bash
# Try different thresholds
for threshold in -30 -28 -26 -24 -22 -20; do
  echo "=== Threshold: $threshold ==="
  python scripts/quick_test.py video.mp4 --threshold $threshold 2>&1 | grep -E "Found|extracted"
done
```

### Compare Confidence Levels

```bash
# Test different confidence thresholds
for confidence in 0.3 0.5 0.7 0.9; do
  python scripts/quick_test.py video.mp4 \
    --confidence $confidence \
    --output "./output_conf_${confidence}"
done

# Compare results
ls -1 output_conf_*/
```

---

## Troubleshooting

### Script not found
```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run explicitly
python scripts/quick_test.py video.mp4
```

### "No module named diveanalyzer"
```bash
# Install package
pip install -e .

# Or add to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

---

## Examples

### Test Your 8-Minute Session

```bash
python scripts/quick_test.py ~/Videos/diving_session.mp4 \
  --output ./session_output \
  --verbose
```

### Test All 10-Second Clips

```bash
python scripts/batch_test.py \
  --folder ~/Videos/diving_clips \
  --pattern "*.mp4" \
  --output ./clips_output \
  --report clips_results.json
```

### Find Best Threshold for Your Camera

```bash
python scripts/quick_test.py test.mp4 --threshold -25
python scripts/quick_test.py test.mp4 --threshold -24
python scripts/quick_test.py test.mp4 --threshold -26
# Compare results and pick the one with most accurate detections
```

### Generate Report for All Videos

```bash
python scripts/batch_test.py \
  --folder ~/Videos/Session1 \
  --folder ~/Videos/Session2 \
  --report summary.json \
  --output ./all_extracted
```

---

## Performance Tips

### Speed Up Testing

```bash
# Batch test is faster for multiple videos
# ✓ Good: Test 5 videos at once
python scripts/batch_test.py --folder ./videos

# ✗ Avoid: Testing each video individually
for video in videos/*; do
  python scripts/quick_test.py "$video"
done
```

### Reduce Storage During Testing

```bash
# Use lower confidence to skip low-confidence dives
python scripts/quick_test.py video.mp4 --confidence 0.8

# Only extract high-quality detections
```

### Parallel Testing

```bash
# Test multiple videos in parallel (requires GNU parallel)
find ./videos -name "*.mp4" | \
  parallel python scripts/quick_test.py {} --output ./output_parallel
```

---

## Scripts Reference

| Script | Purpose | Speed | Output |
|--------|---------|-------|--------|
| `test_installation.py` | Verify setup | ~1s | Pass/fail |
| `quick_test.py` | Test single video | ~20s for 8min | Extracted clips |
| `batch_test.py` | Test multiple videos | ~15s per video | JSON report + clips |

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test DiveAnalyzer

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: |
          sudo apt install ffmpeg libsndfile1-dev
          pip install -r requirements.txt
          pip install -e .
      - run: python scripts/test_installation.py
      - run: python scripts/batch_test.py --folder test_videos
```

---

## More Help

- **Usage Guide**: See `README_V2.md`
- **Technical Design**: See `ARCHITECTURE_PLAN.md`
- **Quick Start**: See `QUICK_TEST_GUIDE.md`
- **Phase 1 Details**: See `PHASE_1_COMPLETE.md`

---

Last updated: January 2026
