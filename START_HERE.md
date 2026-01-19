# 🚀 START HERE - Phase 1 Quick Start

Get DiveAnalyzer v2.0 running in **10 minutes** on your diving videos.

---

## 1️⃣ Install (5 minutes)

```bash
cd DiveAnalyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install CLI
pip install -e .
```

### Verify Installation

```bash
python scripts/test_installation.py
```

Expected output:
```
✅ All tests passed! Installation is ready.
```

---

## 2️⃣ Test on Your Videos (5 minutes)

### Option A: Single Video (Recommended First)

Copy one of your videos to the project folder:

```bash
# Copy your 8-minute session or a 10-second clip
cp ~/Videos/diving_session.mp4 ./test_video.mp4
```

Then test it:

```bash
# Quick test to see what happens
python scripts/quick_test.py test_video.mp4

# With verbose output to see details
python scripts/quick_test.py test_video.mp4 --verbose
```

### Option B: Test Multiple Videos

```bash
# Create a folder with your videos
mkdir test_videos
cp ~/Videos/session*.mp4 test_videos/
cp ~/Videos/clip*.mp4 test_videos/

# Run batch test
python scripts/batch_test.py --folder test_videos --report results.json
```

---

## 3️⃣ Tune Parameters (Optional)

Not detecting dives correctly? Adjust the threshold:

```bash
# Too few dives detected? Lower threshold:
python scripts/quick_test.py test_video.mp4 --threshold -30

# Too many false positives? Raise threshold:
python scripts/quick_test.py test_video.mp4 --threshold -20

# Find optimal (usually -22 to -28):
python scripts/quick_test.py test_video.mp4 --threshold -24
```

---

## 📊 What You'll See

### Example Output

```
🧪 DiveAnalyzer Phase 1 Quick Test
=========================================================================
  Testing: diving_session.mp4
=========================================================================

📹 Video: diving_session.mp4
⏱️  Duration: 00:08:00

[1/4] 🔊 Extracting audio...
      ✓ Audio extracted (5.2s)

[2/4] 🌊 Detecting splashes (threshold: -25dB)...
      ✓ Detection complete (2.1s)
      Found 8 potential splashes

[3/4] 🔗 Creating dive events...
      ✓ Created 8 dive events

[4/4] ✂️  Extracting 8 dive clips...
      ✓ Successfully extracted 8/8 clips

📊 Test Results Summary
=========================================================================
✅ Test PASSED

📹 Video: diving_session.mp4
⏱️  Processing time: 18.5s
🌊 Splashes detected: 8
✂️  Clips extracted: 8
📁 Output: ./test_output
💾 Total output size: 1003.2MB
```

Extracted clips: `./test_output/dive_001.mp4`, `./test_output/dive_002.mp4`, etc.

---

## 🎯 Common Commands

### Process a Video
```bash
diveanalyzer process video.mp4 -o ./dives
```

### Dry Run (Just Detect)
```bash
diveanalyzer detect video.mp4
```

### Analyze Audio Directly
```bash
diveanalyzer analyze-audio audio.wav
```

---

## 📁 Files You Need to Know

| File | Purpose |
|------|---------|
| `QUICK_TEST_GUIDE.md` | Detailed step-by-step testing guide |
| `scripts/quick_test.py` | Test single video |
| `scripts/batch_test.py` | Test multiple videos |
| `README_V2.md` | Full usage documentation |
| `ARCHITECTURE_PLAN.md` | Technical architecture (50KB) |

---

## ✅ Quick Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Installed FFmpeg (`brew install ffmpeg` or `apt install ffmpeg`)
- [ ] Installation test passed (`python scripts/test_installation.py`)
- [ ] Tested on at least one video (`python scripts/quick_test.py video.mp4`)
- [ ] Got extracted dive clips in output folder
- [ ] Checked that clips play and have audio

---

## 🚀 Next Steps

### If Everything Works ✅
- **Phase 2**: Motion detection + proxy workflow (see ARCHITECTURE_PLAN.md)
- **Advanced**: Tune parameters for your specific camera setup

### If Something Doesn't Work ❌
1. Check error message
2. See troubleshooting section below
3. Check QUICK_TEST_GUIDE.md for detailed help

---

## 🐛 Troubleshooting

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### "No module named diveanalyzer"
```bash
# Install package
pip install -e .
```

### "No dives detected"
```bash
# Try lower threshold
python scripts/quick_test.py video.mp4 --threshold -30

# Check audio quality
diveanalyzer analyze-audio extracted_audio.wav
```

### "Extracted clips have no audio"
```bash
# Re-run without --no-audio flag
diveanalyzer process video.mp4  # Default keeps audio
```

---

## 📊 Performance Expectations

### For Your Videos

**8-minute session (4K):**
- Processing time: ~20 seconds
- Output: 8-12 individual dive clips
- Total output size: ~1.2GB

**10-second clip (HD):**
- Processing time: ~5 seconds
- Output: 1 individual dive clip
- Output size: ~60MB

**1-hour session (4K):**
- Processing time: ~3 minutes
- Output: 60-120 individual dive clips
- Total output size: ~12GB

---

## 💡 Tips

### 1. Start with Dry Run
```bash
# Just detect, don't extract
diveanalyzer detect video.mp4

# Find optimal threshold first
for t in -30 -28 -26 -24 -22 -20; do
  diveanalyzer detect video.mp4 --threshold $t
done
```

### 2. Test on Small Video First
- Start with a 10-second clip
- Then test on full 8-minute session
- Verify parameters before batch processing

### 3. Filter Low Confidence
```bash
# Only extract high-confidence dives
diveanalyzer process video.mp4 --confidence 0.7
```

### 4. Batch Process Multiple Videos
```bash
python scripts/batch_test.py --folder ./videos --report summary.json
```

---

## 📚 Documentation Map

```
START_HERE.md (you are here)
    ↓
QUICK_TEST_GUIDE.md (step-by-step testing)
    ↓
README_V2.md (full user guide)
    ↓
ARCHITECTURE_PLAN.md (technical details + Phase 2-4 roadmap)
    ↓
PHASE_1_COMPLETE.md (implementation details)
    ↓
scripts/README.md (testing scripts reference)
```

---

## 🎯 Success Criteria

You've successfully tested Phase 1 when:

1. ✅ Installation test passes
2. ✅ Detect command finds dives in your video
3. ✅ Process command extracts MP4 clips
4. ✅ Extracted clips have audio and can be played
5. ✅ Processing time is < 30 seconds for 8-min video

---

## 🎉 Ready to Go!

You now have a fully functional dive clip extraction system!

**Next?**
- Use it to process your diving sessions
- Tune parameters for your camera/audio quality
- Plan Phase 2 upgrade (motion validation + proxy workflow)

---

## 📞 Need Help?

1. **Installation issues**: See `scripts/README.md`
2. **Testing issues**: See `QUICK_TEST_GUIDE.md`
3. **Usage questions**: See `README_V2.md`
4. **Technical details**: See `ARCHITECTURE_PLAN.md`

---

## 🏃 TL;DR (Too Long; Didn't Read)

```bash
# 1. Install
pip install -r requirements.txt && pip install -e .

# 2. Verify
python scripts/test_installation.py

# 3. Test
python scripts/quick_test.py your_video.mp4

# 4. Check output
ls -lh test_output/
```

Done! 🎉

---

**DiveAnalyzer v2.0 Phase 1** - Ready to extract your diving videos!

Last updated: January 2026
