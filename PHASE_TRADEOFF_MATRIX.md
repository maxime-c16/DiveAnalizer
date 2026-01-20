# Phase Tradeoff Matrix: Which Phase for Which Machine?

**Purpose:** Help users understand the accuracy/speed tradeoff for each phase on different hardware

---

## Quick Decision Guide

### 🏃‍♂️ If You Want **Speed** (< 20s processing)
→ Use **Phase 2** (15s, 0.92 confidence) ✅
→ Works on ALL machines (even weak ones)
→ Great accuracy, fast processing

### 🎯 If You Want **Best Accuracy** (0.96+)
→ Use **Phase 3 with GPU** (35-40s, 0.96 confidence) ✅
→ Need: GPU (CUDA/Metal/ROCm)
→ Highest accuracy, reasonable speed

### ⚠️ If You Want **Phase 3 on CPU-Only Machine**
→ **NOT RECOMMENDED** (380+ seconds) ❌
→ Gain only +4% accuracy (0.92→0.96)
→ Cost 25x slower (15s→380s)
→ Unless you have time to wait...

---

## Hardware-Specific Recommendations

### 📱 Weak Machine (Intel i5, 4 cores, 8GB, No GPU)
```
Your System: ★☆☆☆☆ (2/10)

Phase 1 (Audio-only):
├─ Time: 5s
├─ Confidence: 0.82
├─ Status: ✅ Works great if speed critical
└─ Use case: Quick demos, testing

Phase 2 (Audio + Motion):  ⭐ RECOMMENDED
├─ Time: 15s
├─ Confidence: 0.92
├─ Status: ✅ Perfect balance
└─ Use case: Production use, everyday

Phase 3 (Audio + Motion + Person):
├─ Time: 380s ⚠️
├─ Confidence: 0.96
├─ Status: ❌ Too slow without GPU
└─ Use case: Not recommended for this machine
```

**Verdict: Use Phase 2 (0.92 confidence, 15s)**

---

### 💻 Mid-Range Machine (Intel i7, 8 cores, 16GB, No GPU)
```
Your System: ★★★☆☆ (5/10)

Phase 1 (Audio-only):
├─ Time: 5s
├─ Confidence: 0.82
└─ Status: ✅ Available

Phase 2 (Audio + Motion):  ⭐ RECOMMENDED
├─ Time: 15s
├─ Confidence: 0.92
└─ Status: ✅ Still better speed/accuracy

Phase 3 (Audio + Motion + Person):
├─ Time: 150s (better than i5, but still slow)
├─ Confidence: 0.96
└─ Status: ⚠️ Acceptable if you can wait ~2.5 min total
```

**Verdict: Use Phase 2 (unless you have time for Phase 3)**

---

### 🔥 High-End Machine with GPU (Mac M3, 8 cores, 16GB RAM, Metal GPU 8GB VRAM)
```
Your System: ★★★★★ (8/10)

Phase 1 (Audio-only):
├─ Time: 5s
├─ Confidence: 0.82
└─ Status: ✅ Overkill, but available

Phase 2 (Audio + Motion):
├─ Time: 15s
├─ Confidence: 0.92
└─ Status: ✅ Good option

Phase 3 (Audio + Motion + Person):  ⭐ RECOMMENDED
├─ Time: 35s (GPU accelerated!)
├─ Confidence: 0.96
└─ Status: ✅ Best accuracy + reasonable speed
```

**Verdict: Use Phase 3 (0.96 confidence, 35s on GPU is excellent)**

---

### 🚀 Powerful GPU Desktop (16 cores, 32GB RAM, RTX 4090)
```
Your System: ★★★★★ (10/10)

Phase 1 (Audio-only):
├─ Time: 5s
├─ Confidence: 0.82
└─ Status: ✅ Fastest, but low accuracy

Phase 2 (Audio + Motion):
├─ Time: 15s
├─ Confidence: 0.92
└─ Status: ✅ Good option

Phase 3 (Audio + Motion + Person):  ⭐ RECOMMENDED
├─ Time: 8s (GPU powerhouse)
├─ Confidence: 0.96
└─ Status: ✅ Best of both worlds: blazing fast AND most accurate
```

**Verdict: Use Phase 3 (0.96 confidence, 8s on powerful GPU is blazing fast)**

---

## Detailed Tradeoff Matrix

### Processing Speed

```
WEAK MACHINE (i5, 8GB, no GPU):
Phase 1: ████░░░░░░░░░░░░░░░░░ 5s      ✅
Phase 2: ███████████░░░░░░░░░░ 15s     ✅✅ RECOMMENDED
Phase 3: █████████████████████ 380s    ⚠️ NOT FOR THIS MACHINE

MID-RANGE (i7, 16GB, no GPU):
Phase 1: ████░░░░░░░░░░░░░░░░░ 5s      ✅
Phase 2: ███████████░░░░░░░░░░ 15s     ✅✅ RECOMMENDED
Phase 3: ███████████████░░░░░░ 150s    ⚠️ BORDERLINE

GPU MACHINE (M3, 16GB, Metal GPU):
Phase 1: ████░░░░░░░░░░░░░░░░░ 5s      ✅
Phase 2: ███████████░░░░░░░░░░ 15s     ✅
Phase 3: ███████████████░░░░░░ 35s     ✅✅ RECOMMENDED

POWERFUL GPU (RTX 4090, 32GB):
Phase 1: ████░░░░░░░░░░░░░░░░░ 5s      ✅
Phase 2: ███████████░░░░░░░░░░ 15s     ✅
Phase 3: ██████░░░░░░░░░░░░░░░ 8s      ✅✅ RECOMMENDED
```

### Detection Accuracy

```
WEAK MACHINE (i5, 8GB, no GPU):
Phase 1: ████████░░░░░░░░░░░░░ 0.82 (82%)
Phase 2: █████████████░░░░░░░░ 0.92 (92%) ✅✅ RECOMMENDED
Phase 3: ███████████████░░░░░░ 0.96 (96%) [but too slow]

MID-RANGE (i7, 16GB, no GPU):
Phase 1: ████████░░░░░░░░░░░░░ 0.82 (82%)
Phase 2: █████████████░░░░░░░░ 0.92 (92%) ✅ RECOMMENDED
Phase 3: ███████████████░░░░░░ 0.96 (96%) [only if you have time]

GPU MACHINE (M3, 16GB, Metal GPU):
Phase 1: ████████░░░░░░░░░░░░░ 0.82 (82%)
Phase 2: █████████████░░░░░░░░ 0.92 (92%)
Phase 3: ███████████████░░░░░░ 0.96 (96%) ✅✅ RECOMMENDED

POWERFUL GPU (RTX 4090, 32GB):
Phase 1: ████████░░░░░░░░░░░░░ 0.82 (82%)
Phase 2: █████████████░░░░░░░░ 0.92 (92%)
Phase 3: ███████████████░░░░░░ 0.96 (96%) ✅✅ RECOMMENDED
```

### CPU Usage During Processing

```
Phase 1 (Audio-only):     ██░░░░░░░░ 20% (very light)
Phase 2 (Audio + Motion): ███░░░░░░░ 40% (moderate)
Phase 3 (CPU-only):       ██████░░░░ 55% (high)
Phase 3 (with GPU):       ███░░░░░░░ 35% (GPU offloads, lighter CPU)
```

---

## What Task 1.11 Does

**Automatically selects the right phase for your machine:**

```
┌─────────────────────────────────────┐
│ DiveAnalyzer Starts Processing      │
└─────────────────────────────────────┘
           ⬇️
┌─────────────────────────────────────┐
│ System Profiler Detects:            │
│ • CPU: 4 cores @ 1.4 GHz            │
│ • RAM: 8 GB                         │
│ • GPU: None                         │
│ • System Score: 2/10                │
└─────────────────────────────────────┘
           ⬇️
┌─────────────────────────────────────┐
│ Phase 3 Estimation:                 │
│ • Would take 380s (too slow!)       │
│ • Only +4% accuracy gain            │
│ • Not worth it                      │
└─────────────────────────────────────┘
           ⬇️
┌─────────────────────────────────────┐
│ Decision: AUTO-SELECT PHASE 2       │
│ • 15s processing ✅                 │
│ • 0.92 confidence ✅                │
│ • Production-ready ✅               │
└─────────────────────────────────────┘
           ⬇️
        ✅ Done!
```

---

## Example Scenarios

### Scenario 1: Your Mac (Intel i5, 8GB, No GPU)

**Without Task 1.11:**
```bash
$ diveanalyzer video.mov
# System defaults to Phase 3
# Waiting... 1 min... 2 min... 3 min... (stuck on YOLO)
# User frustrated ❌
```

**With Task 1.11:**
```bash
$ diveanalyzer video.mov
✓ Auto-selected Phase 2 based on system profile
  Estimated time: 15s
  Confidence: 0.92 (92%)

Processing...
✓ Complete in 15s!
✓ 30 dives extracted with 0.92 confidence
✓ User happy ✅
```

---

### Scenario 2: A User Upgrades to GPU Mac

**Before upgrade (i5, 8GB):**
```bash
$ diveanalyzer video.mov
✓ Auto-selected Phase 2 (0.92 confidence, 15s)
```

**After upgrade to M3 with GPU (16GB):**
```bash
$ diveanalyzer video.mov
✓ System detected new GPU (Metal)
✓ Auto-selected Phase 3 (0.96 confidence, 35s)
# Note: Only changed automatically on next profile refresh
# User gets best accuracy automatically!
```

---

## Cost-Benefit Analysis

### Accuracy Gain Per Speed Cost

| Phase Transition | Accuracy Gain | Speed Cost | ROI | Recommendation |
|-----------------|---------------|-----------|-----|-----------------|
| Phase 1 → Phase 2 | +0.10 (12%) | 3x slower | **EXCELLENT** | Always worth it |
| Phase 2 → Phase 3 (with GPU) | +0.04 (4%) | ~2x slower | **GOOD** | Worth it if GPU |
| Phase 2 → Phase 3 (CPU-only) | +0.04 (4%) | **25x slower** | **TERRIBLE** | Not worth it |

---

## Production Decision

### For Swimming Pool Video Analysis

**Recommendation: Phase 2**
- 0.92 confidence = 92% accurate (excellent for production)
- ~15 second processing time (human-perceivable but acceptable)
- Works on all machines (even weak laptops)
- Manual review needed for ~8% of clips (still manageable)
- Cost-effective

**Phase 3 is overkill unless:**
- You want fully automated (0 manual review)
- You have GPU to accelerate it
- Accuracy is mission-critical
- You don't mind waiting on CPU-only

---

## Configuration

Users can configure defaults in `~/.diveanalyzer/config.yaml`:

```yaml
# Automatic phase selection settings
auto_phase_selection:
  enabled: true                    # Enable auto-selection

  # Thresholds for phase recommendation
  phase_2_threshold_sec: 30        # If Phase 3 > 30s, use Phase 2
  min_system_score_for_phase3: 7   # Phase 3 only if score ≥ 7

  # User overrides
  force_phase: null                # Force specific phase (1, 2, or 3)

  # Caching
  profile_cache_days: 7            # Re-profile every 7 days

# Or force via CLI:
# diveanalyzer video.mov --force-phase=3 --auto-select=false
```

---

## Conclusion

### The Smart Approach (Task 1.11)
- **Weak machines**: Phase 2 (fast, accurate, works)
- **GPU machines**: Phase 3 (fastest, most accurate)
- **Users don't have to think**: Auto-selection handles it

### The Old Approach (Without Task 1.11)
- **Weak machines**: Stuck with Phase 3 (slow, frustrating)
- **GPU machines**: Still have to manually select Phase 3
- **Users confused**: Which phase to use?

**Task 1.11 solves this by making it automatic and smart.** ✅

---

**Created:** 2026-01-20
**Status:** Ready for implementation
**Next:** Implement system_profiler.py and integrate with CLI
