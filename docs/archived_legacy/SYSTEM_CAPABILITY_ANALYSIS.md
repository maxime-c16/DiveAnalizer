# System Capability Analysis & Adaptive Phase Selection

**Date:** 2026-01-20
**Status:** Sprint 1.11 New Ticket Added to Roadmap

---

## Your System Profile

### Hardware
```
MacBook Pro 15,4
├─ CPU: Intel Core i5 @ 1.4 GHz (4 cores)
├─ RAM: 8 GB
├─ GPU: None (Intel Iris UHD 630 - not compute-capable)
├─ System Score: 2/10 (below minimum for Phase 3 without GPU)
└─ Storage: Adequate
```

### Performance Reality

| Phase | Processing Time | Confidence | Status |
|-------|-----------------|-----------|--------|
| **Phase 1** | ~5s | 0.82 | ✅ BLAZING FAST |
| **Phase 2** | ~15s | 0.92 | ✅ EXCELLENT |
| **Phase 3 (CPU)** | **~380s** ❌ | 0.96 | ⚠️ **7X SLOWER!** |
| **Phase 3 (GPU)** | ~35-40s | 0.96 | ✅ Would be perfect with GPU |

---

## The Problem

**Without GPU, Phase 3 adds only +4% accuracy (0.92→0.96) but costs 25x the processing time.**

```
Your Mac on Phase 3 (CPU-only):
├─ Audio detection:    1s
├─ Splash analysis:    0.3s
├─ Proxy generation:   60s (one-time, cached)
├─ Motion detection:   13s
├─ Person detection:   **350s** ← This is the killer
├─ Clip extraction:    3-5 min
└─ TOTAL: **6+ minutes** for only +4% accuracy gain

Vs Phase 2:
├─ Audio detection:    1s
├─ Splash analysis:    0.3s
├─ Proxy generation:   60s (one-time, cached)
├─ Motion detection:   13s
├─ Fusion:             0.0s
├─ Clip extraction:    3-5 min
└─ TOTAL: **3-5 minutes** with 0.92 confidence ✅
```

---

## The Solution: Sprint 1.11 - Adaptive Phase Selection ⭐

**New ticket added to SPRINT_ROADMAP.md to automatically select the best phase based on system capabilities.**

### How It Works

1. **System Profiling**: Detects CPU cores, RAM, GPU type, calculates system "score" (0-10)
2. **Phase Estimation**: Estimates how long Phase 3 would take
3. **Smart Selection**:
   - Phase 3 estimated > 30s? → Auto-select Phase 2 (0.92 confidence, fast)
   - System score < 7? → Auto-select Phase 2 (safeguard)
   - GPU available? → Auto-select Phase 3 (0.96 confidence, fast due to GPU)
4. **User Control**: Flags to override (`--force-phase=1|2|3`)
5. **Transparency**: Show users the tradeoff clearly

### Demo Output (Your Machine)

```
System Profile (macOS Intel i5, 8GB RAM):
├─ CPU: Quad-Core Intel i5 @ 1.4GHz
├─ RAM: 8.0 GB
├─ GPU: None (Intel Iris UHD 630 - not suitable for inference)
├─ Storage: 450GB free
├─ System Score: 2/10 (below Phase 3 minimum)
│
├─ Phase Timing Estimates (for 10-min video):
│  ├─ Phase 1: 5s (0.82 confidence)
│  ├─ Phase 2: 15s (0.92 confidence) ← RECOMMENDED
│  └─ Phase 3: 380s ⚠️ (too slow without GPU)
│
├─ Decision: Using PHASE 2 for optimal speed/accuracy
├─ Accuracy: 0.92 (92% confidence - excellent for production)
├─ Speed: 15s processing + 3-5min extraction = 3-5min total
│
└─ Options:
   • To use Phase 3 anyway: diveanalyzer video.mov --force-phase=3
   • To use Phase 1 (fastest): diveanalyzer video.mov --force-phase=1
   • To view system profile: diveanalyzer --profile
   • To update profile cache: diveanalyzer --profile --refresh
```

---

## Key Benefits

| For You (Intel i5, 8GB) | For GPU Users |
|------------------------|--------------|
| ✅ Auto-select Phase 2 (0.92 confidence) | ✅ Auto-select Phase 3 (0.96 confidence) |
| ✅ ~15s processing (not 350s) | ✅ ~35-40s processing |
| ✅ Usable performance | ✅ Best accuracy |
| ✅ Can still override if needed | ✅ Can still override if needed |

---

## What This Means

### Before Sprint 1.11
- Your machine might attempt Phase 3 by default
- Results: 380s processing, unusable lag, user frustration

### After Sprint 1.11
- Your machine detects it doesn't have GPU
- Automatically uses Phase 2: 0.92 confidence, ~15s processing
- User sees why: "Phase 3 would take 380s, so we're using Phase 2 (15s) for you"
- If user really wants Phase 3, they can `--force-phase=3`

---

## For Comparison: Different Machines

### Machine A: MacBook Pro M3 (16GB, Metal GPU)
```
System Score: 8/10
Phase 3 Estimate: 35s (with GPU acceleration)
→ Auto-selects Phase 3 ✓ (0.96 confidence, acceptable speed)
```

### Machine B: Desktop with NVIDIA GPU (32GB, CUDA)
```
System Score: 10/10
Phase 3 Estimate: 8s (with GPU acceleration)
→ Auto-selects Phase 3 ✓ (0.96 confidence, blazing fast)
```

### Machine C: Low-end Windows Laptop (4GB, no GPU)
```
System Score: 0/10
Phase 3 Estimate: 600+ seconds (unusable)
→ Auto-selects Phase 1 or 2 ✓ (prevents disaster)
```

---

## Implementation Details

### New Module: `diveanalyzer/utils/system_profiler.py`

```python
class SystemProfile:
    cpu_count: int
    cpu_freq_ghz: float
    total_ram_gb: float
    gpu_type: str  # 'cuda', 'metal', 'rocm', 'none'
    gpu_memory_gb: float
    system_score: int  # 0-10 scale
    phase_3_estimate_sec: float
    recommended_phase: int

def profile_system() -> SystemProfile:
    """Profile current system capabilities."""
    # Detects CPU/RAM/GPU, calculates score, estimates Phase 3 time

def estimate_phase_3_time(cpu_count, cpu_freq, gpu_type) -> float:
    """Estimate Phase 3 processing time for 10-min video."""
    # Baseline: 350s for 4-core 1.4GHz CPU
    # Scales with CPU power and GPU availability

def calculate_system_score(cpu_count, cpu_freq, total_ram, gpu_type) -> int:
    """Calculate 0-10 system capability score."""
    # CPU (max 4 pts) + RAM (max 3 pts) + GPU (max 3 pts)
```

### CLI Integration

```bash
# View system profile
$ diveanalyzer --profile
System Profile (macOS Intel i5, 8GB RAM):
├─ System Score: 2/10
├─ Recommended Phase: 2
└─ ...

# Force a specific phase
$ diveanalyzer video.mov --force-phase=3  # Use Phase 3 anyway
$ diveanalyzer video.mov --force-phase=1  # Use Phase 1 only (fastest)

# Refresh profile cache (normally cached 7 days)
$ diveanalyzer --profile --refresh
```

### Configuration

```yaml
# ~/.diveanalyzer/auto_phase.yaml
system_profile:
  cpu_count: 4
  cpu_freq_ghz: 1.4
  total_ram_gb: 8.0
  gpu_type: "none"
  system_score: 2
  recommended_phase: 2
  phase_3_estimate_sec: 380
  profile_date: "2026-01-20T15:30:00"

auto_select: true                 # Enable auto-selection
phase_2_threshold_sec: 30         # If Phase 3 > 30s, use Phase 2
```

---

## Testing Strategy

- Unit tests: CPU/RAM/GPU detection with mocked values
- Integration tests: Full pipeline with auto-selection
- Edge cases: Very weak machines, very powerful machines
- Benchmark validation: Ensure estimates match real performance

---

## Success Criteria

✓ System profiler accurately detects CPU/RAM/GPU
✓ Low-end machines auto-select Phase 2 (0.92 confidence, fast)
✓ High-end machines auto-select Phase 3 (0.96 confidence, acceptable speed)
✓ Auto-selection prevents 7x slowdowns on weak machines
✓ Users understand the tradeoff clearly
✓ Override flags allow manual control

---

## Documentation Updates

- README: Add "System Requirements" section
- CLI help: Add `--force-phase`, `--profile` flag descriptions
- New file: "System Compatibility Guide"

---

## Related Tasks

This is **Task 1.11 in Sprint 1** - sits alongside:
- Task 1.1: GPU detection (companion piece)
- Task 1.2-1.10: GPU optimization for Phase 3
- Task 1.11: Smart fallback for low-end machines (NEW)

---

## Summary

**Your machine will get Phase 2 automatically** (0.92 confidence, ~15s processing) instead of attempting Phase 3 (0.96 confidence, 380s processing).

This is the **right tradeoff** for your hardware: gain 92% accuracy without the 7x performance hit.

If you upgrade to a Mac with GPU (M3/M4) or a GPU desktop, the system will automatically detect it and use Phase 3 for the highest accuracy.

---

**Next Steps:**
1. ✅ Ticket added to SPRINT_ROADMAP.md (Task 1.11)
2. ⏳ Implement system profiler module
3. ⏳ Add CLI flags and configuration
4. ⏳ Integration testing
5. ⏳ Documentation
