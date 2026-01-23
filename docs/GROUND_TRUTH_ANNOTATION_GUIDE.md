# Ground Truth Annotation Guide
**Version**: 2.0 (Frame-Based)
**Purpose**: Standardize dive detection ground truth data collection using frame indices

---

## Important: Frame Indices vs Elapsed Time

This guide uses **FRAME INDICES** (frame numbers) instead of elapsed time in seconds.

**Why?**
- Frame indices are more precise (exact frame, not approximation)
- Easy to identify frame-by-frame in video player
- Objective and unambiguous
- Conversion to seconds is simple: `elapsed_time = frame_index / fps`

**All test fixtures are 30 fps**, so:
- Frame 30 = 1 second
- Frame 900 = 30 seconds
- Frame 3600 = 120 seconds

---

## Dive Event Definitions

### Key Frame Indices

#### 1. START TIME - When diver leaves the platform/springboard
- **Definition**: The moment when the diver's feet lose contact with the board/platform
- **Why this point?**
  - Clearest, most objective moment (binary: feet on board vs airborne)
  - Marks the beginning of the dive action
  - Motion detection should activate around this time

- **How to identify in video**:
  - Watch diver approach board
  - Look for the exact frame when feet lift off
  - This is when vertical motion begins

- **Example**:
  ```
  Frame 145: Diver standing on board (feet visible on surface)
  Frame 146: Diver's feet have lifted off, body airborne
  ← START TIME is at frame 146 timestamp
  ```

---

#### 2. SPLASH TIME - When diver's body makes first contact with water
- **Definition**: The moment when ANY part of the diver's body first touches the water surface
- **Why this point?**
  - This creates the acoustic splash signal we detect with audio
  - Matches the audio peak detection timestamp
  - Objective: either touching water or not

- **How to identify in video**:
  - Watch for first ripples or disturbance on water surface
  - Look for hands/head making contact (usually hands first)
  - Listen for the splash sound

- **Example**:
  ```
  Frame 287: Hands approaching water surface
  Frame 288: Hands break the water surface ← SPLASH happens here
  SPLASH TIME is at frame 288 timestamp
  ```

- **Audio Alignment**:
  - The audio peak we detect should occur at approximately this timestamp
  - Small variations (±0.1-0.2 seconds) are normal due to:
    - Audio processing delay
    - Camera frame rate vs audio sample rate
    - Speed of sound travel in water

---

#### 3. END TIME - When the diver's body is fully submerged
- **Definition**: The moment when the last visible part of the diver's body disappears below the water surface
- **Why this point?**
  - Captures complete entry into water
  - Defines natural boundary of dive action
  - Consistent with extraction requirement (capture full dive)

- **How to identify in video**:
  - Watch diver enter water after splash
  - Track the last part of body visible (usually feet/legs)
  - Mark when it fully submerges

- **Example**:
  ```
  Frame 310: Feet still visible above water line
  Frame 311: Feet have disappeared, fully submerged
  ← END TIME is at frame 311 timestamp
  ```

- **Note**: This is typically 0.5-2 seconds after splash, depending on:
  - Diver size (tall person takes longer to fully enter)
  - Dive type (back dives may show body longer)
  - Camera angle

---

## Duration Relationships

**Typical dive characteristics**:
```
START (feet leave board)
  ↓
  (airborne phase: 0.3-0.8 seconds)
  ↓
SPLASH (hands/head hit water)
  ↓
  (entry phase: 0.5-2.0 seconds)
  ↓
END (fully submerged)
```

**Typical total duration**: START → END = 1-3 seconds

**For extraction**:
- We add 10 seconds **before** START (pre-splash buffer)
- We add 3 seconds **after** END (post-splash buffer)
- Total extracted clip: ~14-16 seconds

---

## Special Cases

### Back-to-Back Dives (< 5 seconds apart)

**Definition**: Two dives where the diver does NOT fully surface/exit water between them

**How to handle**:
- Record SPLASH times for both dives separately
- If gaps between dives < 5 seconds, they may merge in post-processing
- This is INTENTIONAL - we want to know about very close dives

**Example**:
```
Dive 1: splash@10.0s
Diver starts climbing out...
Dive 2: splash@12.5s (gap: 2.5 seconds)
↑
These are recorded as 2 separate dives
But may be merged if extraction buffer overlaps
```

---

### Failed Dive (Abort - no splash)

**Definition**: Diver leaves platform but does NOT make contact with water

**Why this happens**:
- Diver changes mind mid-jump
- Loss of balance causes belly flop avoidance
- Intentional abort

**How to record**:
- Record START time when feet leave board
- Record MOTION BURST if visible
- Note that **there is NO splash** (this is the key!)
- Leave END time blank or mark as "no_splash"

**Example**:
```
"failed_dive.mp4": {
  "duration_sec": 40,
  "total_dives": 0,
  "dives": [],
  "motion_bursts": [
    {
      "start_time_sec": 5.0,
      "end_time_sec": 8.0,
      "splash_present": false,
      "notes": "Diver jumped but did not enter water"
    }
  ]
}
```

---

### False Positive (Loud noise, no dive)

**Definition**: Loud sound in audio but no actual diver or dive happening

**Why this matters**:
- Phase 1 (audio-only) detects the sound
- Phase 2/3 should reject it because no motion or person visible
- This validates our multi-phase approach

**How to record**:
```
"false_positive.mp4": {
  "duration_sec": 30,
  "total_dives": 0,
  "dives": [],
  "false_positives": [
    {
      "time_sec": 5.0,
      "reason": "equipment crash, background noise, door slam",
      "severity": "loud"
    }
  ],
  "notes": "No actual dive occurred"
}
```

---

### No Audio Case

**Definition**: Video with no audio track (we use Phase 2 motion detection instead)

**How to record**:
- Record start/splash/end times **by watching motion**
- Should be same as corresponding dive in audio version
- Validates Phase 2 fallback works

**Example**:
```
"edge_cases/no_audio.mp4": {
  "duration_sec": 40,
  "total_dives": 1,
  "no_audio_track": true,
  "dives": [
    {
      "id": 1,
      "start_time_sec": 5.0,
      "splash_time_sec": 10.0,
      "end_time_sec": 15.0,
      "detection_method": "motion_only",
      "notes": "Same timing as short_dive.mp4, validates Phase 2"
    }
  ]
}
```

---

## Data Format (JSON)

### Standard Dive Entry

```json
{
  "id": 1,
  "start_time_sec": 5.0,
  "splash_time_sec": 10.0,
  "end_time_sec": 15.0,
  "dive_type": "forward",
  "confidence": "high",
  "notes": ""
}
```

**Fields**:
- `id`: Sequential dive number (1, 2, 3, ...)
- `start_time_sec`: When feet leave platform (float, seconds from video start)
- `splash_time_sec`: When hands/head hits water (float)
- `end_time_sec`: When fully submerged (float)
- `dive_type`: Optional - "forward", "backward", "twist", "unknown"
- `confidence`: Optional - "high", "medium", "low" based on clarity
- `notes`: Any observations (noisy video, unclear moment, etc)

### Validation Rules

```
Check: start_time < splash_time < end_time
Check: splash_time - start_time ≈ 0.3-0.8s (airborne phase)
Check: end_time - splash_time ≈ 0.5-2.0s (entry phase)
Check: end_time - start_time ≈ 1-3s (total dive duration)
```

---

## Measurement Technique

### Using Video Player

**Prerequisites**:
- Use player that shows frame-by-frame timestamps
- Know your video's frame rate (usually 30fps or 60fps)
- Zoom in on important moments

**Steps**:

1. **Find START TIME**:
   - Play video at normal speed
   - When you see diver approach board, switch to frame-by-frame
   - Go frame-by-frame as they prepare
   - Click on frame where feet leave board
   - Record timestamp (or frame number, convert later)

2. **Find SPLASH TIME**:
   - Continue frame-by-frame after START
   - Watch for first water disturbance
   - Click on frame where contact begins
   - Record timestamp
   - Verify: typically 0.3-0.8 seconds after START

3. **Find END TIME**:
   - Continue frame-by-frame after SPLASH
   - Watch last visible body part
   - Click on frame where last body part submerges
   - Record timestamp
   - Verify: typically 0.5-2.0 seconds after SPLASH

---

## Expected Results After Standardization

### short_dive.mp4 (40 seconds)
Expected pattern:
- 1 diver visible
- 1 clear dive sequence
- Audio should have 1 major peak (maybe some noise)
- Clean case for validation

### multi_dive.mp4 (300 seconds, CRITICAL)
Expected pattern:
- Multiple dives across 5 minutes
- Dives should be 5-15 seconds apart
- Audio will have 37 peaks (echoes and actual dives mixed)
- Challenge: distinguish real dives from echoes/reflections
- Our fix groups peaks by splash time - expect 3-7 real dives

**Hypothesis for 37 peaks**:
- ~5-7 actual dives (real people jumping)
- ~30 peaks are echoes and reflections in the pool
- Our merging (gap < 5s) should consolidate these

### back_to_back.mp4 (60 seconds)
Expected pattern:
- 2 dives very close (< 5 seconds apart)
- Diver may not fully surface between them
- Tests our handling of close dives
- Should keep as 2 separate events (unless < 5s gap causes merge)

### false_positive.mp4 (30 seconds)
Expected pattern:
- Loud sound(s) but NO diver visible
- NO motion burst corresponding to sound
- NO person entering frame
- Phase 1 might detect as peak, Phase 2/3 should reject
- Validates multi-phase approach

### no_audio.mp4 (40 seconds)
Expected pattern:
- Same content as short_dive.mp4 but audio removed
- Phase 1 fails (no audio)
- Phase 2 detects motion burst
- Should find same dive as short_dive.mp4

---

## Annotation Workflow

1. **Open test video** in video player with frame-by-frame capability
2. **Play at normal speed** to get overall understanding
3. **For each dive** (in order):
   a. Pause when diver approaches board
   b. Frame-by-frame to feet leaving board → **START TIME**
   c. Frame-by-frame to first water contact → **SPLASH TIME**
   d. Frame-by-frame to full submersion → **END TIME**
   e. Record dive_type if recognizable
4. **Record all three timestamps** in JSON
5. **Verify timestamps** make sense (durations match expectations)
6. **Double-check** audio peaks match splash times (±0.2s)

---

## Troubleshooting

### Q: The splash is hard to see on camera
**A**: Listen to the audio instead
- Where's the splash sound?
- That's approximately the splash time
- Visual + audio should align within 0.2 seconds

### Q: There's an echo - is that a second dive?
**A**: Probably not
- Pool echoes audio
- Usually within 0.5 seconds of original splash
- If gap < 5 seconds AND no new diver visible, it's an echo
- Our merging logic groups these together

### Q: The diver doesn't fully submerge (shallow dive)
**A**: Mark when body disappears
- Even if diver is standing in shallow water, find where last part submerges
- This is the "end" of the dive action

### Q: Multiple people in frame
**A**: Track each person separately
- Each person = separate dive entry
- Record timings for each independently

---

## Quality Checklist

Before submitting ground truth, verify:

- [ ] All timestamps are in seconds (float format)
- [ ] start_time < splash_time < end_time
- [ ] All dives are in chronological order
- [ ] dive_type field is filled (or "unknown")
- [ ] Total count matches number of entries
- [ ] Audio peaks roughly match splash times
- [ ] No negative timestamps
- [ ] File is valid JSON (can be parsed)
- [ ] No comments or extra fields (clean JSON)

---

## Summary Table

| Timestamp | What happens | Visual cue | Audio cue |
|-----------|--------------|-----------|----------|
| START | Feet leave board | Body airborne | Motion sound |
| SPLASH | First body-water contact | Water ripples | Splash peak |
| END | Fully submerged | No body visible | Splash decays |

**Typical gaps**:
- START→SPLASH: 0.3-0.8s (flight time)
- SPLASH→END: 0.5-2.0s (entry time)
- START→END: 1-3s (total dive)

---

## Questions?

If you encounter edge cases not covered here, note them with:
- Video timestamp
- What you observed
- Why you're unsure
- Best guess for classification

This helps us improve the guide for next iteration.

---

**Version**: 1.0 (2026-01-20)
**Status**: Ready for annotation
