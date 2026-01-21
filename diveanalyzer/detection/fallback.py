"""Phase fallback logic for graceful degradation.

Implements Phase 3 → Phase 2 → Phase 1 fallback on detection failures.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .fusion import (
    DiveEvent,
    fuse_signals_audio_only,
    fuse_signals_audio_motion,
    fuse_signals_audio_motion_person,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)
from .audio import extract_audio, detect_splash_peaks
from .motion import detect_motion_bursts
from .person import detect_person_frames


@dataclass
class PhaseFallbackStats:
    """Statistics about phase fallback."""
    target_phase: int  # Original target phase (1, 2, or 3)
    actual_phase: int  # Phase that actually completed
    fallbacks: List[Dict[str, Any]]  # List of fallbacks attempted
    errors: List[str]  # Error messages during fallbacks
    confidence_level: float  # Final confidence of results
    dives_detected: int  # Number of dives found

    def __str__(self) -> str:
        """Format as string."""
        if self.target_phase == self.actual_phase:
            return f"Phase {self.actual_phase} (no fallback needed, {self.confidence_level:.0%} confidence)"
        else:
            fallback_str = " → ".join([str(f["from_phase"]) for f in self.fallbacks] + [str(self.actual_phase)])
            return f"Phase {fallback_str} (fallback due to {self.fallbacks[-1]['reason']})"


def detect_dives_with_fallback(
    video_path: str,
    target_phase: int = 3,
    audio_threshold: float = -25.0,
    motion_sample_fps: float = 5.0,
    person_sample_fps: float = 5.0,
    use_gpu: bool = False,
    force_cpu: bool = False,
    use_fp16: bool = False,
    batch_size: int = 16,
    min_confidence: float = 0.5,
    verbose: bool = False,
) -> Tuple[List[DiveEvent], PhaseFallbackStats]:
    """
    Detect dives with automatic fallback on phase failures.

    Attempts detection in phases: Phase 3 → Phase 2 → Phase 1.
    Automatically falls back to earlier phases if GPU/resource issues occur.

    Args:
        video_path: Path to input video
        target_phase: Target detection phase (1, 2, or 3)
        audio_threshold: Audio threshold in dB
        motion_sample_fps: Motion detection sample rate
        person_sample_fps: Person detection sample rate
        use_gpu: Use GPU for person detection
        force_cpu: Force CPU (overrides use_gpu)
        use_fp16: Use FP16 quantization
        batch_size: Batch size for inference
        min_confidence: Minimum confidence filter
        verbose: Print detailed messages

    Returns:
        Tuple of (list of DiveEvent, PhaseFallbackStats)
    """
    video_path = str(Path(video_path).resolve())
    stats = PhaseFallbackStats(
        target_phase=target_phase,
        actual_phase=target_phase,
        fallbacks=[],
        errors=[],
        confidence_level=0.0,
        dives_detected=0,
    )

    # Always start with Phase 1 (audio)
    if verbose:
        print(f"🔊 Phase 1: Extracting audio...")

    try:
        audio_path = extract_audio(video_path)
        peaks = detect_splash_peaks(
            audio_path,
            threshold_db=audio_threshold,
            min_distance_sec=5.0,
            prominence=5.0,
        )
        phase1_dives = fuse_signals_audio_only(peaks)
        current_phase = 1

        if verbose:
            print(f"   ✓ Found {len(phase1_dives)} potential dives (audio)")

    except Exception as e:
        error_msg = f"Phase 1 failed: {str(e)[:100]}"
        stats.errors.append(error_msg)
        if verbose:
            print(f"   ❌ {error_msg}")
        return [], stats

    # Try Phase 2 if requested
    if target_phase >= 2:
        if verbose:
            print(f"🎬 Phase 2: Detecting motion bursts...")

        try:
            # Motion needs the original video
            motion_events = detect_motion_bursts(
                video_path,
                sample_fps=motion_sample_fps,
            )

            phase2_dives = fuse_signals_audio_motion(
                peaks,
                motion_events,
            )
            current_phase = 2

            if verbose:
                print(f"   ✓ Found {len(motion_events)} motion bursts")
                print(f"   ✓ Fused to {len(phase2_dives)} dives (audio + motion)")

        except Exception as e:
            error_msg = f"Phase 2 failed: {str(e)[:100]}"
            stats.errors.append(error_msg)
            if verbose:
                print(f"   ⚠️  {error_msg}")
                print(f"   📉 Falling back to Phase 1")
            phase2_dives = phase1_dives
            current_phase = 1

    else:
        phase2_dives = phase1_dives

    # Try Phase 3 if requested
    if target_phase >= 3 and current_phase == 2:
        if verbose:
            print(f"👤 Phase 3: Detecting persons...")

        try:
            person_events = detect_person_frames(
                video_path,
                sample_fps=person_sample_fps,
                use_gpu=use_gpu,
                force_cpu=force_cpu,
                use_fp16=use_fp16,
                batch_size=batch_size,
            )

            phase3_dives = fuse_signals_audio_motion_person(
                peaks,
                motion_events,
                person_events,
            )
            current_phase = 3

            if verbose:
                print(f"   ✓ Detected {sum(1 for _, p, _ in person_events if p)} person frames")
                print(f"   ✓ Fused to {len(phase3_dives)} dives (audio + motion + person)")

        except Exception as e:
            error_msg = f"Phase 3 failed: {str(e)[:100]}"
            stats.errors.append(error_msg)

            # Determine if should fallback
            should_fallback = _should_fallback(e, current_phase)

            if should_fallback and current_phase >= 2:
                if verbose:
                    print(f"   ⚠️  {error_msg}")
                    print(f"   📉 GPU/resource error - falling back to Phase 2")

                stats.fallbacks.append({
                    "from_phase": 3,
                    "to_phase": 2,
                    "reason": _get_fallback_reason(e),
                })
                phase3_dives = phase2_dives
                current_phase = 2

            else:
                # Can't recover - use best available
                if verbose:
                    print(f"   ❌ {error_msg}")

                phase3_dives = phase2_dives
                current_phase = 2

    else:
        phase3_dives = phase2_dives

    # Merge and filter final results
    final_dives = merge_overlapping_dives(phase3_dives)
    filtered_dives = filter_dives_by_confidence(final_dives, min_confidence=min_confidence)

    # Calculate final confidence
    if filtered_dives:
        avg_confidence = sum(d.confidence for d in filtered_dives) / len(filtered_dives)
    else:
        avg_confidence = 0.0

    # Update stats
    stats.actual_phase = current_phase
    stats.confidence_level = avg_confidence
    stats.dives_detected = len(filtered_dives)

    if verbose:
        print()
        print(f"📊 Results: {stats}")

    return filtered_dives, stats


def _should_fallback(error: Exception, current_phase: int) -> bool:
    """Determine if we should fallback from current phase.

    Args:
        error: Exception that occurred
        current_phase: Current phase (1, 2, or 3)

    Returns:
        True if should fallback to earlier phase
    """
    error_str = str(error).lower()

    # GPU/memory errors: always fallback
    gpu_errors = (
        "cuda",
        "out of memory",
        "oom",
        "gpu",
        "mps",
        "metal",
    )
    if any(msg in error_str for msg in gpu_errors):
        return current_phase > 1

    # Import/dependency errors: can fallback from Phase 3 to Phase 2
    if "module" in error_str or "import" in error_str:
        return current_phase == 3

    # Timeout: try earlier phase
    if "timeout" in error_str:
        return current_phase > 1

    return False


def _get_fallback_reason(error: Exception) -> str:
    """Get human-readable reason for fallback.

    Args:
        error: Exception that caused fallback

    Returns:
        String explaining the reason
    """
    error_str = str(error).lower()

    if "cuda" in error_str or "cuda out of memory" in error_str:
        return "CUDA out of memory - GPU too small"

    if "out of memory" in error_str or "oom" in error_str:
        return "Out of memory - system under pressure"

    if "mps" in error_str or "metal" in error_str:
        return "Metal/GPU error - not supported"

    if "timeout" in error_str:
        return "Operation timeout - too slow"

    if "module" in error_str or "import" in error_str:
        return "Missing dependency"

    return "Detection error"


def get_confidence_for_phase(phase: int) -> float:
    """Get expected confidence for a phase.

    Args:
        phase: Detection phase (1, 2, or 3)

    Returns:
        Expected confidence value
    """
    confidence_map = {
        1: 0.82,  # Audio-only
        2: 0.92,  # Audio + motion
        3: 0.96,  # Audio + motion + person
    }
    return confidence_map.get(phase, 0.5)
