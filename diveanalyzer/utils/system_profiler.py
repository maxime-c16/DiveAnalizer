"""
System profiler for detecting hardware capabilities and recommending detection phases.

Intelligently selects between Phase 1/2/3 based on CPU/RAM/GPU availability
to optimize performance vs accuracy tradeoff.
"""

import json
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class SystemProfile:
    """Complete system capability profile."""

    # Hardware specs
    cpu_count: int
    cpu_freq_ghz: float
    total_ram_gb: float
    available_ram_gb: float

    # GPU info
    gpu_type: str  # 'cuda', 'metal', 'rocm', 'none'
    gpu_memory_gb: float

    # System scoring
    system_score: int  # 0-10 scale
    phase_3_estimate_sec: float  # Estimated time for 10-min video

    # Recommendation
    recommended_phase: int  # 1, 2, or 3
    profile_date: str = field(default_factory=lambda: datetime.now().isoformat())

    # OS info
    os_type: str = field(default="")
    os_version: str = field(default="")

    def __str__(self) -> str:
        """Human-readable profile summary."""
        return f"""
System Profile ({self.os_type} {self.os_version}):
├─ CPU: {self.cpu_count}-Core @ {self.cpu_freq_ghz:.1f} GHz
├─ RAM: {self.total_ram_gb:.1f} GB ({self.available_ram_gb:.1f} GB available)
├─ GPU: {self.gpu_type.upper()}{f' ({self.gpu_memory_gb:.1f} GB VRAM)' if self.gpu_memory_gb > 0 else ' (None)'}
├─ System Score: {self.system_score}/10
│
├─ Phase Timing Estimates (10-min video):
│  ├─ Phase 1: ~5s (0.82 confidence)
│  ├─ Phase 2: ~15s (0.92 confidence){' ← RECOMMENDED' if self.recommended_phase == 2 else ''}
│  └─ Phase 3: ~{self.phase_3_estimate_sec:.0f}s (0.96 confidence){' ← RECOMMENDED' if self.recommended_phase == 3 else ''}
│
└─ Recommendation: PHASE {self.recommended_phase}
""".strip()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SystemProfile":
        """Load from dictionary."""
        return cls(**data)


class SystemProfiler:
    """Detect system capabilities and recommend detection phases."""

    # Baseline specs for estimation (Intel i5, 1.4 GHz, 4 cores)
    BASELINE_PHASE3_TIME_SEC = 350.0
    BASELINE_CPU_CORES = 4
    BASELINE_CPU_FREQ_GHZ = 1.4

    # GPU speedup factors
    GPU_SPEEDUP_FACTORS = {
        "cuda": 12.0,  # 8.5x-15x speedup on NVIDIA
        "metal": 8.0,  # 7-10x speedup on Apple Silicon
        "rocm": 10.0,  # 8-12x speedup on AMD
        "none": 1.0,   # No GPU
    }

    # Phase thresholds
    PHASE_2_THRESHOLD_SEC = 30.0  # Use Phase 2 if Phase 3 > 30s
    MIN_SYSTEM_SCORE_PHASE3 = 7  # Min score for Phase 3

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize profiler with optional cache directory."""
        self.cache_dir = cache_dir or Path.home() / ".diveanalyzer"
        self.cache_file = self.cache_dir / "system_profile.json"
        self.cache_duration = timedelta(days=7)

    def get_profile(self, refresh: bool = False) -> SystemProfile:
        """
        Get system profile (cached or fresh).

        Args:
            refresh: Force fresh profile instead of using cache

        Returns:
            SystemProfile with current system capabilities
        """
        # Check cache first
        if not refresh and self.cache_file.exists():
            try:
                cached_profile = self._load_from_cache()
                if self._is_cache_valid(cached_profile):
                    return cached_profile
            except Exception as e:
                print(f"⚠️  Failed to load cached profile: {e}")

        # Generate fresh profile
        profile = self._profile_system()

        # Save to cache
        self._save_to_cache(profile)

        return profile

    def _profile_system(self) -> SystemProfile:
        """Profile current system and return detailed profile."""
        if psutil is None:
            # Fallback if psutil not available
            return self._profile_system_fallback()

        # Detect CPU
        cpu_count = psutil.cpu_count(logical=False) or 4
        cpu_freq = psutil.cpu_freq()
        cpu_freq_ghz = (cpu_freq.current / 1000.0) if cpu_freq else 2.0

        # Detect RAM
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

        # Detect GPU
        gpu_type, gpu_memory_gb = self._detect_gpu()

        # Calculate system score
        system_score = self._calculate_system_score(
            cpu_count, cpu_freq_ghz, total_ram_gb, gpu_type
        )

        # Estimate Phase 3 time
        phase_3_estimate = self._estimate_phase_3_time(
            cpu_count, cpu_freq_ghz, gpu_type
        )

        # Recommend phase
        recommended_phase = self._recommend_phase(system_score, phase_3_estimate)

        # Get OS info
        os_type = platform.system()
        os_version = platform.release()

        return SystemProfile(
            cpu_count=cpu_count,
            cpu_freq_ghz=cpu_freq_ghz,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb,
            system_score=system_score,
            phase_3_estimate_sec=phase_3_estimate,
            recommended_phase=recommended_phase,
            os_type=os_type,
            os_version=os_version,
        )

    def _profile_system_fallback(self) -> SystemProfile:
        """Fallback profile when psutil not available."""
        import os

        cpu_count = os.cpu_count() or 4
        total_ram_gb = 8.0  # Conservative estimate
        available_ram_gb = 4.0

        gpu_type = "none"
        gpu_memory_gb = 0.0

        os_type = platform.system()
        os_version = platform.release()

        system_score = 2  # Conservative
        phase_3_estimate = 300.0  # Conservative

        recommended_phase = 2

        return SystemProfile(
            cpu_count=cpu_count,
            cpu_freq_ghz=1.4,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb,
            system_score=system_score,
            phase_3_estimate_sec=phase_3_estimate,
            recommended_phase=recommended_phase,
            os_type=os_type,
            os_version=os_version,
        )

    def _detect_gpu(self) -> Tuple[str, float]:
        """
        Detect available GPU.

        Returns:
            Tuple of (gpu_type, gpu_memory_gb)
        """
        # Try NVIDIA CUDA
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                        1024 ** 3
                    )
                    return ("cuda", gpu_memory)
        except ImportError:
            pass

        # Try Apple Metal (M1/M2/M3/etc)
        if platform.system() == "Darwin":
            try:
                import torch

                if torch.backends.mps.is_available():
                    return ("metal", 8.0)  # Assume ~8GB VRAM for Metal
            except ImportError:
                pass

        # Try AMD ROCm
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                return ("rocm", 8.0)
        except ImportError:
            pass

        # No GPU detected
        return ("none", 0.0)

    def _calculate_system_score(
        self, cpu_count: int, cpu_freq_ghz: float, total_ram_gb: float, gpu_type: str
    ) -> int:
        """
        Calculate 0-10 system capability score.

        Higher score = more capable machine.
        Used to decide if Phase 3 is feasible.
        """
        score = 0

        # CPU contribution (max 4 points)
        # Normalized to baseline (4-core @ 1.4 GHz = baseline)
        cpu_power = (cpu_count / self.BASELINE_CPU_CORES) * (
            cpu_freq_ghz / self.BASELINE_CPU_FREQ_GHZ
        )
        cpu_score = min(4, int(cpu_power * 3.0))  # Conservative scaling
        score += cpu_score

        # RAM contribution (max 3 points)
        if total_ram_gb >= 16:
            score += 3
        elif total_ram_gb >= 8:
            score += 2
        else:
            score += 1

        # GPU contribution (max 3 points)
        if gpu_type in ["cuda", "rocm"]:
            score += 3  # Powerful GPUs
        elif gpu_type == "metal":
            score += 2  # Apple Metal
        else:
            score += 0  # No GPU

        return min(10, score)

    def _estimate_phase_3_time(
        self, cpu_count: int, cpu_freq_ghz: float, gpu_type: str
    ) -> float:
        """
        Estimate Phase 3 processing time for 10-minute video.

        Based on CPU cores/frequency and GPU availability.
        """
        # Get GPU speedup factor
        speedup = self.GPU_SPEEDUP_FACTORS.get(gpu_type, 1.0)

        # Calculate CPU power index
        cpu_power_index = (cpu_count / self.BASELINE_CPU_CORES) * (
            cpu_freq_ghz / self.BASELINE_CPU_FREQ_GHZ
        )

        # Estimate time
        estimated_time = (self.BASELINE_PHASE3_TIME_SEC / cpu_power_index) / speedup

        return estimated_time

    def _recommend_phase(
        self, system_score: int, phase_3_estimate_sec: float
    ) -> int:
        """
        Recommend detection phase based on system capabilities.

        Returns:
            Phase number (1, 2, or 3)
        """
        # If Phase 3 would be too slow, use Phase 2
        if phase_3_estimate_sec > self.PHASE_2_THRESHOLD_SEC:
            return 2

        # If system score too low, use Phase 2
        if system_score < self.MIN_SYSTEM_SCORE_PHASE3:
            return 2

        # Otherwise use Phase 3
        return 3

    def _is_cache_valid(self, profile: SystemProfile) -> bool:
        """Check if cached profile is still valid."""
        profile_time = datetime.fromisoformat(profile.profile_date)
        return (datetime.now() - profile_time) < self.cache_duration

    def _load_from_cache(self) -> Optional[SystemProfile]:
        """Load cached profile from file."""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                return SystemProfile.from_dict(data)
        return None

    def _save_to_cache(self, profile: SystemProfile) -> None:
        """Save profile to cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def clear_cache(self) -> None:
        """Clear cached profile."""
        if self.cache_file.exists():
            self.cache_file.unlink()


def get_system_profile(
    cache_dir: Optional[Path] = None, refresh: bool = False
) -> SystemProfile:
    """Convenience function to get system profile."""
    profiler = SystemProfiler(cache_dir)
    return profiler.get_profile(refresh=refresh)
