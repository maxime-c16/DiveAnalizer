"""Error handling and graceful degradation for DiveAnalyzer.

Handles GPU failures, timeouts, and implements phase fallback logic.
"""

import signal
import functools
from typing import Optional, Callable, Any
from pathlib import Path


class TimeoutError(Exception):
    """Operation exceeded timeout."""
    pass


class PhaseFailover:
    """Handle phase fallback when detection fails."""

    @staticmethod
    def should_fallback_phase(current_phase: int, error: Exception) -> bool:
        """Determine if we should fallback to earlier phase.

        Args:
            current_phase: Current detection phase (1, 2, or 3)
            error: Exception that occurred

        Returns:
            True if should fallback to earlier phase
        """
        # GPU errors: fallback to CPU (Phase 3 → Phase 2 → Phase 1)
        gpu_errors = (
            "CUDA out of memory",
            "out of memory",
            "GPU",
            "torch.cuda",
            "MPS",
        )

        if any(msg in str(error) for msg in gpu_errors):
            return current_phase > 1

        # Timeout: try earlier phase
        if "timeout" in str(error).lower():
            return current_phase > 1

        # Missing dependency: fallback
        dependency_errors = (
            "No module named",
            "import",
            "YOLO",
            "ultralytics",
        )
        if any(msg in str(error) for msg in dependency_errors):
            return current_phase == 3  # Can fallback from Phase 3 to Phase 2

        return False

    @staticmethod
    def get_fallback_phase(current_phase: int) -> int:
        """Get the phase to fallback to.

        Args:
            current_phase: Current phase (1, 2, or 3)

        Returns:
            Next lower phase to try (or 1 if at minimum)
        """
        if current_phase > 1:
            return current_phase - 1
        return 1

    @staticmethod
    def get_fallback_reason(phase: int, error: Exception) -> str:
        """Get human-readable reason for fallback.

        Args:
            phase: Phase we're falling back from
            error: Exception that caused fallback

        Returns:
            String explaining the fallback
        """
        error_str = str(error).lower()

        if "cuda" in error_str or "gpu" in error_str or "out of memory" in error_str:
            return f"Phase {phase} GPU error: {str(error)[:100]}"

        if "timeout" in error_str:
            return f"Phase {phase} timeout: operation took too long"

        if "module" in error_str or "import" in error_str:
            return f"Phase {phase} missing dependency: {str(error)[:100]}"

        return f"Phase {phase} failed: {str(error)[:100]}"


def timeout_handler(timeout_seconds: int):
    """Decorator to add timeout to function.

    Args:
        timeout_seconds: Max seconds to run
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def timeout_handler_signal(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

            # Set alarm (UNIX only, doesn't work on Windows)
            try:
                signal.signal(signal.SIGALRM, timeout_handler_signal)
                signal.alarm(timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)  # Cancel alarm
                return result
            except ValueError:
                # signal.alarm not supported on this platform (Windows)
                # Just run without timeout
                return func(*args, **kwargs)

        return wrapper
    return decorator


class GracefulDegradation:
    """Wrapper for operations that can degrade gracefully."""

    def __init__(self, verbose: bool = False):
        """Initialize degradation handler.

        Args:
            verbose: Print degradation messages
        """
        self.verbose = verbose
        self.fallbacks_used = []

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"⚠️  {message}")

    def handle_gpu_error(self, error: Exception, context: str = "") -> Optional[str]:
        """Handle GPU-related error.

        Args:
            error: The GPU error
            context: Context about what was running

        Returns:
            Message about fallback, or None if can't handle
        """
        error_str = str(error).lower()

        # OOM errors: suggest fallback
        if "out of memory" in error_str:
            self.log(f"GPU out of memory {context}")
            self.log("Suggestions:")
            self.log("  1. Reduce batch size: diveanalyzer process video.mov --batch-size=8")
            self.log("  2. Use FP32 instead of FP16: remove --use-fp16 flag")
            self.log("  3. Fall back to CPU: remove --use-gpu flag")
            self.fallbacks_used.append("gpu_oom")
            return "GPU OOM - reduce batch size or use CPU"

        # CUDA not found
        if "cuda" in error_str:
            self.log(f"CUDA error {context}")
            self.log("Falling back to CPU inference (slower)")
            self.fallbacks_used.append("no_cuda")
            return "CUDA not available - using CPU"

        # Metal errors (Apple Silicon)
        if "mps" in error_str or "metal" in error_str:
            self.log(f"Metal error {context}")
            self.log("Falling back to CPU inference (slower)")
            self.fallbacks_used.append("no_metal")
            return "Metal not available - using CPU"

        return None

    def handle_timeout_error(self, timeout_seconds: int, context: str = "") -> str:
        """Handle timeout error.

        Args:
            timeout_seconds: How long operation took
            context: Context about what was running

        Returns:
            Message about what to do
        """
        self.log(f"Operation timeout after {timeout_seconds}s {context}")
        self.log("Try:")
        self.log("  1. Reduce processing resolution: --proxy-height=360")
        self.log("  2. Disable Phase 3: --force-phase=2")
        self.log("  3. Use CPU inference: remove --use-gpu")
        self.fallbacks_used.append("timeout")
        return f"Timeout after {timeout_seconds}s - try simpler options"

    def handle_missing_dependency(self, dependency: str, context: str = "") -> str:
        """Handle missing dependency error.

        Args:
            dependency: Name of missing dependency
            context: Context about what was running

        Returns:
            Message about installation
        """
        self.log(f"Missing dependency: {dependency} {context}")

        install_commands = {
            "librosa": "pip install librosa soundfile",
            "ultralytics": "pip install ultralytics",
            "opencv": "pip install opencv-python",
            "torch": "pip install torch",
            "scipy": "pip install scipy",
        }

        if dependency in install_commands:
            self.log(f"Install with: {install_commands[dependency]}")

        self.fallbacks_used.append(f"missing_{dependency}")
        return f"Missing {dependency} - install with pip"

    def get_summary(self) -> str:
        """Get summary of fallbacks used."""
        if not self.fallbacks_used:
            return "No fallbacks needed"

        unique = list(set(self.fallbacks_used))
        return f"Used {len(unique)} fallback(s): {', '.join(unique)}"


class ErrorContext:
    """Context manager for detailed error reporting."""

    def __init__(self, operation: str, verbose: bool = False):
        """Initialize error context.

        Args:
            operation: Name of operation (e.g., "GPU inference")
            verbose: Print context on error
        """
        self.operation = operation
        self.verbose = verbose
        self.start_time = None

    def __enter__(self):
        """Enter context."""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle errors."""
        import time

        if exc_type is None:
            return

        elapsed = time.time() - self.start_time

        if self.verbose:
            print(f"\n❌ {self.operation} failed after {elapsed:.1f}s")
            print(f"   Error: {exc_type.__name__}: {exc_val}")

            # Provide suggestions
            error_str = str(exc_val).lower()

            if "cuda" in error_str or "gpu" in error_str:
                print("   💡 Try removing --use-gpu flag to use CPU")

            if "out of memory" in error_str:
                print("   💡 Try reducing --batch-size (e.g., 8 or 4)")

            if "timeout" in error_str:
                print("   💡 Try reducing resolution or enabling Phase 2 only")

        return False  # Don't suppress exception


def suggest_fixes_for_error(error: Exception) -> list:
    """Suggest fixes for a given error.

    Args:
        error: The exception

    Returns:
        List of suggested fixes
    """
    error_str = str(error).lower()
    fixes = []

    # GPU errors
    if "cuda out of memory" in error_str:
        fixes.append("Reduce batch size: --batch-size=8")
        fixes.append("Use FP32: remove --use-fp16")
        fixes.append("Use CPU: remove --use-gpu")
        fixes.append("Process shorter videos first")

    elif "cuda" in error_str or "not available" in error_str:
        fixes.append("Ensure GPU drivers are installed")
        fixes.append("Try CPU mode: remove --use-gpu")

    # Timeout errors
    elif "timeout" in error_str:
        fixes.append("Reduce resolution: --proxy-height=360")
        fixes.append("Use Phase 2 only: --force-phase=2")
        fixes.append("Use CPU: remove --use-gpu")

    # Memory errors
    elif "memory" in error_str:
        fixes.append("Close other applications")
        fixes.append("Restart the system")
        fixes.append("Process smaller videos")

    # Missing dependencies
    elif "module" in error_str or "import" in error_str:
        module = error_str.split("'")[-2] if "'" in error_str else "unknown"
        fixes.append(f"Install: pip install {module}")

    # Default
    if not fixes:
        fixes.append("Check logs for details")
        fixes.append("Try simpler settings (Phase 1 or 2)")

    return fixes
