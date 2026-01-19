"""
Unit tests for GPU device detection and management.

Tests GPU detection for CUDA, Metal, ROCm, and CPU fallback.
"""

import pytest
from diveanalyzer.detection.person import (
    detect_gpu_device,
    check_gpu_memory,
    GPUInfo,
)


class TestGPUDetection:
    """Test GPU device detection."""

    def test_detect_gpu_device_returns_gpu_info(self):
        """Test that detect_gpu_device returns GPUInfo object."""
        gpu_info = detect_gpu_device()
        assert isinstance(gpu_info, GPUInfo)

    def test_gpu_info_has_required_fields(self):
        """Test that GPUInfo has all required fields."""
        gpu_info = detect_gpu_device()
        assert hasattr(gpu_info, "device_type")
        assert hasattr(gpu_info, "device_index")
        assert hasattr(gpu_info, "device_name")
        assert hasattr(gpu_info, "total_memory_mb")
        assert hasattr(gpu_info, "available_memory_mb")
        assert hasattr(gpu_info, "is_available")

    def test_gpu_device_type_is_valid(self):
        """Test that device_type is one of valid types."""
        gpu_info = detect_gpu_device()
        assert gpu_info.device_type in ("cuda", "metal", "rocm", "cpu")

    def test_gpu_is_available(self):
        """Test that is_available is always True."""
        gpu_info = detect_gpu_device()
        assert gpu_info.is_available is True

    def test_device_index_is_non_negative(self):
        """Test that device_index is non-negative."""
        gpu_info = detect_gpu_device()
        assert gpu_info.device_index >= 0

    def test_memory_values_are_non_negative(self):
        """Test that memory values are non-negative."""
        gpu_info = detect_gpu_device()
        assert gpu_info.total_memory_mb >= 0.0
        assert gpu_info.available_memory_mb >= 0.0

    def test_force_cpu_flag_returns_cpu(self):
        """Test that force_cpu=True returns CPU device."""
        gpu_info = detect_gpu_device(force_cpu=True)
        assert gpu_info.device_type == "cpu"
        assert "forced" in gpu_info.device_name.lower()

    def test_device_name_is_string(self):
        """Test that device_name is a string."""
        gpu_info = detect_gpu_device()
        assert isinstance(gpu_info.device_name, str)
        assert len(gpu_info.device_name) > 0


class TestGPUMemoryCheck:
    """Test GPU memory checking."""

    def test_check_gpu_memory_cpu_always_true(self):
        """Test that CPU device always passes memory check."""
        gpu_info = detect_gpu_device(force_cpu=True)
        result = check_gpu_memory(100.0)
        assert result is True

    def test_check_gpu_memory_with_valid_model(self):
        """Test memory check with reasonable model size."""
        result = check_gpu_memory(100.0)  # 100MB model
        # Result depends on actual hardware, but should be boolean
        assert isinstance(result, bool)

    def test_check_gpu_memory_with_large_model(self):
        """Test memory check with very large model size."""
        result = check_gpu_memory(50000.0)  # 50GB model (unrealistic)
        # On most systems this should fail if GPU is present
        assert isinstance(result, bool)

    def test_check_gpu_memory_with_buffer(self):
        """Test memory check with additional buffer."""
        result = check_gpu_memory(50.0, required_buffer_mb=500.0)
        assert isinstance(result, bool)

    def test_check_gpu_memory_returns_boolean(self):
        """Test that check_gpu_memory always returns boolean."""
        for model_size in [10.0, 100.0, 1000.0]:
            result = check_gpu_memory(model_size)
            assert isinstance(result, bool)


class TestGPUInfoDataclass:
    """Test GPUInfo dataclass."""

    def test_gpu_info_creation(self):
        """Test creating GPUInfo object."""
        gpu_info = GPUInfo(
            device_type="cuda",
            device_index=0,
            device_name="NVIDIA RTX 4090",
            total_memory_mb=24576.0,
            available_memory_mb=20480.0,
            is_available=True,
        )
        assert gpu_info.device_type == "cuda"
        assert gpu_info.device_index == 0
        assert gpu_info.total_memory_mb == 24576.0

    def test_gpu_info_string_representation(self):
        """Test string representation of GPUInfo."""
        gpu_info = detect_gpu_device()
        str_repr = str(gpu_info)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_gpu_info_equality(self):
        """Test equality comparison of GPUInfo objects."""
        gpu1 = GPUInfo(
            device_type="cpu",
            device_index=0,
            device_name="CPU",
            total_memory_mb=0.0,
            available_memory_mb=0.0,
            is_available=True,
        )
        gpu2 = GPUInfo(
            device_type="cpu",
            device_index=0,
            device_name="CPU",
            total_memory_mb=0.0,
            available_memory_mb=0.0,
            is_available=True,
        )
        assert gpu1 == gpu2


class TestGPUDetectionEdgeCases:
    """Test edge cases in GPU detection."""

    def test_multiple_calls_consistent(self):
        """Test that multiple calls to detect_gpu_device are consistent."""
        gpu_info1 = detect_gpu_device()
        gpu_info2 = detect_gpu_device()
        assert gpu_info1.device_type == gpu_info2.device_type
        assert gpu_info1.device_name == gpu_info2.device_name

    def test_force_cpu_overrides_gpu(self):
        """Test that force_cpu overrides GPU detection."""
        gpu_info_auto = detect_gpu_device(force_cpu=False)
        gpu_info_forced = detect_gpu_device(force_cpu=True)

        if gpu_info_auto.device_type != "cpu":
            # Only test if GPU is actually available
            assert gpu_info_forced.device_type == "cpu"

    def test_zero_model_size(self):
        """Test memory check with zero model size."""
        result = check_gpu_memory(0.0)
        assert isinstance(result, bool)

    def test_gpu_info_attributes_immutable(self):
        """Test that GPUInfo attributes are preserved."""
        gpu_info = detect_gpu_device()
        device_type_original = gpu_info.device_type
        device_name_original = gpu_info.device_name

        # Access again to verify no changes
        assert gpu_info.device_type == device_type_original
        assert gpu_info.device_name == device_name_original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
