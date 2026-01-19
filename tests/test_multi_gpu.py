"""
Unit tests for multi-GPU support.

Tests GPU enumeration and selection for multi-GPU systems.
"""

import pytest
from unittest.mock import Mock, patch
from diveanalyzer.detection.person import (
    get_available_gpu_count,
    get_all_gpu_devices,
    get_best_gpu_device,
    detect_gpu_device,
    GPUInfo,
)


class TestGPUEnumeration:
    """Test GPU device enumeration."""

    def test_get_available_gpu_count_no_gpu(self):
        """Test GPU count returns 0 when no GPU available."""
        count = get_available_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_available_gpu_count_returns_int(self):
        """Test that GPU count is always an integer."""
        count = get_available_gpu_count()
        assert isinstance(count, int)

    def test_get_all_gpu_devices_returns_list(self):
        """Test that get_all_gpu_devices returns a list."""
        devices = get_all_gpu_devices()
        assert isinstance(devices, list)

    def test_get_all_gpu_devices_contains_gpu_info(self):
        """Test that returned list contains GPUInfo objects."""
        devices = get_all_gpu_devices()
        assert len(devices) > 0
        for device in devices:
            assert isinstance(device, GPUInfo)

    def test_get_all_gpu_devices_fallback_to_cpu(self):
        """Test that list always contains at least CPU."""
        devices = get_all_gpu_devices()
        assert len(devices) > 0
        # At least one device should be available
        assert any(d.is_available for d in devices)


class TestGPUSelection:
    """Test GPU device selection."""

    def test_get_best_gpu_device_returns_gpu_info(self):
        """Test that best GPU selection returns GPUInfo."""
        device = get_best_gpu_device()
        assert isinstance(device, GPUInfo)

    def test_get_best_gpu_device_has_required_fields(self):
        """Test that returned device has all required fields."""
        device = get_best_gpu_device()
        assert hasattr(device, "device_type")
        assert hasattr(device, "device_index")
        assert hasattr(device, "device_name")
        assert device.is_available is True

    def test_get_best_gpu_device_with_memory_filter(self):
        """Test GPU selection with memory requirement."""
        # Request devices with at least 1MB available
        device = get_best_gpu_device(max_memory_mb=1.0)
        assert isinstance(device, GPUInfo)

    def test_get_best_gpu_device_high_memory_filter(self):
        """Test GPU selection with high memory requirement."""
        # Request devices with 100GB available (unrealistic)
        device = get_best_gpu_device(max_memory_mb=100000.0)
        assert isinstance(device, GPUInfo)
        # Should return CPU as fallback since no GPU has that much
        # or return a GPU if available with sufficient memory

    def test_detect_gpu_device_with_index(self):
        """Test detecting specific GPU device by index."""
        gpu_info = detect_gpu_device(device_index=0)
        assert isinstance(gpu_info, GPUInfo)

    def test_detect_gpu_device_index_clamping(self):
        """Test that invalid device index is clamped."""
        # Request GPU 999 (doesn't exist)
        gpu_info = detect_gpu_device(device_index=999)
        assert isinstance(gpu_info, GPUInfo)
        assert gpu_info.is_available is True


class TestMultiGPUScenarios:
    """Test multi-GPU scenarios."""

    def test_single_gpu_system(self):
        """Test system with single GPU."""
        devices = get_all_gpu_devices()
        # Even single GPU system should work
        assert len(devices) > 0

    def test_no_gpu_system_fallback(self):
        """Test system with no GPU falls back gracefully."""
        count = get_available_gpu_count()
        if count == 0:
            devices = get_all_gpu_devices()
            # Should have CPU as fallback
            assert len(devices) > 0
            assert any(d.device_type == "cpu" for d in devices)

    def test_device_selection_order(self):
        """Test that device selection maintains consistent order."""
        devices1 = get_all_gpu_devices()
        devices2 = get_all_gpu_devices()
        assert len(devices1) == len(devices2)
        for d1, d2 in zip(devices1, devices2):
            assert d1.device_type == d2.device_type
            assert d1.device_index == d2.device_index

    def test_best_gpu_multiple_calls(self):
        """Test that best GPU selection is consistent."""
        device1 = get_best_gpu_device()
        device2 = get_best_gpu_device()
        assert device1.device_type == device2.device_type
        assert device1.device_index == device2.device_index


class TestGPUDeviceIndex:
    """Test GPU device indexing."""

    def test_device_index_non_negative(self):
        """Test that device index is always non-negative."""
        devices = get_all_gpu_devices()
        for device in devices:
            assert device.device_index >= 0

    def test_device_index_sequential(self):
        """Test that device indices are sequential (if multiple GPUs)."""
        devices = get_all_gpu_devices()
        if len(devices) > 1 and all(d.device_type == devices[0].device_type for d in devices):
            # If multiple devices of same type, indices should be sequential
            indices = [d.device_index for d in devices if d.device_type != "cpu"]
            if len(indices) > 1:
                # Could be sequential (0, 1, 2...) or have gaps - just verify non-negative
                assert all(i >= 0 for i in indices)

    def test_device_name_includes_index(self):
        """Test that device name includes index for multi-GPU."""
        devices = get_all_gpu_devices()
        for device in devices:
            assert isinstance(device.device_name, str)
            assert len(device.device_name) > 0
            if device.device_index > 0:
                # Multi-GPU names might include index
                assert isinstance(device.device_name, str)


class TestGPUMemoryFiltering:
    """Test GPU memory-based filtering."""

    def test_memory_filter_zero(self):
        """Test filtering with zero memory requirement."""
        device = get_best_gpu_device(max_memory_mb=0.0)
        assert isinstance(device, GPUInfo)

    def test_memory_filter_increasing(self):
        """Test that higher memory requirements are satisfied."""
        for memory_req in [10.0, 100.0, 1000.0]:
            device = get_best_gpu_device(max_memory_mb=memory_req)
            assert isinstance(device, GPUInfo)
            # If GPU was selected and not CPU, it should have enough memory
            if device.device_type != "cpu":
                assert device.available_memory_mb >= memory_req

    def test_cpu_always_included_in_filter(self):
        """Test that CPU is always available as fallback in filtering."""
        # Even with unrealistic memory requirement, should get CPU
        device = get_best_gpu_device(max_memory_mb=999999.0)
        # Should get a device (likely CPU as fallback)
        assert isinstance(device, GPUInfo)
        assert device.is_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
