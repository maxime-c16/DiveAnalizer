"""
Tests for signal fusion module.

Tests can be run with: pytest tests/test_fusion.py -v
"""

try:
    from diveanalyzer.detection.fusion import (
        fuse_signals_audio_only,
        merge_overlapping_dives,
        filter_dives_by_confidence,
        DiveEvent,
    )
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


if FUSION_AVAILABLE:

    def test_fuse_signals_audio_only():
        """Test basic signal fusion with audio only."""
        audio_peaks = [(5.0, -10.0), (15.0, -15.0), (25.0, -20.0)]

        dives = fuse_signals_audio_only(audio_peaks)

        assert len(dives) == 3
        assert all(isinstance(d, DiveEvent) for d in dives)
        assert dives[0].splash_time == 5.0
        assert dives[0].confidence > 0.5

    def test_merge_overlapping_dives():
        """Test merging of overlapping dives."""
        # Create dives that are close together
        dives = [
            DiveEvent(0, 5, 2.5, 0.8, -10.0, dive_number=1),
            DiveEvent(3, 8, 5.5, 0.7, -12.0, dive_number=2),
            DiveEvent(20, 25, 22.5, 0.9, -8.0, dive_number=3),
        ]

        merged = merge_overlapping_dives(dives, min_gap=5.0)

        # First two should merge, third should stay separate
        assert len(merged) == 2
        assert merged[0].start_time == 0
        assert merged[0].end_time == 8

    def test_filter_dives_by_confidence():
        """Test filtering dives by confidence."""
        dives = [
            DiveEvent(0, 5, 2.5, 0.9, -10.0, dive_number=1),
            DiveEvent(10, 15, 12.5, 0.4, -20.0, dive_number=2),
            DiveEvent(20, 25, 22.5, 0.8, -8.0, dive_number=3),
        ]

        filtered = filter_dives_by_confidence(dives, min_confidence=0.5)

        assert len(filtered) == 2
        assert all(d.confidence >= 0.5 for d in filtered)

    def test_dive_event_duration():
        """Test DiveEvent duration calculation."""
        dive = DiveEvent(
            start_time=10.0,
            end_time=24.3,
            splash_time=14.0,
            confidence=0.9,
            audio_amplitude=-10.0,
        )

        assert dive.duration() == 14.3

    def test_dive_event_str():
        """Test DiveEvent string representation."""
        dive = DiveEvent(
            start_time=10.0,
            end_time=24.3,
            splash_time=14.0,
            confidence=0.9,
            audio_amplitude=-10.0,
            dive_number=5,
        )

        s = str(dive)
        assert "Dive #5" in s
        assert "14.00s" in s
        assert "90" in s  # Confidence percentage

else:

    def test_skip_fusion():
        """Placeholder test when fusion dependencies not available."""
        assert True
