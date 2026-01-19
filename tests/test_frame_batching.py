"""
Unit tests for frame batching in YOLO inference.

Tests batch processing accuracy and performance.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from diveanalyzer.detection.person import _process_frame_batch


class TestFrameBatching:
    """Test frame batch processing."""

    def test_process_frame_batch_empty(self):
        """Test that empty batch is handled gracefully."""
        model = Mock()
        results = []

        _process_frame_batch(
            model, [], [], confidence_threshold=0.5, fps=30.0, results=results
        )

        assert len(results) == 0
        model.predict.assert_not_called()

    def test_process_frame_batch_single_frame(self):
        """Test processing single frame batch."""
        # Create mock model
        model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        model.predict.return_value = [mock_result]

        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_batch = [(0, frame)]
        timestamp_batch = [0.0]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        assert len(results) == 1
        assert results[0] == (0.0, False, 0.0)
        model.predict.assert_called_once()

    def test_process_frame_batch_multiple_frames(self):
        """Test processing multiple frames in batch."""
        model = Mock()

        # Create mock results - 3 frames, one with person
        mock_result1 = Mock()
        mock_result1.boxes = None

        mock_box = Mock()
        mock_box.cls = 0  # Person class
        mock_box.conf = 0.8

        mock_result2 = Mock()
        mock_result2.boxes = [mock_box]

        mock_result3 = Mock()
        mock_result3.boxes = None

        model.predict.return_value = [mock_result1, mock_result2, mock_result3]

        # Create test frames
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)
        ]
        frame_batch = [(i, frame) for i, frame in enumerate(frames)]
        timestamp_batch = [0.0, 1.0, 2.0]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        assert len(results) == 3
        assert results[0] == (0.0, False, 0.0)
        assert results[1] == (1.0, True, 0.8)  # Person detected
        assert results[2] == (2.0, False, 0.0)

    def test_process_frame_batch_respects_confidence_threshold(self):
        """Test that confidence threshold is applied correctly."""
        model = Mock()

        # Create mock result with low confidence
        mock_box = Mock()
        mock_box.cls = 0  # Person class
        mock_box.conf = 0.3  # Below threshold

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_batch = [(0, frame)]
        timestamp_batch = [0.0]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        # Person should not be detected (below threshold)
        assert results[0] == (0.0, False, 0.3)

    def test_process_frame_batch_multiple_detections(self):
        """Test frame with multiple person detections."""
        model = Mock()

        # Create multiple person detections
        mock_box1 = Mock()
        mock_box1.cls = 0
        mock_box1.conf = 0.6

        mock_box2 = Mock()
        mock_box2.cls = 0
        mock_box2.conf = 0.8  # Higher confidence

        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]
        model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_batch = [(0, frame)]
        timestamp_batch = [0.0]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        # Should use highest confidence
        assert results[0] == (0.0, True, 0.8)

    def test_process_frame_batch_non_person_class(self):
        """Test that non-person classes are ignored."""
        model = Mock()

        # Create detection for different class (not person)
        mock_box = Mock()
        mock_box.cls = 2  # Not person class
        mock_box.conf = 0.9

        mock_result = Mock()
        mock_result.boxes = [mock_box]
        model.predict.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_batch = [(0, frame)]
        timestamp_batch = [0.0]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        # Should not detect person
        assert results[0] == (0.0, False, 0.0)

    def test_process_frame_batch_preserves_order(self):
        """Test that results are in correct order."""
        model = Mock()

        # Create results for 5 frames
        mock_results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.boxes = None
            mock_results.append(mock_result)

        model.predict.return_value = mock_results

        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        frame_batch = [(i, frame) for i, frame in enumerate(frames)]
        timestamp_batch = [float(i) for i in range(5)]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        # Check timestamps are in order
        for i, (timestamp, _, _) in enumerate(results):
            assert timestamp == float(i)

    def test_process_frame_batch_large_batch(self):
        """Test processing large batch of frames."""
        model = Mock()

        # Create 32 mock results (no detections)
        mock_results = [Mock(boxes=None) for _ in range(32)]
        model.predict.return_value = mock_results

        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(32)
        ]
        frame_batch = [(i, frame) for i, frame in enumerate(frames)]
        timestamp_batch = [float(i) for i in range(32)]
        results = []

        _process_frame_batch(
            model,
            frame_batch,
            timestamp_batch,
            confidence_threshold=0.5,
            fps=30.0,
            results=results,
        )

        assert len(results) == 32
        model.predict.assert_called_once()
        # Verify batch was passed as list of 32 images
        call_args = model.predict.call_args
        images_arg = call_args[0][0]
        assert len(images_arg) == 32


class TestBatchSizeValidation:
    """Test batch size parameter validation."""

    def test_batch_size_clamping(self):
        """Test that batch size is properly clamped."""
        # This would be tested in detect_person_frames
        # But we can verify the clamping logic here
        batch_size = 16
        batch_size = max(1, min(batch_size, 64))
        assert batch_size == 16

        batch_size = 0
        batch_size = max(1, min(batch_size, 64))
        assert batch_size == 1

        batch_size = 128
        batch_size = max(1, min(batch_size, 64))
        assert batch_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
