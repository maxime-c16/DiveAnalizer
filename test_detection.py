#!/usr/bin/env python3
"""
Quick test to verify diver detection is working properly.
"""

import cv2
import slAIcer

def test_diver_detection(video_path, max_frames=200):
    """Test basic diver detection functionality"""

    print(f"🧪 Testing diver detection on {video_path}")
    print(f"Processing first {max_frames} frames...")

    # Use known working zones
    diver_zone = (0.131, 0.505, 0.206, 0.667)  # From recent successful test
    splash_zone_top = 0.721
    splash_zone_bottom = 0.789
    board_y = 0.593
    water_y = 0.950

    print(f"Using zones: diver={diver_zone}")

    # Test with optimization disabled first
    print("\n📊 Test 1: Without pose optimization")
    frame_gen = slAIcer.frame_generator(video_path)
    limited_gen = ((idx, frame) for idx, frame in enumerate(frame_gen) if idx < max_frames)

    start_idx, end_idx, confidence = slAIcer.find_next_dive(
        limited_gen, board_y, water_y,
        splash_zone_top_norm=splash_zone_top, splash_zone_bottom_norm=splash_zone_bottom,
        diver_zone_norm=diver_zone, debug=False, use_threading=True,
        enable_pose_optimization=False, video_fps=30.0
    )

    print(f"Result: start={start_idx}, end={end_idx}, confidence={confidence}")

    # Test with optimization enabled
    print("\n📊 Test 2: With pose optimization")
    frame_gen = slAIcer.frame_generator(video_path)
    limited_gen = ((idx, frame) for idx, frame in enumerate(frame_gen) if idx < max_frames)

    start_idx2, end_idx2, confidence2 = slAIcer.find_next_dive(
        limited_gen, board_y, water_y,
        splash_zone_top_norm=splash_zone_top, splash_zone_bottom_norm=splash_zone_bottom,
        diver_zone_norm=diver_zone, debug=False, use_threading=True,
        enable_pose_optimization=True, video_fps=30.0
    )

    print(f"Result: start={start_idx2}, end={end_idx2}, confidence={confidence2}")

    # Compare results
    print("\n🔍 Comparison:")
    if start_idx == start_idx2 and end_idx == end_idx2:
        print("✅ Both methods produced identical results!")
        return True
    else:
        print("❌ Results differ between optimized and non-optimized versions")
        print(f"  Non-optimized: ({start_idx}, {end_idx})")
        print(f"  Optimized:     ({start_idx2}, {end_idx2})")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        success = test_diver_detection(video_path)
        print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
    else:
        print("Usage: python3 test_detection.py <video_path>")
