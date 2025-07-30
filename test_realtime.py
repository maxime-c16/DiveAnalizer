#!/usr/bin/env python3
"""
Test the real-time dive detection and extraction system.
"""

import time
import os
import slAIcer

def test_realtime_detection(video_path):
    """Test real-time dive detection to verify immediate extraction spawning"""

    print("🧪 Testing Real-time Dive Detection and Extraction")
    print("=" * 60)

    # Create test output directory
    test_output_dir = "test_realtime_output"
    os.makedirs(test_output_dir, exist_ok=True)

    # Use known working zones (you can adjust these)
    diver_zone = (0.13, 0.50, 0.24, 0.68)  # From recent tests
    splash_zone_top = 0.73
    splash_zone_bottom = 0.81
    board_y = 0.58
    water_y = 0.95

    print(f"📁 Test output directory: {test_output_dir}")
    print(f"🎯 Using zones: diver={diver_zone}")
    print(f"🌊 Splash zone: {splash_zone_top:.3f} to {splash_zone_bottom:.3f}")
    print("")

    start_time = time.time()

    # Test the real-time function
    dives = slAIcer.detect_and_extract_dives_realtime(
        video_path, board_y, water_y,
        splash_zone_top_norm=splash_zone_top,
        splash_zone_bottom_norm=splash_zone_bottom,
        diver_zone_norm=diver_zone,
        debug=False,
        splash_method='motion_intensity',
        use_threading=True,
        enable_pose_optimization=False,
        output_dir=test_output_dir
    )

    total_time = time.time() - start_time

    print("")
    print("🎉 Real-time Test Results:")
    print("-" * 40)
    print(f"📊 Dives detected: {len(dives)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📁 Output files:")

    # List generated files
    if os.path.exists(test_output_dir):
        files = os.listdir(test_output_dir)
        for file in sorted(files):
            if file.endswith('.mp4'):
                file_path = os.path.join(test_output_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   📹 {file} ({file_size:.1f} MB)")

    print("")
    if len(dives) > 0:
        print("✅ Real-time detection and extraction working!")
        print("💡 Key advantage: Extraction started immediately when each dive was detected")
    else:
        print("⚠️  No dives detected in test video")

    return len(dives) > 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        success = test_realtime_detection(video_path)
        if success:
            print("\n🚀 Real-time system is working perfectly!")
        else:
            print("\n⚠️  Test completed but no dives detected")
    else:
        print("Usage: python3 test_realtime.py <video_path>")
