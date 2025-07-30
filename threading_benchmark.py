#!/usr/bin/env python3
"""
Performance benchmark comparing threaded vs non-threaded dive detection.

This script tests the performance improvements achieved through threading
in the dive analysis pipeline.
"""

import time
import cv2
import argparse
import slAIcer

def setup_detection_zones(video_path):
    """
    Interactive setup of detection zones (same as slAIcer.py main function).

    Returns:
        (board_y_norm, water_y_norm, splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm)
    """
    print("Setting up detection zones for benchmark...")

    # Get the first frame for zone selection
    frame_gen = slAIcer.frame_generator(video_path)
    _, first_frame = next(frame_gen)

    # Step 1: user specifies the board line
    print("Please specify the diving board position.")
    board_y_px = slAIcer.get_board_y_px(first_frame)
    if board_y_px is None:
        print("No board line selected. Using default position.")
        board_y_px = int(0.3 * first_frame.shape[0])
    else:
        print(f"✓ Board line selected at y={board_y_px}px")

    # Step 2: user specifies diver detection zone
    print("Please specify the diver detection zone.")
    diver_zone_coords = slAIcer.get_diver_detection_zone(first_frame)
    if diver_zone_coords is None:
        print("No diver detection zone selected. Using full frame detection.")
        diver_zone_norm = None
    else:
        left, top, right, bottom = diver_zone_coords
        h, w = first_frame.shape[:2]
        diver_zone_norm = (left/w, top/h, right/w, bottom/h)
        zone_width = right - left
        zone_height = bottom - top
        print(f"✓ Diver detection zone selected: ({left},{top}) to ({right},{bottom})")
        print(f"  Zone size: {zone_width}x{zone_height}px")
        print(f"  Normalized coordinates: ({diver_zone_norm[0]:.3f},{diver_zone_norm[1]:.3f}) to ({diver_zone_norm[2]:.3f},{diver_zone_norm[3]:.3f})")

    # Step 3: user specifies splash zone
    print("Please specify the splash detection zone.")
    splash_top_px, splash_bottom_px = slAIcer.get_splash_zone(first_frame)
    if splash_top_px is None or splash_bottom_px is None:
        print("No splash zone selected. Using default zone.")
        splash_zone_top_norm = None
        splash_zone_bottom_norm = None
    else:
        splash_zone_top_norm = splash_top_px / first_frame.shape[0]
        splash_zone_bottom_norm = splash_bottom_px / first_frame.shape[0]
        zone_height = splash_bottom_px - splash_top_px
        print(f"✓ Splash zone selected: {splash_top_px} to {splash_bottom_px} pixels (height: {zone_height}px)")
        print(f"  Normalized coordinates: {splash_zone_top_norm:.3f} to {splash_zone_bottom_norm:.3f}")

    # Convert to normalized coordinates
    board_y_norm = board_y_px / first_frame.shape[0]
    water_y_norm = 0.95  # bottom 5% is water

    return board_y_norm, water_y_norm, splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm

def benchmark_detection(video_path, board_y_norm, water_y_norm, splash_zone_top_norm, splash_zone_bottom_norm,
                       diver_zone_norm, use_threading=True, max_frames=1000):
    """
    Benchmark dive detection performance with provided zone parameters.

    Args:
        video_path: Path to test video
        board_y_norm, water_y_norm, splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm: Detection zones
        use_threading: Whether to use threaded processing
        max_frames: Maximum frames to process for testing

    Returns:
        (processing_time, fps_achieved)
    """
    print(f"🔬 Benchmarking {'threaded' if use_threading else 'sequential'} processing...")

    # Get video info
    video_fps = slAIcer.get_video_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    test_frames = min(max_frames, total_frames)
    print(f"  Testing with {test_frames} frames from video ({video_fps:.1f}fps)")

    start_time = time.time()

    try:
        # Create frame generator with limited frames
        frame_gen = slAIcer.frame_generator(video_path)
        limited_frames = []
        for i, (idx, frame) in enumerate(frame_gen):
            if i >= test_frames:
                break
            limited_frames.append((idx, frame))

        # Convert back to generator
        def limited_frame_gen():
            for frame_data in limited_frames:
                yield frame_data

        # Run detection
        start_idx, end_idx, confidence = slAIcer.find_next_dive(
            limited_frame_gen(),
            board_y_norm=board_y_norm,
            water_y_norm=water_y_norm,
            splash_zone_top_norm=splash_zone_top_norm,
            splash_zone_bottom_norm=splash_zone_bottom_norm,
            diver_zone_norm=diver_zone_norm,
            debug=False,
            splash_method='motion_intensity',
            video_fps=video_fps,
            use_threading=use_threading
        )

        processing_time = time.time() - start_time
        fps_achieved = test_frames / processing_time

        print(f"  Processed {test_frames} frames in {processing_time:.2f}s")
        print(f"  Achieved {fps_achieved:.1f} fps (target: {video_fps:.1f} fps)")
        print(f"  Performance ratio: {fps_achieved/video_fps:.2f}x real-time")

        return processing_time, fps_achieved

    except Exception as e:
        print(f"  ❌ Error during benchmark: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Benchmark threading performance improvements")
    parser.add_argument("video_path", help="Path to test video file")
    parser.add_argument("--frames", type=int, default=500, help="Max frames to test (default: 500)")
    parser.add_argument("--skip-setup", action="store_true", help="Skip zone selection and use default zones")
    args = parser.parse_args()

    print("🚀 Threading Performance Benchmark")
    print("=" * 50)

    # Setup detection zones interactively (unless skipped)
    if args.skip_setup:
        print("⚡ Using default detection zones (skipping interactive setup)")
        # Use default mock parameters
        board_y_norm = 0.3
        water_y_norm = 0.95
        splash_zone_top_norm = 0.85
        splash_zone_bottom_norm = 0.95
        diver_zone_norm = (0.3, 0.1, 0.7, 0.6)
    else:
        print("🎯 Interactive zone setup (same as slAIcer.py)")
        board_y_norm, water_y_norm, splash_zone_top_norm, splash_zone_bottom_norm, diver_zone_norm = setup_detection_zones(args.video_path)

    print("\n📋 Detection zones configured:")
    print(f"  Board Y: {board_y_norm:.3f}")
    print(f"  Water Y: {water_y_norm:.3f}")
    if splash_zone_top_norm and splash_zone_bottom_norm:
        print(f"  Splash zone: {splash_zone_top_norm:.3f} to {splash_zone_bottom_norm:.3f}")
    else:
        print("  Splash zone: Default")
    if diver_zone_norm:
        print(f"  Diver zone: ({diver_zone_norm[0]:.3f},{diver_zone_norm[1]:.3f}) to ({diver_zone_norm[2]:.3f},{diver_zone_norm[3]:.3f})")
    else:
        print("  Diver zone: Full frame")

    # Test threaded processing FIRST to catch errors quickly
    print("\n📊 Testing Threaded Processing (checking for errors first):")
    thread_time, thread_fps = benchmark_detection(
        args.video_path, board_y_norm, water_y_norm, splash_zone_top_norm,
        splash_zone_bottom_norm, diver_zone_norm, use_threading=True, max_frames=args.frames
    )

    # Test sequential processing
    print("\n📊 Testing Sequential Processing:")
    seq_time, seq_fps = benchmark_detection(
        args.video_path, board_y_norm, water_y_norm, splash_zone_top_norm,
        splash_zone_bottom_norm, diver_zone_norm, use_threading=False, max_frames=args.frames
    )

    # Calculate improvements
    if seq_time and thread_time:
        print("\n🎯 Performance Comparison:")
        print("-" * 30)
        speedup = seq_time / thread_time
        fps_improvement = thread_fps / seq_fps
        time_saved = seq_time - thread_time

        print(f"Sequential:  {seq_time:.2f}s ({seq_fps:.1f} fps)")
        print(f"Threaded:    {thread_time:.2f}s ({thread_fps:.1f} fps)")
        print(f"Speedup:     {speedup:.2f}x faster")
        print(f"Time saved:  {time_saved:.2f}s ({time_saved/seq_time*100:.1f}%)")
        print(f"FPS boost:   {fps_improvement:.2f}x")

        if speedup > 1.2:
            print("✅ Threading provides significant performance improvement!")
        elif speedup > 1.05:
            print("👍 Threading provides modest performance improvement")
        else:
            print("⚠️  Threading overhead may not be beneficial for this scenario")
    else:
        print("❌ Could not complete performance comparison")

if __name__ == "__main__":
    main()
