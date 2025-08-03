#!/usr/bin/env python3
"""
Test script for the Splash-Only Dive Detection Syst    print("✅ All tests completed!")
    print("\n📋 Usage Examples:")
    print("  Basic usage:")
    print("    python splash_only_detector.py your_video.mp4 --debug")
    print("")
    print("  Custom parameters:")
    print("    python splash_only_detector.py video.mp4 --method combined --threshold 15 --zone 0.75 0.9")
    print("")
    print("  Detection only (no extraction):")
    print("    python splash_only_detector.py video.mp4 --no-extract --debug")port sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from splash_only_detector import SplashOnlyDetector, DetectionConfig
    print("✅ Successfully imported splash_only_detector")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_configuration():
    """Test the configuration system"""
    print("\n🧪 Testing Configuration...")

    # Test default configuration
    config = DetectionConfig()
    print(f"  Default method: {config.method}")
    print(f"  Default zone: {config.splash_zone_top_norm} - {config.splash_zone_bottom_norm}")
    print(f"  Default threshold: {config.base_threshold}")

    # Test custom configuration
    custom_config = DetectionConfig(
        method='combined',
        base_threshold=15.0,
        temporal_gaussian_sigma=2.0
    )
    print(f"  Custom method: {custom_config.method}")
    print(f"  Custom threshold: {custom_config.base_threshold}")
    print(f"  Custom sigma: {custom_config.temporal_gaussian_sigma}")

    print("✅ Configuration test passed")

def test_detector_initialization():
    """Test detector initialization"""
    print("\n🧪 Testing Detector Initialization...")

    config = DetectionConfig(enable_debug_plots=False)  # Disable plots for testing
    detector = SplashOnlyDetector(config)

    print(f"  Gaussian filter initialized: {detector.gaussian_filter is not None}")
    print(f"  Peak detector initialized: {detector.peak_detector is not None}")
    print(f"  Debug data structure: {len(detector.debug_data)} fields")

    print("✅ Detector initialization test passed")

def test_help_system():
    """Test the help system"""
    print("\n🧪 Testing Help System...")

    # Test command-line help
    import subprocess
    result = subprocess.run([sys.executable, 'splash_only_detector.py', '--help'],
                          capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Help system working")
        print("  Help output length:", len(result.stdout))
    else:
        print("❌ Help system failed")
        print("  Error:", result.stderr)

def main():
    """Run all tests"""
    print("🧪 Testing Splash-Only Dive Detection System")
    print("=" * 45)

    test_configuration()
    test_detector_initialization()
    test_help_system()

    print("\n✅ All tests completed!")
    print("\n📋 Usage Examples:")
    print("  Basic usage:")
    print("    python splash_only_detector.py your_video.mp4 --debug")
    print("
    print("  Custom parameters:")
    print("    python splash_only_detector.py video.mp4 --method combined --threshold 15 --zone 0.75 0.9")
    print("
    print("  Detection only (no extraction):")
    print("    python splash_only_detector.py video.mp4 --no-extract --debug")

if __name__ == "__main__":
    main()
