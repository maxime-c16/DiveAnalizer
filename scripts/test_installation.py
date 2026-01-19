#!/usr/bin/env python3
"""
Test installation script - Verify all dependencies and core functionality.

Run with: python scripts/test_installation.py
"""

import sys
import subprocess
from pathlib import Path


def check_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def test_imports():
    """Test if all required modules can be imported."""
    print("\n🔍 Testing Python imports...")

    modules = [
        ("numpy", "NumPy"),
        ("librosa", "librosa"),
        ("scipy", "SciPy"),
        ("click", "Click"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - {e}")
            all_ok = False

    return all_ok


def test_external_tools():
    """Test if external tools are available."""
    print("\n🔧 Testing external tools...")

    tools = [
        ("ffmpeg -version", "FFmpeg"),
        ("ffprobe -version", "ffprobe"),
    ]

    all_ok = True
    for cmd, name in tools:
        if check_command(cmd):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - Not found in PATH")
            all_ok = False

    return all_ok


def test_diveanalyzer_imports():
    """Test if DiveAnalyzer modules can be imported."""
    print("\n📦 Testing DiveAnalyzer modules...")

    modules = [
        ("diveanalyzer", "diveanalyzer"),
        ("diveanalyzer.config", "config"),
        ("diveanalyzer.detection.audio", "detection.audio"),
        ("diveanalyzer.detection.fusion", "detection.fusion"),
        ("diveanalyzer.extraction.ffmpeg", "extraction.ffmpeg"),
        ("diveanalyzer.cli", "cli"),
    ]

    all_ok = True
    for module_path, display_name in modules:
        try:
            __import__(module_path)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - {e}")
            all_ok = False

    return all_ok


def test_cli():
    """Test if CLI is available."""
    print("\n💻 Testing CLI...")

    try:
        result = subprocess.run(
            ["diveanalyzer", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            print("  ✓ diveanalyzer command available")
            return True
        else:
            print(f"  ✗ diveanalyzer command failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  ⚠ diveanalyzer command not found (install with: pip install -e .)")
        return False
    except Exception as e:
        print(f"  ✗ Error testing CLI: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DiveAnalyzer v2.0 - Installation Test")
    print("=" * 60)

    results = [
        ("Python imports", test_imports()),
        ("External tools", test_external_tools()),
        ("DiveAnalyzer modules", test_diveanalyzer_imports()),
        ("CLI", test_cli()),
    ]

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    all_pass = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")
        if not result:
            all_pass = False

    print("=" * 60)

    if all_pass:
        print("\n✅ All tests passed! Installation is ready.")
        print("\n📖 Next steps:")
        print("  1. Read README_V2.md for usage instructions")
        print("  2. Try: diveanalyzer process --help")
        print("  3. Process a video: diveanalyzer process video.mp4")
        return 0
    else:
        print("\n❌ Some tests failed. See above for details.")
        print("\n💡 To fix:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Install FFmpeg: brew install ffmpeg (or apt install ffmpeg)")
        print("  3. Install CLI: pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
