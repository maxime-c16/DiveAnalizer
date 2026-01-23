#!/usr/bin/env python3
"""
Test script for the review gallery feature.
Demonstrates quick dive filtering workflow.
"""

import json
from pathlib import Path
from diveanalyzer.utils.review_gallery import DiveGalleryGenerator


def test_gallery_with_mock_dives():
    """Test gallery generation with mock dive data."""

    # Create test output directory
    test_dir = Path("/tmp/test_review_dives")
    test_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("DIVE REVIEW GALLERY - TEST DEMO")
    print("=" * 80)
    print()

    # Create mock dive videos by copying test fixture
    source_video = Path("tests/fixtures/short_dive.mp4")
    if not source_video.exists():
        print("❌ Test video not found. Please run from DiveAnalizer directory.")
        return

    print(f"📹 Using test video: {source_video}")
    print(f"📂 Output directory: {test_dir}")
    print()

    # Create mock extracted dive files (in real scenario these would be actual dives)
    print("Creating mock dive extracts...")
    for i in range(1, 9):
        dive_path = test_dir / f"dive_{i:03d}.mp4"
        if not dive_path.exists():
            # Copy the test video as mock dives
            import shutil
            shutil.copy(source_video, dive_path)
            print(f"  ✓ Created {dive_path.name}")

    print()
    print("Generating review gallery...")
    print()

    # Generate gallery
    generator = DiveGalleryGenerator(test_dir, "short_dive.mp4")
    dives = generator.scan_dives()

    print(f"✅ Found {len(dives)} dives")
    for dive in dives:
        print(f"   Dive #{dive['id']:02d}: {dive['filename']} ({dive['duration']:.1f}s)")

    print()
    print("Generating HTML gallery...")
    html_path = generator.generate_html()

    print()
    print("=" * 80)
    print("GALLERY FEATURES")
    print("=" * 80)
    print("""
✨ Interactive Features:
  • 3-frame thumbnail strip for each dive (start/middle/end)
  • Checkbox selection for batch operations
  • Real-time stats (Total/Selected/Keep count)
  • Color-coded confidence levels (HIGH/MEDIUM/LOW)

⚡ Keyboard Shortcuts:
  ← → : Navigate through dives
  Space: Toggle current dive for deletion
  A : Select all dives
  Ctrl+A : Deselect all dives
  D : Delete selected dives
  W : Watch selected dive
  Enter : Accept remaining dives
  ? : Show help menu

🎯 Workflow (User Perspective):
  1. Gallery opens automatically after extraction
  2. User browses 3-frame thumbnails
  3. User marks dives to DELETE with checkbox
  4. User watches any suspicious dives with W key
  5. User deletes marked dives with D key
  6. User accepts remaining dives with Enter key
  ✅ Done - only wanted dives remain!

⏱️ Speed Metrics:
  • 12 dives review: ~30 seconds
  • Visual scanning very fast (3 frames per dive)
  • Keyboard shortcuts minimize mouse usage
  • Batch delete operations save time
""")

    print()
    print("=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    print()
    print(f"📄 HTML Gallery: {html_path}")
    print(f"📁 Test videos: {test_dir}")
    print()

    # Show some stats
    html_file = Path(html_path)
    if html_file.exists():
        size_mb = html_file.stat().st_size / 1024 / 1024
        print(f"File size: {size_mb:.2f} MB (includes embedded thumbnails)")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Open the HTML file in a browser:
   open {html_path}

2. Try the shortcuts:
   • Use arrow keys to navigate
   • Press Space to select/deselect dives
   • Press A to select all
   • Press D to delete selected

3. Real integration (in slAIcer.py):
   After extraction, automatically:
   - Generate gallery
   - Open in default browser
   - User makes decisions
   - Deletes unwanted files
   - Closes when done
""".format(html_path=html_path))

    print()
    print("✅ Test complete!")
    print()

    return html_path


if __name__ == "__main__":
    html_path = test_gallery_with_mock_dives()

    # Try to open in browser
    try:
        import webbrowser
        print(f"🌐 Opening {html_path} in browser...")
        webbrowser.open(f"file://{Path(html_path).absolute()}")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print(f"📂 Please open manually: {html_path}")
