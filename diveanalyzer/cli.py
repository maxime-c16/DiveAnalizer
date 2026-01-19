"""
Command-line interface for DiveAnalyzer.

Main entry point for the application.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from . import __version__
from .config import get_config, DetectionConfig
from .detection.audio import extract_audio, detect_splash_peaks
from .detection.fusion import (
    fuse_signals_audio_only,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)
from .extraction.ffmpeg import extract_multiple_dives, get_video_duration, format_time


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    DiveAnalyzer - Automatic diving video clip extraction.

    Extract individual dive clips from full session videos using audio detection.
    """
    pass


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./dives",
    help="Output directory for extracted clips",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=-25.0,
    help="Audio threshold in dB for splash detection",
)
@click.option(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="Minimum confidence (0-1) to extract dive",
)
@click.option(
    "--no-audio",
    is_flag=True,
    default=False,
    help="Don't include audio in extracted clips",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output",
)
def process(
    video_path: str,
    output: str,
    threshold: float,
    confidence: float,
    no_audio: bool,
    verbose: bool,
):
    """
    Process a video and extract all detected dives.

    VIDEO_PATH: Path to input video file (or iCloud folder)

    Example:
        diveanalyzer process session.mp4 -o ./dives
        diveanalyzer process ~/iCloud\\ Drive/Diving/ -c 0.7
    """
    try:
        video_path = str(Path(video_path).resolve())

        if not Path(video_path).exists():
            click.echo(f"❌ Video not found: {video_path}", err=True)
            sys.exit(1)

        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"🎬 DiveAnalyzer v{__version__}")
        click.echo(f"📹 Input: {video_path}")
        click.echo(f"📁 Output: {output_dir}")
        click.echo()

        # Get video info
        try:
            duration = get_video_duration(video_path)
            click.echo(f"⏱️  Video duration: {format_time(duration)}")
        except Exception as e:
            click.echo(f"⚠️  Could not determine video length: {e}")
            duration = None

        # Phase 1: Extract audio
        click.echo("\n🔊 Extracting audio track...")
        try:
            audio_path = extract_audio(video_path)
            click.echo(f"✓ Audio extracted to: {audio_path}")
        except Exception as e:
            click.echo(f"❌ Failed to extract audio: {e}", err=True)
            sys.exit(1)

        # Phase 2: Detect splash peaks
        click.echo("\n🌊 Detecting splashes...")
        try:
            peaks = detect_splash_peaks(
                audio_path,
                threshold_db=threshold,
                min_distance_sec=5.0,
                prominence=5.0,
            )
            click.echo(f"✓ Found {len(peaks)} potential splashes")

            if verbose:
                for i, (time, amp) in enumerate(peaks, 1):
                    click.echo(f"  {i}. {time:7.2f}s (amplitude {amp:6.1f}dB)")
        except Exception as e:
            click.echo(f"❌ Failed to detect splashes: {e}", err=True)
            sys.exit(1)

        # Phase 3: Fuse signals (Phase 1: audio only)
        click.echo("\n🔗 Fusing detection signals...")
        try:
            dives = fuse_signals_audio_only(peaks)
            click.echo(f"✓ Created {len(dives)} dive events")

            # Merge overlapping dives
            dives_merged = merge_overlapping_dives(dives)
            if len(dives_merged) < len(dives):
                click.echo(f"✓ Merged overlapping dives: {len(dives)} → {len(dives_merged)}")
                dives = dives_merged

            # Filter by confidence
            dives_filtered = filter_dives_by_confidence(dives, min_confidence=confidence)
            if len(dives_filtered) < len(dives):
                filtered_out = len(dives) - len(dives_filtered)
                click.echo(f"⊘ Filtered {filtered_out} low-confidence dives")
                dives = dives_filtered

            click.echo(f"✓ Final dive count: {len(dives)}")
        except Exception as e:
            click.echo(f"❌ Failed to fuse signals: {e}", err=True)
            sys.exit(1)

        if not dives:
            click.echo("\n⚠️  No dives detected. Try adjusting --threshold or --confidence")
            sys.exit(0)

        # Phase 4: Extract clips
        click.echo(f"\n✂️  Extracting {len(dives)} dive clips...")
        try:
            results = extract_multiple_dives(
                video_path,
                dives,
                output_dir,
                audio_enabled=not no_audio,
                verbose=verbose,
            )

            success_count = sum(1 for s, _, _ in results.values() if s)
            click.echo(f"✓ Successfully extracted {success_count}/{len(dives)} clips")

            # Show summary
            click.echo("\n📊 Summary:")
            click.echo(f"  Total dives: {len(dives)}")
            click.echo(f"  Extracted: {success_count}")

            if success_count < len(dives):
                click.echo(f"  Failed: {len(dives) - success_count}")

            click.echo(f"  Output folder: {output_dir}")

            for dive_num in sorted(results.keys()):
                success, path, error = results[dive_num]
                if success:
                    size_info = ""
                    if Path(path).exists():
                        size_mb = Path(path).stat().st_size / (1024 * 1024)
                        size_info = f" ({size_mb:.1f}MB)"
                    click.echo(f"  ✓ dive_{dive_num:03d}.mp4{size_info}")
                else:
                    click.echo(f"  ✗ dive_{dive_num:03d}.mp4 - {error}")

        except Exception as e:
            click.echo(f"❌ Failed to extract clips: {e}", err=True)
            sys.exit(1)

        click.echo("\n✅ Done!")

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=-25.0,
    help="Audio threshold in dB",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Verbose output",
)
def detect(video_path: str, threshold: float, verbose: bool):
    """
    Detect dives without extracting (dry run).

    Useful for tuning detection parameters.
    """
    try:
        video_path = str(Path(video_path).resolve())

        click.echo(f"🔍 Detecting dives in: {video_path}")

        # Extract audio
        click.echo("Extracting audio...")
        audio_path = extract_audio(video_path)

        # Detect peaks
        click.echo(f"Detecting splashes (threshold: {threshold}dB)...")
        peaks = detect_splash_peaks(audio_path, threshold_db=threshold)

        click.echo(f"\nFound {len(peaks)} potential splashes:\n")

        for i, (time, amp) in enumerate(peaks, 1):
            click.echo(
                f"  {i:3d}. Time: {time:7.2f}s  |  Amplitude: {amp:6.1f}dB  |  Confidence: {(amp+40)/40*100:5.1f}%"
            )

        click.echo(f"\nTotal: {len(peaks)} splashes detected")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=-25.0,
    help="Audio threshold in dB",
)
def analyze_audio(audio_path: str, threshold: float):
    """
    Analyze audio file for splashes.

    Useful for debugging audio-based detection.
    """
    try:
        from .detection.audio import get_audio_properties

        audio_path = str(Path(audio_path).resolve())

        # Get audio info
        props = get_audio_properties(audio_path)
        click.echo(f"📊 Audio Properties:")
        click.echo(f"  Duration: {format_time(props['duration'])}")
        click.echo(f"  Sample rate: {props['sample_rate']} Hz")
        click.echo(f"  Channels: {props['channels']}")
        click.echo(f"  Total samples: {props['total_samples']:,}")

        # Detect peaks
        click.echo(f"\n🌊 Detecting splashes (threshold: {threshold}dB)...")
        peaks = detect_splash_peaks(audio_path, threshold_db=threshold)

        click.echo(f"Found {len(peaks)} peaks:\n")
        for i, (time, amp) in enumerate(peaks, 1):
            click.echo(f"  {i:3d}. {time:7.2f}s  →  {amp:6.1f}dB")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
