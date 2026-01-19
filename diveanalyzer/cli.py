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
from .detection.fusion import (
    fuse_signals_audio_only,
    fuse_signals_audio_motion,
    fuse_signals_audio_motion_person,
    merge_overlapping_dives,
    filter_dives_by_confidence,
)
from .detection.motion import detect_motion_bursts
from .extraction.ffmpeg import extract_multiple_dives, get_video_duration, format_time
from .extraction.proxy import generate_proxy, is_proxy_generation_needed
from .storage.cache import CacheManager
from .storage.icloud import find_icloud_videos

# Lazy imports for audio (requires librosa)
def get_audio_functions():
    """Import audio functions on-demand to handle librosa dependency gracefully."""
    try:
        from .detection.audio import extract_audio, detect_splash_peaks
        return extract_audio, detect_splash_peaks
    except ImportError:
        raise ImportError(
            "librosa not installed. Install with: pip install librosa soundfile"
        )


def get_or_generate_proxy(video_path: str, proxy_height: int = 480, enable_cache: bool = True, verbose: bool = False):
    """Get or generate 480p proxy for motion/person detection.

    Returns the proxy path, or original video path if proxy not needed.

    Args:
        video_path: Path to source video
        proxy_height: Proxy resolution height in pixels
        enable_cache: Use cache system for proxy storage
        verbose: Print progress

    Returns:
        Path to proxy video (or original if not needed)
    """
    import tempfile

    # Check if proxy generation is recommended (only for large videos)
    if not is_proxy_generation_needed(video_path, size_threshold_mb=500):
        if verbose:
            click.echo("    Video is small enough, skipping proxy generation")
        return video_path

    # Try to get cached proxy
    if enable_cache:
        cache = CacheManager()
        cached_proxy = cache.get_proxy(video_path, height=proxy_height)
        if cached_proxy:
            if verbose:
                click.echo(f"    ✓ Using cached proxy: {Path(cached_proxy).name}")
            return cached_proxy

    # Generate new proxy
    if verbose:
        click.echo(f"    Generating {proxy_height}p proxy...")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_proxy_path = tmp.name

    try:
        generate_proxy(video_path, tmp_proxy_path, height=proxy_height, verbose=verbose)

        # Cache the proxy if caching enabled
        if enable_cache:
            cache = CacheManager()
            proxy_path = cache.put_proxy(video_path, tmp_proxy_path, height=proxy_height)
            if verbose:
                proxy_size_mb = Path(proxy_path).stat().st_size / (1024 * 1024)
                click.echo(f"    ✓ Cached proxy ({proxy_size_mb:.0f}MB)")
            return proxy_path
        else:
            return tmp_proxy_path

    except Exception as e:
        if Path(tmp_proxy_path).exists():
            try:
                Path(tmp_proxy_path).unlink()
            except:
                pass
        raise


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
@click.option(
    "--enable-motion",
    is_flag=True,
    default=False,
    help="Enable Phase 2 motion-based validation (requires OpenCV)",
)
@click.option(
    "--enable-cache",
    is_flag=True,
    default=True,
    help="Cache audio and proxy files for reuse",
)
@click.option(
    "--proxy-height",
    type=int,
    default=480,
    help="Proxy video height for motion detection (480, 360, 240)",
)
@click.option(
    "--enable-person",
    is_flag=True,
    default=False,
    help="Enable Phase 3 person detection (requires ultralytics)",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Use GPU for person detection (NVIDIA/Apple/AMD GPU)",
)
@click.option(
    "--force-cpu",
    is_flag=True,
    default=False,
    help="Force CPU usage even if GPU is available",
)
@click.option(
    "--use-fp16",
    is_flag=True,
    default=False,
    help="Use FP16 half-precision on GPU (faster, lower memory)",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for frame inference (16-32 recommended)",
)
def process(
    video_path: str,
    output: str,
    threshold: float,
    confidence: float,
    no_audio: bool,
    verbose: bool,
    enable_motion: bool,
    enable_cache: bool,
    proxy_height: int,
    enable_person: bool,
    use_gpu: bool,
    force_cpu: bool,
    use_fp16: bool,
    batch_size: int,
):
    """
    Process a video and extract all detected dives.

    VIDEO_PATH: Path to input video file

    Phase 1: Audio-based detection (always enabled)
    Phase 2: Motion validation (optional, with --enable-motion)

    Examples:
        diveanalyzer process session.mov -o ./dives
        diveanalyzer process session.mov -o ./dives --enable-motion
        diveanalyzer process session.mov -c 0.7 --threshold -20
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

        # Load audio functions
        try:
            extract_audio, detect_splash_peaks = get_audio_functions()
        except ImportError as e:
            click.echo(f"❌ {e}", err=True)
            sys.exit(1)

        # Phase 1: Extract audio
        click.echo("\n🔊 Extracting audio track...")
        try:
            audio_path = extract_audio(video_path)
            click.echo(f"✓ Audio extracted")
        except Exception as e:
            click.echo(f"❌ Failed to extract audio: {e}", err=True)
            sys.exit(1)

        # Phase 1: Detect splash peaks
        click.echo("\n🌊 Detecting splashes (threshold {:.0f}dB)...".format(threshold))
        try:
            peaks = detect_splash_peaks(
                audio_path,
                threshold_db=threshold,
                min_distance_sec=5.0,
                prominence=5.0,
            )
            click.echo(f"✓ Found {len(peaks)} splash peaks")

            if verbose:
                for i, (time, amp) in enumerate(peaks, 1):
                    click.echo(f"  {i}. {time:7.2f}s (amplitude {amp:6.1f}dB)")
        except Exception as e:
            click.echo(f"❌ Failed to detect splashes: {e}", err=True)
            sys.exit(1)

        # Phase 2: Motion detection (optional)
        motion_events = []
        if enable_motion:
            click.echo("\n🎬 Phase 2: Motion-Based Validation...")
            try:
                # Get or generate proxy for motion detection (uses cache)
                if verbose:
                    click.echo("  Getting proxy for motion detection...")
                motion_video = get_or_generate_proxy(
                    video_path,
                    proxy_height=proxy_height,
                    enable_cache=enable_cache,
                    verbose=verbose
                )

                click.echo("  Detecting motion bursts...")
                motion_events = detect_motion_bursts(motion_video, sample_fps=5.0)
                click.echo(f"  ✓ Found {len(motion_events)} motion bursts")

                if verbose and motion_events:
                    click.echo("    Motion details (first 5):")
                    for i, (start, end, intensity) in enumerate(motion_events[:5], 1):
                        click.echo(f"      {i}. {start:7.2f}s - {end:7.2f}s (intensity {intensity:.1f})")

            except ImportError:
                click.echo("  ⚠️  OpenCV not installed. Skipping motion detection.")
                click.echo("     Install with: pip install opencv-python")
                enable_motion = False
            except Exception as e:
                click.echo(f"  ⚠️  Motion detection failed: {e}")
                enable_motion = False

        # Phase 3: Person detection (optional)
        person_departures = []
        if enable_person:
            click.echo("\n👤 Phase 3: Person Detection & Validation...")
            try:
                from .detection.person import (
                    detect_person_frames,
                    smooth_person_timeline,
                    find_person_zone_departures,
                )

                # Use same proxy as motion detection
                person_video = motion_video if enable_motion else video_path

                if verbose:
                    click.echo("  Detecting person frames...")
                person_timeline = detect_person_frames(
                    person_video,
                    sample_fps=5.0,
                    confidence_threshold=0.5,
                    use_gpu=use_gpu,
                    force_cpu=force_cpu,
                    use_fp16=use_fp16,
                    batch_size=batch_size,
                )

                # Smooth timeline to remove jitter
                person_timeline = smooth_person_timeline(person_timeline, window_size=2)

                # Find zone departures (potential dive starts)
                click.echo("  Finding person zone departures...")
                person_departures = find_person_zone_departures(
                    person_timeline,
                    min_absence_duration=0.5,
                )

                click.echo(f"  ✓ Found {len(person_departures)} person departures")

                if verbose and person_departures:
                    click.echo("    Person departures (first 5):")
                    for i, (dep_time, dep_conf) in enumerate(person_departures[:5], 1):
                        click.echo(
                            f"      {i}. {dep_time:7.2f}s (confidence {dep_conf:.1%})"
                        )

            except ImportError:
                click.echo("  ⚠️  ultralytics not installed. Skipping person detection.")
                click.echo("     Install with: pip install ultralytics")
                enable_person = False
            except Exception as e:
                click.echo(f"  ⚠️  Person detection failed: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                enable_person = False

        # Phase 4: Fuse signals
        click.echo("\n🔗 Fusing detection signals...")
        try:
            if enable_person and person_departures:
                # Phase 3: Audio + Motion + Person fusion
                dives = fuse_signals_audio_motion_person(
                    peaks, motion_events, person_departures
                )
                signal_type = "audio + motion + person"
            elif enable_motion and motion_events:
                # Phase 2: Audio + Motion fusion
                dives = fuse_signals_audio_motion(peaks, motion_events)
                signal_type = "audio + motion"
            else:
                # Phase 1: Audio only
                dives = fuse_signals_audio_only(peaks)
                signal_type = "audio only"

            click.echo(f"✓ Created {len(dives)} dive events ({signal_type})")

            # Merge overlapping dives
            dives_merged = merge_overlapping_dives(dives)
            if len(dives_merged) < len(dives):
                click.echo(f"✓ Merged overlapping dives: {len(dives)} → {len(dives_merged)}")
                dives = dives_merged

            # Filter by confidence
            dives_filtered = filter_dives_by_confidence(dives, min_confidence=confidence)
            if len(dives_filtered) < len(dives):
                filtered_out = len(dives) - len(dives_filtered)
                click.echo(f"⊘ Filtered {filtered_out} low-confidence dives (confidence < {confidence})")
                dives = dives_filtered

            # Statistics
            if enable_person and person_departures:
                three_signal = sum(1 for d in dives if d.notes == "3-signal")
                two_signal = sum(1 for d in dives if d.notes == "2-signal")
                audio_only = sum(1 for d in dives if d.notes == "audio-only")
                click.echo(f"  ├─ 3-signal (audio+motion+person): {three_signal}")
                click.echo(f"  ├─ 2-signal (audio+motion/person): {two_signal}")
                click.echo(f"  └─ Audio-only: {audio_only}")
            elif enable_motion and motion_events:
                validated = sum(1 for d in dives if d.notes == "audio+motion")
                click.echo(f"  └─ Motion validated: {validated}/{len(dives)}")

            click.echo(f"✓ Final dive count: {len(dives)}")
        except Exception as e:
            click.echo(f"❌ Failed to fuse signals: {e}", err=True)
            import traceback
            if verbose:
                traceback.print_exc()
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
        extract_audio, detect_splash_peaks = get_audio_functions()

        video_path = str(Path(video_path).resolve())

        click.echo(f"🔍 Detecting dives in: {Path(video_path).name}")

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


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--sample-fps",
    type=float,
    default=5.0,
    help="Frames per second to sample",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed motion burst info",
)
def analyze_motion(video_path: str, sample_fps: float, verbose: bool):
    """
    Analyze motion patterns in a video.

    Phase 2: Debug motion detection parameters.
    """
    try:
        video_path = str(Path(video_path).resolve())

        click.echo(f"🎬 Analyzing motion in: {Path(video_path).name}")
        click.echo(f"   Sample rate: {sample_fps} FPS")

        motion_events = detect_motion_bursts(video_path, sample_fps=sample_fps)

        click.echo(f"\nFound {len(motion_events)} motion bursts:\n")

        if motion_events:
            total_motion_time = sum(end - start for start, end, _ in motion_events)
            click.echo(f"Total motion time: {total_motion_time:.1f}s\n")

            for i, (start, end, intensity) in enumerate(motion_events, 1):
                duration = end - start
                click.echo(
                    f"  {i:3d}. {start:7.2f}s - {end:7.2f}s "
                    f"({duration:5.2f}s)  intensity: {intensity:6.1f}"
                )
        else:
            click.echo("No motion bursts detected. Try adjusting sample-fps or video.")

    except ImportError:
        click.echo("❌ OpenCV not installed. Install with: pip install opencv-python", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting",
)
def clear_cache(dry_run: bool):
    """
    Clear the local cache.

    Removes cached audio, proxy videos, and metadata (older than 7 days).
    """
    try:
        from .storage.cleanup import cleanup_expired_cache

        click.echo("🧹 Cache Cleanup")
        click.echo("=" * 50)

        if dry_run:
            stats = cleanup_expired_cache(dry_run=True)
            click.echo(f"\n Would delete {stats['expired_count']} expired entries")
            click.echo("(Use without --dry-run to actually delete)")
        else:
            stats = cleanup_expired_cache()
            click.echo(f"\n✓ Deleted {stats['expired_count']} expired entries")
            click.echo(f"✓ Freed {stats['freed_size_mb']:.1f} MB")
            click.echo(f"✓ {stats['entries_after']} active cache entries remaining")

        # Show cache stats
        cache = CacheManager()
        cache_stats = cache.get_cache_stats()
        click.echo(f"\nCache stats:")
        click.echo(f"  Total size: {cache_stats['total_size_mb']:.1f} MB")
        click.echo(f"  Total entries: {cache_stats['entry_count']}")
        click.echo(f"  Cache dir: {cache_stats['cache_dir']}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
