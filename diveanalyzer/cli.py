"""
Command-line interface for DiveAnalyzer.

Main entry point for the application.
"""

import sys
import threading
import webbrowser
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
from .utils.system_profiler import SystemProfiler
from .storage.cleanup import cleanup_expired_cache, get_cache_stats, check_disk_space
from .utils.review_gallery import generate_thumbnails_deferred, DiveGalleryGenerator
from .server import EventServer

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
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    help="Force refresh system profile (ignores cache)",
)
def profile(refresh: bool):
    """Show system profile and recommended detection phase."""
    click.echo(f"🎬 DiveAnalyzer v{__version__}\n")

    profiler = SystemProfiler()
    sys_profile = profiler.get_profile(refresh=refresh)

    click.echo(sys_profile)
    click.echo()
    click.echo("📊 Interpretation:")
    click.echo(f"  • Phase 1: ~5s processing, 0.82 confidence (audio-only)")
    click.echo(f"  • Phase 2: ~15s processing, 0.92 confidence (audio + motion)")
    click.echo(f"  • Phase 3: ~{sys_profile.phase_3_estimate_sec:.0f}s processing, 0.96 confidence (audio + motion + person)")
    click.echo()
    click.echo("ℹ️  Run 'diveanalyzer process video.mov' to analyze a video.")
    click.echo(f"   It will automatically use Phase {sys_profile.recommended_phase}.")
    click.echo("   Override with --force-phase=1|2|3 if desired.")


@cli.group()
def cache():
    """Manage cache (audio, proxy, metadata)."""
    pass


@cache.command()
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be deleted without actually deleting",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print details about deleted files",
)
def cleanup(dry_run: bool, verbose: bool):
    """Delete cache files older than 7 days."""
    click.echo(f"🎬 DiveAnalyzer v{__version__}\n")

    if dry_run:
        click.echo("🔍 Checking for expired cache entries (dry run)...")
        stats = cleanup_expired_cache(dry_run=True, verbose=verbose)
        click.echo()
        click.echo(f"Would delete: {stats['expired_count']} entries")

        if stats["expired_count"] == 0:
            click.echo("✓ No expired entries found")
        else:
            click.echo()
            click.echo("To actually delete them, run: diveanalyzer cache cleanup")
    else:
        click.echo("🧹 Cleaning up expired cache...")
        stats = cleanup_expired_cache(dry_run=False, verbose=verbose)
        click.echo()
        click.echo(f"✓ Deleted: {stats['expired_count']} entries")
        click.echo(f"  Freed: {stats['freed_size_mb']:.1f} MB")
        click.echo(f"  Cache size: {stats['total_size_before_mb']:.1f} MB → {stats['total_size_after_mb']:.1f} MB")


@cache.command()
@click.option(
    "--detailed",
    is_flag=True,
    default=False,
    help="Show breakdown by cache type",
)
def stats(detailed: bool):
    """Show cache usage statistics."""
    click.echo(f"🎬 DiveAnalyzer v{__version__}\n")

    # Get cache stats
    cache_stats = get_cache_stats(detailed=detailed)
    disk_info = check_disk_space()

    click.echo("💾 Cache Statistics:")
    click.echo(f"  Total entries: {cache_stats['total_entries']}")
    click.echo(f"  Valid entries: {cache_stats['valid_entries']}")
    click.echo(f"  Expired entries: {cache_stats['expired_entries']}")
    click.echo(f"  Total size: {cache_stats['total_size_mb']:.1f} MB")

    if detailed and "by_type" in cache_stats:
        click.echo()
        click.echo("  Breakdown by type:")
        for type_name, type_stats in cache_stats["by_type"].items():
            click.echo(f"    • {type_name:10s}: {type_stats['count']:3d} files, {type_stats['size_mb']:6.1f} MB")

    click.echo()
    click.echo("💿 Disk Space:")
    click.echo(f"  Total disk: {disk_info['total_gb']:.1f} GB")
    click.echo(f"  Used: {disk_info['used_gb']:.1f} GB ({disk_info['percent_full']:.1f}%)")
    click.echo(f"  Free: {disk_info['free_gb']:.1f} GB")

    if disk_info["status"] == "LOW_SPACE":
        click.echo()
        click.secho(f"  ⚠️  {disk_info['warning']}", fg="yellow")


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
@click.option(
    "--force-phase",
    type=click.Choice(["1", "2", "3"], case_sensitive=False),
    default=None,
    help="Force specific detection phase (1=audio, 2=audio+motion, 3=full). Overrides auto-selection.",
)
@click.option(
    "--auto-select",
    is_flag=True,
    default=True,
    help="Enable automatic phase selection based on system capabilities (default: True)",
)
@click.option(
    "--enable-server",
    is_flag=True,
    default=False,
    help="Enable HTTP server with live review interface (http://localhost:8765)",
)
@click.option(
    "--server-port",
    type=int,
    default=8765,
    help="Port for HTTP server (default: 8765)",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Don't automatically open browser when --enable-server is used",
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
    force_phase: Optional[str],
    auto_select: bool,
    enable_server: bool,
    server_port: int,
    no_open: bool,
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

        # Prepare server config if requested (will start early to receive all events)
        server: Optional[EventServer] = None
        server_config = None
        if enable_server:
            gallery_path = output_dir / "review_gallery.html"
            server_config = {
                'gallery_path': str(gallery_path),
                'host': 'localhost',
                'port': server_port,
                'log_level': 'INFO' if verbose else 'WARNING',
            }

            # Create placeholder gallery immediately so server can serve something
            # while detection and extraction happen in the background
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                placeholder_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DiveAnalyzer - Live Review</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏊</text></svg>">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { font-size: 28px; margin-bottom: 5px; color: #333; }
        .header p { color: #999; font-size: 14px; }
        .status-dashboard { background: white; border-radius: 8px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .phase-indicator { display: flex; align-items: center; gap: 15px; margin-bottom: 30px; }
        .phase-icon { font-size: 48px; }
        .phase-info h2 { font-size: 20px; margin-bottom: 5px; color: #333; }
        .phase-info p { color: #999; font-size: 14px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric { background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; }
        .metric-label { color: #999; font-size: 12px; text-transform: uppercase; margin-bottom: 10px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #0066cc; }
        .progress-container { margin-top: 30px; }
        .progress-label { display: flex; justify-content: space-between; font-size: 12px; color: #999; margin-bottom: 10px; }
        .progress-bar-wrapper { width: 100%; height: 8px; background: #e5e5e5; border-radius: 4px; overflow: hidden; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #0066cc, #00a8ff); transition: width 0.3s ease; }
        .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #e5e5e5; border-top-color: #0066cc; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .message-box { background: #f0f4f8; border-left: 4px solid #0066cc; padding: 15px; border-radius: 4px; color: #333; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏊 DiveAnalyzer Live Review</h1>
            <p>Processing your video in real-time...</p>
        </div>

        <div class="status-dashboard">
            <div class="phase-indicator">
                <div class="phase-icon">📊</div>
                <div class="phase-info">
                    <h2 id="phaseLabel">Audio Detection</h2>
                    <p id="phaseStatus">Analyzing audio for splash peaks...</p>
                </div>
                <div style="margin-left: auto;">
                    <div class="spinner"></div>
                </div>
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Dives Found</div>
                    <div class="metric-value" id="metricDives">0/0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Processing Speed</div>
                    <div class="metric-value" id="metricSpeed">0.0 dives/min</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Time Remaining</div>
                    <div class="metric-value" id="metricTime">--:--</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Thumbnails Ready</div>
                    <div class="metric-value" id="metricThumbnails">0/0</div>
                </div>
            </div>

            <div class="progress-container">
                <div class="progress-label">
                    <span id="progressLabel">Overall Progress</span>
                    <span id="progressPercent">0%</span>
                </div>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-fill" id="progressBarFill" style="width: 0%"></div>
                </div>
            </div>

            <div class="message-box" id="statusMessage" style="margin-top: 20px;">
                ⏳ Waiting for dives to be detected...
            </div>
        </div>
    </div>

    <script>
        const phaseNames = {
            'phase_1': '🔊 Audio Detection',
            'phase_2': '🎬 Motion Detection',
            'phase_3': '👤 Person Detection',
            'extraction': '✂️ Clip Extraction',
            'thumbnails': '🖼️ Thumbnail Generation'
        };

        const phaseStatuses = {
            'phase_1': 'Analyzing audio for splash peaks...',
            'phase_2': 'Detecting motion bursts in video...',
            'phase_3': 'Detecting person departures...',
            'extraction': 'Extracting dive clips...',
            'thumbnails': 'Generating thumbnails for each dive...'
        };

        function updateStatus(statusData) {
            try {
                const phase = statusData.phase || 'phase_1';
                const phaseName = phaseNames[phase] || statusData.phase_name || 'Processing';
                const phaseStatus = phaseStatuses[phase] || 'Processing...';

                console.log(`Status update: ${phaseName}, Dives: ${statusData.dives_found}/${statusData.dives_expected}, Progress: ${statusData.progress_percent}%`);

                // Update phase info
                const phaseLabel = document.getElementById('phaseLabel');
                const phaseStatusElem = document.getElementById('phaseStatus');
                if (phaseLabel) phaseLabel.textContent = phaseName;
                if (phaseStatusElem) phaseStatusElem.textContent = phaseStatus;

                // Update metrics with proper null checking
                const diveText = `${statusData.dives_found || 0}/${statusData.dives_expected || 0}`;
                const speedText = `${(statusData.processing_speed || 0).toFixed(2)} dives/min`;
                const thumbText = `${statusData.thumbnails_ready || 0}/${statusData.thumbnails_expected || 0}`;
                const timeText = calculateTimeRemaining(statusData);

                const metricDives = document.getElementById('metricDives');
                const metricSpeed = document.getElementById('metricSpeed');
                const metricTime = document.getElementById('metricTime');
                const metricThumbs = document.getElementById('metricThumbnails');

                if (metricDives) metricDives.textContent = diveText;
                if (metricSpeed) metricSpeed.textContent = speedText;
                if (metricTime) metricTime.textContent = timeText;
                if (metricThumbs) metricThumbs.textContent = thumbText;

                // Update progress bar
                const percent = Math.min(100, statusData.progress_percent || 0);
                const progressFill = document.getElementById('progressBarFill');
                const progressPercent = document.getElementById('progressPercent');
                if (progressFill) progressFill.style.width = percent + '%';
                if (progressPercent) progressPercent.textContent = Math.round(percent) + '%';

                // Update status message
                const statusMsg = document.getElementById('statusMessage');
                if (statusMsg) {
                    if (statusData.dives_found > 0) {
                        statusMsg.textContent = `✅ Found ${statusData.dives_found} dive(s) - Processing...`;
                        statusMsg.style.borderLeftColor = '#28a745';
                        statusMsg.style.background = '#e8f5e9';
                    } else {
                        statusMsg.textContent = `⏳ Detecting dives...`;
                        statusMsg.style.borderLeftColor = '#0066cc';
                        statusMsg.style.background = '#f0f4f8';
                    }
                }
            } catch (err) {
                console.error('Error in updateStatus:', err);
            }
        }

        function calculateTimeRemaining(statusData) {
            if (statusData.processing_speed <= 0 || statusData.dives_expected <= 0) {
                return '--:--';
            }
            const remaining = Math.max(0, statusData.dives_expected - statusData.dives_found);
            if (remaining <= 0) return '0:00';
            const timeMinutes = remaining / statusData.processing_speed;
            const mins = Math.floor(timeMinutes);
            const secs = Math.round((timeMinutes - mins) * 60);
            return `${mins}:${String(secs).padStart(2, '0')}`;
        }

        // Connect to SSE events
        const serverUrl = window.location.origin;
        try {
            const eventSource = new EventSource(serverUrl + '/events');

            eventSource.addEventListener('status_update', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    updateStatus(data);
                } catch (err) {
                    console.error('Error parsing status_update:', err);
                }
            });

            eventSource.addEventListener('connected', (e) => {
                console.log('Connected to live event stream');
                document.getElementById('statusMessage').textContent = '🔗 Connected to server';
            });

            eventSource.addEventListener('gallery_ready', (e) => {
                try {
                    console.log('Gallery ready - refreshing page...');
                    document.getElementById('statusMessage').textContent = '✅ Gallery ready - Loading...';
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } catch (err) {
                    console.error('Error handling gallery_ready:', err);
                }
            });

            eventSource.onerror = (e) => {
                // Suppress repetitive error logging but handle connection loss
                if (eventSource.readyState === EventSource.CLOSED) {
                    console.warn('SSE connection closed by server');
                    document.getElementById('statusMessage').textContent =
                        '✅ Processing complete - preparing gallery...';
                } else if (eventSource.readyState === EventSource.CONNECTING) {
                    // Attempting to reconnect - suppress error
                    console.debug('Reconnecting to event stream...');
                }
                // Don't log error object (too verbose in console)
            };
        } catch (err) {
            console.warn('Could not connect to event stream:', err.message);
        }
    </script>
</body>
</html>"""
                gallery_path.write_text(placeholder_html, encoding='utf-8')

                # Start server immediately to capture all events from detection onward
                click.echo(f"🌐 Starting HTTP server for live review...")
                server = EventServer(**server_config)
                if server.start():
                    click.echo(f"✓ Server running at {server.get_url()}")
                    click.echo(f"  Events: {server.get_events_url()}")

                    # FEAT-06: Auto-launch browser immediately (don't wait for processing)
                    if not no_open:
                        try:
                            webbrowser.open(f"http://localhost:{server_port}")
                            click.echo(f"🌐 Opening browser at http://localhost:{server_port}")
                        except Exception as e:
                            if verbose:
                                click.echo(f"ℹ️  Could not open browser automatically: {e}")
                else:
                    click.echo("⚠️  Failed to start server, continuing without live review")
                    server = None
            except Exception as e:
                click.echo(f"⚠️  Could not prepare server: {e}")
                server = None

        # Get video info
        try:
            duration = get_video_duration(video_path)
            click.echo(f"⏱️  Video duration: {format_time(duration)}")
        except Exception as e:
            click.echo(f"⚠️  Could not determine video length: {e}")
            duration = None

        # Adaptive Phase Selection (Task 1.11)
        click.echo()
        selected_phase = 1  # Default to Phase 1
        if auto_select and not force_phase:
            click.echo("🔍 Analyzing system capabilities...")
            try:
                profiler = SystemProfiler()
                sys_profile = profiler.get_profile()
                selected_phase = sys_profile.recommended_phase

                click.echo(f"   System Score: {sys_profile.system_score}/10")
                click.echo(f"   CPU: {sys_profile.cpu_count}-core @ {sys_profile.cpu_freq_ghz:.1f} GHz")
                click.echo(f"   RAM: {sys_profile.total_ram_gb:.1f} GB")
                click.echo(f"   GPU: {sys_profile.gpu_type.upper()}")
                click.echo()
                click.echo(f"   ✓ Auto-selected Phase {selected_phase}")
                click.echo(f"   Estimated time: ~{sys_profile.phase_3_estimate_sec if selected_phase == 3 else (15 if selected_phase == 2 else 5):.0f}s")
                click.echo(f"   Expected confidence: {(0.96 if selected_phase == 3 else (0.92 if selected_phase == 2 else 0.82)):.0%}")

                if verbose:
                    click.echo()
                    click.echo(f"   Phase details:")
                    click.echo(f"   • Phase 1: 5s, 0.82 confidence")
                    click.echo(f"   • Phase 2: 15s, 0.92 confidence")
                    click.echo(f"   • Phase 3: {sys_profile.phase_3_estimate_sec:.0f}s, 0.96 confidence")
            except Exception as e:
                click.echo(f"   ⚠️  Could not profile system: {e}")
                selected_phase = 1
        elif force_phase:
            selected_phase = int(force_phase)
            click.echo(f"⚠️  Forced Phase {selected_phase} (user override)")

        # Apply phase selection to enable_motion and enable_person
        if selected_phase >= 2 and not force_phase:
            enable_motion = True
        if selected_phase >= 3 and not force_phase:
            enable_person = True

        click.echo()

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

            # FEAT-05: Emit status update for Phase 1
            if server:
                server.emit("status_update", {
                    "phase": "phase_1",
                    "phase_name": "Audio Detection",
                    "dives_found": len(peaks),
                    "dives_expected": len(peaks),
                    "thumbnails_ready": 0,
                    "thumbnails_expected": 0,
                    "processing_speed": 0,
                    "elapsed_seconds": 0,
                    "progress_percent": 33,
                })
                server.emit("splash_detection_complete", {
                    "peak_count": len(peaks),
                    "threshold_db": threshold,
                })

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

                # FEAT-05: Emit status update for Phase 2
                if server:
                    server.emit("status_update", {
                        "phase": "phase_2",
                        "phase_name": "Motion Detection",
                        "dives_found": len(motion_events),
                        "dives_expected": len(peaks),
                        "thumbnails_ready": 0,
                        "thumbnails_expected": 0,
                        "processing_speed": 0,
                        "elapsed_seconds": 0,
                        "progress_percent": 66,
                    })
                    server.emit("motion_detection_complete", {
                        "burst_count": len(motion_events),
                        "proxy_height": proxy_height,
                    })

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

                # FEAT-05: Emit status update for Phase 3
                if server:
                    server.emit("status_update", {
                        "phase": "phase_3",
                        "phase_name": "Person Detection",
                        "dives_found": len(person_departures),
                        "dives_expected": len(peaks),
                        "thumbnails_ready": 0,
                        "thumbnails_expected": 0,
                        "processing_speed": 0,
                        "elapsed_seconds": 0,
                        "progress_percent": 90,
                    })
                    server.emit("person_detection_complete", {
                        "departure_count": len(person_departures),
                        "confidence_threshold": 0.5,
                    })

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

            # FEAT-05: Emit event for server with final detection status
            if server:
                # Determine final phase
                final_phase = "phase_1"
                if enable_person and person_departures:
                    final_phase = "phase_3"
                elif enable_motion and motion_events:
                    final_phase = "phase_2"

                server.emit("status_update", {
                    "phase": final_phase,
                    "phase_name": "Detection Complete",
                    "dives_found": len(dives),
                    "dives_expected": len(dives),
                    "thumbnails_ready": 0,
                    "thumbnails_expected": len(dives),
                    "processing_speed": 0,
                    "elapsed_seconds": 0,
                    "progress_percent": 95,
                })
                server.emit("dives_detected", {
                    "dive_count": len(dives),
                    "signal_type": signal_type,
                    "confidence_threshold": confidence,
                })
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

            # FEAT-05: Emit extraction and final status update for server
            if server:
                server.emit("status_update", {
                    "phase": "complete",
                    "phase_name": "Extraction Complete",
                    "dives_found": success_count,
                    "dives_expected": len(dives),
                    "thumbnails_ready": success_count,
                    "thumbnails_expected": len(dives),
                    "processing_speed": 0,
                    "elapsed_seconds": 0,
                    "progress_percent": 100,
                })
                server.emit("extraction_complete", {
                    "total_dives": len(dives),
                    "successful": success_count,
                    "failed": len(dives) - success_count,
                })

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

            # Create review gallery HTML with placeholders for detected dives
            # This must be done before the server starts serving requests
            if success_count > 0:
                try:
                    click.echo(f"\n📸 Creating review gallery...")
                    generator = DiveGalleryGenerator(output_dir, Path(video_path).name)
                    generator.scan_dives()
                    gallery_path = generator.generate_html()

                    click.echo(f"✓ Gallery created: {gallery_path}")

                    # Emit event to notify browser that real gallery is ready
                    # Browser should refresh to see actual dive cards instead of placeholder
                    if server:
                        server.emit("gallery_ready", {
                            "message": "Real gallery is ready. Refreshing browser...",
                            "dives_count": success_count,
                        })
                except Exception as e:
                    click.echo(f"⚠️  Could not create gallery: {e}")
                    server_config = None  # Don't try to start server without gallery

            # If no server was requested, open gallery in browser directly
            if success_count > 0 and not enable_server:
                try:
                    generator.open_in_browser(Path(gallery_path))
                except Exception as e:
                    if verbose:
                        click.echo(f"ℹ️  Could not open browser: {e}")

            # FEAT-07: Start background thumbnail generation after extraction
            # This allows the gallery to appear immediately with placeholders,
            # while thumbnails are generated in the background
            if server and success_count > 0:
                click.echo(f"\n🖼️  Generating thumbnails in background...")

                # Emit status update so browser dashboard shows dive counts
                server.emit("status_update", {
                    "phase": "thumbnail_generation",
                    "phase_name": "Generating Thumbnails",
                    "dives_found": success_count,
                    "dives_expected": success_count,
                    "thumbnails_ready": 0,
                    "thumbnails_expected": success_count,
                    "progress_percent": 0,
                    "message": f"Generating thumbnails for {success_count} dive{'s' if success_count != 1 else ''}...",
                })

                # Prepare dive list for background generation
                dive_list_for_thumbnails = [
                    (dive_num, str(output_dir / f"dive_{dive_num:03d}.mp4"))
                    for dive_num in sorted(results.keys())
                    if results[dive_num][0]  # Only successful dives
                ]

                # Start background thread for thumbnail generation
                # This emits thumbnail_ready events as each one completes
                thumbnail_thread = threading.Thread(
                    target=generate_thumbnails_deferred,
                    args=(dive_list_for_thumbnails, output_dir, server),
                    kwargs={"timeout_sec": 1200.0},
                    daemon=True
                )
                thumbnail_thread.start()
                click.echo(f"✓ Background thumbnail generation started")

        except Exception as e:
            click.echo(f"❌ Failed to extract clips: {e}", err=True)
            if server:
                server.stop()
            sys.exit(1)

        # Emit final event and keep server running for live updates
        if server:
            server.emit("processing_complete", {
                "status": "success",
                "output_directory": str(output_dir),
            })

            # FEAT-07: Wait for background thumbnail generation to complete
            # Keep server running while thumbnails generate so browser can receive updates
            import time
            if success_count > 0 and 'thumbnail_thread' in locals():
                click.echo(f"\n🖼️  Waiting for thumbnail generation to complete...")
                click.echo(f"   Server running at {server.get_url()} - Keep browser open")

                # Wait for thread to complete with timeout
                thumbnail_thread.join(timeout=120.0)  # Wait up to 2 minutes

                if thumbnail_thread.is_alive():
                    click.echo("⚠️  Thumbnail generation still running (timeout reached)")
                else:
                    click.echo("✓ Thumbnail generation complete")
            else:
                click.echo("\n🌐 No background thumbnails to wait for")

            click.echo("\n🌐 Server is running for live review until you shut it down.")
            click.echo(f"  • To stop the server: visit {server.get_url()}/shutdown in your browser or click 'Accept All & Close' in the gallery.")
            click.echo("  • Or stop the process with Ctrl+C in this terminal.")

        click.echo("\n✅ Done!")

    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()

        # Cleanup server on error
        if 'server' in locals() and server:
            server.stop()

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
def detect(video_path: str, threshold: float):
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
        from .detection.audio import get_audio_properties, detect_splash_peaks

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
def analyze_motion(video_path: str, sample_fps: float):
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
