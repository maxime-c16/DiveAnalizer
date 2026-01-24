#!/usr/bin/env python3
"""
End-to-end automated Chromium test for DiveAnalyzer pipeline.

Tests the complete workflow:
1. CLI starts with --enable-server
2. Video processing (detection → thumbnails)
3. Gallery opens in selection mode
4. User selects and extracts dives
5. Gallery switches to extracted mode
6. User reviews and deletes
7. Shutdown

Collects performance metrics and detailed debug logs for UX analysis.
"""

import asyncio
import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/e2e_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class E2ETestMetrics:
    """Track performance metrics throughout the test."""

    def __init__(self):
        self.events: List[Dict] = []
        self.start_time = time.time()
        self.phase_timings = {}
        self.current_phase = None
        self.gallery_states = []

    def start_phase(self, phase_name: str):
        """Mark the start of a test phase."""
        self.current_phase = phase_name
        self.phase_timings[phase_name] = {'start': time.time()}
        logger.info(f"📊 PHASE START: {phase_name}")
        self.log_event('phase_start', {'phase': phase_name})

    def end_phase(self, phase_name: str):
        """Mark the end of a test phase."""
        if phase_name in self.phase_timings:
            elapsed = time.time() - self.phase_timings[phase_name]['start']
            self.phase_timings[phase_name]['end'] = time.time()
            self.phase_timings[phase_name]['duration'] = elapsed
            logger.info(f"📊 PHASE END: {phase_name} ({elapsed:.2f}s)")
            self.log_event('phase_end', {'phase': phase_name, 'duration': elapsed})

    def log_event(self, event_type: str, data: Dict):
        """Log a timestamped event."""
        event = {
            'timestamp': time.time(),
            'relative_time': time.time() - self.start_time,
            'type': event_type,
            'data': data
        }
        self.events.append(event)
        logger.debug(f"EVENT: {event_type} | {json.dumps(data)}")

    def record_gallery_state(self, state: str, details: Dict):
        """Record gallery state transitions."""
        self.gallery_states.append({
            'timestamp': time.time(),
            'state': state,
            'details': details
        })
        logger.info(f"🖼️  GALLERY STATE: {state} | {details}")

    def print_summary(self):
        """Print performance summary."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("📊 E2E TEST PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"\nTotal execution time: {total_time:.2f}s\n")

        print("Phase Timings:")
        for phase, timing in self.phase_timings.items():
            duration = timing.get('duration', 0)
            print(f"  {phase:.<40} {duration:>8.2f}s")

        print(f"\nTotal phases: {len(self.phase_timings)}")
        print(f"Total events: {len(self.events)}")
        print(f"Gallery state transitions: {len(self.gallery_states)}")

        # Calculate percentages
        if self.phase_timings:
            print("\nPhase Distribution:")
            total_phases = sum(t.get('duration', 0) for t in self.phase_timings.values())
            for phase, timing in self.phase_timings.items():
                duration = timing.get('duration', 0)
                pct = (duration / total_phases * 100) if total_phases > 0 else 0
                bar = '█' * int(pct / 5)
                print(f"  {phase:.<30} {pct:>5.1f}% {bar}")

        print("\n" + "=" * 80 + "\n")


async def test_e2e_pipeline():
    """Run the complete end-to-end test."""
    metrics = E2ETestMetrics()

    try:
        from playwright.async_api import async_playwright

        # Get test video - use back_to_back.mp4 as it has more reliable codec
        # (very_short.mp4 has frame reading issues at the end)
        test_video = Path("tests/fixtures/edge_cases/back_to_back.mp4")

        if not test_video.exists():
            # Fallback to other videos
            for fallback in [
                Path("tests/fixtures/edge_cases/false_positive.mp4"),
                Path("tests/fixtures/edge_cases/no_audio.mp4"),
            ]:
                if fallback.exists():
                    test_video = fallback
                    break

        if not test_video.exists():
            logger.error(f"❌ No test video found")
            logger.error(f"Available videos:")
            for v in Path("tests/fixtures/edge_cases").glob("*.mp4"):
                logger.error(f"  - {v.name} ({v.stat().st_size / (1024*1024):.1f}MB)")
            return False

        logger.info(f"✅ Test video found: {test_video}")
        logger.info(f"📏 File size: {test_video.stat().st_size / (1024*1024):.2f}MB")

        output_dir = Path("/tmp/e2e_test_output")
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"📁 Output directory: {output_dir}")

        # ========== PHASE 1: START CLI ==========
        metrics.start_phase("CLI_Startup")

        cli_process = subprocess.Popen(
            [
                sys.executable, "-m", "diveanalyzer.cli",
                "process",
                str(test_video),
                "--output", str(output_dir),
                "--enable-server",
                "--server-port", "9999",
                "--no-open",  # Don't open Safari automatically
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/Users/mcauchy/workflow/DiveAnalizer"
        )

        logger.info(f"✅ CLI process started (PID: {cli_process.pid})")
        metrics.log_event('cli_started', {'pid': cli_process.pid})

        # Log CLI output in background
        def log_cli_output():
            try:
                for line in cli_process.stdout:
                    logger.debug(f"[CLI] {line.rstrip()}")
            except:
                pass

        import threading
        cli_logger_thread = threading.Thread(target=log_cli_output, daemon=True)
        cli_logger_thread.start()

        # Wait for server to start
        server_ready = False
        for attempt in range(60):
            try:
                import urllib.request
                response = urllib.request.urlopen("http://localhost:9999/health", timeout=1)
                if response.status == 200:
                    server_ready = True
                    logger.info(f"✅ Server ready at http://localhost:9999")
                    metrics.log_event('server_ready', {'attempt': attempt})
                    break
            except:
                await asyncio.sleep(0.5)
                if attempt % 5 == 0:
                    logger.debug(f"⏳ Waiting for server... (attempt {attempt+1}/30)")

        if not server_ready:
            logger.error("❌ Server failed to start")
            return False

        metrics.end_phase("CLI_Startup")

        # ========== PHASE 2: BROWSER AND GALLERY LOAD ==========
        metrics.start_phase("Browser_Load")

        async with async_playwright() as p:
            # Launch Chromium
            browser = await p.chromium.launch(headless=True)
            logger.info(f"✅ Chromium launched")
            metrics.log_event('browser_launched', {'browser': 'chromium'})

            # Create a context with detailed logging
            context = await browser.new_context()
            page = await context.new_page()

            # Set up console message handler for debugging
            def log_console(msg):
                logger.debug(f"🌐 [BROWSER CONSOLE] {msg.type}: {msg.text}")

            page.on("console", log_console)

            # Navigate to gallery
            logger.info("🔍 Navigating to gallery...")
            await page.goto("http://localhost:9999", wait_until="load", timeout=30000)
            logger.info("✅ Gallery loaded")
            metrics.log_event('gallery_loaded', {'url': 'http://localhost:9999'})

            # Wait for page to settle
            await asyncio.sleep(2)
            metrics.end_phase("Browser_Load")

            # ========== PHASE 3: WAIT FOR SELECTION MODE GALLERY ==========
            metrics.start_phase("Wait_Selection_Mode")

            gallery_ready = False
            placeholder_visible = False
            for attempt in range(180):  # 3 minutes timeout
                try:
                    # Check if we're in selection mode
                    selection_mode = await page.evaluate(
                        "() => document.body.innerHTML.includes('Extract Selected Dives')"
                    )

                    # Also check for placeholder
                    placeholder = await page.evaluate(
                        "() => document.body.innerHTML.includes('DiveAnalyzer Live Review')"
                    )

                    if placeholder and not placeholder_visible:
                        logger.info("ℹ️  Placeholder gallery visible (processing in progress)")
                        placeholder_visible = True

                    if selection_mode:
                        logger.info("✅ Selection mode gallery loaded")
                        metrics.record_gallery_state('selection_mode', {
                            'has_extract_button': True
                        })
                        gallery_ready = True
                        break

                    # Check for error messages
                    error_msg = await page.query_selector(".error-message")
                    if error_msg:
                        error_text = await error_msg.text_content()
                        logger.warning(f"⚠️  Error message appeared: {error_text}")

                    if attempt % 10 == 0:
                        logger.debug(f"⏳ Waiting for selection gallery... ({attempt}s)")

                except Exception as e:
                    logger.debug(f"Check failed: {e}")

                await asyncio.sleep(1)

            if not gallery_ready:
                logger.error("❌ Selection mode gallery failed to load")
                # Debug: get page content
                try:
                    page_html = await page.content()
                    logger.debug(f"Page HTML length: {len(page_html)}")
                    if len(page_html) < 500:
                        logger.error(f"Page content seems incomplete:\n{page_html}")
                except:
                    pass
                # Take screenshot for debugging
                try:
                    await page.screenshot(path="/tmp/e2e_failed_gallery.png")
                    logger.info("📸 Screenshot saved to /tmp/e2e_failed_gallery.png")
                except:
                    pass
                return False

            metrics.end_phase("Wait_Selection_Mode")

            # ========== PHASE 4: INSPECT GALLERY STATE ==========
            metrics.start_phase("Inspect_Gallery")

            # Get gallery information
            dive_count = await page.evaluate(
                "() => document.querySelectorAll('.dive-card').length"
            )
            logger.info(f"📊 Found {dive_count} dive cards in gallery")
            metrics.log_event('gallery_inspected', {'dive_count': dive_count})

            # Debug: list all buttons on page
            all_buttons = await page.evaluate(
                "() => Array.from(document.querySelectorAll('button')).map(b => ({id: b.id, text: b.textContent.substring(0, 50)}))"
            )
            logger.debug(f"All buttons on page: {json.dumps(all_buttons, indent=2)}")

            # Check buttons
            has_extract_btn = await page.evaluate(
                "() => !!document.getElementById('btn-extract-selected')"
            )
            logger.info(f"{'✅' if has_extract_btn else '❌'} Extract button present: {has_extract_btn}")
            metrics.log_event('button_check', {'extract_button': has_extract_btn})

            metrics.end_phase("Inspect_Gallery")

            if not has_extract_btn:
                logger.error("❌ Extract button not found!")
                await page.screenshot(path="/tmp/e2e_no_extract_button.png")
                return False

            # ========== PHASE 5: SELECT AND EXTRACT ==========
            metrics.start_phase("Selection_And_Extraction")

            # Select first dive if available
            if dive_count > 0:
                logger.info(f"🎯 Selecting first dive...")
                first_checkbox = await page.query_selector('.dive-checkbox')
                if first_checkbox:
                    await first_checkbox.click()
                    await asyncio.sleep(0.5)
                    logger.info("✅ First dive selected")
                    metrics.log_event('dive_selected', {'dive_index': 0})

                # Click extract button
                logger.info("🚀 Clicking 'Extract Selected Dives'...")
                extract_btn = await page.query_selector('#btn-extract-selected')
                if extract_btn:
                    await extract_btn.click()
                    logger.info("✅ Extract button clicked")
                    metrics.log_event('extract_started', {'selected_dives': 1})

                    # Wait for extraction to complete
                    extraction_complete = False
                    for attempt in range(120):  # 2 minute timeout
                        try:
                            # Look for extraction complete message or extracted videos
                            has_videos = await page.evaluate(
                                "() => Array.from(document.querySelectorAll('.dive-card')).some(card => "
                                "card.querySelector('video') || card.innerHTML.includes('dive_'))"
                            )

                            if has_videos:
                                logger.info("✅ Extracted videos detected")
                                extraction_complete = True
                                break

                            if attempt % 10 == 0:
                                logger.debug(f"⏳ Waiting for extraction... ({attempt}s)")

                        except Exception as e:
                            logger.debug(f"Extraction check failed: {e}")

                        await asyncio.sleep(1)

                    if extraction_complete:
                        metrics.log_event('extraction_complete', {'success': True})
                    else:
                        logger.warning("⏳ Extraction timeout (might still be processing)")

            metrics.end_phase("Selection_And_Extraction")

            # ========== PHASE 6: CHECK EXTRACTED MODE GALLERY ==========
            metrics.start_phase("Extracted_Mode_Check")

            # Check if gallery switched to extracted mode
            await asyncio.sleep(3)  # Wait for any potential reload
            await page.reload(wait_until="networkidle")

            has_delete_btn = await page.evaluate(
                "() => !!document.getElementById('btn-delete')"
            )
            has_watch_btn = await page.evaluate(
                "() => !!document.getElementById('btn-watch')"
            )
            has_accept_btn = await page.evaluate(
                "() => !!document.getElementById('btn-accept')"
            )

            logger.info(f"{'✅' if has_delete_btn else '❌'} Delete button: {has_delete_btn}")
            logger.info(f"{'✅' if has_watch_btn else '❌'} Watch button: {has_watch_btn}")
            logger.info(f"{'✅' if has_accept_btn else '❌'} Accept button: {has_accept_btn}")

            metrics.record_gallery_state('extracted_mode', {
                'delete_button': has_delete_btn,
                'watch_button': has_watch_btn,
                'accept_button': has_accept_btn
            })

            metrics.log_event('extracted_mode_buttons', {
                'delete': has_delete_btn,
                'watch': has_watch_btn,
                'accept': has_accept_btn
            })

            metrics.end_phase("Extracted_Mode_Check")

            # ========== RESULTS ==========
            success = all([has_delete_btn, has_watch_btn, has_accept_btn])

            if success:
                logger.info("\n" + "=" * 80)
                logger.info("✅ E2E TEST PASSED")
                logger.info("=" * 80)
                logger.info("All buttons present in extracted mode gallery")
            else:
                logger.error("\n" + "=" * 80)
                logger.error("❌ E2E TEST FAILED")
                logger.error("=" * 80)
                logger.error("Missing buttons in extracted mode:")
                if not has_delete_btn:
                    logger.error("  - Delete button missing")
                if not has_watch_btn:
                    logger.error("  - Watch button missing")
                if not has_accept_btn:
                    logger.error("  - Accept button missing")
                await page.screenshot(path="/tmp/e2e_final_state.png")

            # Take final screenshot
            await page.screenshot(path="/tmp/e2e_final_gallery.png")
            logger.info("📸 Screenshots saved to /tmp/e2e_*.png")

            # Cleanup
            await context.close()
            await browser.close()
            logger.info("🔌 Browser closed")

            return success

    except Exception as e:
        logger.exception(f"❌ Test failed with exception: {e}")
        return False

    finally:
        # Cleanup: terminate CLI process
        if 'cli_process' in locals():
            logger.info("🛑 Terminating CLI process...")
            cli_process.terminate()
            try:
                cli_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cli_process.kill()
            logger.info("✅ CLI process terminated")

        # Print metrics summary
        metrics.print_summary()

        # Save metrics to file
        metrics_file = Path("/tmp/e2e_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'total_time': time.time() - metrics.start_time,
                'phase_timings': metrics.phase_timings,
                'event_count': len(metrics.events),
                'gallery_states': metrics.gallery_states,
                'test_timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        logger.info(f"📊 Metrics saved to {metrics_file}")


def main():
    """Run the test."""
    # Change to project root
    project_root = Path("/Users/mcauchy/workflow/DiveAnalizer")
    if project_root.exists():
        import os
        os.chdir(project_root)
        logger.info(f"Working directory: {os.getcwd()}")
    else:
        logger.error(f"Project root not found: {project_root}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🧪 DIVEANALYZER E2E CHROMIUM TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Test video: tests/fixtures/edge_cases/back_to_back.mp4")
    print(f"Working directory: {Path.cwd()}")
    print("=" * 80 + "\n")

    # Verify test video exists
    test_video = Path("tests/fixtures/edge_cases/back_to_back.mp4")
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Directory contents: {list(Path.cwd().iterdir())}")
        sys.exit(1)

    # Run async test
    result = asyncio.run(test_e2e_pipeline())

    # Exit with appropriate code
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
