"""
Interactive HTML gallery for reviewing and filtering detected dives.
Allows users to quickly accept/reject dives with keyboard shortcuts.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import base64
import tempfile
import os


class DiveGalleryGenerator:
    """Generates interactive HTML gallery for dive review."""

    def __init__(self, output_dir: Path, video_name: str = ""):
        """Initialize gallery generator.

        Args:
            output_dir: Directory containing extracted dive videos
            video_name: Name of source video file
        """
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        self.dives = []
        self.thumbnails = {}

    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1",
                   str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0

    def extract_thumbnail(self, video_path: Path, position: str = "middle",
                          width: int = 320, height: int = 240, quality: int = 5,
                          percentage: float = None) -> str:
        """Extract a thumbnail frame from video as base64 PNG.

        Args:
            video_path: Path to video file
            position: "start" (20% - early flight), "middle" (65% - splash), or "end" (95% - submersion)
            width: Thumbnail width (default 320)
            height: Thumbnail height (default 240)
            quality: JPEG quality (default 5, lower = better but slower)
            percentage: Specific percentage of video (0.0 to 1.0), overrides position

        Returns:
            Base64 encoded PNG data URL, or None if failed
        """
        try:
            duration = self.get_video_duration(video_path)

            if duration <= 0:
                print(f"Warning: Could not determine duration for {video_path}")
                return None

            # Determine time within video for thumbnail
            if percentage is not None:
                time_sec = duration * percentage
            elif position == "start":
                time_sec = duration * 0.20
            elif position == "end":
                time_sec = duration * 0.95
            else:
                time_sec = duration * 0.65

            # Create temp file for thumbnail
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                thumb_path = tmp.name

            # Extract thumbnail using ffmpeg
            cmd = [
                "ffmpeg",
                "-ss", str(time_sec),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                "-q:v", str(quality),
                "-y",
                str(thumb_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Convert to base64 if produced
            if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                with open(thumb_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                try:
                    os.unlink(thumb_path)
                except:
                    pass
                return f"data:image/jpeg;base64,{img_data}"
            else:
                print(f"Warning: ffmpeg did not produce output for {video_path} at {position}")

        except subprocess.TimeoutExpired:
            print(f"Warning: ffmpeg timeout for {video_path}")
        except Exception as e:
            print(f"Warning: Could not extract thumbnail for {video_path}: {e}")

        return None

    def extract_timeline_frames(self, video_path: Path) -> List[str]:
        """Extract 8 evenly-spaced frames from dive video for timeline.

        Frames at: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
        Returns a list of base64 data URLs or None entries for failures.
        """
        frames = []
        percentages = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        for pct in percentages:
            try:
                frame_data = self.extract_thumbnail(
                    video_path,
                    position="middle",
                    width=720,
                    height=1280,
                    quality=3,
                    percentage=pct,
                )
            except Exception:
                frame_data = None

            frames.append(frame_data)

        return frames

    def scan_dives(self) -> List[Dict[str, Any]]:
        """Scan output directory for extracted dive videos.

        Returns:
            List of dive info dicts
        """
        dives = []
        dive_files = sorted(self.output_dir.glob("dive_*.mp4"))

        for i, video_path in enumerate(dive_files, 1):
            # Extract duration
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1",
                   str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                duration = float(result.stdout.strip())
            except:
                duration = 0

            dive_info = {
                "id": i,
                "filename": video_path.name,
                "path": str(video_path),
                "duration": duration,
                "timestamp": "unknown",
                "confidence": "high" if i % 2 == 0 else "medium",  # Placeholder
                "thumbnails": {
                    "start": self.extract_thumbnail(video_path, "start"),
                    "middle": self.extract_thumbnail(video_path, "middle"),
                    "end": self.extract_thumbnail(video_path, "end"),
                },
                "timeline_thumbnails": self.extract_timeline_frames(video_path)
            }
            dives.append(dive_info)

        self.dives = dives
        return dives

    def generate_html(self, output_path: Path = None) -> str:
        """Generate interactive HTML gallery.

        Args:
            output_path: Optional path to save HTML file

        Returns:
            HTML content
        """
        if output_path is None:
            output_path = self.output_dir / "review_gallery.html"

        # Generate HTML
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dive Review Gallery</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            color: #1e3c72;
            margin-bottom: 5px;
        }}

        .header p {{
            color: #666;
            font-size: 14px;
        }}

        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 10px;
            font-size: 14px;
        }}

        .stat {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .stat-value {{
            font-weight: bold;
            color: #1e3c72;
            font-size: 16px;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .dive-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
        }}

        .dive-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }}

        .dive-card.selected {{
            outline: 3px solid #4CAF50;
            outline-offset: -3px;
        }}

        .dive-card.deleted {{
            opacity: 0.5;
            outline: 3px solid #f44336;
            outline-offset: -3px;
        }}

        .checkbox {{
            position: absolute;
            top: 10px;
            left: 10px;
            width: 24px;
            height: 24px;
            cursor: pointer;
            z-index: 10;
        }}

        .checkbox input {{
            width: 100%;
            height: 100%;
            cursor: pointer;
        }}

        .thumbnails {{
            display: flex;
            height: 200px;
            gap: 3px;
            background: #f5f5f5;
            overflow: hidden;
        }}

        .thumbnail {{
            flex: 1;
            object-fit: cover;
            background: #ddd;
        }}

        .dive-info {{
            padding: 15px;
        }}

        .dive-number {{
            font-weight: bold;
            color: #1e3c72;
            font-size: 16px;
            margin-bottom: 8px;
        }}

        .dive-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 13px;
            color: #666;
        }}

        .detail-row {{
            display: flex;
            justify-content: space-between;
        }}

        .detail-label {{
            font-weight: 500;
        }}

        .detail-value {{
            text-align: right;
        }}

        .confidence {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 8px;
        }}

        .confidence.high {{
            background: #c8e6c9;
            color: #2e7d32;
        }}

        .confidence.medium {{
            background: #fff9c4;
            color: #f57f17;
        }}

        .confidence.low {{
            background: #ffccbc;
            color: #d84315;
        }}

        .footer {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}

        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            font-size: 14px;
        }}

        .btn-delete {{
            background: #f44336;
            color: white;
        }}

        .btn-delete:hover {{
            background: #d32f2f;
        }}

        .btn-accept {{
            background: #4CAF50;
            color: white;
        }}

        .btn-accept:hover {{
            background: #388E3C;
        }}

        .btn-select-all {{
            background: #2196F3;
            color: white;
        }}

        .btn-select-all:hover {{
            background: #1565C0;
        }}

        .btn-deselect-all {{
            background: #757575;
            color: white;
        }}

        .btn-deselect-all:hover {{
            background: #616161;
        }}

        .btn-watch {{
            background: #FF9800;
            color: white;
        }}

        .btn-watch:hover {{
            background: #F57C00;
        }}

        .shortcuts {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
            font-size: 12px;
        }}

        .shortcuts h4 {{
            color: #1e3c72;
            margin-bottom: 10px;
        }}

        .shortcut-list {{
            columns: 2;
            gap: 20px;
        }}

        .shortcut {{
            display: flex;
            gap: 10px;
            margin-bottom: 8px;
        }}

        .shortcut-key {{
            background: #1e3c72;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-family: monospace;
            font-weight: bold;
            min-width: 30px;
            text-align: center;
        }}

        .status-message {{
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: none;
        }}

        .status-message.show {{
            display: block;
        }}

        .status-message.success {{
            background: #c8e6c9;
            color: #2e7d32;
            border: 1px solid #4CAF50;
        }}

        .status-message.error {{
            background: #ffccbc;
            color: #d84315;
            border: 1px solid #f44336;
        }}

        .video-player {{
            margin-top: 20px;
            display: none;
        }}

        .video-player.show {{
            display: block;
        }}

        .video-player video {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}

        /* Modal Styles */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .modal-overlay.show {{
            display: flex;
            opacity: 1;
            align-items: center;
            justify-content: center;
        }}

        .modal-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 90vw;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            opacity: 0;
            transform: scale(0.95);
            transition: all 0.3s ease;
            z-index: 1001;
        }}

        .modal-overlay.show .modal-container {{
            opacity: 1;
            transform: scale(1);
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            background: #f9f9f9;
        }}

        .modal-title {{
            font-size: 20px;
            font-weight: bold;
            color: #1e3c72;
        }}

        .modal-close {{
            background: none;
            border: none;
            font-size: 28px;
            cursor: pointer;
            color: #999;
            padding: 0;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        }}

        .modal-close:hover {{
            background: #e0e0e0;
            color: #333;
        }}

        .modal-content {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}

        .timeline-section {{
            margin-bottom: 20px;
        }}

        .timeline-label {{
            font-size: 12px;
            font-weight: bold;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }}

        .timeline-frames {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
            padding: 12px;
            border-radius: 4px;
            background: #f5f5f5;
        }}

        .timeline-frame {{
            flex: 0 0 calc(50% - 6px);
            min-width: 180px;
            aspect-ratio: 9 / 16;
            border-radius: 4px;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s ease;
            background: #ddd;
        }}

        .timeline-frame img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #f0f0f0;
        }}

        .timeline-frame:hover {{
            transform: scale(1.05);
            border-color: #2196F3;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
        }}

        .info-panel {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
        }}

        .info-row:last-child {{
            margin-bottom: 0;
        }}

        .info-label {{
            font-weight: 600;
            color: #666;
        }}

        .info-value {{
            color: #333;
            text-align: right;
        }}

        .modal-actions {{
            display: flex;
            gap: 10px;
            padding: 15px 20px;
            border-top: 1px solid #e0e0e0;
            background: #f9f9f9;
        }}

        .modal-btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
            transition: all 0.2s ease;
            flex: 1;
        }}

        .modal-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .modal-btn-keep {{
            background: #4CAF50;
            color: white;
        }}

        .modal-btn-keep:hover:not(:disabled) {{
            background: #388E3C;
        }}

        .modal-btn-delete {{
            background: #f44336;
            color: white;
        }}

        .modal-btn-delete:hover:not(:disabled) {{
            background: #d32f2f;
        }}

        .modal-btn-cancel {{
            background: #757575;
            color: white;
        }}

        .modal-btn-cancel:hover:not(:disabled) {{
            background: #616161;
        }}

        .keyboard-hint {{
            font-size: 11px;
            color: #999;
            margin-top: 4px;
        }}

        /* FEAT-02: Connection Status Indicator */
        .connection-status {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 15px;
            border-radius: 8px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 500;
            font-size: 13px;
            font-weight: 500;
        }}

        .connection-status.connected {{
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }}

        .connection-status.disconnected {{
            background: #ffebee;
            border-left: 4px solid #f44336;
        }}

        .connection-status.connecting {{
            background: #fff3e0;
            border-left: 4px solid #FF9800;
        }}

        .status-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }}

        .status-indicator.connected {{
            background: #4CAF50;
        }}

        .status-indicator.disconnected {{
            background: #f44336;
            animation: none;
        }}

        .status-indicator.connecting {{
            background: #FF9800;
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.5;
            }}
        }}

        .status-address {{
            color: #666;
            font-size: 12px;
            margin-left: 4px;
        }}

        /* FEAT-03: Placeholder Styles for Skeleton Loading */
        .placeholder-card {{
            animation: fadeIn 0.2s ease-in;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .placeholder-thumbnails {{
            display: flex;
            height: 200px;
            gap: 3px;
            background: #f5f5f5;
            overflow: hidden;
        }}

        .placeholder-thumbnail {{
            flex: 1;
            background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
            border-radius: 2px;
        }}

        @keyframes shimmer {{
            0% {{
                background-position: 200% 0;
            }}
            100% {{
                background-position: -200% 0;
            }}
        }}

        .placeholder-info {{
            padding: 15px;
        }}

        .placeholder-number {{
            height: 16px;
            background: #d0d0d0;
            border-radius: 4px;
            margin-bottom: 8px;
            width: 80px;
            animation: shimmer 2s infinite;
        }}

        .placeholder-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}

        .placeholder-detail {{
            height: 12px;
            background: #e8e8e8;
            border-radius: 3px;
            animation: shimmer 2s infinite;
        }}

        .placeholder-confidence {{
            height: 20px;
            width: 60px;
            background: #d0d0d0;
            border-radius: 4px;
            margin-top: 8px;
            animation: shimmer 2s infinite;
        }}

        .empty-gallery-message {{
            text-align: center;
            padding: 40px 20px;
            color: #999;
            font-size: 16px;
            grid-column: 1 / -1;
        }}

        .empty-gallery-message-icon {{
            font-size: 48px;
            margin-bottom: 10px;
        }}

        /* FEAT-02: Event Log Display */
        .event-log-container {{
            display: none;
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            max-height: 300px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 499;
            border: 1px solid #e0e0e0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .event-log-container.show {{
            display: flex;
        }}

        .event-log-header {{
            padding: 10px 15px;
            background: #f5f5f5;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
            font-size: 13px;
            color: #333;
        }}

        .event-log-close {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 16px;
            color: #999;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .event-log-close:hover {{
            color: #333;
        }}

        .event-log {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            font-size: 11px;
            font-family: monospace;
            line-height: 1.4;
        }}

        .event-log-entry {{
            padding: 6px 8px;
            margin-bottom: 6px;
            border-radius: 3px;
            background: #f9f9f9;
            border-left: 3px solid #999;
            color: #555;
            word-break: break-word;
        }}

        .event-log-entry.info {{
            border-left-color: #2196F3;
            background: #e3f2fd;
            color: #1565C0;
        }}

        .event-log-entry.success {{
            border-left-color: #4CAF50;
            background: #e8f5e9;
            color: #2e7d32;
        }}

        .event-log-entry.warning {{
            border-left-color: #FF9800;
            background: #fff3e0;
            color: #e65100;
        }}

        .event-log-entry.error {{
            border-left-color: #f44336;
            background: #ffebee;
            color: #c62828;
        }}

        .event-log-timestamp {{
            font-size: 10px;
            opacity: 0.7;
            margin-right: 4px;
        }}

        .toggle-event-log {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 498;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }}

        .toggle-event-log:hover {{
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        .toggle-event-log.hidden {{
            display: none;
        }}

        /* FEAT-05: Status Dashboard & Progress Tracking */
        .status-dashboard {{
            position: sticky;
            top: 0;
            z-index: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-bottom: 2px solid rgba(0,0,0,0.1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-radius: 0 0 8px 8px;
        }}

        .status-dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            gap: 20px;
        }}

        .status-dashboard-title {{
            font-size: 16px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .status-dashboard-phase {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            font-size: 13px;
            font-weight: 500;
        }}

        .status-dashboard-phase.phase-1 {{
            background: rgba(59, 130, 246, 0.3);
            border: 1px solid rgba(59, 130, 246, 0.5);
        }}

        .status-dashboard-phase.phase-2 {{
            background: rgba(234, 179, 8, 0.3);
            border: 1px solid rgba(234, 179, 8, 0.5);
        }}

        .status-dashboard-phase.phase-3 {{
            background: rgba(34, 197, 94, 0.3);
            border: 1px solid rgba(34, 197, 94, 0.5);
        }}

        .status-dashboard-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}

        .status-metric {{
            background: rgba(255,255,255,0.1);
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .status-metric-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
            margin-bottom: 4px;
        }}

        .status-metric-value {{
            font-size: 16px;
            font-weight: bold;
        }}

        .progress-container {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .progress-bar-wrapper {{
            flex: 1;
        }}

        .progress-bar {{
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .progress-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            border-radius: 3px;
            transition: width 0.3s ease;
            width: 0%;
        }}

        .progress-percent {{
            font-size: 13px;
            font-weight: bold;
            min-width: 45px;
            text-align: right;
        }}

        .status-connection-banner {{
            display: none;
            position: fixed;
            top: 70px;
            left: 20px;
            right: 20px;
            background: #ff9800;
            color: white;
            padding: 12px 15px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 599;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideDown 0.3s ease;
        }}

        .status-connection-banner.show {{
            display: flex;
        }}

        .status-connection-banner.error {{
            background: #f44336;
        }}

        .status-spinner {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }}

        @keyframes slideDown {{
            from {{
                transform: translateY(-100%);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}

        @keyframes spin {{
            to {{
                transform: rotate(360deg);
            }}
        }}

        @media (max-width: 640px) {{
            .status-dashboard {{
                padding: 15px;
                margin-bottom: 15px;
            }}

            .status-dashboard-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}

            .status-dashboard-metrics {{
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }}

            .status-metric {{
                padding: 10px;
                font-size: 12px;
            }}

            .status-metric-value {{
                font-size: 14px;
            }}

            .progress-container {{
                gap: 8px;
            }}

            .progress-percent {{
                min-width: 35px;
                font-size: 12px;
            }}

            .status-connection-banner {{
                top: 15px;
                left: 10px;
                right: 10px;
                padding: 10px 12px;
                font-size: 12px;
            }}
        }}

        @media (max-width: 640px) {{
            .connection-status {{
                top: 10px;
                right: 10px;
                font-size: 12px;
                padding: 8px 12px;
            }}

            .event-log-container {{
                bottom: 10px;
                right: 10px;
                width: calc(100% - 20px);
                max-height: 250px;
            }}

            .toggle-event-log {{
                bottom: 10px;
                right: 10px;
            }}
        }}

        @media (max-width: 640px) {{
            .modal-container {{
                width: 95vw;
                max-height: 85vh;
            }}

            .timeline-frame {{
                flex: 0 0 calc(100% - 12px);
                min-width: 140px;
                aspect-ratio: 9 / 16;
            }}

            .modal-actions {{
                flex-direction: column;
            }}

            .modal-btn {{
                width: 100%;
            }}
        }}

        /* ===== FEAT-04: Placeholder Styles ===== */
        .placeholder-card {{
            animation: slideIn 0.3s ease-out;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .placeholder-thumbnails {{
            display: flex;
            height: 200px;
            gap: 3px;
            background: #f5f5f5;
            overflow: hidden;
            padding: 8px;
        }}

        .placeholder-thumbnail {{
            flex: 1;
            background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 4px;
        }}

        @keyframes loading {{
            0% {{
                background-position: 200% 0;
            }}
            100% {{
                background-position: -200% 0;
            }}
        }}

        .placeholder-info {{
            padding: 12px;
            background: #fafafa;
        }}

        .placeholder-number {{
            height: 16px;
            width: 80%;
            background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 4px;
            margin-bottom: 8px;
        }}

        .placeholder-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .placeholder-detail {{
            height: 12px;
            background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 4px;
        }}

        .placeholder-confidence {{
            height: 20px;
            width: 60px;
            background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 4px;
        }}
        /* ===== END FEAT-04 ===== */
    </style>
</head>
<body>
    <!-- FEAT-05: Status Dashboard & Progress Tracking -->
    <div class="status-dashboard" id="statusDashboard">
        <div class="status-dashboard-header">
            <div class="status-dashboard-title">
                <span id="dashboardTitle">📊 Processing Status</span>
            </div>
            <div class="status-dashboard-phase" id="phaseIndicator">
                <span id="phaseLabel">Phase 1 - Audio Detection</span>
            </div>
        </div>

        <div class="status-dashboard-metrics">
            <div class="status-metric">
                <div class="status-metric-label">Dives Found</div>
                <div class="status-metric-value" id="metricDives">0/0</div>
            </div>
            <div class="status-metric">
                <div class="status-metric-label">Processing Speed</div>
                <div class="status-metric-value" id="metricSpeed">0.0 dives/min</div>
            </div>
            <div class="status-metric">
                <div class="status-metric-label">Time Remaining</div>
                <div class="status-metric-value" id="metricTime">--:--</div>
            </div>
            <div class="status-metric">
                <div class="status-metric-label">Thumbnails Ready</div>
                <div class="status-metric-value" id="metricThumbnails">0/0</div>
            </div>
        </div>

        <div class="progress-container">
            <div class="progress-bar-wrapper">
                <div class="progress-bar">
                    <div class="progress-bar-fill" id="progressBarFill"></div>
                </div>
            </div>
            <div class="progress-percent" id="progressPercent">0%</div>
        </div>
    </div>

    <!-- FEAT-08: Connection Status Banner -->
    <div class="status-connection-banner" id="connectionBanner">
        <span class="status-spinner" id="bannerSpinner"></span>
        <span id="bannerText">Reconnecting...</span>
        <button id="bannerRetryBtn" style="margin-left: auto; background: rgba(255,255,255,0.2); border: 1px solid white; color: white; padding: 4px 8px; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: 500;">Retry</button>
    </div>

    <div class="container">
        <div class="header">
            <h1>🤿 Dive Review Gallery</h1>
            <p>Quick review and filter detected dives</p>
            <div class="stats">
                <div class="stat">
                    <span>Total Dives:</span>
                    <span class="stat-value" id="total-dives">{len(self.dives)}</span>
                </div>
                <div class="stat">
                    <span>Selected for Delete:</span>
                    <span class="stat-value" id="selected-count">0</span>
                </div>
                <div class="stat">
                    <span>To Keep:</span>
                    <span class="stat-value" id="keep-count">{len(self.dives)}</span>
                </div>
            </div>
        </div>

        <div class="gallery" id="gallery">
            <!-- FEAT-03: Initial empty gallery message -->
            <div class="empty-gallery-message" id="emptyMessage">
                <div class="empty-gallery-message-icon">⏳</div>
                <div>Waiting for dives...</div>
            </div>
"""

        # Replace simple Python placeholders inside the static HTML (avoids f-string parsing of JS braces)
        html = html.replace('{len(self.dives)}', str(len(self.dives)))

        # Add dive cards
        for dive in self.dives:
            thumbnails_html = ""
            for thumb in [dive["thumbnails"]["start"], dive["thumbnails"]["middle"], dive["thumbnails"]["end"]]:
                if thumb:
                    thumbnails_html += f'<img class="thumbnail" src="{thumb}" alt="frame">'
                else:
                    thumbnails_html += '<div class="thumbnail" style="background: #ddd;"></div>'

            # Prepare timeline thumbnails for embedding in data attribute
            timeline_data = dive.get("timeline_thumbnails", [])
            # Filter out None values and escape for JSON
            timeline_json = json.dumps(timeline_data)

            html += f"""
            <div class="dive-card" data-id="{dive['id']}" data-file="{dive['filename']}" data-timeline='{timeline_json}'>
                <div class="checkbox">
                    <input type="checkbox" class="dive-checkbox">
                </div>
                <div class="thumbnails">
                    {thumbnails_html}
                </div>
                <div class="dive-info">
                    <div class="dive-number">Dive #{dive['id']:02d}</div>
                    <div class="dive-details">
                        <div class="detail-row">
                            <span class="detail-label">Duration:</span>
                            <span class="detail-value">{dive['duration']:.1f}s</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Status:</span>
                            <span class="detail-value">Extracted</span>
                        </div>
                    </div>
                    <div class="confidence {dive['confidence']}">{dive['confidence'].upper()}</div>
                </div>
            </div>
"""

        html += """
        </div>

        <div class="footer">
            <div class="status-message" id="status-message"></div>

            <div class="controls">
                <button class="btn-select-all" id="btn-select-all">Select All</button>
                <button class="btn-deselect-all" id="btn-deselect-all">Deselect All</button>
                <button class="btn-watch" id="btn-watch">Watch Selected</button>
                <button class="btn-delete" id="btn-delete">Delete Selected</button>
                <button class="btn-accept" id="btn-accept">Accept All & Close</button>
            </div>

            <div class="video-player" id="video-player">
                <video width="640" height="360" controls style="width: 100%; max-width: 640px;">
                    <source id="video-source" src="" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>

            <div class="shortcuts">
                <h4>⌨️ Keyboard Shortcuts</h4>
                <div class="shortcut-list">
                    <div class="shortcut">
                        <span class="shortcut-key">←→</span>
                        <span>Navigate dives</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">Space</span>
                        <span>Toggle current dive</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">D</span>
                        <span>Delete selected</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">A</span>
                        <span>Select all</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">Ctrl+A</span>
                        <span>Deselect all</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">W</span>
                        <span>Watch selected</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">Enter</span>
                        <span>Accept all</span>
                    </div>
                    <div class="shortcut">
                        <span class="shortcut-key">?</span>
                        <span>Show help</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- FEAT-02: Connection Status Indicator -->
    <div class="connection-status" id="connectionStatus">
        <span class="status-indicator connecting"></span>
        <span id="statusText">Connecting...</span>
        <span class="status-address" id="statusAddress"></span>
    </div>

    <!-- FEAT-02: Event Log Display -->
    <div class="event-log-container" id="eventLogContainer">
        <div class="event-log-header">
            <span>Live Events</span>
            <button class="event-log-close" id="closeEventLog">&times;</button>
        </div>
        <div class="event-log" id="eventLog"></div>
    </div>

    <!-- FEAT-02: Toggle Event Log Button -->
    <button class="toggle-event-log" id="toggleEventLog" title="Show/hide live events">📋</button>

    <!-- Detailed Review Modal -->
    <div class="modal-overlay" id="diveModal">
        <div class="modal-container">
            <div class="modal-header">
                <div class="modal-title" id="modalTitle">Dive #001</div>
                <button class="modal-close" id="modalCloseBtn">&times;</button>
            </div>
            <div class="modal-content">
                <div class="timeline-section">
                    <div class="timeline-label">Timeline</div>
                    <div class="timeline-frames" id="timelineFrames">
                        <!-- 8 frames will be inserted here by JavaScript -->
                    </div>
                </div>
                <div class="info-panel">
                    <div class="info-row">
                        <span class="info-label">Duration:</span>
                        <span class="info-value" id="modalDuration">0.0s</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Confidence:</span>
                        <span class="info-value" id="modalConfidence">HIGH</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">File:</span>
                        <span class="info-value" id="modalFilename">dive_001.mp4</span>
                    </div>
                </div>
            </div>
            <div class="modal-actions">
                <button class="modal-btn modal-btn-keep" id="modalKeepBtn">
                    Keep <span class="keyboard-hint">(K)</span>
                </button>
                <button class="modal-btn modal-btn-delete" id="modalDeleteBtn">
                    Delete <span class="keyboard-hint">(D)</span>
                </button>
                <button class="modal-btn modal-btn-cancel" id="modalCancelBtn">
                    Cancel <span class="keyboard-hint">(Esc)</span>
                </button>
            </div>
        </div>
    </div>

    <script>
        let currentDiveIndex = 0;
        let cards = [];

        // ===== FEAT-05: Status Dashboard Management =====
        class StatusDashboard {{
            constructor() {{
                this.data = {{
                    phase: 'phase_1',
                    phase_name: 'Audio Detection',
                    dives_found: 0,
                    dives_expected: 0,
                    thumbnails_ready: 0,
                    thumbnails_expected: 0,
                    processing_speed: 0,
                    elapsed_seconds: 0,
                    progress_percent: 0
                }};
                this.phaseColors = {{
                    'phase_1': '#3b82f6',
                    'phase_2': '#eab308',
                    'phase_3': '#22c55e'
                }};
                this.phaseNames = {{
                    'phase_1': 'Phase 1 - Audio Detection',
                    'phase_2': 'Phase 2 - Motion Detection',
                    'phase_3': 'Phase 3 - Person Detection'
                }};
            }}

            update(statusData) {{
                // Merge new data
                Object.assign(this.data, statusData);
                this._render();
            }}

            _render() {{
                requestAnimationFrame(() => {{
                    const dashboard = document.getElementById('statusDashboard');
                    if (!dashboard) return;

                    // Update phase indicator
                    const phaseIndicator = document.getElementById('phaseIndicator');
                    const phaseLabel = document.getElementById('phaseLabel');
                    if (phaseIndicator && phaseLabel) {{
                        phaseIndicator.className = `status-dashboard-phase ${{this.data.phase}}`;
                        phaseLabel.textContent = this.data.phase_name || this.phaseNames[this.data.phase];
                    }}

                    // Update metrics
                    const diveText = `${{this.data.dives_found}}/${{this.data.dives_expected}}`;
                    const speedText = `${{this.data.processing_speed.toFixed(2)}} dives/min`;
                    const timeText = this._formatTimeRemaining();
                    const thumbText = `${{this.data.thumbnails_ready}}/${{this.data.thumbnails_expected}}`;

                    document.getElementById('metricDives').textContent = diveText;
                    document.getElementById('metricSpeed').textContent = speedText;
                    document.getElementById('metricTime').textContent = timeText;
                    document.getElementById('metricThumbnails').textContent = thumbText;

                    // Update progress bar
                    const percent = Math.min(100, this.data.progress_percent || 0);
                    const fill = document.getElementById('progressBarFill');
                    const percentLabel = document.getElementById('progressPercent');
                    if (fill) fill.style.width = percent + '%';
                    if (percentLabel) percentLabel.textContent = Math.round(percent) + '%';
                }});
            }}

            _formatTimeRemaining() {{
                if (this.data.processing_speed <= 0 || this.data.dives_expected <= 0) {{
                    return '--:--';
                }}

                // Calculate remaining dives
                const remaining = this.data.dives_expected - this.data.dives_found;
                if (remaining <= 0) return '0:00';

                // Calculate time in minutes
                const timeMinutes = remaining / this.data.processing_speed;
                const mins = Math.floor(timeMinutes);
                const secs = Math.round((timeMinutes - mins) * 60);

                return `${{mins}}:${{String(secs).padStart(2, '0')}}`;
            }}
        }}

        // ===== FEAT-08: Enhanced EventStreamConsumer with Reconnection =====
        class EventStreamConsumer {{
            constructor(serverUrl = null) {{
                this.serverUrl = serverUrl || this._detectServerUrl();
                this.eventSource = null;
                this.isConnected = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.baseReconnectDelay = 1000;  // Start at 1s
                this.eventLog = [];
                this.maxLogEntries = 100;
                this.cachedEvents = [];
                this.maxCachedEvents = 500;
                this.lastEventId = null;
                this.pollingInterval = null;
                this.pollingActive = false;
                this.statusDashboard = null;
                this._initializeCache();
            }}

            _initializeCache() {{
                // Load cached events from localStorage
                try {{
                    const cached = localStorage.getItem('diveanalyzer_events');
                    if (cached) {{
                        this.cachedEvents = JSON.parse(cached);
                        console.log(`SSE: Loaded ${{this.cachedEvents.length}} cached events from localStorage`);
                    }}
                }} catch (e) {{
                    console.warn('SSE: Could not load cached events:', e);
                }}
            }}

            _saveToCache(event) {{
                try {{
                    // Skip caching very large events (thumbnails) to avoid localStorage quota errors
                    if (event && event.event_type && event.event_type.startsWith && event.event_type.startsWith('thumbnail')) {{
                        // Don't cache thumbnail_frame_ready or thumbnail_ready events
                        return;
                    }}

                    // Avoid caching extremely large payloads
                    let approx = '';
                    try {{ approx = JSON.stringify(event); }} catch (e) {{ approx = ''; }}
                    if (approx.length > 100000) {{
                        console.warn('SSE: Skipping caching of oversized event');
                        return;
                    }}

                    // Add to in-memory cache
                    this.cachedEvents.push(event);

                    // Trim if too large
                    if (this.cachedEvents.length > this.maxCachedEvents) {{
                        this.cachedEvents.shift();
                    }}

                    // Save to localStorage (best-effort)
                    try {{
                        localStorage.setItem('diveanalyzer_events', JSON.stringify(this.cachedEvents));
                    }} catch (e) {{
                        console.warn('SSE: Could not save to localStorage:', e);
                    }}
                }} catch (e) {{
                    console.warn('SSE: _saveToCache internal error:', e);
                }}
            }}

            setStatusDashboard(dashboard) {{
                this.statusDashboard = dashboard;
            }}

            _detectServerUrl() {{
                // Try to detect server URL from current location
                // Default: http://localhost:8765
                if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {{
                    // If running on a different host, use that host
                    return `http://${{window.location.hostname}}:8765`;
                }}
                return 'http://localhost:8765';
            }}

            connect() {{
                console.log(`SSE: Attempting to connect to ${{this.serverUrl}}/events (attempt ${{this.reconnectAttempts + 1}}/${{this.maxReconnectAttempts}})`);
                this._updateStatus('connecting', `Connecting (attempt ${{this.reconnectAttempts + 1}})`);

                try {{
                    this.eventSource = new EventSource(`${{this.serverUrl}}/events`);

                    // Handle different event types
                    const eventTypes = [
                        'connected',
                        'splash_detection_complete',
                        'motion_detection_complete',
                        'person_detection_complete',
                        'dives_detected',
                        'extraction_complete',
                        'processing_complete',
                        'status_update',
                        'dive_detected',
                        // FEAT-07 & FEAT-04: Thumbnail generation events
                        'thumbnail_ready',
                        'thumbnail_frame_ready',
                        'thumbnail_generation_complete'
                    ];

                    eventTypes.forEach(eventType => {{
                        this.eventSource.addEventListener(eventType, (event) => {{
                            this._handleEvent(eventType, event);
                        }});
                    }});

                    // Handle generic message events
                    this.eventSource.addEventListener('message', (event) => {{
                        try {{
                            const data = JSON.parse(event.data);
                            this._handleEvent('message', event);
                        }} catch (e) {{
                            console.warn('SSE: Could not parse message event:', e);
                        }}
                    }});

                    // Handle connection open
                    this.eventSource.onopen = () => {{
                        console.log('SSE: Connection opened');
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this._updateStatus('connected', 'Connected');
                        this._hideConnectionBanner();
                        this._stopPolling();
                        this._logEvent('connected', 'Connected to server', 'success');
                    }};

                    // Handle connection errors
                    this.eventSource.onerror = (error) => {{
                        console.error('SSE: Connection error:', error);
                        this._handleConnectionError();
                    }};

                }} catch (error) {{
                    console.error('SSE: Failed to create EventSource:', error);
                    this._handleConnectionError();
                }}
            }}

            _getReconnectDelay(attempt) {{
                // Exponential backoff: 1s, 2s, 4s, 8s, 8s
                const delays = [1000, 2000, 4000, 8000, 8000];
                return delays[Math.min(attempt, delays.length - 1)];
            }}

            _handleEvent(eventType, event) {{
                try {{
                    const data = JSON.parse(event.data);
                    console.log(`SSE: Event received - ${{eventType}}:`, data);

                    // Cache all events
                    this._saveToCache({{ event_type: eventType, data, timestamp: new Date().toISOString() }});

                    // Update last event timestamp and message
                    this._updateLatestEvent(eventType, data);

                    // Log the event
                    this._logEvent(
                        eventType,
                        `${{eventType}}: ${{JSON.stringify(data).substring(0, 100)}}`,
                        this._getEventLogType(eventType)
                    );

                    // FEAT-05: Handle status_update events for dashboard
                    if (eventType === 'status_update' && this.statusDashboard) {{
                        this.statusDashboard.update(data);
                    }}

                    // FEAT-03: Handle dive_detected events to render placeholders
                    if (eventType === 'dive_detected' && data.dive_index !== undefined) {{
                        renderDiveCardPlaceholder({{
                            dive_index: data.dive_index,
                            dive_id: data.dive_id || `dive_${{data.dive_index}}`,
                            duration: data.duration || 0,
                            confidence: data.confidence || 0
                        }});
                    }}

                    // FEAT-04: Handle thumbnail_ready events for progressive loading
                    if (eventType === 'thumbnail_ready' && data.dive_id !== undefined) {{
                        updateThumbnailInPlace(data.dive_id, data.frames);
                    }}

                    // FEAT-09: Handle files_deleted events to keep UI in sync
                    if (eventType === 'files_deleted' && data.deleted) {{
                        try {{
                            const deletedFiles = data.deleted;
                            console.log('SSE: files_deleted event received:', deletedFiles);
                            deletedFiles.forEach(fname => {{
                                const card = document.querySelector(`.dive-card[data-file="${{fname}}"]`);
                                if (card) {{
                                    card.style.opacity = '0';
                                    card.style.transform = 'scale(0.9)';
                                    setTimeout(() => {{ card.style.display = 'none'; updateStats(); }}, 300);
                                }}
                            }});
                        }} catch (e) {{
                            console.warn('Error handling files_deleted event:', e);
                        }}
                    }}

                    // FEAT-04: Handle thumbnail_frame_ready events for individual frame updates
                    if (eventType === 'thumbnail_frame_ready' && data.dive_id !== undefined) {{
                        updateThumbnailFrame(data.dive_id, data.frame_index, data.frame_data);
                    }}

                    // FEAT-07: Handle thumbnail_generation_complete for final status
                    if (eventType === 'thumbnail_generation_complete') {{
                        console.log(`Thumbnail generation complete: ${{data.completed_count}}/${{data.total_dives}} thumbnails`);
                    }}

                }} catch (error) {{
                    console.warn(`SSE: Error parsing event data:`, error);
                }}
            }}

            _handleConnectionError() {{
                this.isConnected = false;
                this._updateStatus('disconnected', 'Disconnected');
                this._logEvent('error', 'Connection lost', 'error');

                if (this.reconnectAttempts < this.maxReconnectAttempts) {{
                    this.reconnectAttempts++;
                    const delay = this._getReconnectDelay(this.reconnectAttempts - 1);
                    console.log(`SSE: Attempting to reconnect in ${{delay}}ms (attempt ${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}})`);
                    this._showConnectionBanner('reconnecting', this.reconnectAttempts);

                    setTimeout(() => {{
                        if (!this.isConnected) {{
                            this.connect();
                        }}
                    }}, delay);
                }} else {{
                    console.error('SSE: Max reconnection attempts exceeded');
                    this._updateStatus('disconnected', 'Connection Lost');
                    this._showConnectionBanner('error', this.reconnectAttempts);
                    this._logEvent('error', 'Connection lost - starting fallback polling', 'error');
                    // Start polling fallback
                    this._startPolling();
                }}
            }}

            _updateLatestEvent(eventType, data) {{
                requestAnimationFrame(() => {{
                    const statusEl = document.getElementById('statusText');
                    if (statusEl) {{
                        const timestamp = new Date().toLocaleTimeString();
                        statusEl.textContent = `${{timestamp}} - ${{eventType}}`;
                    }}
                }});
            }}

            _updateStatus(status, message) {{
                requestAnimationFrame(() => {{
                    const statusEl = document.getElementById('connectionStatus');
                    if (!statusEl) return;

                    // Update status classes
                    statusEl.classList.remove('connected', 'disconnected', 'connecting');
                    statusEl.classList.add(status);

                    // Update indicator
                    const indicator = statusEl.querySelector('.status-indicator');
                    if (indicator) {{
                        indicator.classList.remove('connected', 'disconnected', 'connecting');
                        indicator.classList.add(status);
                    }}

                    // Update text
                    const textEl = document.getElementById('statusText');
                    if (textEl) {{
                        textEl.textContent = message;
                    }}

                    // Update address
                    const addressEl = document.getElementById('statusAddress');
                    if (addressEl) {{
                        addressEl.textContent = this.serverUrl;
                    }}
                }});
            }}

            _logEvent(type, message, logType = 'info') {{
                const now = new Date();
                const timeStr = now.toLocaleTimeString();
                const entry = {{ type, message, logType, timestamp: timeStr }};

                this.eventLog.push(entry);
                if (this.eventLog.length > this.maxLogEntries) {{
                    this.eventLog.shift();
                }}

                requestAnimationFrame(() => {{
                    this._renderEventLog();
                }});
            }}

            _renderEventLog() {{
                const logEl = document.getElementById('eventLog');
                if (!logEl) return;

                logEl.innerHTML = this.eventLog.map(entry => `
                    <div class="event-log-entry ${{entry.logType}}">
                        <span class="event-log-timestamp">${{entry.timestamp}}</span>
                        <span>${{entry.message}}</span>
                    </div>
                `).join('');

                // Auto-scroll to bottom
                logEl.scrollTop = logEl.scrollHeight;
            }}

            _getEventLogType(eventType) {{
                if (eventType === 'connected') return 'success';
                if (eventType === 'error') return 'error';
                if (eventType.includes('complete') || eventType.includes('detected')) return 'info';
                return 'info';
            }}

            _startPolling() {
                if (this.pollingActive) return;

                console.log('SSE: Starting fallback polling every 3 seconds');
                this.pollingActive = true;
                this._pollOnce();  // Start immediately
            }

            _pollOnce() {
                if (!this.pollingActive) return;

                fetch(`${this.serverUrl}/events-history`)
                    .then(res => res.json())
                    .then(data => {
                        if (data && data.events && Array.isArray(data.events)) {
                            // Process new events not yet cached
                            data.events.forEach(event => {
                                const cached = this.cachedEvents.some(e =>
                                    e.timestamp === event.timestamp && e.event_type === event.event_type
                                );
                                if (!cached) {
                                    this._handleEvent(event.event_type, { data: event.data });
                                }
                            });
                        }
                    })
                    .catch(err => {
                        console.error('Polling failed:', err);
                        this._logEvent('error', 'Polling failed', 'error');
                    })
                    .finally(() => {
                        if (this.pollingActive) {
                            this.pollingInterval = setTimeout(() => this._pollOnce(), 3000);
                        }
                    });
            }

            _stopPolling() {
                if (this.pollingInterval) {
                    clearTimeout(this.pollingInterval);
                    this.pollingInterval = null;
                }
                this.pollingActive = false;
                console.log('SSE: Stopped polling');
            }

            _showConnectionBanner(state, attemptNumber = 0) {{
                const banner = document.getElementById('connectionBanner');
                const text = document.getElementById('bannerText');
                const spinner = document.getElementById('bannerSpinner');
                const btn = document.getElementById('bannerRetryBtn');

                if (!banner) return;

                if (state === 'reconnecting') {{
                    banner.classList.add('show');
                    banner.classList.remove('error');
                    text.textContent = `Reconnecting... (${{attemptNumber}}/${{this.maxReconnectAttempts}})`;
                    spinner.style.display = 'inline-block';
                    btn.style.display = 'none';
                }} else if (state === 'error') {{
                    banner.classList.add('show', 'error');
                    text.textContent = 'Connection lost - Using cached data';
                    spinner.style.display = 'none';
                    btn.style.display = 'inline-block';
                }}
            }}

            _hideConnectionBanner() {{
                const banner = document.getElementById('connectionBanner');
                if (banner) {{
                    banner.classList.remove('show', 'error');
                }}
            }}

            disconnect() {{
                if (this.eventSource) {{
                    this.eventSource.close();
                    this.eventSource = null;
                    this.isConnected = false;
                    console.log('SSE: Disconnected');
                    this._updateStatus('disconnected', 'Disconnected');
                }}
                this._stopPolling();
            }}

            getCachedEvents() {{
                return this.cachedEvents;
            }}

            getLogEntries() {{
                return this.eventLog;
            }}

            clearLog() {{
                this.eventLog = [];
                this._renderEventLog();
            }}
        }}

        // ===== FEAT-03: Dive Card Placeholder System =====

        /**
         * Render a placeholder dive card with shimmer animation
         * @param {{dive_index: number, dive_id: string, duration: number, confidence: number}} diveData
         */
        function renderDiveCardPlaceholder(diveData) {{
            const gallery = document.getElementById('gallery');
            if (!gallery) {{
                console.warn('Gallery element not found');
                return;
            }}

            // Remove empty gallery message on first card
            const emptyMessage = document.getElementById('emptyMessage');
            if (emptyMessage && document.querySelectorAll('.dive-card:not(.placeholder-card)').length === 0) {{
                emptyMessage.remove();
                console.log('FEAT-03: Empty gallery message removed');
            }}

            // Create placeholder card element
            const card = document.createElement('div');
            card.className = 'dive-card placeholder-card';
            card.dataset.id = diveData.dive_index;

            // Create placeholder content
            const placeholderHTML = `
                <div class="checkbox">
                    <input type="checkbox" class="dive-checkbox">
                </div>
                <div class="placeholder-thumbnails">
                    <div class="placeholder-thumbnail"></div>
                    <div class="placeholder-thumbnail"></div>
                    <div class="placeholder-thumbnail"></div>
                </div>
                <div class="placeholder-info">
                    <div class="placeholder-number"></div>
                    <div class="placeholder-details">
                        <div class="placeholder-detail"></div>
                        <div class="placeholder-detail"></div>
                    </div>
                    <div class="placeholder-confidence"></div>
                </div>
            `;

            card.innerHTML = placeholderHTML;

            // Add to gallery with smooth fade-in
            gallery.appendChild(card);

            // Update gallery with new card (exclude placeholder cards from cards array)
            cards = Array.from(document.querySelectorAll('.dive-card:not(.placeholder-card)'));

            // Log for debugging
            console.log('FEAT-03: Placeholder card rendered for dive ' + diveData.dive_index + ', total cards: ' + document.querySelectorAll('.dive-card').length);
        }}

        // ===== END FEAT-03 =====

        // ===== FEAT-04: Progressive Thumbnail Loading =====

        /**
         * Update thumbnail in place with fade-in animation (FEAT-04).
         *
         * This function updates a placeholder card with actual thumbnail frames
         * when they become available from the background generation thread.
         * It performs a smooth fade-out/fade-in transition.
         *
         * @param {{number}} diveId - The dive ID to update
         * @param {{Array}} frames - Array of base64 frame data URLs
         */
        function updateThumbnailInPlace(diveId, frames) {{
            const card = document.querySelector(`[data-id="${{diveId}}"]`);
            if (!card) {{
                console.warn(`FEAT-04: Could not find card for dive ${{diveId}}`);
                return;
            }}

            const thumbArea = card.querySelector('.placeholder-thumbnails') || card.querySelector('.thumbnails');
            if (!thumbArea) {{
                console.warn(`FEAT-04: Could not find thumbnail area for dive ${{diveId}}`);
                return;
            }}

            // Start fade-out animation
            thumbArea.style.transition = 'opacity 0.2s ease-out';
            thumbArea.style.opacity = '0';

            // Swap images after fade-out completes
            setTimeout(() => {{
                try {{
                    // Create frame grid HTML
                    let frameHTML = '';
                    if (Array.isArray(frames) && frames.length > 0) {{
                        frameHTML = frames.map((frame, idx) => {{
                            if (frame) {{
                                return `<img src="${{frame}}" class="thumbnail" style="flex: 1; object-fit: cover;" alt="Frame ${{idx}}" />`;
                            }}
                            return '';
                        }}).join('');
                    }}

                    // Update the thumbnails container
                    thumbArea.innerHTML = frameHTML || '<div style="flex: 1; background: #ddd;"></div>';
                    thumbArea.className = 'thumbnails';  // Replace placeholder-thumbnails with thumbnails

                    // Fade back in
                    thumbArea.style.opacity = '1';
                    thumbArea.style.transition = 'opacity 0.3s ease-in';

                    console.log(`FEAT-04: Updated thumbnails for dive ${{diveId}} (${{frames.length}} frames)`);
                }} catch (error) {{
                    console.error(`FEAT-04: Error updating thumbnails for dive ${{diveId}}:`, error);
                }}
            }}, 200);  // Match CSS transition duration
        }}

        /**
         * Update individual thumbnail frame in place (FEAT-04).
         *
         * This function updates a single frame in the thumbnail grid as it becomes
         * available from the background generation thread. Allows for progressive
         * frame-by-frame updates.
         *
         * @param {{number}} diveId - The dive ID to update
         * @param {{number}} frameIndex - Index of the frame (0-7)
         * @param {{string}} frameData - Base64 frame data URL
         */
        function updateThumbnailFrame(diveId, frameIndex, frameData) {{
            const card = document.querySelector(`[data-id="${{diveId}}"]`);
            if (!card) return;

            const thumbArea = card.querySelector('.placeholder-thumbnails') || card.querySelector('.thumbnails');
            if (!thumbArea) return;

            // Get or create frames container
            let frames = thumbArea.querySelectorAll('img');

            if (frameIndex === 0 && frames.length === 0) {{
                // First frame - convert placeholder to actual frames container
                thumbArea.className = 'thumbnails';
                thumbArea.innerHTML = '';
                // Create 8 empty slots
                for (let i = 0; i < 8; i++) {{
                    const img = document.createElement('img');
                    img.className = 'thumbnail';
                    img.style.flex = '1';
                    img.style.objectFit = 'cover';
                    img.style.background = '#ddd';
                    img.style.opacity = '0.5';
                    thumbArea.appendChild(img);
                }}
                frames = thumbArea.querySelectorAll('img');
            }}

            // Update specific frame with fade-in
            if (frameIndex < frames.length && frameData) {{
                const img = frames[frameIndex];
                img.src = frameData;
                img.style.opacity = '1';
                img.style.transition = 'opacity 0.3s ease-in';
            }}
        }}

        // ===== END FEAT-04 =====

        // Initialize event consumer and status dashboard
        let eventConsumer = null;
        let statusDashboard = null;

        function initializeEventConsumer() {{
            // Detect server URL
            const serverUrl = window.location.hostname === 'file:'
                ? 'http://localhost:8765'
                : `http://${{window.location.hostname}}:8765`;

            // FEAT-05: Initialize status dashboard
            statusDashboard = new StatusDashboard();

            eventConsumer = new EventStreamConsumer(serverUrl);
            eventConsumer.setStatusDashboard(statusDashboard);

            // Expose server URL globally for other functions (delete requests)
            try {{
                window.SERVER_URL = serverUrl;
                console.log('Server URL set to', window.SERVER_URL);
            }} catch (e) {{
                // ignore if window not available
            }}

            // Set up retry button in connection banner
            const retryBtn = document.getElementById('bannerRetryBtn');
            if (retryBtn) {{
                retryBtn.addEventListener('click', () => {{
                    console.log('Manual reconnect requested');
                    eventConsumer.reconnectAttempts = 0;
                    eventConsumer.connect();
                }});
            }}

            // Set up event log controls
            const toggleBtn = document.getElementById('toggleEventLog');
            const closeBtn = document.getElementById('closeEventLog');
            const container = document.getElementById('eventLogContainer');

            if (toggleBtn) {{
                toggleBtn.addEventListener('click', () => {{
                    if (container.classList.contains('show')) {{
                        container.classList.remove('show');
                        toggleBtn.classList.add('hidden');
                    }} else {{
                        container.classList.add('show');
                        toggleBtn.classList.remove('hidden');
                    }}
                }});
            }}

            if (closeBtn) {{
                closeBtn.addEventListener('click', () => {{
                    container.classList.remove('show');
                    if (toggleBtn) {{
                        toggleBtn.classList.remove('hidden');
                    }}
                }});
            }}

            // Try to connect to event stream
            try {{
                eventConsumer.connect();
            }} catch (error) {{
                console.warn('SSE: Could not initialize event consumer:', error);
                // Continue anyway - gallery works without live events
            }}
        }}

        // ===== END FEAT-02 =====

        // Initialize after DOM is ready
        function initGallery() {{
            cards = Array.from(document.querySelectorAll('.dive-card'));

            // FEAT-02: Initialize real-time event consumer
            initializeEventConsumer();

            // Make entire card clickable to toggle selection or open modal
            cards.forEach((card, index) => {{
                card.style.cursor = 'pointer';

                card.addEventListener('click', (e) => {{
                    // Don't handle if clicking the checkbox directly
                    if (e.target.matches('input[type="checkbox"]')) {{
                        return;
                    }}

                    // Check if double-click or using Ctrl/Cmd to open modal
                    if (e.detail === 2 || e.ctrlKey || e.metaKey) {{
                        // Double-click or Ctrl+click: open detailed modal
                        openDiveModal(index);
                    }} else {{
                        // Single click: toggle checkbox
                        const checkbox = card.querySelector('.dive-checkbox');
                        checkbox.checked = !checkbox.checked;
                        updateStats();

                        // Focus this card
                        currentDiveIndex = index;
                        focusCard(index);
                    }}
                }});
            }});

            // Add change handler to checkboxes (for manual clicks)
            document.querySelectorAll('.dive-checkbox').forEach(checkbox => {{
                checkbox.addEventListener('change', (e) => {{
                    e.stopPropagation();
                    updateStats();
                }});
            }});

            // Add button event listeners
            document.getElementById('btn-select-all').addEventListener('click', selectAll);
            document.getElementById('btn-deselect-all').addEventListener('click', deselectAll);
            document.getElementById('btn-watch').addEventListener('click', watchSelected);
            document.getElementById('btn-delete').addEventListener('click', deleteSelected);
            document.getElementById('btn-accept').addEventListener('click', acceptAll);

            // Initialize modal handlers
            initModalHandlers();

            // Set focus on container for keyboard events
            document.body.tabIndex = 0;
            document.body.focus();

            focusCard(0);
        }}

        // Keyboard shortcuts - Works in Safari & all browsers
        window.addEventListener('keydown', function(e) {{
            const key = e.key;
            const code = e.code;
            const modalOpen = document.getElementById('diveModal').classList.contains('show');

            // Debug: log keyboard events
            console.log('Key pressed:', key, code, 'Modal open:', modalOpen);

            // Modal-specific shortcuts (FEAT-07: Enhanced keyboard navigation)
            if (modalOpen) {{
                if (key === 'Escape') {{
                    e.preventDefault();
                    handleCancel();
                    console.log('Escape pressed - closing modal');
                }}
                else if (key.toLowerCase() === 'k' && !e.ctrlKey && !e.metaKey) {{
                    e.preventDefault();
                    handleKeep();
                    console.log('K pressed - keeping dive and advancing');
                }}
                else if (key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey) {{
                    e.preventDefault();
                    handleDelete();
                    console.log('D pressed - deleting dive and advancing');
                }}
                else if (key === 'ArrowRight') {{
                    e.preventDefault();
                    const nextIndex = getNextUndeleted(currentModalDiveIndex);
                    if (nextIndex !== null && !isTransitioning) {{
                        isTransitioning = true;
                        currentModalDiveIndex = nextIndex;
                        openDiveModal(nextIndex);
                        console.log('Right arrow - navigating to next dive:', nextIndex);
                        setTimeout(() => {{ isTransitioning = false; }}, 300);
                    }}
                }}
                else if (key === 'ArrowLeft') {{
                    e.preventDefault();
                    const prevIndex = getPrevUndeleted(currentModalDiveIndex);
                    if (prevIndex !== null && !isTransitioning) {{
                        isTransitioning = true;
                        currentModalDiveIndex = prevIndex;
                        openDiveModal(prevIndex);
                        console.log('Left arrow - navigating to prev dive:', prevIndex);
                        setTimeout(() => {{ isTransitioning = false; }}, 300);
                    }}
                }}
                return;  // Don't process gallery shortcuts when modal is open
            }}

            // Gallery shortcuts (only when modal is not open)
            if (key === 'ArrowRight') {{
                e.preventDefault();
                currentDiveIndex = Math.min(currentDiveIndex + 1, cards.length - 1);
                focusCard(currentDiveIndex);
                console.log('Navigation right, now at:', currentDiveIndex);
            }}
            else if (key === 'ArrowLeft') {{
                e.preventDefault();
                currentDiveIndex = Math.max(currentDiveIndex - 1, 0);
                focusCard(currentDiveIndex);
                console.log('Navigation left, now at:', currentDiveIndex);
            }}
            else if (code === 'Space' || key === ' ') {{
                e.preventDefault();
                const checkbox = cards[currentDiveIndex].querySelector('.dive-checkbox');
                checkbox.checked = !checkbox.checked;
                updateStats();
                console.log('Space: toggled dive', currentDiveIndex, 'checked:', checkbox.checked);
            }}
            else if (key.toLowerCase() === 'd' && !e.ctrlKey && !e.metaKey) {{
                e.preventDefault();
                deleteSelected();
                console.log('Delete key pressed');
            }}
            else if (key.toLowerCase() === 'a' && !e.ctrlKey && !e.metaKey) {{
                e.preventDefault();
                selectAll();
                console.log('Select All key pressed');
            }}
            else if ((e.ctrlKey || e.metaKey) && key.toLowerCase() === 'a') {{
                e.preventDefault();
                deselectAll();
                console.log('Deselect All key pressed');
            }}
            else if (key.toLowerCase() === 'w' && !e.ctrlKey && !e.metaKey) {{
                e.preventDefault();
                watchSelected();
                console.log('Watch key pressed');
            }}
            else if (key === 'Enter') {{
                e.preventDefault();
                acceptAll();
                console.log('Enter key pressed');
            }}
            else if (key === '?') {{
                e.preventDefault();
                showHelp();
            }}
        }});

        function focusCard(index) {{
            cards.forEach(c => {{
                c.style.outline = '';
                c.style.backgroundColor = '';
            }});
            if (cards[index]) {{
                cards[index].style.outline = '3px solid #2196F3';
                cards[index].style.backgroundColor = '#e3f2fd';
                cards[index].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function updateStats() {{
            const checked = document.querySelectorAll('.dive-checkbox:checked').length;
            const total = cards.length;
            document.getElementById('selected-count').textContent = checked;
            document.getElementById('keep-count').textContent = total - checked;

            // Update card styles
            cards.forEach(card => {{
                const checkbox = card.querySelector('.dive-checkbox');
                if (checkbox.checked) {{
                    card.classList.add('deleted');
                }} else {{
                    card.classList.remove('deleted');
                }}
            }});
        }}

        function selectAll() {{
            document.querySelectorAll('.dive-checkbox').forEach(cb => {{
                cb.checked = true;
            }});
            updateStats();
            showMessage('✅ All dives selected for deletion', 'success');
        }}

        function deselectAll() {{
            document.querySelectorAll('.dive-checkbox').forEach(cb => {{
                cb.checked = false;
            }});
            updateStats();
            showMessage('✅ All dives deselected', 'success');
        }}

        function deleteSelected() {{
            const selected = document.querySelectorAll('.dive-checkbox:checked');
            if (selected.length === 0) {{
                showMessage('❌ No dives selected for deletion', 'error');
                return;
            }}
            if (!confirm(`Delete ${{selected.length}} dive(s)? This cannot be undone.`)) {{
                return;
            }}

            const files = [];
            selected.forEach(checkbox => {{
                const card = checkbox.closest('.dive-card');
                files.push(card.dataset.file);
            }});

            console.log('Files to delete:', files);

            // If server URL is available, request deletion from server first
            if (window.SERVER_URL) {{
                showMessage(`🗑️ Deleting ${{files.length}} file(s)...`, 'info');
                console.log('AcceptAll: sending delete request to server:', files);
                fetch(`${{window.SERVER_URL}}/delete`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ files }})
                })
                .then(res => res.json())
                .then(data => {
                    const deleted = data.deleted || [];
                    const failed = data.failed || [];

                    if (deleted.length > 0) {{
                        showMessage(`✅ Deleted ${{deleted.length}} file(s)`, 'success');

                        // Remove deleted files from UI
                        Array.from(selected).forEach((checkbox) => {{
                            const card = checkbox.closest('.dive-card');
                            if (deleted.includes(card.dataset.file)) {{
                                card.style.opacity = '0';
                                card.style.transform = 'scale(0.9)';
                                setTimeout(() => {{
                                    card.style.display = 'none';
                                    updateStats();
                                }}, 300);
                            }}
                        }});
                    }} else {{
                        showMessage('⚠️ No files were deleted', 'warning');
                    }}

                    if (failed.length > 0) {{
                        console.warn('Delete failed for:', failed);
                        showMessage(`⚠️ Failed to delete ${{failed.length}} file(s)`, 'error');
                    }}
                }})
                .catch(err => {{
                    console.error('Delete request failed:', err);
                    showMessage('❌ Delete request failed (server error)', 'error');
                }});
            }} else {{
                // No server: cannot delete files when opened via file://
                showMessage('⚠️ Cannot delete files without server. Run with --enable-server.', 'error');
            }}
        }}

        function watchSelected() {{
            const selected = document.querySelector('.dive-checkbox:checked');
            if (!selected) {{
                showMessage('❌ No dive selected to watch', 'error');
                return;
            }}

            const card = selected.closest('.dive-card');
            const videoFile = card.dataset.file;

            document.getElementById('video-source').src = videoFile;
            const player = document.getElementById('video-player');
            player.classList.add('show');

            const video = player.querySelector('video');
            video.load();
            setTimeout(() => {{
                video.play().catch(e => console.log('Play error:', e));
            }}, 100);
        }}

        function acceptAll() {{
            if (!confirm('Keep remaining dives and close review?')) return;

            // Collect selected files to delete (those checked)
            const selected = document.querySelectorAll('.dive-checkbox:checked');
            const files = [];
            selected.forEach(cb => {{
                const card = cb.closest('.dive-card');
                files.push(card.dataset.file);
            }});

            if (files.length === 0) {{
                // No files to delete - request server shutdown and close
                showMessage('✅ Review complete! Requesting server shutdown...', 'success');
                if (window.SERVER_URL) {{
                    fetch(`${{window.SERVER_URL}}/shutdown`, {{ method: 'POST' }})
                        .finally(() => setTimeout(() => {{ window.close(); }}, 400));
                }} else {{
                    setTimeout(() => {{ window.close(); }}, 400);
                }}
                return;
            }}

            // If server available, request deletion first
            if (window.SERVER_URL) {{
                showMessage(`🗑️ Deleting ${{files.length}} selected file(s) before close...`, 'info');
                fetch(`${{window.SERVER_URL}}/delete`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ files }})
                }})
                .then(res => res.json())
                .then(data => {{
                    const deleted = data.deleted || [];
                    const failed = data.failed || [];

                    if (deleted.length > 0) {{
                        showMessage(`✅ Deleted ${{deleted.length}} file(s). Closing...`, 'success');
                    }} else if (failed.length > 0) {{
                        showMessage(`⚠️ Some files failed to delete. Closing anyway.`, 'warning');
                    }} else {{
                        showMessage('⚠️ No files deleted. Closing...', 'warning');
                    }}

                    setTimeout(() => {{ window.close(); }}, 800);
                }})
                .catch(err => {{
                    console.error('Delete request failed:', err);
                    showMessage('❌ Delete request failed. Closing anyway.', 'error');
                    setTimeout(() => {{ window.close(); }}, 800);
                }});
            }} else {{
                showMessage('⚠️ Cannot delete files without server. Closing without deleting.', 'warning');
                setTimeout(() => {{ window.close(); }}, 800);
            }}
        }}

        function showMessage(text, type) {{
            const msg = document.getElementById('status-message');
            msg.textContent = text;
            msg.className = `status-message show ${{type}}`;
            setTimeout(() => msg.classList.remove('show'), 4000);
        }}

        function showHelp() {{
            alert(`🤿 Dive Review Keyboard Shortcuts:

GALLERY VIEW:
  ← →   Navigate left/right through dives
  Space Toggle current dive for deletion
  A     Select all dives
  ⌘A    Deselect all dives
  D     Delete selected dives
  W     Watch selected dive
  Enter Accept remaining & close
  ?     Show this help

MODAL VIEW (open by double-clicking a dive):
  K     Keep dive and advance to next
  D     Delete dive and auto-advance
  ← →   Navigate between dives without action
  Esc   Close modal and return to gallery
  ?     Show this help`);
        }}

        // ===== MODAL FUNCTIONS =====
        let currentModalDiveIndex = null;
        let isTransitioning = false;  // Prevent double-clicks during transition

        // FEAT-06: Helper functions for checking card state
        function isCardDeleted(index) {{
            return index < cards.length && cards[index].classList.contains('deleted');
        }}

        // FEAT-05: Find next undeleted dive
        function getNextUndeleted(currentIndex) {{
            for (let i = currentIndex + 1; i < cards.length; i++) {{
                if (!isCardDeleted(i)) {{
                    return i;
                }}
            }}
            return null;
        }}

        // FEAT-07: Find previous undeleted dive
        function getPrevUndeleted(currentIndex) {{
            for (let i = currentIndex - 1; i >= 0; i--) {{
                if (!isCardDeleted(i)) {{
                    return i;
                }}
            }}
            return null;
        }}

        function openDiveModal(diveIndex) {{
            if (isTransitioning) return;

            currentModalDiveIndex = diveIndex;
            const card = cards[diveIndex];
            if (!card) {{
                console.log('Card not found for index:', diveIndex);
                return;
            }}

            // Extract duration - parse from card's detail section
            let durationText = '0.0s';
            const durationElements = card.querySelectorAll('.dive-details .detail-value');
            if (durationElements.length > 0) {{
                durationText = durationElements[0].textContent.trim();
            }}

            const confidenceEl = card.querySelector('.confidence');
            const confidenceText = confidenceEl ? confidenceEl.textContent.trim() : 'UNKNOWN';

            const diveData = {{
                id: card.dataset.id,
                filename: card.dataset.file,
                duration: durationText,
                confidence: confidenceText,
                timeline_thumbnails: []
            }};

            console.log('Opening modal for dive:', diveData);

            // Extract timeline frames from card data
            const timelineAttr = card.dataset.timeline;
            if (timelineAttr) {{
                try {{
                    diveData.timeline_thumbnails = JSON.parse(timelineAttr);
                    console.log('Loaded', diveData.timeline_thumbnails.length, 'timeline frames');
                }} catch(e) {{
                    console.log('Could not parse timeline data:', e);
                }}
            }}

            // Ensure modal is shown
            const modal = document.getElementById('diveModal');
            if (!modal.classList.contains('show')) {{
                modal.classList.add('show');
            }}

            // CRITICAL: Reset scroll to top when loading new modal
            const modalContent = modal.querySelector('.modal-content');
            if (modalContent) {{
                modalContent.scrollTop = 0;
                console.log('Modal scroll reset to top');
            }}

            renderTimelineFrames(diveData);
            console.log('Modal updated with new dive data');
        }}

        function closeModal() {{
            const modal = document.getElementById('diveModal');
            modal.classList.remove('show');
            currentModalDiveIndex = null;
        }}

        function renderTimelineFrames(diveData) {{
            const container = document.getElementById('timelineFrames');
            container.innerHTML = '';

            const frames = diveData.timeline_thumbnails || [];

            // If no frames available, show placeholder
            if (frames.length === 0) {{
                container.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">No timeline frames available</div>';
                return;
            }}

            // Add 8 frames
            frames.forEach((frameData, index) => {{
                const frameDiv = document.createElement('div');
                frameDiv.className = 'timeline-frame';

                if (frameData) {{
                    const img = document.createElement('img');
                    img.src = frameData;
                    img.alt = `Frame ${{index + 1}}`;
                    img.title = `${{Math.round(index * 12.5)}}%`;
                    frameDiv.appendChild(img);
                }} else {{
                    frameDiv.style.background = '#ddd';
                    frameDiv.style.display = 'flex';
                    frameDiv.style.alignItems = 'center';
                    frameDiv.style.justifyContent = 'center';
                    frameDiv.textContent = 'N/A';
                }}

                container.appendChild(frameDiv);
            }});

            // Update dive info (FEAT-06: Enhanced info panel)
            document.getElementById('modalTitle').textContent = `Dive #${{String(diveData.id).padStart(3, '0')}}`;
            document.getElementById('modalDuration').textContent = diveData.duration;
            document.getElementById('modalConfidence').textContent = diveData.confidence;
            document.getElementById('modalFilename').textContent = diveData.filename;
        }}

        // FEAT-05: Auto-advance on delete - the core UX feature
        function deleteAndAdvance() {{
            if (isTransitioning || currentModalDiveIndex === null) return;

            isTransitioning = true;
            const currentCard = cards[currentModalDiveIndex];

            // Fade out current card with smooth animation
            currentCard.style.transition = 'all 0.3s ease';
            currentCard.style.opacity = '0';
            currentCard.style.transform = 'scale(0.95)';

            // Find next undeleted dive
            const nextIndex = getNextUndeleted(currentModalDiveIndex);

            // Execute after fade-out completes
            setTimeout(() => {{
                // Hide the card from layout
                currentCard.style.display = 'none';

                // CRITICAL: Set isTransitioning = false BEFORE calling openDiveModal
                // so it doesn't get blocked by the guard check
                isTransitioning = false;

                if (nextIndex !== null) {{
                    // Found next dive - auto-open it
                    currentModalDiveIndex = nextIndex;
                    openDiveModal(nextIndex);
                    showMessage('✅ Next dive loaded', 'success');
                    console.log('Auto-advanced to dive index:', nextIndex);
                }} else {{
                    // No more dives - show completion and close
                    closeModal();
                    showMessage('✅ All decisions made! Review complete.', 'success');
                    console.log('All dives processed');
                }}
            }}, 300);
        }}

        function handleKeep() {{
            if (isTransitioning) return;

            // FEAT-07: Auto-advance with Keep key (like delete but without marking checkbox)
            isTransitioning = true;
            const nextIndex = getNextUndeleted(currentModalDiveIndex);

            if (nextIndex !== null) {{
                // Found next dive - transition to it smoothly
                showMessage('✅ Dive kept, moving to next...', 'success');
                setTimeout(() => {{
                    isTransitioning = false;
                    currentModalDiveIndex = nextIndex;
                    openDiveModal(nextIndex);
                    console.log('Keep: Advanced to dive index:', nextIndex);
                }}, 200);
            }} else {{
                // No more dives - close modal
                closeModal();
                showMessage('✅ All dives reviewed!', 'success');
                isTransitioning = false;
                console.log('Keep: No more dives, review complete');
            }}
        }}

        function handleDelete() {{
            if (isTransitioning || currentModalDiveIndex === null) return;

            const card = cards[currentModalDiveIndex];
            const checkbox = card.querySelector('.dive-checkbox');

            // Mark for deletion
            checkbox.checked = true;
            updateStats();

            // FEAT-05: Auto-advance to next dive
            deleteAndAdvance();
        }}

        function handleCancel() {{
            if (!isTransitioning) {{
                closeModal();
            }}
        }}

        // Modal event listeners - Add to initGallery function
        function initModalHandlers() {{
            document.getElementById('modalCloseBtn').addEventListener('click', handleCancel);
            document.getElementById('modalKeepBtn').addEventListener('click', handleKeep);
            document.getElementById('modalDeleteBtn').addEventListener('click', handleDelete);
            document.getElementById('modalCancelBtn').addEventListener('click', handleCancel);

            // Close modal when clicking overlay
            const overlay = document.getElementById('diveModal');
            overlay.addEventListener('click', (e) => {{
                if (e.target === overlay && !isTransitioning) {{
                    handleCancel();
                }}
            }});
        }}

        // Start when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initGallery);
        }} else {{
            initGallery();
        }}
    </script>
</body>
</html>
"""

        # Normalize doubled braces (produced to escape Python f-strings) back to single.
        # Collapse repeated sequences iteratively so patterns like '{{{' -> '{'
        while '{{' in html:
            html = html.replace('{{', '{')
        while '}}' in html:
            html = html.replace('}}', '}')

        # Save HTML
        with open(output_path, "w") as f:
            f.write(html)

        print(f"✅ Gallery saved to: {output_path}")
        return str(output_path)

    def open_in_browser(self, html_path: Path):
        """Open HTML gallery in default browser.

        Args:
            html_path: Path to HTML file
        """
        try:
            import webbrowser
            webbrowser.open(f"file://{html_path.absolute()}")
            print(f"🌐 Opened gallery in browser: {html_path}")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")
            print(f"📂 Manual open: {html_path}")


def extract_timeline_frames_background(
    video_path: Path,
    dive_id: int,
    server=None,
    width: int = 720,
    height: int = 1280,
    quality: int = 3
) -> List[str]:
    """Extract 8 evenly-spaced frames from dive video for timeline (background).

    FEAT-07 & FEAT-04: Called in background thread to generate thumbnails progressively.
    Emits events as each frame is ready, then a final batch event when complete.

    Frame positions: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
    Resolution: 720x1280 (portrait, optimized for gallery display)
    Quality: JPEG q=3 (best quality, ~200KB per frame)

    Args:
        video_path: Path to dive video file
        dive_id: Dive ID for event emission (used as data-id in HTML)
        server: EventServer instance for event emission (optional)
        width: Frame width in pixels (default 720)
        height: Frame height in pixels (default 1280)
        quality: JPEG quality 1-5 (default 3, best quality)

    Returns:
        List of 8 base64 data URLs (may contain None for failed frames),
        or None if video couldn't be read

    Events emitted:
    - thumbnail_frame_ready: Each frame as it completes (for progressive updates)
    - thumbnail_ready: Final batch when all frames complete (FEAT-04 uses this)
    """
    import tempfile
    import time as time_module

    frames = []
    percentages = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    frame_start_time = time_module.time()

    try:
        # Get video duration
        generator = DiveGalleryGenerator(Path(video_path).parent)
        duration = generator.get_video_duration(video_path)
        if duration <= 0:
            print(f"[FEAT-07] Could not determine duration for {video_path.name}")
            return None
    except Exception as e:
        print(f"[FEAT-07] Error getting video duration: {e}")
        return None

    # Extract each frame at specified percentage
    for frame_index, pct in enumerate(percentages):
        try:
            time_sec = duration * pct

            # Create temp file for thumbnail
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                thumb_path = tmp.name

            try:
                # Extract single frame using ffmpeg
                cmd = [
                    "ffmpeg",
                    "-ss", str(time_sec),
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                    "-q:v", str(quality),
                    "-y",  # Overwrite output
                    thumb_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                # Convert to base64 if successful
                if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                    with open(thumb_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()

                    frame_data_url = f"data:image/jpeg;base64,{img_data}"
                    frames.append(frame_data_url)

                    # FEAT-04: Emit individual frame ready event for progressive loading
                    if server:
                        server.emit("thumbnail_frame_ready", {
                            "dive_id": dive_id,
                            "frame_index": frame_index,
                            "total_frames": len(percentages),
                            "frame_data": frame_data_url,
                            "timestamp": time_module.time()
                        })
                else:
                    # Frame generation failed
                    frames.append(None)

            finally:
                # Clean up temp file
                try:
                    if os.path.exists(thumb_path):
                        os.unlink(thumb_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            print(f"[FEAT-07] ffmpeg timeout for dive {dive_id} frame {frame_index}")
            frames.append(None)
        except Exception as e:
            print(f"[FEAT-07] Error extracting frame {frame_index} for dive {dive_id}: {e}")
            frames.append(None)

    # FEAT-04: Emit complete thumbnail batch event
    # Frontend uses this to update the entire grid at once
    if server and any(f is not None for f in frames):
        valid_frames = [f for f in frames if f is not None]
        server.emit("thumbnail_ready", {
            "dive_id": dive_id,
            "type": "grid",
            "frames": frames,  # Include all frames (with None for failed ones)
            "frame_count": len(valid_frames),
            "total_frames": len(percentages),
            "generation_time_sec": time_module.time() - frame_start_time
        })

    return frames if any(f is not None for f in frames) else None


def generate_thumbnails_deferred(
    dives: List[tuple],
    output_dir: Path,
    server=None,
    timeout_sec: float = 20.0
) -> None:
    """Generate thumbnails in background thread after all dives detected (FEAT-07).

    This function is called in a background thread after Phase 1 (audio detection) completes
    and all video clips are extracted. It generates thumbnails for each dive and emits
    thumbnail_ready events to the server progressively.

    The gallery shows placeholders immediately while thumbnails are generated in background.
    This ensures the UI appears responsive (<3s) before thumbnail generation begins.

    Args:
        dives: List of (dive_num, dive_path) tuples from extract_multiple_dives
        output_dir: Directory containing extracted dive videos (used for validation)
        server: EventServer instance for event emission (optional)
        timeout_sec: Maximum time to spend on thumbnail generation (default: 20s)

    Timeline:
    - Each thumbnail takes ~1-2s to generate (8 frames @ 720x1280 resolution)
    - With timeout_sec=20, expect 10-20 thumbnails per run
    - Remaining thumbnails generated if server still connected
    - Emits thumbnail_ready event after all 8 frames ready for each dive
    """
    import time

    start_time = time.time()
    dive_count = 0
    total_dives = len(dives)

    for dive_num, dive_path in sorted(dives):
        # Check timeout - allow graceful exit
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"[FEAT-07] Thumbnail generation timeout after {elapsed:.1f}s ({dive_count}/{total_dives} completed)")
            break

        try:
            # Validate dive video exists
            video_path = Path(dive_path)
            if not video_path.exists():
                print(f"[FEAT-07] Dive video not found: {video_path}")
                continue

            # Extract 8-frame timeline with server event emission
            # This emits thumbnail_frame_ready for each frame, then thumbnail_ready when complete
            frames = extract_timeline_frames_background(
                video_path,
                dive_id=dive_num,
                server=server
            )

            if frames and any(f is not None for f in frames):
                dive_count += 1
                frame_count = sum(1 for f in frames if f is not None)
                print(f"[FEAT-07] Generated {frame_count}/8 frames for dive {dive_num} ({elapsed:.1f}s elapsed)")
            else:
                print(f"[FEAT-07] Failed to generate frames for dive {dive_num}")

        except Exception as e:
            print(f"[FEAT-07] Error generating thumbnails for dive {dive_num}: {e}")
            continue

    # Emit completion event for frontend status update
    if server:
        elapsed = time.time() - start_time
        server.emit("thumbnail_generation_complete", {
            "completed_count": dive_count,
            "total_dives": total_dives,
            "elapsed_seconds": elapsed,
            "timeout_reached": dive_count < total_dives
        })

    print(f"[FEAT-07] Thumbnail generation complete: {dive_count}/{total_dives} in {elapsed:.1f}s")


def create_review_gallery(output_dir: Path, video_name: str = "") -> Path:
    """Create review gallery for extracted dives.

    Args:
        output_dir: Directory with extracted dive videos
        video_name: Original video filename

    Returns:
        Path to generated HTML file
    """
    generator = DiveGalleryGenerator(output_dir, video_name)
    generator.scan_dives()
    html_path = generator.generate_html()
    generator.open_in_browser(Path(html_path))
    return Path(html_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
        video_name = sys.argv[2] if len(sys.argv) > 2 else ""
        create_review_gallery(output_dir, video_name)
    else:
        print("Usage: python review_gallery.py <output_dir> [video_name]")
