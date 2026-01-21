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

            # Calibrated frame positions based on ground truth analysis:
            # - Flight phase: 22-37 frames at start (20-25% of typical 1-1.5s dive)
            # - Splash moment: around 65-70% (entry technique visible)
            # - Submersion: 95%+ (completion of entry)
            if percentage is not None:
                # Use explicit percentage (0.0 to 1.0)
                time_sec = duration * percentage
            elif position == "start":
                time_sec = duration * 0.20  # 20% - early flight phase showing takeoff
            elif position == "end":
                time_sec = duration * 0.95  # 95% - near complete submersion
            else:  # middle
                time_sec = duration * 0.65  # 65% - splash/entry moment

            # Create temp file for thumbnail
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                thumb_path = tmp.name

            # Extract thumbnail - higher resolution for bigger display
            cmd = [
                "ffmpeg",
                "-ss", str(time_sec),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                "-q:v", str(quality),  # Quality (lower = better, but slower)
                "-y",  # Overwrite
                thumb_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # Convert to base64
            if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                with open(thumb_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                # Clean up temp file
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
        Resolution: 720x1280 (portrait, high quality for small display)
        Quality: 3 (best quality)

        Args:
            video_path: Path to dive video file

        Returns:
            List of 8 base64 data URLs, or empty list if failed
        """
        frames = []
        # 8 frames at these percentages
        percentages = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        for pct in percentages:
            frame_data = self.extract_thumbnail(
                video_path,
                position="middle",  # ignored when percentage is set
                width=720,
                height=1280,
                quality=3,
                percentage=pct
            )
            if frame_data:
                frames.append(frame_data)
            else:
                # Use a placeholder on failure
                frames.append(None)

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
        html = f"""<!DOCTYPE html>
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
    </style>
</head>
<body>
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

        html += f"""
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

        // ===== FEAT-02: SSE Real-Time Event Consumer =====
        class EventStreamConsumer {{
            constructor(serverUrl = null) {{
                this.serverUrl = serverUrl || this._detectServerUrl();
                this.eventSource = null;
                this.isConnected = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 2000;
                this.eventLog = [];
                this.maxLogEntries = 100;
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
                console.log(`SSE: Attempting to connect to ${{this.serverUrl}}/events`);
                this._updateStatus('connecting', `Connecting to ${{this.serverUrl}}`);

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
                        'processing_complete'
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

            _handleEvent(eventType, event) {{
                try {{
                    const data = JSON.parse(event.data);
                    console.log(`SSE: Event received - ${{eventType}}:`, data);

                    // Update last event timestamp and message
                    this._updateLatestEvent(eventType, data);

                    // Log the event
                    this._logEvent(
                        eventType,
                        `${{eventType}}: ${{JSON.stringify(data).substring(0, 100)}}`,
                        this._getEventLogType(eventType)
                    );

                    // FEAT-03: Handle dive_detected events to render placeholders
                    if (eventType === 'dive_detected' && data.dive_index !== undefined) {{
                        renderDiveCardPlaceholder({{
                            dive_index: data.dive_index,
                            dive_id: data.dive_id || `dive_${{data.dive_index}}`,
                            duration: data.duration || 0,
                            confidence: data.confidence || 0
                        }});
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
                    const delay = this.reconnectDelay * this.reconnectAttempts;
                    console.log(`SSE: Attempting to reconnect in ${{delay}}ms (attempt ${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}})`);
                    this._updateStatus('connecting', `Reconnecting... (${{this.reconnectAttempts}}/${{this.maxReconnectAttempts}})`);

                    setTimeout(() => {{
                        if (!this.isConnected) {{
                            this.connect();
                        }}
                    }}, delay);
                }} else {{
                    console.error('SSE: Max reconnection attempts exceeded');
                    this._updateStatus('disconnected', 'Server not available');
                    this._logEvent('error', 'Server not available, using local mode', 'warning');
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

            disconnect() {{
                if (this.eventSource) {{
                    this.eventSource.close();
                    this.eventSource = null;
                    this.isConnected = false;
                    console.log('SSE: Disconnected');
                    this._updateStatus('disconnected', 'Disconnected');
                }}
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
            console.log(`FEAT-03: Placeholder card rendered for dive ${diveData.dive_index}, total cards: ${document.querySelectorAll('.dive-card').length}`);
        }}

        // ===== END FEAT-03 =====

        // Initialize event consumer
        let eventConsumer = null;

        function initializeEventConsumer() {{
            // Detect server URL
            const serverUrl = window.location.hostname === 'file:'
                ? 'http://localhost:8765'
                : `http://${{window.location.hostname}}:8765`;

            eventConsumer = new EventStreamConsumer(serverUrl);

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

            if (confirm(`Delete ${{selected.length}} dive(s)? This cannot be undone.`)) {{
                const files = [];
                selected.forEach(checkbox => {{
                    const card = checkbox.closest('.dive-card');
                    files.push(card.dataset.file);
                }});

                console.log('Files to delete:', files);
                showMessage(`✅ Deleted ${{selected.length}} dive(s)`, 'success');

                // Remove from UI with animation
                Array.from(selected).forEach((checkbox, i) => {{
                    const card = checkbox.closest('.dive-card');
                    setTimeout(() => {{
                        card.style.opacity = '0';
                        card.style.transform = 'scale(0.9)';
                        setTimeout(() => {{
                            card.style.display = 'none';
                            updateStats();
                        }}, 300);
                    }}, i * 50);
                }});
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
            if (confirm('Keep remaining dives and close review?')) {{
                showMessage('✅ Review complete! Closing...', 'success');
                setTimeout(() => {{
                    window.close();
                }}, 1500);
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
