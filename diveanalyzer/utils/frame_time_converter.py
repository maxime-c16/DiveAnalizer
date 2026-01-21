"""
Frame index to elapsed time converter for ground truth annotations.

Converts between frame indices and elapsed time in seconds.
"""

from typing import Union, Tuple, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class FrameTimeEntry:
    """A single dive's frame-based timestamps."""

    id: int
    start_frame: int
    splash_frame: int
    end_frame: int
    dive_type: str = "unknown"
    confidence: str = "medium"
    notes: str = ""

    def to_seconds(self, fps: float) -> Dict[str, float]:
        """Convert frames to elapsed time in seconds.

        Args:
            fps: Frames per second of the video

        Returns:
            Dict with start_time_sec, splash_time_sec, end_time_sec
        """
        return {
            "start_time_sec": self.start_frame / fps,
            "splash_time_sec": self.splash_frame / fps,
            "end_time_sec": self.end_frame / fps,
        }

    def validate(self) -> Tuple[bool, str]:
        """Validate frame order and relationships.

        Returns:
            (is_valid, message)
        """
        if not (self.start_frame < self.splash_frame < self.end_frame):
            return False, f"Frame order invalid: {self.start_frame} < {self.splash_frame} < {self.end_frame}"
        return True, "Valid"


@dataclass
class VideoMetadata:
    """Metadata for a video with frame-based ground truth."""

    filename: str
    fps: float
    duration_sec: float
    total_dives: int
    dives: list  # List of FrameTimeEntry
    confidence_level: str = "unknown"
    notes: str = ""

    def validate_all(self) -> Dict[str, Tuple[bool, str]]:
        """Validate all dives in this video.

        Returns:
            Dict mapping dive_id to (is_valid, message)
        """
        results = {}
        for dive in self.dives:
            is_valid, msg = dive.validate()

            # Additional validation with FPS
            # Constraints calibrated from actual ground truth data
            if is_valid:
                flight_time = (dive.splash_frame - dive.start_frame) / self.fps
                entry_time = (dive.end_frame - dive.splash_frame) / self.fps
                total_time = (dive.end_frame - dive.start_frame) / self.fps

                # Realistic constraints based on actual dive data
                # Flight: 0.3-1.5s (covers normal dives and some over-rotations)
                # Entry: 0.06-1.0s (covers belly flops, quick entries, and slow entries)
                # Total: 0.8-3.3s (covers compressed and extended dives)
                if not (0.3 <= flight_time <= 1.5):
                    msg = f"Flight time {flight_time:.2f}s out of range (0.3-1.5s)"
                    is_valid = False
                elif not (0.06 <= entry_time <= 1.0):
                    msg = f"Entry time {entry_time:.2f}s out of range (0.06-1.0s)"
                    is_valid = False
                elif not (0.8 <= total_time <= 3.3):
                    msg = f"Total time {total_time:.2f}s out of range (0.8-3.3s)"
                    is_valid = False

            results[f"dive_{dive.id}"] = (is_valid, msg)

        return results

    def to_seconds_dict(self) -> Dict[str, Any]:
        """Convert entire video metadata to seconds format.

        Returns:
            Dict in same structure but with elapsed times instead of frames
        """
        dives_in_seconds = []

        for dive in self.dives:
            entry = dive.to_seconds(self.fps)
            entry.update({
                "id": dive.id,
                "dive_type": dive.dive_type,
                "confidence": dive.confidence,
                "notes": dive.notes,
            })
            dives_in_seconds.append(entry)

        return {
            "filename": self.filename,
            "fps": self.fps,
            "duration_sec": self.duration_sec,
            "total_dives": self.total_dives,
            "dives": dives_in_seconds,
            "confidence_level": self.confidence_level,
            "notes": self.notes,
        }


class FrameTimeConverter:
    """Utility for converting between frame indices and elapsed time."""

    @staticmethod
    def _is_placeholder(value: Any) -> bool:
        """Check if a value is a placeholder (not actual numeric frame data).

        Args:
            value: Value to check

        Returns:
            True if value is a placeholder like "FILL_ME_IN" or non-numeric
        """
        if isinstance(value, int):
            return False
        if isinstance(value, str):
            # Check for common placeholder patterns
            placeholder_patterns = ["FILL_ME_IN", "TODO", "FIXME", "?", "N/A", "none", "null"]
            value_lower = value.lower().strip()
            return any(pattern.lower() in value_lower for pattern in placeholder_patterns)
        return True

    @staticmethod
    def frame_to_seconds(frame_index: int, fps: float) -> float:
        """Convert frame index to elapsed time.

        Args:
            frame_index: Frame number (0-indexed)
            fps: Frames per second

        Returns:
            Elapsed time in seconds
        """
        return frame_index / fps

    @staticmethod
    def seconds_to_frame(elapsed_time: float, fps: float) -> int:
        """Convert elapsed time to frame index.

        Args:
            elapsed_time: Time in seconds
            fps: Frames per second

        Returns:
            Frame index (rounded to nearest integer)
        """
        return round(elapsed_time * fps)

    @staticmethod
    def timecode_to_frame(timecode: str, fps: float) -> int:
        """Convert HH:MM:SS.MS timecode to frame index.

        Args:
            timecode: String in format "HH:MM:SS.MS" or "MM:SS.MS"
            fps: Frames per second

        Returns:
            Frame index
        """
        parts = timecode.split(':')

        if len(parts) == 3:
            hours, minutes, seconds = parts
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            total_seconds = int(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(parts[0])

        return round(total_seconds * fps)

    @staticmethod
    def frame_to_timecode(frame_index: int, fps: float) -> str:
        """Convert frame index to HH:MM:SS.MS timecode.

        Args:
            frame_index: Frame number
            fps: Frames per second

        Returns:
            String in format "HH:MM:SS.MS"
        """
        total_seconds = frame_index / fps

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    @staticmethod
    def load_ground_truth_frames(json_file: Path, skip_incomplete: bool = True) -> Dict[str, VideoMetadata]:
        """Load frame-based ground truth from JSON file.

        Args:
            json_file: Path to ground truth JSON file
            skip_incomplete: If True, skip dives with placeholder values (FILL_ME_IN, etc)

        Returns:
            Dict mapping filename to VideoMetadata
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        videos = {}

        for filename, video_data in data.items():
            if filename == "metadata" or filename.startswith("_"):
                continue

            dives = []
            for dive_data in video_data.get("dives", []):
                # Check if this dive has placeholder values
                start_val = dive_data.get("start_frame")
                splash_val = dive_data.get("splash_frame")
                end_val = dive_data.get("end_frame")

                # Skip incomplete dives if requested
                if skip_incomplete:
                    if (FrameTimeConverter._is_placeholder(start_val) or
                        FrameTimeConverter._is_placeholder(splash_val) or
                        FrameTimeConverter._is_placeholder(end_val)):
                        continue

                # Try to convert to integers
                try:
                    dive = FrameTimeEntry(
                        id=int(dive_data["id"]),
                        start_frame=int(start_val),
                        splash_frame=int(splash_val),
                        end_frame=int(end_val),
                        dive_type=dive_data.get("dive_type", "unknown"),
                        confidence=dive_data.get("confidence", "medium"),
                        notes=dive_data.get("notes", ""),
                    )
                    dives.append(dive)
                except (ValueError, TypeError):
                    # Skip dives that can't be converted to integers
                    if not skip_incomplete:
                        raise ValueError(f"Could not parse dive data: {dive_data}")
                    continue

            video = VideoMetadata(
                filename=filename,
                fps=float(video_data["fps"]),
                duration_sec=float(video_data["duration_sec"]),
                total_dives=len(dives),  # Actual count of filled dives
                dives=dives,
                confidence_level=video_data.get("confidence_level", "unknown"),
                notes=video_data.get("notes", ""),
            )

            videos[filename] = video

        return videos

    @staticmethod
    def convert_to_seconds_json(json_file: Path, output_file: Path = None) -> Dict[str, Any]:
        """Convert frame-based ground truth to seconds-based JSON.

        Args:
            json_file: Input ground truth file (frame-based)
            output_file: Optional output file. If provided, saves converted data.

        Returns:
            Dict with all videos converted to seconds
        """
        videos = FrameTimeConverter.load_ground_truth_frames(json_file)

        converted = {
            "metadata": {
                "source": str(json_file),
                "note": "Converted from frame indices to elapsed time",
            }
        }

        for filename, video in videos.items():
            converted[filename] = video.to_seconds_dict()

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(converted, f, indent=2)

        return converted


def validate_ground_truth_frames(json_file: Path, verbose: bool = True) -> Dict[str, Dict]:
    """Validate a frame-based ground truth JSON file.

    Args:
        json_file: Path to ground truth file
        verbose: Print detailed validation output

    Returns:
        Dict with validation results
    """
    converter = FrameTimeConverter()

    # Load with skip_incomplete=False to see ALL dives
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Count incomplete dives before loading
    incomplete_count = 0
    complete_count = 0
    for filename, video_data in data.items():
        if filename == "metadata" or filename.startswith("_"):
            continue
        for dive_data in video_data.get("dives", []):
            start_val = dive_data.get("start_frame")
            splash_val = dive_data.get("splash_frame")
            end_val = dive_data.get("end_frame")
            if (converter._is_placeholder(start_val) or
                converter._is_placeholder(splash_val) or
                converter._is_placeholder(end_val)):
                incomplete_count += 1
            else:
                complete_count += 1

    # Load only complete dives for validation
    videos = converter.load_ground_truth_frames(json_file, skip_incomplete=True)

    results = {
        "valid": True,
        "videos": {},
        "summary": {
            "total_files": len([k for k in data.keys() if k != "metadata" and not k.startswith("_")]),
            "complete_dives": complete_count,
            "incomplete_dives": incomplete_count,
        }
    }

    for filename, video in videos.items():
        if verbose:
            print(f"\n📹 {filename}")
            print(f"   FPS: {video.fps}, Duration: {video.duration_sec}s, Dives: {len(video.dives)} (complete)")

        if len(video.dives) == 0:
            if verbose:
                print(f"   ⏳ No completed dives yet (waiting for annotation)")
            results["videos"][filename] = {
                "valid": None,  # Not validated - no data
                "validation": {},
            }
            continue

        validation = video.validate_all()
        video_valid = all(is_valid for is_valid, _ in validation.values())

        results["videos"][filename] = {
            "valid": video_valid,
            "validation": validation,
        }

        if not video_valid:
            results["valid"] = False

        if verbose:
            for dive_id, (is_valid, msg) in validation.items():
                status = "✓" if is_valid else "✗"
                print(f"   {status} {dive_id}: {msg}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Summary: {complete_count} complete, {incomplete_count} incomplete")
        if complete_count == 0:
            print("⏳ No completed dives yet. Fill in frame indices to validate.")
        elif results["valid"]:
            print("✅ All completed dives validated successfully!")
        else:
            print("❌ Some completed dives failed validation. See details above.")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ground_truth_file = Path(sys.argv[1])

        print(f"Validating: {ground_truth_file}")
        print("="*70)

        validate_ground_truth_frames(ground_truth_file, verbose=True)
    else:
        print("Usage: python frame_time_converter.py <ground_truth.json>")
        print("\nExample conversions:")
        print(f"  Frame 100 @ 30fps = {FrameTimeConverter.frame_to_seconds(100, 30):.3f}s")
        print(f"  10.5s @ 30fps = Frame {FrameTimeConverter.seconds_to_frame(10.5, 30)}")
        print(f"  Frame 450 @ 30fps = {FrameTimeConverter.frame_to_timecode(450, 30)}")
