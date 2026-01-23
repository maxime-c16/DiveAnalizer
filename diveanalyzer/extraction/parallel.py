"""
Parallel extraction orchestrator for processing multiple dives concurrently.

Provides:
- ExtractionWorker: Wrapper for single dive extraction
- ParallelExtractionOrchestrator: Manages parallel extraction with callbacks
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from diveanalyzer.extraction.ffmpeg import extract_dive_clip

logger = logging.getLogger(__name__)


class ExtractionWorker:
    """Worker for extracting a single dive clip.

    Encapsulates the extraction logic and error handling for one dive.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        dive_id: int,
        start_time: float,
        end_time: float,
        audio_enabled: bool = True,
    ):
        """Initialize extraction worker.

        Args:
            video_path: Path to source video file
            output_dir: Directory to save extracted clip
            dive_id: Unique dive identifier
            start_time: Start time in seconds
            end_time: End time in seconds
            audio_enabled: Whether to include audio in output
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.dive_id = dive_id
        self.start_time = start_time
        self.end_time = end_time
        self.audio_enabled = audio_enabled
        self.output_filename = f"dive_{dive_id:03d}.mp4"
        self.output_path = self.output_dir / self.output_filename

    def extract(self) -> Tuple[bool, str, float, Optional[str]]:
        """Extract dive clip from video.

        Returns:
            Tuple of (success, filepath, file_size_mb, error_message)
            - success: True if extraction completed successfully
            - filepath: Path to extracted file (or None if failed)
            - file_size_mb: Size of extracted file in MB (or 0 if failed)
            - error_message: Error description (or None if successful)

        Raises:
            No exceptions raised; errors captured in return tuple
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Extract dive clip using FFmpeg
            success = extract_dive_clip(
                video_path=self.video_path,
                start_time=self.start_time,
                end_time=self.end_time,
                output_path=str(self.output_path),
                audio_enabled=self.audio_enabled,
                verbose=False,
            )

            if not success:
                return False, None, 0.0, "FFmpeg returned False"

            # Verify output file exists and get size
            if not self.output_path.exists():
                return False, None, 0.0, "Output file was not created"

            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)

            logger.info(
                f"Successfully extracted dive {self.dive_id}: "
                f"{self.output_filename} ({file_size_mb:.2f}MB)"
            )

            return True, str(self.output_path), file_size_mb, None

        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            logger.error(f"Dive {self.dive_id}: {error_msg}")
            return False, None, 0.0, error_msg


class ParallelExtractionOrchestrator:
    """Orchestrates parallel extraction of multiple dives with progress callbacks.

    Manages ThreadPoolExecutor with configurable workers and emits events
    as dives complete.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        audio_enabled: bool = True,
        max_workers: int = 4,
    ):
        """Initialize parallel extraction orchestrator.

        Args:
            video_path: Path to source video file
            output_dir: Directory to save extracted dives
            audio_enabled: Whether to include audio in extracts (default True)
            max_workers: Number of parallel worker threads (default 4)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_enabled = audio_enabled
        self.max_workers = max_workers
        self.total_count = 0
        self.extracted_count = 0
        self.failed_dives: List[Dict[str, Any]] = []

    def extract_selected(
        self,
        dives_list: List[Tuple[int, float, float]],
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Extract selected dives in parallel with progress callbacks.

        Args:
            dives_list: List of (dive_id, start_time, end_time) tuples
            event_callback: Optional callback function called after each dive
                          Receives dict: {
                              "dive_id": int,
                              "success": bool,
                              "filename": str,
                              "size_mb": float,
                              "extracted_count": int,
                              "total_count": int,
                              "error": str or None
                          }

        Returns:
            Dict with extraction results:
            {
                "total": int,
                "successful": int,
                "failed": int,
                "results": {
                    dive_id: {
                        "success": bool,
                        "path": str,
                        "size_mb": float,
                        "error": str or None
                    },
                    ...
                },
                "failed_dives": [
                    {"dive_id": int, "error": str},
                    ...
                ]
            }
        """
        self.total_count = len(dives_list)
        self.extracted_count = 0
        self.failed_dives = []
        results = {}

        logger.info(f"Starting parallel extraction of {self.total_count} dives (max_workers={self.max_workers})")

        if self.total_count == 0:
            logger.warning("No dives to extract")
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "results": {},
                "failed_dives": [],
            }

        # Use ThreadPoolExecutor for parallel extraction
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all extraction tasks
            future_to_dive = {}
            for dive_id, start_time, end_time in dives_list:
                worker = ExtractionWorker(
                    video_path=self.video_path,
                    output_dir=self.output_dir,
                    dive_id=dive_id,
                    start_time=start_time,
                    end_time=end_time,
                    audio_enabled=self.audio_enabled,
                )
                future = executor.submit(worker.extract)
                future_to_dive[future] = dive_id

            # Process results as they complete (not in order)
            for future in as_completed(future_to_dive):
                dive_id = future_to_dive[future]

                try:
                    success, filepath, size_mb, error = future.result()

                    if success:
                        self.extracted_count += 1
                        filename = Path(filepath).name if filepath else None
                        results[dive_id] = {
                            "success": True,
                            "path": filepath,
                            "size_mb": size_mb,
                            "error": None,
                        }

                        # Emit progress callback
                        if event_callback:
                            event_callback({
                                "dive_id": dive_id,
                                "success": True,
                                "filename": filename,
                                "size_mb": f"{size_mb:.2f}",
                                "extracted_count": self.extracted_count,
                                "total_count": self.total_count,
                                "error": None,
                            })

                    else:
                        results[dive_id] = {
                            "success": False,
                            "path": None,
                            "size_mb": 0.0,
                            "error": error,
                        }
                        self.failed_dives.append({
                            "dive_id": dive_id,
                            "error": error,
                        })

                        # Emit failure callback
                        if event_callback:
                            event_callback({
                                "dive_id": dive_id,
                                "success": False,
                                "filename": None,
                                "size_mb": 0.0,
                                "extracted_count": self.extracted_count,
                                "total_count": self.total_count,
                                "error": error,
                            })

                        logger.error(f"Dive {dive_id} extraction failed: {error}")

                except Exception as e:
                    # Handle unexpected errors in future.result()
                    error_msg = f"Unexpected error: {str(e)}"
                    results[dive_id] = {
                        "success": False,
                        "path": None,
                        "size_mb": 0.0,
                        "error": error_msg,
                    }
                    self.failed_dives.append({
                        "dive_id": dive_id,
                        "error": error_msg,
                    })

                    if event_callback:
                        event_callback({
                            "dive_id": dive_id,
                            "success": False,
                            "filename": None,
                            "size_mb": 0.0,
                            "extracted_count": self.extracted_count,
                            "total_count": self.total_count,
                            "error": error_msg,
                        })

                    logger.error(f"Dive {dive_id} unexpected error: {error_msg}")

        failed_count = self.total_count - self.extracted_count
        logger.info(
            f"Extraction complete: {self.extracted_count}/{self.total_count} successful, "
            f"{failed_count} failed"
        )

        return {
            "total": self.total_count,
            "successful": self.extracted_count,
            "failed": failed_count,
            "results": results,
            "failed_dives": self.failed_dives,
        }
