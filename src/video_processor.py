"""
Video Processing Pipeline for Cement Bag Counter.
Reads input video, performs detection & tracking frame-by-frame,
applies line crossing logic, annotates frames, and writes output video.
"""

import os
import time
import logging
from typing import Callable, Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

from src.detector import CementBagDetector
from src.tracker import BagTracker
from src.counter import BagCounter

logger = logging.getLogger("CementBagCounter.VideoProcessor")

class VideoProcessor:
    """
    Main processing engine for reading input video, running YOLO detection & tracking,
    counting line crossing, rendering HUD annotations, and saving output MP4 video.
    """

    def __init__(
        self,
        detector: CementBagDetector,
        counter: BagCounter,
        tracker_type: str = "bytetrack.yaml",
        debug_mode: bool = False
    ):
        self.detector = detector
        self.counter = counter
        self.tracker_type = tracker_type
        self.debug_mode = debug_mode
        self.tracker = BagTracker(
            yolo_model=detector.model,
            tracker_type=tracker_type,
            conf_threshold=detector.conf_threshold,
            target_class_id=detector.target_class_id
        )

    def draw_annotations(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[int, Tuple[float, float, float, float], float, int]],
        newly_counted: List[int],
        current_fps: float,
        frame_num: int
    ) -> np.ndarray:
        """
        Draw visual annotations: bounding boxes, tracking IDs, confidence scores,
        counting line, ROI boundary, live count HUD, and FPS.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw ROI Polygon if configured
        if (
            self.counter.mode != "POLYGON"
            and self.counter.roi_polygon
            and len(self.counter.roi_polygon) >= 3
        ):
            pts = np.array(self.counter.roi_polygon, dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            if self.debug_mode:
                cv2.putText(
                    annotated, "ROI", (pts[0][0], max(pts[0][1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

        # 2. Draw Counting Element (Rectangle Box or Line)
        if self.counter.mode == "POLYGON" and self.counter.roi_polygon:
            pts = np.array(self.counter.roi_polygon, dtype=np.int32)
            polygon_color = (0, 255, 0) if newly_counted else (0, 255, 255)
            cv2.polylines(annotated, [pts], isClosed=True, color=polygon_color, thickness=4)
            cv2.putText(
                annotated,
                "POLYGON COUNTING ZONE",
                (int(pts[0][0]), max(int(pts[0][1]) - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                polygon_color,
                2,
                cv2.LINE_AA
            )
        elif self.counter.mode == "RECTANGLE_BOX" and self.counter.counting_box:
            bx1, by1, bx2, by2 = self.counter.counting_box
            if len(newly_counted) > 0:
                box_color = (0, 255, 0)      # Bright green flash on count
                thickness = 6
            else:
                box_color = (0, 255, 255)    # Yellow counting box
                thickness = 4

            cv2.rectangle(annotated, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, thickness)
            cv2.putText(
                annotated,
                "COUNTING BOX ZONE",
                (int(bx1) + 8, max(int(by1) - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
                cv2.LINE_AA
            )
        else:
            l_start = self.counter.line_start
            l_end = self.counter.line_end
            
            # Flash line green if a bag was newly counted in this frame
            if len(newly_counted) > 0:
                line_color = (0, 255, 0)      # Bright green flash on count
                line_thickness = 6
            else:
                line_color = (0, 255, 255)    # Yellow counting line
                line_thickness = 4

            cv2.line(annotated, l_start, l_end, line_color, line_thickness)

            # Draw direction label on line
            mid_x = int((l_start[0] + l_end[0]) / 2)
            mid_y = int((l_start[1] + l_end[1]) / 2)
            cv2.putText(
                annotated,
                f"LINE ({self.counter.direction})",
                (max(10, mid_x - 70), max(25, mid_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2,
                cv2.LINE_AA
            )

        # 3. Draw Tracked Bounding Boxes
        for track_id, bbox, conf, cls_id in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            center_pt = self.counter.calculate_center(bbox)
            cx, cy = int(center_pt[0]), int(center_pt[1])

            is_already_counted = track_id in self.counter.counted_track_ids
            is_just_counted = track_id in newly_counted

            # Color scheme:
            # Bright Green if just counted, Gold if already counted, Vibrant Blue/Cyan if active
            if is_just_counted:
                box_color = (0, 255, 0)      # Green glow
                thickness = 4
            elif is_already_counted:
                box_color = (0, 215, 255)    # Gold
                thickness = 2
            else:
                box_color = (255, 191, 0)    # Deep Cyan/Blue
                thickness = 2

            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Draw Center Dot
            cv2.circle(annotated, (cx, cy), 5, box_color, -1)

            # Label text
            label = f"Bag #{track_id} ({conf:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Label background box
            label_y1 = max(y1 - text_h - 8, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 8, label_y1 + text_h + baseline + 4),
                box_color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            # Debug mode: draw center trajectory trace
            if self.debug_mode and track_id in self.counter.track_history:
                prev_pt = self.counter.track_history[track_id]
                cv2.line(
                    annotated,
                    (int(prev_pt[0]), int(prev_pt[1])),
                    (cx, cy),
                    (255, 0, 255),
                    2
                )

        # 4. Draw HUD Header Bar (Top Left: Total Bags, Top Right: FPS)
        # Background box for Total Count
        count_str = f"CEMENT BAGS: {self.counter.total_count}"
        (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (15, 15), (25 + cw, 30 + ch + 15), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, 15), (25 + cw, 30 + ch + 15), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            count_str,
            (25, 25 + ch),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # FPS Box
        fps_str = f"FPS: {current_fps:.1f}"
        (fw, fh), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (w - fw - 35, 15), (w - 15, 30 + fh + 10), (20, 20, 20), -1)
        cv2.rectangle(annotated, (w - fw - 35, 15), (w - 15, 30 + fh + 10), (255, 255, 255), 1)
        cv2.putText(
            annotated,
            fps_str,
            (w - fw - 25, 25 + fh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Debug overlay (Frame number & status)
        if self.debug_mode:
            debug_str = f"Frame: {frame_num} | Tracks: {len(tracks)}"
            cv2.putText(
                annotated,
                debug_str,
                (25, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return annotated

    def process_video(
        self,
        input_path: str,
        output_path: str,
        conf_threshold: Optional[float] = None,
        target_class_id: Optional[int] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, float, int, np.ndarray], bool]] = None
    ) -> Dict[str, Any]:
        """
        Process full video from input_path to output_path.
        Returns dictionary of processing metrics.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {input_path}")

        # Extract video metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Fallback default FPS
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        logger.info(f"Video opened: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create VideoWriter with mp4v / avc1 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out_writer.isOpened():
            # Try alternate fallback codec
            fourcc_alt = cv2.VideoWriter_fourcc(*'avc1')
            out_writer = cv2.VideoWriter(output_path, fourcc_alt, fps, (width, height))
            if not out_writer.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to initialize VideoWriter for path: {output_path}")

        # Reset counter state before processing
        self.counter.reset()

        frame_index = 0
        processed_frames = 0
        start_time = time.time()
        last_proc_fps = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                
                # Perform tracking on frame with spatial zone filtering
                tracks = self.tracker.track(
                    frame=frame,
                    conf_threshold=conf_threshold,
                    target_class_id=target_class_id,
                    y_min=y_min,
                    y_max=y_max,
                    x_min=x_min,
                    x_max=x_max
                )

                # Update counter logic
                newly_counted = self.counter.update_frame_tracks(tracks)

                # Calculate current processing FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    last_proc_fps = processed_frames / elapsed

                # Annotate frame
                annotated_frame = self.draw_annotations(
                    frame=frame,
                    tracks=tracks,
                    newly_counted=newly_counted,
                    current_fps=last_proc_fps if last_proc_fps > 0 else fps,
                    frame_num=frame_index
                )

                # Write frame to output video
                out_writer.write(annotated_frame)
                processed_frames += 1

                # Send progress updates to Streamlit UI via callback
                if progress_callback is not None:
                    # Callback returns False if cancellation requested
                    should_continue = progress_callback(
                        frame_index,
                        total_frames,
                        last_proc_fps,
                        self.counter.total_count,
                        annotated_frame
                    )
                    if should_continue is False:
                        logger.info("Processing cancelled by user.")
                        break

        except Exception as e:
            logger.error(f"Error during video processing loop: {e}", exc_info=True)
            raise e
        finally:
            cap.release()
            out_writer.release()

        total_elapsed = time.time() - start_time
        avg_processing_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0.0

        logger.info(
            f"Processing completed: total_bags={self.counter.total_count} | "
            f"frames={processed_frames} | avg_fps={avg_processing_fps:.2f}"
        )

        return {
            "input_path": input_path,
            "output_path": output_path,
            "total_bags": self.counter.total_count,
            "counted_ids": list(self.counter.counted_track_ids),
            "video_resolution": (width, height),
            "video_fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "video_duration_sec": duration,
            "processing_duration_sec": total_elapsed,
            "processing_fps": avg_processing_fps,
            "is_custom_model": self.detector.is_custom_model,
            "warning_message": self.detector.warning_message
        }
