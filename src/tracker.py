"""
Multi-Object Tracker wrapper for Cement Bag Counter.
Uses ByteTrack / BoT-SORT tracker integrated with Ultralytics YOLO model,
with optional spatial zone filtering (Y-Min, Y-Max, X-Min, X-Max).
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("CementBagCounter.Tracker")

class BagTracker:
    """
    Object tracking module assigning unique persistent tracking IDs to detected bags across frames.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        target_class_id: int = -1
    ):
        self.model = yolo_model
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self._fallback_next_id = -1
        self._fallback_tracks = {}

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _fallback_id(self, bbox):
        best_id = None
        best_iou = 0.0
        for track_id, previous_bbox in self._fallback_tracks.items():
            overlap = self._iou(bbox, previous_bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_id = track_id
        if best_id is None or best_iou < 0.15:
            best_id = self._fallback_next_id
            self._fallback_next_id -= 1
        self._fallback_tracks[best_id] = bbox
        return best_id

    def track(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        target_class_id: Optional[int] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None
    ) -> List[Tuple[int, Tuple[float, float, float, float], float, int]]:
        """
        Track objects in frame.
        Optionally filter by spatial detection zone (y_min, y_max, x_min, x_max).
        Returns list of (track_id, bbox, confidence, class_id)
        where bbox = (x1, y1, x2, y2).
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        cls_id = target_class_id if target_class_id is not None else self.target_class_id

        classes_arg = [cls_id] if (cls_id is not None and cls_id >= 0) else None

        # Run tracking using Ultralytics persist=True
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=conf,
            classes=classes_arg,
            verbose=False
        )

        tracks = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            xyxy_boxes = boxes.xyxy.cpu().numpy()

            for t_id, (x1, y1, x2, y2), c_val, c_id in zip(track_ids, xyxy_boxes, confidences, class_ids):
                bbox = (float(x1), float(y1), float(x2), float(y2))
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                if y_min is not None and cy < y_min:
                    continue
                if y_max is not None and cy > y_max:
                    continue
                if x_min is not None and cx < x_min:
                    continue
                if x_max is not None and cx > x_max:
                    continue

                stable_id = int(t_id) if t_id is not None else self._fallback_id(bbox)
                tracks.append((stable_id, bbox, float(c_val), int(c_id)))

        return tracks
