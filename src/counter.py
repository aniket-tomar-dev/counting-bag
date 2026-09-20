"""
Line, Rectangle Box, and Polygon zone counting algorithm for Cement Bag Counter.
Implements robust rectangle zone entry detection, line crossing detection,
direction validation, margin tolerance, and duplicate count prevention.
"""

import logging
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import cv2

logger = logging.getLogger("CementBagCounter.Counter")

class BagCounter:
    """
    Tracks centers/bboxes of detected bags across frames and counts them as they enter
    a user-configured Counting Rectangle/Polygon Zone or cross a Line in a specified direction.
    """

    def __init__(
        self,
        mode: str = "RECTANGLE_BOX",  # "RECTANGLE_BOX", "POLYGON", or "LINE"
        counting_box: Optional[Tuple[int, int, int, int]] = None,  # (x_min, y_min, x_max, y_max)
        line_start: Tuple[int, int] = (100, 400),
        line_end: Tuple[int, int] = (1200, 400),
        direction: str = "ANY",
        roi_polygon: Optional[List[Tuple[int, int]]] = None,
        margin_percent: float = 0.20  # 20% line length margin tolerance
    ):
        self.mode = mode.upper()
        self.counting_box = counting_box
        self.line_start = line_start
        self.line_end = line_end
        self.direction = direction
        self.roi_polygon = roi_polygon
        self.margin_percent = margin_percent
        
        # Track state
        self.counted_track_ids: Set[int] = set()
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.polygon_inside_state: Dict[int, bool] = {}
        self.polygon_counted_objects: List[Dict[str, object]] = []
        self.frame_index: int = 0
        self.total_count: int = 0
        
        # Track IDs that were counted in the current frame (for visual feedback)
        self.recently_counted: Set[int] = set()

    def reset(self):
        """Reset counter state."""
        self.counted_track_ids.clear()
        self.track_history.clear()
        self.polygon_inside_state.clear()
        self.polygon_counted_objects.clear()
        self.frame_index = 0
        self.total_count = 0
        self.recently_counted.clear()

    @staticmethod
    def calculate_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate the center point (cx, cy) of a bounding box [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _cross_product(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """
        Cross product / determinant of vector (p1 -> p2) and (p1 -> p3).
        """
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float]
    ) -> bool:
        """
        Robust line crossing check between trajectory segment p1-p2 and line segment q1-q2.
        Includes margin tolerance.
        """
        cp1 = self._cross_product(q1, q2, p1)
        cp2 = self._cross_product(q1, q2, p2)

        if cp1 * cp2 <= 0:
            denom = cp1 - cp2
            if abs(denom) > 1e-6:
                t = cp1 / denom
                min_t = -self.margin_percent
                max_t = 1.0 + self.margin_percent
                if min_t <= t <= max_t:
                    cp3 = self._cross_product(p1, p2, q1)
                    cp4 = self._cross_product(p1, p2, q2)
                    if cp3 * cp4 <= 0 or abs(cp3) < 1e-3 or abs(cp4) < 1e-3:
                        return True
            else:
                min_p_x, max_p_x = min(p1[0], p2[0]), max(p1[0], p2[0])
                min_q_x, max_q_x = min(q1[0], q2[0]), max(q1[0], q2[0])
                min_p_y, max_p_y = min(p1[1], p2[1]), max(p1[1], p2[1])
                min_q_y, max_q_y = min(q1[1], q2[1]), max(q1[1], q2[1])
                return (max_p_x >= min_q_x and max_q_x >= min_p_x) and \
                       (max_p_y >= min_q_y and max_q_y >= min_p_y)

        return False

    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside ROI polygon."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return True

        pts = np.array(self.roi_polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return result >= 0

    def _is_valid_direction(
        self,
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float]
    ) -> bool:
        """Verify movement direction."""
        direction = self.direction.upper()
        if direction in ("ANY", "EITHER"):
            return True

        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]

        if direction == "LEFT_TO_RIGHT":
            return dx >= 0
        elif direction == "RIGHT_TO_LEFT":
            return dx <= 0
        elif direction == "TOP_TO_BOTTOM":
            return dy >= 0
        elif direction == "BOTTOM_TO_TOP":
            return dy <= 0
        
        return True

    def _is_inside_counting_box(
        self,
        curr_center: Tuple[float, float],
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Check if object center or bounding box overlaps/enters the configured counting box.
        """
        if not self.counting_box:
            return False

        bx1, by1, bx2, by2 = self.counting_box
        cx, cy = curr_center
        x1, y1, x2, y2 = bbox

        # Check center inside box
        center_inside = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
        # Check bbox bounding box overlap with counting box
        bbox_overlap = (x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1)

        return center_inside or bbox_overlap

    def _is_inside_counting_polygon(
        self,
        curr_center: Tuple[float, float],
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check whether the bag center is inside the polygon zone."""
        if not self.roi_polygon or len(self.roi_polygon) < 3:
            return False

        return self.is_point_in_roi(curr_center)

    @staticmethod
    def _box_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _matches_counted_polygon_object(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Dict[str, object]]:
        """Match a new tracker ID to a recently counted physical bag."""
        x1, y1, x2, y2 = bbox
        center_x, center_y = self.calculate_center(bbox)
        max_dimension = max(x2 - x1, y2 - y1)
        for counted_object in self.polygon_counted_objects:
            previous_bbox = counted_object["bbox"]
            previous_center = counted_object["center"]
            if self.frame_index - int(counted_object["last_seen"]) > 15:
                continue
            previous_x, previous_y = previous_center
            center_distance = float(
                np.hypot(center_x - previous_x, center_y - previous_y)
            )
            if (
                self._box_iou(bbox, previous_bbox) >= 0.10
                or center_distance <= max(40.0, max_dimension * 0.75)
            ):
                return counted_object
        return None

    def update_track(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Update tracking position for object track_id and evaluate entry into Counting Box / Line.
        Returns True if this update resulted in a NEW count.
        """
        curr_center = self.calculate_center(bbox)
        
        # Ensure track_history exists for this ID
        if track_id not in self.track_history:
            self.track_history[track_id] = [curr_center]

        history = self.track_history[track_id]
        prev_center = history[-1]
        polygon_inside = (
            self.mode == "POLYGON"
            and self._is_inside_counting_polygon(curr_center, bbox)
        )
        # A newly observed track is treated as outside until its first
        # polygon observation is evaluated. This allows bags whose detector
        # track starts at the polygon boundary to be counted immediately.
        previous_polygon_inside = self.polygon_inside_state.get(track_id, False)
        self.polygon_inside_state[track_id] = polygon_inside
        matched_counted_object = (
            self._matches_counted_polygon_object(bbox)
            if self.mode == "POLYGON" and polygon_inside else None
        )
        if matched_counted_object is not None:
            matched_counted_object["bbox"] = bbox
            matched_counted_object["center"] = curr_center
            matched_counted_object["last_seen"] = self.frame_index

        # A polygon counting zone must evaluate the full bag box, not only its
        # center; otherwise a large bag can visibly overlap the zone but never
        # be counted while its center remains outside.
        if self.mode != "POLYGON" and not self.is_point_in_roi(curr_center):
            history.append(curr_center)
            if len(history) > 30:
                history.pop(0)
            return False

        new_count = False

        # 1. Prevent double counting immediately
        if track_id not in self.counted_track_ids:
            if self.mode == "RECTANGLE_BOX" and self.counting_box:
                # Mode A: Counting Rectangle Box
                if self._is_inside_counting_box(curr_center, bbox):
                    self.total_count += 1
                    self.counted_track_ids.add(track_id)
                    self.recently_counted.add(track_id)
                    new_count = True
                    logger.info(f"Bag counted (Rectangle Box Entry): track_id={track_id}, total={self.total_count}")
            elif self.mode == "POLYGON" and self.roi_polygon:
                entered_polygon = (
                    polygon_inside
                    and previous_polygon_inside is False
                )
                if entered_polygon and matched_counted_object is None:
                    self.total_count += 1
                    self.counted_track_ids.add(track_id)
                    self.recently_counted.add(track_id)
                    self.polygon_counted_objects.append({
                        "bbox": bbox,
                        "center": curr_center,
                        "last_seen": self.frame_index,
                    })
                    new_count = True
                    logger.info(f"Bag counted (Polygon Zone Entry): track_id={track_id}, total={self.total_count}")
            else:
                # Mode B: Line Crossing
                for past_pt in (prev_center, history[0]):
                    intersects = self._segments_intersect(
                        past_pt, curr_center,
                        self.line_start, self.line_end
                    )
                    valid_direction = self._is_valid_direction(past_pt, curr_center)

                    if intersects and valid_direction:
                        self.total_count += 1
                        self.counted_track_ids.add(track_id)
                        self.recently_counted.add(track_id)
                        new_count = True
                        logger.info(f"Bag counted (Line Crossing): track_id={track_id}, total={self.total_count}")
                        break

        # Update position history (keep max 30 points)
        history.append(curr_center)
        if len(history) > 30:
            history.pop(0)

        return new_count

    def update_frame_tracks(
        self,
        tracks: List[Tuple[int, Tuple[float, float, float, float], float, int]]
    ) -> List[int]:
        """
        Process all active tracks in a frame.
        tracks: list of (track_id, bbox, confidence, class_id)
        Returns list of track_ids that were newly counted in this frame.
        """
        self.recently_counted.clear()
        self.frame_index += 1
        self.polygon_counted_objects = [
            counted_object
            for counted_object in self.polygon_counted_objects
            if self.frame_index - int(counted_object["last_seen"]) <= 15
        ]
        newly_counted = []
        for track_id, bbox, conf, cls_id in tracks:
            if self.update_track(track_id, bbox):
                newly_counted.append(track_id)
        return newly_counted
