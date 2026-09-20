"""
YOLO Object Detection wrapper for Cement Bag Counter.
Supports custom trained models (models/best.pt), standard fallbacks,
and spatial detection zone filtering (Y-Min, Y-Max, X-Min, X-Max).
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("CementBagCounter.Detector")

class CementBagDetector:
    """
    Wrapper around Ultralytics YOLO for detecting cement bags.
    Configurable model path, class IDs, and spatial detection zones.
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        fallback_model: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        target_class_id: int = -1
    ):
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        
        self.is_custom_model: bool = False
        self.warning_message: Optional[str] = None
        self.model: Optional[YOLO] = None
        
        self._load_model()

    def _load_model(self):
        """Load custom model if available, else load fallback model with warning."""
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading custom cement bag model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.is_custom_model = True
                self.warning_message = None
                return
            except Exception as e:
                logger.error(f"Failed to load custom model from {self.model_path}: {e}")
        
        # Fallback handling
        logger.warning(
            f"Custom model '{self.model_path}' not found or failed to load. "
            f"Falling back to pretrained '{self.fallback_model}'."
        )
        try:
            self.model = YOLO(self.fallback_model)
            self.is_custom_model = False
            self.warning_message = (
                "⚠️ Detection model is not suitable for cement bags.\n"
                "A custom cement-bag detection model is required at 'models/best.pt'. "
                "Currently using pretrained YOLOv8 for pipeline validation."
            )
        except Exception as e:
            logger.critical(f"Failed to load fallback model {self.fallback_model}: {e}")
            raise RuntimeError(f"Unable to load any YOLO model: {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        target_class_id: Optional[int] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None
    ) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """
        Run object detection on a single video frame.
        Optionally filter by target_class_id and spatial detection zone (y_min, y_max, x_min, x_max).
        Returns list of (bbox, confidence, class_id) where bbox = (x1, y1, x2, y2).
        """
        if self.model is None:
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        cls_id = target_class_id if target_class_id is not None else self.target_class_id

        # Run inference
        results = self.model(frame, conf=conf, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                c_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Filter by target class ID if specified (and >= 0)
                if cls_id is not None and cls_id >= 0 and c_id != cls_id:
                    continue
                    
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                
                # Spatial Detection Zone Filtering (Center point of box)
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

                detections.append((bbox, confidence, c_id))

        return detections
