
"""
YOLO Object Detection wrapper for Cement Bag Counter.
Supports custom trained models, fallback models,
and spatial detection zone filtering.
"""

import os
import logging
from typing import List, Tuple, Optional

import numpy as np
from ultralytics import YOLO


logger = logging.getLogger("CementBagCounter.Detector")


class CementBagDetector:
    """
    Wrapper around Ultralytics YOLO for detecting cement bags.
    Supports custom models, fallback models, class filtering,
    and spatial detection zones.
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

        # Load model
        self._load_model()

    def _load_model(self):
        """
        Load custom YOLO model if available.
        If unavailable or invalid, load fallback model.
        Logs the exact custom model loading error.
        """

        absolute_model_path = os.path.abspath(self.model_path)

        logger.info(
            f"Current working directory: {os.getcwd()}"
        )
        logger.info(
            f"Model path: {self.model_path}"
        )
        logger.info(
            f"Absolute model path: {absolute_model_path}"
        )
        logger.info(
            f"Model exists: {os.path.exists(absolute_model_path)}"
        )

        # ---------------------------------
        # Load custom model
        # ---------------------------------
        if not os.path.exists(absolute_model_path):
            logger.warning(
                f"Custom model file does not exist: "
                f"{absolute_model_path}"
            )
        else:
            try:
                logger.info(
                    f"Loading custom cement bag model: "
                    f"{absolute_model_path}"
                )

                self.model = YOLO(absolute_model_path)
                self.is_custom_model = True
                self.warning_message = None

                logger.info(
                    "Custom model loaded successfully."
                )

                return

            except Exception:
                logger.exception(
                    f"Failed to load custom model: "
                    f"{absolute_model_path}"
                )

        # ---------------------------------
        # Load fallback model
        # ---------------------------------
        logger.warning(
            f"Falling back to pretrained model: "
            f"{self.fallback_model}"
        )

        try:
            self.model = YOLO(self.fallback_model)
            self.is_custom_model = False

            self.warning_message = (
                "Detection model is not suitable for cement bags. "
                "Custom cement-bag model is required. "
                "Currently using pretrained YOLO for validation."
            )

            logger.info(
                "Fallback YOLO model loaded successfully."
            )

        except Exception:
            logger.exception(
                f"Failed to load fallback model: "
                f"{self.fallback_model}"
            )

            raise RuntimeError(
                "Neither custom model nor fallback model "
                "could be loaded."
            )

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        target_class_id: Optional[int] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None
    ) -> List[
        Tuple[
            Tuple[float, float, float, float],
            float,
            int
        ]
    ]:
        """
        Run object detection on a single video frame.

        Supports:
        - Confidence threshold
        - Target class filtering
        - Y-Min / Y-Max zone filtering
        - X-Min / X-Max zone filtering

        Returns:
            List of:
            (bbox, confidence, class_id)

            bbox = (x1, y1, x2, y2)
        """

        if self.model is None:
            logger.error(
                "YOLO model is not initialized."
            )
            return []

        conf = (
            conf_threshold
            if conf_threshold is not None
            else self.conf_threshold
        )

        cls_id = (
            target_class_id
            if target_class_id is not None
            else self.target_class_id
        )

        # Run YOLO inference
        results = self.model(
            frame,
            conf=conf,
            verbose=False
        )

        detections = []

        if (
            len(results) > 0
            and results[0].boxes is not None
        ):
            boxes = results[0].boxes

            for box in boxes:
                # Class ID
                c_id = int(
                    box.cls[0].item()
                )

                # Confidence
                confidence = float(
                    box.conf[0].item()
                )

                # Filter by target class
                if (
                    cls_id is not None
                    and cls_id >= 0
                    and c_id != cls_id
                ):
                    continue

                # Bounding box
                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                bbox = (
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3])
                )

                # Center point of bounding box
                cx = (
                    bbox[0] + bbox[2]
                ) / 2.0

                cy = (
                    bbox[1] + bbox[3]
                ) / 2.0

                # Y-Min filter
                if (
                    y_min is not None
                    and cy < y_min
                ):
                    continue

                # Y-Max filter
                if (
                    y_max is not None
                    and cy > y_max
                ):
                    continue

                # X-Min filter
                if (
                    x_min is not None
                    and cx < x_min
                ):
                    continue

                # X-Max filter
                if (
                    x_max is not None
                    and cx > x_max
                ):
                    continue

                # Add valid detection
                detections.append(
                    (
                        bbox,
                        confidence,
                        c_id
                    )
                )

        return detections