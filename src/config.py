"""
Configuration module for Cement Bag Counter MVP.
Defines default settings, paths, and line crossing parameters.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class AppConfig:
    # Model Configuration
    MODEL_PATH: str = "models/best.pt"
    FALLBACK_MODEL_PATH: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.50
    TARGET_CLASS_ID: int = 0
    TRACKER_TYPE: str = "bytetrack.yaml"
    
    # Counting Line Configuration
    LINE_START: Tuple[int, int] = (100, 400)
    LINE_END: Tuple[int, int] = (1200, 400)
    COUNT_DIRECTION: str = "LEFT_TO_RIGHT"  # LEFT_TO_RIGHT, RIGHT_TO_LEFT, TOP_TO_BOTTOM, BOTTOM_TO_TOP
    
    # Region of Interest (ROI) Configuration
    # Polygon represented as list of (x, y) tuples. If empty/None, ROI is disabled.
    ROI_POLYGON: Optional[List[Tuple[int, int]]] = None
    
    # Application Paths
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MODELS_DIR: str = "models"
    
    # UI / Debug Settings
    DEBUG_MODE: bool = False
    
    def ensure_directories(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)

config = AppConfig()
config.ensure_directories()
