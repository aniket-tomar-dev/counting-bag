"""
End-to-end synthetic video processing test for Cement Bag Counter.
Generates a short synthetic MP4 video, runs VideoProcessor, and checks output video creation.
"""

import os
import cv2
import numpy as np
import pytest

from src.detector import CementBagDetector
from src.counter import BagCounter
from src.video_processor import VideoProcessor

def test_end_to_end_video_processing(tmp_path):
    # 1. Create a synthetic video with 30 frames
    video_path = str(tmp_path / "synthetic_conveyor.mp4")
    output_path = str(tmp_path / "processed_output.mp4")

    width, height, fps = 640, 480, 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Generate 30 frames
    for i in range(30):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving square simulating a bag moving from left (x=50) to right (x=450)
        x_pos = 50 + i * 13
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 60, 260), (200, 200, 200), -1)
        out.write(frame)
    out.release()

    assert os.path.exists(video_path)

    # 2. Setup Detector, Counter, and Processor
    detector = CementBagDetector(
        model_path="models/best.pt",
        fallback_model="yolov8n.pt",
        conf_threshold=0.20,
        target_class_id=0
    )

    counter = BagCounter(
        line_start=(300, 0),
        line_end=(300, 480),
        direction="LEFT_TO_RIGHT"
    )

    processor = VideoProcessor(
        detector=detector,
        counter=counter,
        tracker_type="bytetrack.yaml",
        debug_mode=True
    )

    # 3. Process video
    metrics = processor.process_video(
        input_path=video_path,
        output_path=output_path
    )

    # 4. Assert metrics and output existence
    assert os.path.exists(output_path)
    assert metrics["processed_frames"] == 30
    assert metrics["video_resolution"] == (640, 480)
    assert metrics["video_fps"] == 25.0
