"""
Unit tests for line-crossing counting algorithm in src/counter.py.
Runs independently of YOLO and OpenCV.
"""

import pytest
from src.counter import BagCounter

def test_no_crossing():
    counter = BagCounter(line_start=(200, 0), line_end=(200, 1000), direction="LEFT_TO_RIGHT")
    
    # Bounding box centers: (100, 400) then (150, 400) - both on left side of x=200
    bbox1 = (90, 390, 110, 410)   # center (100, 400)
    bbox2 = (140, 390, 160, 410)  # center (150, 400)

    counter.update_track(track_id=1, bbox=bbox1)
    counter.update_track(track_id=1, bbox=bbox2)

    assert counter.total_count == 0

def test_valid_crossing():
    counter = BagCounter(line_start=(200, 0), line_end=(200, 1000), direction="LEFT_TO_RIGHT")

    # Start at x=150, move to x=250 (crosses x=200)
    bbox1 = (140, 390, 160, 410)  # center (150, 400)
    bbox2 = (240, 390, 260, 410)  # center (250, 400)

    res1 = counter.update_track(track_id=1, bbox=bbox1)
    assert res1 is False

    res2 = counter.update_track(track_id=1, bbox=bbox2)
    assert res2 is True
    assert counter.total_count == 1
    assert 1 in counter.counted_track_ids

def test_duplicate_crossing():
    counter = BagCounter(line_start=(200, 0), line_end=(200, 1000), direction="LEFT_TO_RIGHT")

    bbox1 = (140, 390, 160, 410)  # center (150, 400)
    bbox2 = (240, 390, 260, 410)  # center (250, 400)
    bbox3 = (290, 390, 310, 410)  # center (300, 400)
    bbox4 = (140, 390, 160, 410)  # center (150, 400) - moves back and crosses again!

    counter.update_track(track_id=1, bbox=bbox1)
    counter.update_track(track_id=1, bbox=bbox2)  # Count = 1
    assert counter.total_count == 1

    counter.update_track(track_id=1, bbox=bbox3)
    assert counter.total_count == 1

    # Moves back left and then right again
    counter.update_track(track_id=1, bbox=bbox4)
    counter.update_track(track_id=1, bbox=bbox2)
    assert counter.total_count == 1  # Must stay 1!

def test_wrong_direction():
    counter = BagCounter(line_start=(200, 0), line_end=(200, 1000), direction="LEFT_TO_RIGHT")

    # Start right (x=250), move left (x=150) -> RIGHT_TO_LEFT direction when LEFT_TO_RIGHT configured
    bbox1 = (240, 390, 260, 410)  # center (250, 400)
    bbox2 = (140, 390, 160, 410)  # center (150, 400)

    counter.update_track(track_id=1, bbox=bbox1)
    res = counter.update_track(track_id=1, bbox=bbox2)
    
    assert res is False
    assert counter.total_count == 0

def test_multiple_objects():
    counter = BagCounter(line_start=(0, 400), line_end=(1000, 400), direction="TOP_TO_BOTTOM")

    # Three separate tracks moving from y=350 to y=450
    for t_id in [10, 20, 30]:
        bbox_before = (100, 340, 120, 360)  # center y=350
        bbox_after = (100, 440, 120, 460)   # center y=450

        counter.update_track(track_id=t_id, bbox=bbox_before)
        counter.update_track(track_id=t_id, bbox=bbox_after)

    assert counter.total_count == 3
    assert counter.counted_track_ids == {10, 20, 30}

def test_synthetic_spec_test():
    """
    Prompt Section 25 test requirement:
    Track 1: (100, 400) -> (200, 400) -> (300, 400), line at x=200. Expected: COUNT = 1.
    """
    counter = BagCounter(line_start=(200, 0), line_end=(200, 1000), direction="LEFT_TO_RIGHT")

    # Frame 1: center (100, 400)
    counter.update_track(track_id=1, bbox=(90, 390, 110, 410))
    assert counter.total_count == 0

    # Frame 2: center (200, 400) -> right on line or crossing
    # Note: center (195, 390, 205, 410) => center (200, 400)
    counter.update_track(track_id=1, bbox=(195, 390, 205, 410))
    
    # Frame 3: center (300, 400)
    counter.update_track(track_id=1, bbox=(295, 390, 305, 410))

    assert counter.total_count == 1
