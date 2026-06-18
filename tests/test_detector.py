import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import numpy as np
from detector import (
    detect_motion_regions,
    filter_by_size,
    filter_by_roi,
    deduplicate_timestamps,
)


def test_filter_by_size_filters_out_large_contours():
    # contour area 5000 > max_area 2000, should be filtered out
    large = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]], dtype=np.int32)
    small = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], dtype=np.int32)
    result = filter_by_size([large, small], min_area=10, max_area=2000)
    assert len(result) == 1
    assert result[0] is small


def test_filter_by_size_filters_out_small_contours():
    small = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]], dtype=np.int32)
    ok = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    result = filter_by_size([small, ok], min_area=10, max_area=2000)
    assert len(result) == 1
    assert result[0] is ok


def test_filter_by_roi_rejects_outside_motion():
    roi = (100, 60, 300, 200)  # x, y, w, h
    inside = np.array([[[150, 100]], [[160, 100]], [[160, 110]], [[150, 110]]], dtype=np.int32)
    outside = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    result = filter_by_roi([inside, outside], roi)
    assert len(result) == 1


def test_deduplicate_timestamps_merges_close_events():
    timestamps = [1.0, 2.5, 3.0, 10.0, 11.0]
    result = deduplicate_timestamps(timestamps, window=5.0)
    assert result == [1.0, 10.0]


def test_deduplicate_timestamps_empty():
    assert deduplicate_timestamps([], window=5.0) == []
