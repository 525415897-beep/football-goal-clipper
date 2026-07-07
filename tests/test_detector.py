import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import cv2
import numpy as np
import pytest
from detector import (
    detect_motion_regions,
    filter_by_size,
    filter_by_roi,
    check_rois,
    deduplicate_timestamps,
    detect_goals,
    detect_goals_video,
    detect_goals_combined,
    detect_audio_peaks,
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
    # Verify it's the inside contour that survived, not the outside one
    assert result[0] is inside


def test_check_rois_single_pass():
    """check_rois tests two ROIs in one pass without building contour lists."""
    roi_left = (10, 10, 50, 50)
    roi_right = (200, 10, 50, 50)

    # Contour centered in left ROI
    left_contour = np.array([[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32)
    # Contour centered in right ROI
    right_contour = np.array([[[220, 30]], [[230, 30]], [[230, 40]], [[220, 40]]], dtype=np.int32)
    # Contour outside both ROIs
    outside = np.array([[[0, 0]], [[5, 0]], [[5, 5]], [[0, 5]]], dtype=np.int32)

    hit_left, hit_right = check_rois([left_contour], roi_left, roi_right)
    assert hit_left is True
    assert hit_right is False

    hit_left, hit_right = check_rois([right_contour], roi_left, roi_right)
    assert hit_left is False
    assert hit_right is True

    hit_left, hit_right = check_rois([left_contour, right_contour], roi_left, roi_right)
    assert hit_left is True
    assert hit_right is True

    hit_left, hit_right = check_rois([outside], roi_left, roi_right)
    assert hit_left is False
    assert hit_right is False


def test_deduplicate_timestamps_merges_close_events():
    timestamps = [1.0, 2.5, 3.0, 10.0, 11.0]
    result = deduplicate_timestamps(timestamps, window=5.0)
    assert result == [1.0, 10.0]


def test_deduplicate_timestamps_empty():
    assert deduplicate_timestamps([], window=5.0) == []


def test_deduplicate_timestamps_single_element():
    """Single-element list should return the same element."""
    assert deduplicate_timestamps([5.0], window=5.0) == [5.0]


def test_detect_goals_synthetic_video():
    """Integration test for detect_goals using a synthetic video.

    Creates a video with 30 frames. Alternates between black and white frames
    in the left-goal ROI region so motion detection finds contours there.
    """
    fps = 10.0
    num_frames = 30
    width, height = 320, 240

    # Define ROIs
    roi_left = (20, 80, 80, 80)   # top-left region
    roi_right = (220, 80, 80, 80)  # top-right region

    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        video_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        assert writer.isOpened(), "Failed to open VideoWriter"

        for i in range(num_frames):
            # Base frame: dark gray
            frame = np.full((height, width, 3), 30, dtype=np.uint8)

            # On even frames, draw a bright rectangle in the left ROI area
            # to create detectable motion on every other frame
            if i % 2 == 0:
                rx, ry, rw, rh = roi_left
                cv2.rectangle(frame, (rx + 10, ry + 10), (rx + 40, ry + 40), (200, 200, 200), -1)

            writer.write(frame)

        writer.release()

        # Run detection with permissive size thresholds for our small test frame
        timestamps = detect_goals(
            video_path,
            roi_left=roi_left,
            roi_right=roi_right,
            threshold=10,
            min_area=10,
            max_area=5000,
            dedup_window=5.0,
            frame_skip=1,
        )

        # Motion occurs on frames 0, 2, 4, ... (even frames vs previous odd frames).
        # detect_goals starts comparing from frame 1, so we should get hits on
        # frames 1, 3, 5, 7, 9, 11, 13, ...
        # Timestamps = frame_idx / fps = 1/10, 3/10, 5/10, ...
        # With dedup_window=5.0, only the first of each cluster is kept.
        assert len(timestamps) >= 1, f"Expected at least 1 timestamp, got {timestamps}"
        # The first detection should be around frame 1 at 10 fps = 0.1s
        assert timestamps[0] == pytest.approx(0.1, abs=0.05), \
            f"Expected first timestamp around 0.1s, got {timestamps[0]}"
    finally:
        os.unlink(video_path)


def test_detect_goals_is_video_alias():
    """detect_goals should be an alias for detect_goals_video for backward compat."""
    assert detect_goals is detect_goals_video


def test_detect_audio_peaks_need_ffmpeg():
    """detect_audio_peaks requires ffmpeg. Test that it handles missing video."""
    import subprocess as sp
    try:
        detect_audio_peaks("/nonexistent/video.mp4")
    except (RuntimeError, sp.TimeoutExpired, FileNotFoundError) as e:
        # Expected: ffmpeg can't find the file or ffmpeg missing
        assert "ffmpeg" in str(e).lower() or "nonexistent" in str(e).lower() or "No such file" in str(e)


def test_deduplicate_timestamps_unsorted_input():
    """deduplicate_timestamps should sort input internally."""
    result = deduplicate_timestamps([30.0, 1.0, 15.0], window=10.0)
    assert result == [1.0, 15.0, 30.0]


def test_detect_goals_video_nonexistent_file():
    """detect_goals_video should raise RuntimeError for missing files."""
    try:
        detect_goals_video("/nonexistent/video.mp4", (0, 0, 10, 10), (0, 0, 10, 10))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Cannot open video" in str(e)
