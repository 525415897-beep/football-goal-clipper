import cv2
import numpy as np
import json
import sys
from pathlib import Path


def detect_motion_regions(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 25) -> list:
    """Return list of contours in motion regions between two grayscale frames."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_by_size(contours: list, min_area: int = 20, max_area: int = 2000) -> list:
    """Keep contours whose area is within [min_area, max_area]."""
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def filter_by_roi(contours: list, roi: tuple) -> list:
    """Keep contours whose centroid falls within the ROI rect (x, y, w, h)."""
    rx, ry, rw, rh = roi
    result = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            result.append(c)
    return result


def deduplicate_timestamps(timestamps: list, window: float = 5.0) -> list:
    """Merge timestamps within `window` seconds, keeping the first in each cluster."""
    if not timestamps:
        return []
    timestamps = sorted(timestamps)
    result = [timestamps[0]]
    for t in timestamps[1:]:
        if t - result[-1] > window:
            result.append(t)
    return result


def detect_goals(video_path: str, roi_left: tuple, roi_right: tuple,
                 threshold: int = 25, min_area: int = 20, max_area: int = 2000,
                 dedup_window: float = 5.0) -> list:
    """Process video and return list of detected goal timestamps in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    candidates = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = detect_motion_regions(prev_gray, curr_gray, threshold)
        contours = filter_by_size(contours, min_area, max_area)

        hit_left = len(filter_by_roi(contours, roi_left)) > 0
        hit_right = len(filter_by_roi(contours, roi_right)) > 0

        if hit_left or hit_right:
            timestamp = frame_idx / fps
            candidates.append(timestamp)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    return deduplicate_timestamps(candidates, dedup_window)


def main():
    input_data = json.loads(sys.stdin.read())
    video_path = input_data["video_path"]
    roi_left = tuple(input_data["roi_left"])
    roi_right = tuple(input_data["roi_right"])
    params = input_data.get("params", {})

    timestamps = detect_goals(
        video_path,
        roi_left,
        roi_right,
        threshold=params.get("threshold", 25),
        min_area=params.get("min_area", 20),
        max_area=params.get("max_area", 2000),
        dedup_window=params.get("dedup_window", 5.0),
    )

    print(json.dumps({"timestamps": timestamps}))


if __name__ == "__main__":
    main()
