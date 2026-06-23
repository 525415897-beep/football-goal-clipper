import cv2
import numpy as np
import json
import sys


def detect_motion_regions(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 25) -> list:
    """Return list of contours in motion regions between two grayscale frames."""
    diff = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_by_size(contours: list, min_area: int = 20, max_area: int = 2000) -> list:
    """Keep contours whose area is within [min_area, max_area].

    Area thresholds are absolute pixel values by default. For resolution-independent
    thresholds, pass values < 1.0 which will be interpreted as fractions of the
    frame area (width * height). See detect_goals() which applies this scaling.
    """
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def check_rois(contours: list, roi_left: tuple, roi_right: tuple) -> tuple:
    """Check if any contour's centroid falls within either ROI.

    Returns (hit_left, hit_right) booleans. Processes all contours in a single pass.
    """
    rlx, rly, rlw, rlh = roi_left
    rrx, rry, rrw, rrh = roi_right
    hit_left = False
    hit_right = False
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if not hit_left and rlx <= cx <= rlx + rlw and rly <= cy <= rly + rlh:
            hit_left = True
        if not hit_right and rrx <= cx <= rrx + rrw and rry <= cy <= rry + rrh:
            hit_right = True
        if hit_left and hit_right:
            break
    return hit_left, hit_right


def filter_by_roi(contours: list, roi: tuple) -> list:
    """Keep contours whose centroid falls within the ROI rect (x, y, w, h).

    Prefer check_rois() when you need to test two ROIs simultaneously,
    as it avoids iterating contours twice.
    """
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


def _resolve_area_thresholds(min_area, max_area, frame_width, frame_height):
    """Compute absolute area thresholds from values that may be fractions.

    If min_area or max_area is between 0 and 1 (exclusive), it is treated as a
    fraction of the total frame area. Otherwise it is used as an absolute pixel value.
    Returns (min_area, max_area) as integers.
    """
    frame_area = frame_width * frame_height
    if min_area is not None and 0 < min_area < 1:
        min_area = int(min_area * frame_area)
    if max_area is not None and 0 < max_area < 1:
        max_area = int(max_area * frame_area)
    if min_area is None:
        min_area = 20
    if max_area is None:
        max_area = 2000
    return min_area, max_area


def detect_goals(video_path: str, roi_left: tuple, roi_right: tuple,
                 threshold: int = 25, min_area=None, max_area=None,
                 dedup_window: float = 5.0) -> list:
    """Process video and return list of detected goal timestamps in seconds.

    Args:
        min_area: Minimum contour area in pixels. Pass a float < 1.0 to use
                  a fraction of frame area (e.g., 0.0001 for 0.01% of frame).
                  None uses the default of 20 pixels.
        max_area: Maximum contour area in pixels. Pass a float < 1.0 to use
                  a fraction of frame area (e.g., 0.01 for 1% of frame).
                  None uses the default of 2000 pixels.
    """
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

    frame_height, frame_width = prev_frame.shape[:2]
    min_area, max_area = _resolve_area_thresholds(
        min_area, max_area, frame_width, frame_height)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 1

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            contours = detect_motion_regions(prev_gray, curr_gray, threshold)
            contours = filter_by_size(contours, min_area, max_area)

            hit_left, hit_right = check_rois(contours, roi_left, roi_right)

            if hit_left or hit_right:
                timestamp = frame_idx / fps
                candidates.append(timestamp)

            prev_gray = curr_gray
            frame_idx += 1
        except Exception as e:
            # Corrupted frame: skip and continue with next frame
            print(f"Warning: skipping corrupted frame {frame_idx}: {e}", file=sys.stderr)
            ret, frame = cap.read()
            if not ret:
                break
            try:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = curr_gray
                frame_idx += 1
            except Exception:
                frame_idx += 1
                continue

    cap.release()
    return deduplicate_timestamps(candidates, dedup_window)


def main():
    """Read JSON input from stdin, run goal detection, and print JSON results.

    Expected input format:
        {"video_path": "...", "roi_left": [x,y,w,h], "roi_right": [x,y,w,h],
         "params": {"threshold": 25, "min_area": 20, "max_area": 2000,
                    "dedup_window": 5.0}}
    """
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            raise ValueError("No input provided. Expected JSON on stdin.")

        try:
            input_data = json.loads(raw_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

        if not isinstance(input_data, dict):
            raise ValueError(f"Expected a JSON object, got {type(input_data).__name__}")

        video_path = input_data.get("video_path")
        if not video_path or not isinstance(video_path, str):
            raise ValueError("Missing or invalid 'video_path' field")

        roi_left_raw = input_data.get("roi_left")
        roi_right_raw = input_data.get("roi_right")
        if not roi_left_raw or not isinstance(roi_left_raw, (list, tuple)):
            raise ValueError("Missing or invalid 'roi_left' field (expected [x,y,w,h])")
        if not roi_right_raw or not isinstance(roi_right_raw, (list, tuple)):
            raise ValueError("Missing or invalid 'roi_right' field (expected [x,y,w,h])")
        if len(roi_left_raw) != 4 or len(roi_right_raw) != 4:
            raise ValueError("ROI fields must have exactly 4 elements [x, y, w, h]")

        roi_left = tuple(roi_left_raw)
        roi_right = tuple(roi_right_raw)
        params = input_data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a JSON object, got {type(params).__name__}")

        timestamps = detect_goals(
            video_path,
            roi_left,
            roi_right,
            threshold=params.get("threshold", 25),
            min_area=params.get("min_area", None),
            max_area=params.get("max_area", None),
            dedup_window=params.get("dedup_window", 5.0),
        )

        print(json.dumps({"timestamps": timestamps}))

    except FileNotFoundError as e:
        print(json.dumps({"error": f"File not found: {e}"}), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
