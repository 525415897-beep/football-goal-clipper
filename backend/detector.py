import cv2
import numpy as np
import json
import sys
import wave
import subprocess
import tempfile
import os


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


def check_rois(contours: list, roi_left: tuple, roi_right: tuple) -> tuple:
    """Check if any contour's centroid falls within either ROI.

    Returns (hit_left, hit_right) booleans. Single pass over all contours.
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


def _resolve_area_thresholds(min_area, max_area, frame_width, frame_height):
    """Compute absolute area thresholds from values that may be fractions."""
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


def detect_audio_peaks(video_path: str, peak_threshold: float = 4.0,
                       window_ms: int = 200, min_peak_duration_ms: int = 600,
                       ffmpeg_path: str = "ffmpeg") -> list:
    """Detect sudden loudness spikes (cheers/goal celebrations) in video audio.

    Args:
        video_path: Path to the video file.
        peak_threshold: Multiplier above median RMS to trigger (default 4x = cheers).
        window_ms: Analysis window size in milliseconds (default 200ms).
        min_peak_duration_ms: Minimum peak duration to count (default 600ms).
        ffmpeg_path: Path to ffmpeg binary.

    Returns:
        List of peak timestamps in seconds, deduplicated.
    """
    # Extract audio to temp WAV: mono, 16kHz sample rate
    audio_path = tempfile.mktemp(suffix='.wav')
    try:
        cmd = [
            ffmpeg_path, "-y", "-v", "error",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            raise RuntimeError("Extracted audio file is empty or missing")

        # Read audio and compute RMS energy
        with wave.open(audio_path, 'r') as wf:
            nchannels, sampwidth, framerate, nframes = wf.getparams()[:4]
            if nframes == 0:
                return []
            audio = np.frombuffer(wf.readframes(nframes), dtype=np.int16).astype(np.float32)

        # Compute RMS in sliding windows
        window_samples = int(framerate * window_ms / 1000)
        step_samples = window_samples  # non-overlapping windows
        num_windows = max(1, (len(audio) - window_samples) // step_samples)

        rms = np.zeros(num_windows)
        for i in range(num_windows):
            start = i * step_samples
            chunk = audio[start:start + window_samples]
            rms[i] = np.sqrt(np.mean(chunk ** 2))

        if len(rms) == 0:
            return []

        # Background = median RMS (robust to peaks)
        background = float(np.median(rms))
        if background < 1:
            return []

        threshold = background * peak_threshold

        # Find sustained peaks above threshold
        peaks = []
        in_peak = False
        peak_start = 0

        for i in range(num_windows):
            above = rms[i] > threshold
            if above and not in_peak:
                in_peak = True
                peak_start = i
            elif not above and in_peak:
                in_peak = False
                peak_duration = (i - peak_start) * window_ms
                if peak_duration >= min_peak_duration_ms:
                    # Timestamp = center of the peak
                    peak_mid = peak_start + (i - peak_start) // 2
                    timestamp = (peak_mid * window_ms) / 1000.0
                    peaks.append(timestamp)

        # Handle peak at end of audio
        if in_peak:
            peak_duration = (num_windows - peak_start) * window_ms
            if peak_duration >= min_peak_duration_ms:
                peak_mid = peak_start + (num_windows - peak_start) // 2
                timestamp = (peak_mid * window_ms) / 1000.0
                peaks.append(timestamp)

        print(f"音频检测: 背景RMS={background:.0f}, 阈值={threshold:.0f}, "
              f"峰值={len(peaks)}个", file=sys.stderr)

    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    return deduplicate_timestamps(peaks, window=5.0)


def detect_goals_video(video_path: str, roi_left: tuple, roi_right: tuple,
                       threshold: int = 25, min_area=None, max_area=None,
                       dedup_window: float = 5.0, frame_skip: int = 5) -> list:
    """Process video frames and detect motion in goal ROIs.

    This is the video-only detection. Use detect_goals_combined() for
    audio-first detection with optional video verification.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Downscale for speed
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    if orig_w > 960:
        scale = 960.0 / orig_w
        proc_w, proc_h = 960, int(orig_h * scale)
    else:
        proc_w, proc_h = orig_w, orig_h

    roi_left_scaled = tuple(int(v * scale) for v in roi_left)
    roi_right_scaled = tuple(int(v * scale) for v in roi_right)

    candidates = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_frame = cv2.resize(prev_frame, (proc_w, proc_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    min_area, max_area = _resolve_area_thresholds(min_area, max_area, proc_w, proc_h)

    frame_idx = 1
    processed = 0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            frame = cv2.resize(frame, (proc_w, proc_h))
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            contours = detect_motion_regions(prev_gray, curr_gray, threshold)
            contours = filter_by_size(contours, min_area, max_area)
            hit_left, hit_right = check_rois(contours, roi_left_scaled, roi_right_scaled)

            if hit_left or hit_right:
                candidates.append(frame_idx / fps)

            prev_gray = curr_gray
            processed += 1
            frame_idx += 1

            if processed % 500 == 0:
                print(f"视频进度: {frame_idx/fps:.0f}s, 候选: {len(candidates)}", file=sys.stderr)

        except Exception as e:
            print(f"Warning: skip frame {frame_idx}: {e}", file=sys.stderr)
            frame_idx += 1
            continue

    cap.release()
    return deduplicate_timestamps(candidates, dedup_window)


def detect_goals_combined(video_path: str, roi_left: tuple, roi_right: tuple,
                          audio_peak_threshold: float = 4.0,
                          video_verify: bool = True,
                          verify_window: float = 3.0,
                          dedup_window: float = 5.0,
                          **video_kwargs) -> list:
    """Detect goals using audio cheers + optional video verification.

    This is the recommended detection method:
    1. Scan audio for sudden loudness peaks (cheers after a goal)
    2. Optionally verify: for each audio peak, check if there's motion
       in the goal ROI within ±verify_window seconds

    Args:
        video_path: Path to video file.
        roi_left: Left goal ROI (x, y, w, h) in original pixel coords.
        roi_right: Right goal ROI (x, y, w, h) in original pixel coords.
        audio_peak_threshold: RMS multiplier for cheer detection (default 4x).
        video_verify: Whether to verify audio peaks with video motion.
        verify_window: ±seconds around audio peak to check for motion.
        dedup_window: Minimum seconds between detected goals.
        **video_kwargs: Passed to detect_goals_video (threshold, min_area, etc.)

    Returns:
        List of goal timestamps in seconds.
    """
    print("步骤1/2: 音频检测欢呼声...", file=sys.stderr)
    audio_peaks = detect_audio_peaks(
        video_path,
        peak_threshold=audio_peak_threshold,
    )

    if not audio_peaks:
        print("音频检测: 未发现欢呼声", file=sys.stderr)
        return []

    print(f"音频检测: 发现 {len(audio_peaks)} 个候选欢呼时刻", file=sys.stderr)

    if not video_verify:
        return audio_peaks

    # Video verification: check each audio peak
    print("步骤2/2: 视频验证球门区域运动...", file=sys.stderr)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Downscale for speed
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    if orig_w > 960:
        scale = 960.0 / orig_w
        proc_w, proc_h = 960, int(orig_h * scale)
    else:
        proc_w, proc_h = orig_w, orig_h

    roi_left_scaled = tuple(int(v * scale) for v in roi_left)
    roi_right_scaled = tuple(int(v * scale) for v in roi_right)

    min_area, max_area = _resolve_area_thresholds(
        video_kwargs.get('min_area'),
        video_kwargs.get('max_area'),
        proc_w, proc_h
    )
    threshold = video_kwargs.get('threshold', 25)

    confirmed = []
    frame_skip = video_kwargs.get('frame_skip', 3)

    for i, peak_t in enumerate(audio_peaks):
        check_start = max(0, peak_t - verify_window)
        check_end = peak_t + verify_window

        # Seek to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(check_start * fps))
        cap.grab()
        ret, prev_frame = cap.retrieve()
        if not ret:
            continue

        prev_frame = cv2.resize(prev_frame, (proc_w, proc_h))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        found_motion = False
        frame_idx = int(check_start * fps) + 1
        end_frame = int(check_end * fps)

        while frame_idx <= end_frame and not found_motion:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                frame = cv2.resize(frame, (proc_w, proc_h))
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                contours = detect_motion_regions(prev_gray, curr_gray, threshold)
                contours = filter_by_size(contours, min_area, max_area)
                hit_left, hit_right = check_rois(contours, roi_left_scaled, roi_right_scaled)

                if hit_left or hit_right:
                    found_motion = True
                    confirmed.append(peak_t)

                prev_gray = curr_gray
                frame_idx += 1
            except Exception:
                frame_idx += 1
                continue

        side_tag = "✓" if found_motion else "✗"
        m, s = int(peak_t // 60), int(peak_t % 60)
        print(f"  {side_tag} {m:02d}:{s:02d} {'确认' if found_motion else '跳过(无球门运动)'}", file=sys.stderr)

    cap.release()
    print(f"最终确认: {len(confirmed)} 个进球", file=sys.stderr)
    return deduplicate_timestamps(confirmed, dedup_window)


# Backward-compatible alias
detect_goals = detect_goals_video


def main():
    """Read JSON input from stdin, run goal detection, and print JSON results.

    Expected input format:
        {"video_path": "...", "roi_left": [x,y,w,h], "roi_right": [x,y,w,h],
         "params": {"mode": "combined", "audio_peak_threshold": 4.0, ...}}

    Modes:
        "combined" (default) - Audio cheer detection + video motion verification
        "audio" - Audio-only cheer detection (fastest)
        "video" - Video-only motion detection (legacy)
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

        params = input_data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a JSON object, got {type(params).__name__}")

        mode = params.get("mode", "combined")

        if mode in ("combined", "video"):
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

        if mode == "audio":
            timestamps = detect_audio_peaks(
                video_path,
                peak_threshold=params.get("audio_peak_threshold", 4.0),
            )

        elif mode == "combined":
            timestamps = detect_goals_combined(
                video_path,
                roi_left,
                roi_right,
                audio_peak_threshold=params.get("audio_peak_threshold", 4.0),
                video_verify=params.get("video_verify", True),
                verify_window=params.get("verify_window", 3.0),
                dedup_window=params.get("dedup_window", 5.0),
                threshold=params.get("threshold", 25),
                min_area=params.get("min_area", None),
                max_area=params.get("max_area", None),
                frame_skip=params.get("frame_skip", 3),
            )

        elif mode == "video":
            timestamps = detect_goals_video(
                video_path,
                roi_left,
                roi_right,
                threshold=params.get("threshold", 25),
                min_area=params.get("min_area", None),
                max_area=params.get("max_area", None),
                dedup_window=params.get("dedup_window", 5.0),
                frame_skip=params.get("frame_skip", 5),
            )

        else:
            raise ValueError(f"Unknown mode: '{mode}'. Valid: audio, combined, video")

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
