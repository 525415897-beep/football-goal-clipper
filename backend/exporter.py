import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def build_clip_ranges(timestamps: list, before: float = 10.0, after: float = 10.0) -> list:
    """Convert timestamps to (start, end) clip ranges in seconds."""
    return [(max(0, t - before), t + after) for t in timestamps]


def merge_overlapping_ranges(ranges: list) -> list:
    """Merge overlapping time ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: r[0])
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def export_clips(video_path: str, timestamps: list, output_dir: str,
                 before: float = 10.0, after: float = 10.0) -> list:
    """Export clips using ffmpeg stream copy. Returns list of output file paths."""
    ranges = build_clip_ranges(timestamps, before, after)
    ranges = merge_overlapping_ranges(ranges)

    if not ranges:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    outputs = []

    for i, (start, end) in enumerate(ranges, 1):
        duration = end - start
        out_path = out_dir / f"进球_{date_str}_{i:02d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode() if e.stderr else "(no stderr)"
            raise RuntimeError(
                f"ffmpeg failed (exit {e.returncode}): {stderr_text}"
            ) from e
        outputs.append(str(out_path))

    return outputs


def main():
    """Read JSON input from stdin, export clips, and print JSON results.

    Expected input format:
        {"video_path": "...", "timestamps": [...], "output_dir": "...",
         "before": 10.0, "after": 10.0}
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

        timestamps = input_data.get("timestamps")
        if not timestamps or not isinstance(timestamps, list):
            raise ValueError("Missing or invalid 'timestamps' field")

        output_dir = input_data.get("output_dir")
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("Missing or invalid 'output_dir' field")

        before = input_data.get("before", 10.0)
        after = input_data.get("after", 10.0)

        paths = export_clips(video_path, timestamps, output_dir, before, after)
        print(json.dumps({"outputs": paths}))

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
