import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def build_clip_ranges(timestamps: list, before: float = 10.0, after: float = 10.0) -> list:
    """Convert timestamps to (start, end) clip ranges in seconds."""
    return [(max(0, t - before), t + after) for t in timestamps]


def merge_overlapping_ranges(ranges: list) -> list:
    """Merge overlapping time ranges. Input must be sorted by start time."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def export_clips(video_path: str, timestamps: list, output_dir: str,
                 before: float = 10.0, after: float = 10.0) -> list:
    """Export clips using ffmpeg stream copy. Returns list of output file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranges = build_clip_ranges(timestamps, before, after)
    ranges = merge_overlapping_ranges(ranges)

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
        subprocess.run(cmd, check=True, capture_output=True)
        outputs.append(str(out_path))

    return outputs


def main():
    input_data = json.loads(sys.stdin.read())
    video_path = input_data["video_path"]
    timestamps = input_data["timestamps"]
    output_dir = input_data["output_dir"]
    before = input_data.get("before", 10.0)
    after = input_data.get("after", 10.0)

    paths = export_clips(video_path, timestamps, output_dir, before, after)
    print(json.dumps({"outputs": paths}))


if __name__ == "__main__":
    main()
