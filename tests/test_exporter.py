import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import subprocess
import json
from pathlib import Path
import io
import pytest
from exporter import (
    merge_overlapping_ranges,
    build_clip_ranges,
    export_clips,
    main,
)


# --- build_clip_ranges tests ---

def test_build_clip_ranges_basic():
    ranges = build_clip_ranges([10.0, 30.0], before=10, after=10)
    assert ranges == [(0.0, 20.0), (20.0, 40.0)]


def test_build_clip_ranges_clamp_start():
    ranges = build_clip_ranges([3.0], before=10, after=10)
    assert ranges == [(0.0, 13.0)]


def test_build_clip_ranges_empty():
    assert build_clip_ranges([], before=10, after=10) == []


# --- merge_overlapping_ranges tests ---

def test_merge_overlapping_ranges_no_overlap():
    ranges = [(0.0, 20.0), (30.0, 50.0)]
    assert merge_overlapping_ranges(ranges) == [(0.0, 20.0), (30.0, 50.0)]


def test_merge_overlapping_ranges_with_overlap():
    ranges = [(0.0, 20.0), (15.0, 35.0), (40.0, 60.0)]
    result = merge_overlapping_ranges(ranges)
    assert result == [(0.0, 35.0), (40.0, 60.0)]


def test_merge_overlapping_ranges_all_overlap():
    ranges = [(0.0, 20.0), (10.0, 30.0), (15.0, 25.0)]
    assert merge_overlapping_ranges(ranges) == [(0.0, 30.0)]


def test_merge_overlapping_ranges_unsorted_input():
    """merge_overlapping_ranges handles unsorted input (sorts internally)."""
    ranges = [(40.0, 60.0), (0.0, 20.0), (15.0, 35.0)]
    result = merge_overlapping_ranges(ranges)
    assert result == [(0.0, 35.0), (40.0, 60.0)]


def test_merge_overlapping_ranges_empty():
    assert merge_overlapping_ranges([]) == []


# --- export_clips tests ---

def test_export_clips_empty_timestamps(tmp_path):
    """Empty timestamps returns empty list without creating output dir."""
    out_dir = tmp_path / "nonexistent"
    result = export_clips("video.mp4", [], str(out_dir))
    assert result == []
    assert not out_dir.exists()


def test_export_clips_ffmpeg_args(monkeypatch, tmp_path):
    """Verify correct ffmpeg arguments are constructed."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_dir = tmp_path / "clips"
    result = export_clips("video.mp4", [30.0], str(out_dir), before=10, after=10)

    assert len(result) == 1
    assert out_dir.exists()
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[1] == "-y"
    # Find key arguments
    assert "-ss" in cmd
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "video.mp4"
    assert "-t" in cmd
    assert "-c" in cmd
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-avoid_negative_ts" in cmd
    assert cmd[cmd.index("-avoid_negative_ts") + 1] == "make_zero"
    # Verify clip timing (before=10, after=10, timestamp=30 => 20-40)
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "20.0"
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "20.0"


def test_export_clips_overlapping_merged(monkeypatch, tmp_path):
    """Overlapping ranges are merged before ffmpeg calls."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Two timestamps 5 seconds apart with before/after 10s => overlap
    result = export_clips("video.mp4", [30.0, 35.0], str(tmp_path / "clips"),
                          before=10, after=10)

    # They should be merged into one clip (20-45)
    assert len(result) == 1
    assert len(calls) == 1


def test_export_clips_propagates_ffmpeg_stderr(monkeypatch, tmp_path):
    """ffmpeg stderr is included in the error message on failure."""
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, stderr=b"ffmpeg error: No such file"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        export_clips("video.mp4", [30.0], str(tmp_path / "clips"))

    try:
        export_clips("video.mp4", [30.0], str(tmp_path / "clips"))
    except RuntimeError as e:
        assert "ffmpeg error: No such file" in str(e)
        assert "exit 1" in str(e)


# --- main() tests ---

def test_main_bad_json(monkeypatch):
    """main() prints JSON error to stderr and exits 1 on bad JSON input."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    error_output = json.loads(stderr.getvalue())
    assert "error" in error_output
    assert "Invalid JSON" in error_output["error"]


def test_main_empty_input(monkeypatch):
    """main() handles empty stdin."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    error_output = json.loads(stderr.getvalue())
    assert "error" in error_output
    assert "No input" in error_output["error"]


def test_main_missing_required_fields(monkeypatch):
    """main() validates required fields and exits with error."""
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"video_path": "v.mp4"}'))
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    error_output = json.loads(stderr.getvalue())
    assert "error" in error_output


def test_main_success(monkeypatch, tmp_path):
    """main() succeeds with valid input."""
    def fake_export(*args, **kwargs):
        return ["/tmp/clip1.mp4"]

    monkeypatch.setattr("exporter.export_clips", fake_export)
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    input_data = {
        "video_path": "video.mp4",
        "timestamps": [30.0],
        "output_dir": str(tmp_path / "clips"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(input_data)))

    main()
    result = json.loads(stdout.getvalue())
    assert "outputs" in result
    assert result["outputs"] == ["/tmp/clip1.mp4"]


def test_main_non_dict_input(monkeypatch):
    """main() rejects non-object JSON input."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("[1, 2, 3]"))
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    error_output = json.loads(stderr.getvalue())
    assert "error" in error_output
    assert "object" in error_output["error"].lower()
