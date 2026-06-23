import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import subprocess
from pathlib import Path
from exporter import (
    merge_overlapping_ranges,
    build_clip_ranges,
    export_clips,
)


def test_build_clip_ranges_basic():
    ranges = build_clip_ranges([10.0, 30.0], before=10, after=10)
    assert ranges == [(0.0, 20.0), (20.0, 40.0)]


def test_build_clip_ranges_clamp_start():
    ranges = build_clip_ranges([3.0], before=10, after=10)
    assert ranges == [(0.0, 13.0)]


def test_build_clip_ranges_empty():
    assert build_clip_ranges([], before=10, after=10) == []


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
