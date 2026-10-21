import numpy as np
import pytest

from ecgscan import cli, post
from ecgscan.compare import compare_dirs

from test_post import PX_PER_MM, make_lines


def test_parser_defaults():
    args = cli.build_parser().parse_args(["convert", "in", "out"])
    assert (args.layout, args.rhythm, args.speed, args.gain) == ("3x4", "V1", 25.0, "auto")


def test_convert_without_images_returns_1(tmp_path, capsys):
    assert cli.main(["convert", str(tmp_path), str(tmp_path / "out")]) == 1


def test_compare_detects_lag(tmp_path):
    (tmp_path / "scan").mkdir()
    (tmp_path / "ref").mkdir()
    lines, *_ = make_lines()
    data, header, meta = post.process(lines, PX_PER_MM, post.Settings())
    post.write_outputs(data, header, meta, str(tmp_path / "scan/e.csv"), str(tmp_path / "scan/e.json"))
    shifted = data.copy()
    shifted[:, 1:] = np.roll(shifted[:, 1:], 5, axis=0)  # 10 ms à 500 Hz
    post.write_outputs(shifted, header, meta, str(tmp_path / "ref/e.csv"), str(tmp_path / "ref/e.json"))
    res = compare_dirs(tmp_path / "scan", tmp_path / "ref")
    assert len(res) == 13
    assert all(abs(abs(r[3]) - 10.0) < 0.1 for r in res)
    assert all(r[4] > 0.99 for r in res)
