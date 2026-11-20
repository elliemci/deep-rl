"""Resume bookkeeping of units/unit6-a2c-robotics/tqc_launch.py (pure file logic, no RL libraries needed)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tqc_launch", ROOT / "units/unit6-a2c-robotics/tqc_launch.py")
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)


def _exp(run, n, steps, buffers=()):
    d = run / "tqc" / f"PandaPickAndPlace-v3_{n}"
    d.mkdir(parents=True)
    for s in steps:
        (d / f"rl_model_{s}_steps.zip").write_bytes(b"z")
    for s in buffers:
        (d / f"rl_model_replay_buffer_{s}_steps.pkl").write_bytes(b"b")
    return d


def test_prefers_checkpoint_with_replay_buffer(tmp_path):
    d = _exp(tmp_path, 1, [100, 200, 300], buffers=[100, 200])
    steps, zip_path, rb = tl.latest_checkpoint(d)
    assert steps == 200 and zip_path.name == "rl_model_200_steps.zip" and rb.name.endswith("200_steps.pkl")


def test_steps_done_adds_offsets_across_resumes(tmp_path):
    _exp(tmp_path, 1, [100, 200], buffers=[200])
    plan = tl.plan_resume(tmp_path, 1000)
    assert (plan["done"], plan["remaining"], plan["next_exp_name"]) == (200, 800, "PandaPickAndPlace-v3_2")
    _exp(tmp_path, 2, [100], buffers=[100])
    (tmp_path / "tqc" / "offsets.json").write_text(json.dumps({"PandaPickAndPlace-v3_2": 200}))
    assert tl.steps_done(tmp_path) == 300
    assert tl.plan_resume(tmp_path, 1000)["remaining"] == 700
    assert tl.plan_resume(tmp_path, 250)["remaining"] == 0


def test_replay_buffer_symlink_points_to_newest(tmp_path):
    d = _exp(tmp_path, 1, [100, 200], buffers=[100, 200])
    (d / "replay_buffer.pkl").write_bytes(b"old final buffer")
    tl._link_replay_buffer(tl.plan_resume(tmp_path, 500))
    link = d / "replay_buffer.pkl"
    assert link.is_symlink() and link.resolve().name == "rl_model_replay_buffer_200_steps.pkl"


def test_nothing_to_resume(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        tl.plan_resume(tmp_path, 100)
