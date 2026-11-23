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


class _FakeBuf:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


def _fake_loader(path):
    """Stands in for load_from_pkl: file content b'buf:<n>' -> buffer of n transitions; anything else is 'corrupt'."""
    data = Path(path).read_bytes()
    if not data.startswith(b"buf:"):
        raise EOFError("Ran out of input")  # what a truncated/foreign pickle looks like
    return _FakeBuf(int(data[4:]))


def _exp_buf(run, n, steps):
    """Experiment folder whose buffers contain `buf:<steps>` so we can tell them apart."""
    d = _exp(run, n, steps)
    for s in steps:
        (d / f"rl_model_replay_buffer_{s}_steps.pkl").write_bytes(f"buf:{s}".encode())
    return d


def test_resume_loads_exactly_the_checkpoints_own_buffer(tmp_path):
    """The Colab failure: rl_zoo3 loads <dir of -i zip>/replay_buffer.pkl and ignores any other buffer."""
    d = _exp_buf(tmp_path, 1, [1000, 2000])
    (d / "replay_buffer.pkl").write_bytes(b"BROKEN end-of-training file")  # stale/corrupt file on Drive
    plan = tl.plan_resume(tmp_path, 3000)
    staged_zip, n = tl.stage_resume(plan, stage_root=tmp_path, loader=_fake_loader)
    staged_buf = staged_zip.parent / "replay_buffer.pkl"
    assert n == 2000 and staged_buf.read_bytes() == b"buf:2000"        # the 2000-step buffer, not 1000, not the broken one
    assert staged_zip.name == "rl_model_2000_steps.zip" and not staged_buf.is_symlink()
    assert staged_zip.parent != d                                       # staged away from the Drive folder
    assert (d / "replay_buffer.pkl").read_bytes().startswith(b"BROKEN")  # Drive folder untouched


def test_resume_fails_loudly_when_buffer_is_missing(tmp_path):
    import pytest
    _exp(tmp_path, 1, [1000, 2000])  # checkpoints but no buffers at all
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        tl.plan_resume(tmp_path, 3000)


def test_resume_skips_newer_experiment_without_buffer(tmp_path):
    _exp_buf(tmp_path, 1, [1000, 2000])
    _exp(tmp_path, 2, [500])  # a failed resume attempt: checkpoint but no buffer
    plan = tl.plan_resume(tmp_path, 3000)
    assert plan["ckpt"].parent.name.endswith("_1") and plan["next_exp_name"] == "PandaPickAndPlace-v3_3"


def test_resume_fails_loudly_when_buffer_does_not_load(tmp_path):
    import pytest
    d = _exp(tmp_path, 1, [1000])
    (d / "rl_model_replay_buffer_1000_steps.pkl").write_bytes(b"garbage")
    with pytest.raises(SystemExit, match="REFUSING TO RESUME.*does not load"):
        tl.stage_resume(tl.plan_resume(tmp_path, 2000), stage_root=tmp_path, loader=_fake_loader)


def test_resume_refuses_empty_buffer(tmp_path):
    import pytest
    d = _exp(tmp_path, 1, [1000])
    (d / "rl_model_replay_buffer_1000_steps.pkl").write_bytes(b"buf:0")
    with pytest.raises(SystemExit, match="empty"):
        tl.stage_resume(tl.plan_resume(tmp_path, 2000), stage_root=tmp_path, loader=_fake_loader)


def test_publish_file_verifies_and_never_leaves_partial_files(tmp_path):
    import pytest
    src_ok, src_bad = tmp_path / "ok.pkl", tmp_path / "bad.pkl"
    src_ok.write_bytes(b"buf:5"); src_bad.write_bytes(b"truncated")
    out = tmp_path / "drive"; out.mkdir()
    tl.publish_file(src_ok, out / "rl_model_replay_buffer_5_steps.pkl", loader=_fake_loader)
    assert (out / "rl_model_replay_buffer_5_steps.pkl").read_bytes() == b"buf:5"
    with pytest.raises(RuntimeError):
        tl.publish_file(src_bad, out / "rl_model_replay_buffer_9_steps.pkl", loader=_fake_loader)
    assert sorted(p.name for p in out.iterdir()) == ["rl_model_replay_buffer_5_steps.pkl"]  # no bad file, no .part


def test_nothing_to_resume(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        tl.plan_resume(tmp_path, 100)


def test_eval_curve_rebuilt_from_committed_results():
    """results/ must reproduce the learning curve: 40 points 50k..2M, the duplicated 1.85M point taken from run 4."""
    import pytest
    pytest.importorskip("numpy")
    spec2 = importlib.util.spec_from_file_location("plot_tqc_curve", ROOT / "units/unit6-a2c-robotics/plot_tqc_curve.py")
    pc = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pc)
    c = pc.load_curve(ROOT / "units/unit6-a2c-robotics/results")
    assert c["steps"] == list(range(50_000, 2_000_001, 50_000))
    i = c["steps"].index(1_850_000)
    assert c["run"][i] == 4 and c["success_pct"][i] == 70.0
    assert [(g, round(v), r) for g, v, r in c["superseded"]] == [(1_850_000, 50, 3)]
    assert c["starts"] == {1: 0, 2: 900_000, 3: 1_000_000, 4: 1_800_000}
