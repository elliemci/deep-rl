"""Thin launcher around rl_zoo3 for the TQC+HER Colab notebook.

Why it exists
-------------
* rl_zoo3.train only saves the replay buffer when training *finishes* (or on Ctrl-C),
  so a Colab disconnect loses it. `train` here is the same CLI, but its periodic
  checkpoints also save the replay buffer (and only the newest two are kept on Drive).
  Every file is written to LOCAL disk first, copied to Drive, fsync'd, test-loaded
  and only then renamed into place, so a half-written file never has a final name.
* `resume` continues from the latest checkpoint + ITS OWN replay buffer. rl_zoo3 (2.9.2,
  `ExperimentManager._load_pretrained_agent`) ignores which buffer you want: it loads
  `<dirname of the -i zip>/replay_buffer.pkl` if that file exists and silently continues
  with an empty buffer if not. So resume stages the checkpoint + its buffer into a fresh
  LOCAL folder as `<ckpt>.zip` + `replay_buffer.pkl`, verifies the buffer loads, and
  fails loudly instead of ever resuming with an empty buffer.
* `push` is rl_zoo3.push_to_hub with the panda_gym environments registered.

Usage (arguments after the sub-command are passed to rl_zoo3 unchanged):
    python tqc_launch.py train  --algo tqc --env PandaPickAndPlace-v3 ...
    python tqc_launch.py resume --run-dir DIR --target-timesteps N --conf FILE [--seed S]
    python tqc_launch.py push   --env PandaPickAndPlace-v3 --algo tqc -f DIR -orga ORG -name NAME
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ENV_ID = "PandaPickAndPlace-v3"
ALGO = "tqc"
KEEP_REPLAY_BUFFERS = 2
_CKPT = re.compile(r"rl_model_(\d+)_steps\.zip$")
_RB = re.compile(r"rl_model_replay_buffer_(\d+)_steps\.pkl$")


# ----------------------------------------------------------------- run-folder helpers
def exp_dirs(run_dir, algo=ALGO, env_id=ENV_ID):
    """Experiment folders of a run, sorted by id: <run_dir>/<algo>/<env_id>_<id>."""
    pat = re.compile(rf"^{re.escape(env_id)}_(\d+)$")
    base = Path(run_dir) / algo
    found = [(int(m.group(1)), p) for p in base.glob(f"{env_id}_*") if p.is_dir() and (m := pat.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(exp_dir):
    """(steps, zip_path, replay_buffer_path_or_None) of the newest usable checkpoint, else None.
    Prefers a checkpoint that has a replay buffer next to it."""
    exp_dir = Path(exp_dir)
    ckpts = {int(m.group(1)): p for p in exp_dir.glob("rl_model_*_steps.zip") if (m := _CKPT.search(p.name))}
    bufs = {int(m.group(1)): p for p in exp_dir.glob("rl_model_replay_buffer_*_steps.pkl") if (m := _RB.search(p.name))}
    with_buf = sorted(set(ckpts) & set(bufs))
    if with_buf:
        s = with_buf[-1]
        return s, ckpts[s], bufs[s]
    if ckpts:
        s = max(ckpts)
        return s, ckpts[s], None
    return None


def _offsets_path(run_dir, algo=ALGO):
    return Path(run_dir) / algo / "offsets.json"


def _load_offsets(run_dir):
    p = _offsets_path(run_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _exp_id(path):
    return int(Path(path).name.rsplit("_", 1)[1])


def steps_done(run_dir):
    """Total env steps trained so far, across resumes (a resumed run restarts its own counter at 0)."""
    for exp in reversed(exp_dirs(run_dir)):
        ck = latest_checkpoint(exp)
        if ck:
            return _load_offsets(run_dir).get(exp.name, 0) + ck[0]
    return 0


def plan_resume(run_dir, target_timesteps):
    """Pick the newest checkpoint that HAS a replay buffer. Raises SystemExit if there is none:
    resuming with an empty buffer is never allowed."""
    dirs = exp_dirs(run_dir)
    if not dirs:
        raise SystemExit(f"nothing to resume: no experiment folder under {run_dir}/{ALGO}")
    for exp in reversed(dirs):
        ck = latest_checkpoint(exp)
        if ck and ck[2] is not None:
            steps, zip_path, rb_path = ck
            done = _load_offsets(run_dir).get(exp.name, 0) + steps
            return dict(
                exp=exp, ckpt=zip_path, replay_buffer=rb_path, done=done, ckpt_steps=steps,
                remaining=max(int(target_timesteps) - done, 0),
                next_exp_name=f"{ENV_ID}_{_exp_id(dirs[-1]) + 1}",
            )
    raise SystemExit(
        f"REFUSING TO RESUME: no checkpoint with a replay buffer (rl_model_replay_buffer_*_steps.pkl) in {run_dir}/{ALGO}. "
        "Resuming without it would silently restart with an empty buffer. Start a new run instead."
    )


# ------------------------------------------------- safe file handling (local -> Drive)
def _load_buffer(path):
    """Unpickle a replay buffer exactly like SB3/rl_zoo3 do. Returns the buffer."""
    from stable_baselines3.common.save_util import load_from_pkl
    return load_from_pkl(str(path))


def buffer_size(buf):
    return int(buf.size()) if hasattr(buf, "size") else int(buf.buffer_size if buf.full else buf.pos)


def check_buffer(path, loader=None, min_size=1):
    """Test-load a replay buffer file; raise RuntimeError if it is broken or empty. Returns its size."""
    try:
        buf = (loader or _load_buffer)(path)
    except Exception as e:  # UnpicklingError, EOFError, ...
        raise RuntimeError(f"replay buffer {path} does not load: {type(e).__name__}: {e}") from e
    n = buffer_size(buf)
    if n < min_size:
        raise RuntimeError(f"replay buffer {path} loaded but is empty (size {n})")
    return n


def publish_file(src, dst, loader=None):
    """Copy src -> dst via `<dst>.part`: fsync, test-load if it is a replay buffer, then atomic rename.
    A crash or disconnect can only leave a `.part` file, never a half-written file under the real name."""
    src, dst = Path(src), Path(dst)
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    with open(part, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    if dst.name.endswith(".pkl") and "replay_buffer" in dst.name:
        try:
            check_buffer(part, loader)
        except RuntimeError:
            part.unlink()
            raise
    os.replace(part, dst)
    return dst


def stage_resume(plan, stage_root=None, loader=None):
    """Copy the chosen checkpoint + ITS buffer into a fresh LOCAL folder as <ckpt>.zip + replay_buffer.pkl
    (real copies; Google Drive does not support symlinks). Returns (staged_zip, buffer_size)."""
    if plan["replay_buffer"] is None:
        raise SystemExit("REFUSING TO RESUME: the chosen checkpoint has no replay buffer.")
    stage = Path(tempfile.mkdtemp(prefix="resume_stage_", dir=stage_root))
    zip_path = stage / plan["ckpt"].name
    shutil.copyfile(plan["ckpt"], zip_path)
    buf_path = stage / "replay_buffer.pkl"
    shutil.copyfile(plan["replay_buffer"], buf_path)
    try:
        n = check_buffer(buf_path, loader)
    except RuntimeError as e:
        raise SystemExit(f"REFUSING TO RESUME: {e}")
    return zip_path, n


# ------------------------------------------------------------------------ sub-commands
def _patched_zoo_train(argv, expect_buffer_steps=None):
    import panda_gym  # noqa: F401  (registers the Panda*-v3 environments)
    import rl_zoo3.exp_manager as em
    from stable_baselines3.common.callbacks import CheckpointCallback

    class ReplayBufferCheckpoint(CheckpointCallback):
        """Checkpoints (model + replay buffer) written locally, then published to Drive safely."""

        def __init__(self, *args, **kwargs):
            kwargs["save_replay_buffer"] = True
            super().__init__(*args, **kwargs)

        def _on_step(self):
            if self.n_calls % self.save_freq != 0:
                return True
            real = self.save_path
            tmp = Path(tempfile.mkdtemp(prefix="ckpt_"))  # local disk, not Drive
            self.save_path = str(tmp)
            try:
                keep_going = super()._on_step()
            finally:
                self.save_path = real
            Path(real).mkdir(parents=True, exist_ok=True)
            for f in sorted(tmp.iterdir()):
                publish_file(f, Path(real) / f.name)
            shutil.rmtree(tmp, ignore_errors=True)
            bufs = sorted(
                (int(m.group(1)), p) for p in Path(real).glob("rl_model_replay_buffer_*_steps.pkl")
                if (m := _RB.search(p.name))
            )
            for _, old in bufs[:-KEEP_REPLAY_BUFFERS]:
                old.unlink()
            return keep_going

    em.CheckpointCallback = ReplayBufferCheckpoint

    # End-of-training buffer: save locally, publish + test-load, instead of writing straight to Drive.
    orig_save = em.ExperimentManager.save_trained_model

    def save_trained_model(self, model):
        want_buffer, self.save_replay_buffer = self.save_replay_buffer, False
        try:
            orig_save(self, model)
        finally:
            self.save_replay_buffer = want_buffer
        if want_buffer and getattr(model, "replay_buffer", None) is not None:
            tmp = Path(tempfile.mkdtemp(prefix="final_rb_")) / "replay_buffer.pkl"
            model.save_replay_buffer(str(tmp))
            publish_file(tmp, Path(self.save_path) / "replay_buffer.pkl")
            shutil.rmtree(tmp.parent, ignore_errors=True)
            print(f"Saved + verified final replay buffer ({buffer_size(model.replay_buffer):,} transitions)")

    em.ExperimentManager.save_trained_model = save_trained_model

    if expect_buffer_steps is not None:
        orig_load = em.ExperimentManager._load_pretrained_agent

        def load_pretrained_agent(self, hyperparams, env):
            model = orig_load(self, hyperparams, env)
            n = buffer_size(model.replay_buffer)
            print(f"REPLAY_BUFFER_SIZE_AFTER_LOAD={n} EXPECTED~={expect_buffer_steps}", flush=True)
            # HerReplayBuffer truncates the last (unfinished) episode on load, so allow a little slack.
            if n == 0 or n < expect_buffer_steps - 100:
                raise RuntimeError(
                    f"resume loaded a replay buffer of {n} transitions but ~{expect_buffer_steps} were expected: aborting"
                )
            return model

        em.ExperimentManager._load_pretrained_agent = load_pretrained_agent

    sys.argv = ["rl_zoo3.train"] + list(argv)
    from rl_zoo3.train import train
    train()


def cmd_train(argv):
    _patched_zoo_train(argv)


def cmd_resume(argv):
    ap = argparse.ArgumentParser(prog="tqc_launch.py resume")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-timesteps", type=int, required=True, help="total steps wanted, counting those already done")
    ap.add_argument("--conf", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra rl_zoo3.train args, after `--`")
    a = ap.parse_args(argv)
    plan = plan_resume(a.run_dir, a.target_timesteps)
    print(f"latest checkpoint : {plan['ckpt']}")
    print(f"its replay buffer : {plan['replay_buffer']}")
    print(f"steps done / target: {plan['done']:,} / {a.target_timesteps:,}  ->  {plan['remaining']:,} to go")
    if plan["remaining"] == 0:
        print("Target already reached, nothing to resume.")
        return
    if a.dry_run:
        print("(dry run) would stage the checkpoint + buffer locally and call rl_zoo3.train -i <staged zip>")
        return
    staged_zip, n = stage_resume(plan)
    print(f"staged {staged_zip} ; buffer test-load OK: {n:,} transitions")
    cmd = [
        "--algo", ALGO, "--env", ENV_ID, "--gym-packages", "panda_gym", "-conf", a.conf,
        "--save-freq", str(a.save_freq), "--save-replay-buffer", "-f", a.run_dir, "--seed", str(a.seed),
        "-i", str(staged_zip), "-n", str(plan["remaining"]),
    ] + [x for x in a.extra if x != "--"]
    if "--eval-freq" not in cmd:
        cmd += ["--eval-freq", "50000", "--eval-episodes", "20"]
    offsets = _load_offsets(a.run_dir)
    offsets[plan["next_exp_name"]] = plan["done"]
    _offsets_path(a.run_dir).write_text(json.dumps(offsets, indent=1))
    _patched_zoo_train(cmd, expect_buffer_steps=plan["ckpt_steps"])


def cmd_push(argv):
    import runpy
    import panda_gym  # noqa: F401
    sys.argv = ["rl_zoo3.push_to_hub"] + list(argv)
    runpy.run_module("rl_zoo3.push_to_hub", run_name="__main__")


if __name__ == "__main__":
    cmds = {"train": cmd_train, "resume": cmd_resume, "push": cmd_push}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])
