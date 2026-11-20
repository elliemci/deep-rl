"""Thin launcher around rl_zoo3 for the TQC+HER Colab notebook.

Why it exists
-------------
* rl_zoo3.train only saves the replay buffer when training *finishes* (or on Ctrl-C),
  so a Colab disconnect loses it. `train` here is the same CLI, but its periodic
  checkpoints also save the replay buffer (and only the newest two are kept on Drive).
* `resume` continues from the latest checkpoint + replay buffer of a run.
* `push` is rl_zoo3.push_to_hub with the panda_gym environments registered.

Usage (arguments after the sub-command are passed to rl_zoo3 unchanged):
    python tqc_launch.py train  --algo tqc --env PandaPickAndPlace-v3 ...
    python tqc_launch.py resume --run-dir DIR --target-timesteps N --conf FILE [--seed S]
    python tqc_launch.py push   --env PandaPickAndPlace-v3 --algo tqc -f DIR -orga ORG -name NAME
"""
import argparse
import json
import re
import subprocess
import sys
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


def steps_done(run_dir):
    """Total env steps trained so far, across resumes (a resumed run restarts its own counter at 0)."""
    dirs = exp_dirs(run_dir)
    if not dirs:
        return 0
    ck = latest_checkpoint(dirs[-1])
    return _load_offsets(run_dir).get(dirs[-1].name, 0) + (ck[0] if ck else 0)


def plan_resume(run_dir, target_timesteps):
    """Return dict(ckpt, replay_buffer, done, remaining, next_exp_name) or raise."""
    dirs = exp_dirs(run_dir)
    if not dirs:
        raise SystemExit(f"nothing to resume: no experiment folder under {run_dir}/{ALGO}")
    exp = dirs[-1]
    ck = latest_checkpoint(exp)
    if ck is None:
        raise SystemExit(f"no rl_model_*_steps.zip checkpoint in {exp} (was --save-freq set?)")
    steps, zip_path, rb_path = ck
    done = _load_offsets(run_dir).get(exp.name, 0) + steps
    next_id = int(exp.name.rsplit("_", 1)[1]) + 1
    return dict(
        exp=exp, ckpt=zip_path, replay_buffer=rb_path, done=done,
        remaining=max(int(target_timesteps) - done, 0),
        next_exp_name=f"{ENV_ID}_{next_id}",
    )


def _link_replay_buffer(plan):
    """rl_zoo3 `-i` looks for `replay_buffer.pkl` next to the checkpoint: point it at the newest one."""
    link = plan["ckpt"].parent / "replay_buffer.pkl"
    if plan["replay_buffer"] is None:
        print("WARNING: this checkpoint has no replay buffer; resuming with an EMPTY buffer.")
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(plan["replay_buffer"].name)


# ------------------------------------------------------------------------ sub-commands
def _patched_zoo_train(argv):
    import panda_gym  # noqa: F401  (registers the Panda*-v3 environments)
    import rl_zoo3.exp_manager as em
    from stable_baselines3.common.callbacks import CheckpointCallback

    class ReplayBufferCheckpoint(CheckpointCallback):
        """CheckpointCallback that also saves the replay buffer and prunes old buffers."""

        def __init__(self, *args, **kwargs):
            kwargs["save_replay_buffer"] = True
            super().__init__(*args, **kwargs)

        def _on_step(self):
            keep_going = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                bufs = sorted(
                    (int(m.group(1)), p) for p in Path(self.save_path).glob("rl_model_replay_buffer_*_steps.pkl")
                    if (m := _RB.search(p.name))
                )
                for _, old in bufs[:-KEEP_REPLAY_BUFFERS]:
                    old.unlink()
            return keep_going

    em.CheckpointCallback = ReplayBufferCheckpoint
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
    print(f"replay buffer     : {plan['replay_buffer']}")
    print(f"steps done / target: {plan['done']:,} / {a.target_timesteps:,}  ->  {plan['remaining']:,} to go")
    if plan["remaining"] == 0:
        print("Target already reached, nothing to resume.")
        return
    cmd = [
        "--algo", ALGO, "--env", ENV_ID, "--gym-packages", "panda_gym", "-conf", a.conf,
        "--save-freq", str(a.save_freq), "--save-replay-buffer", "-f", a.run_dir, "--seed", str(a.seed),
        "-i", str(plan["ckpt"]), "-n", str(plan["remaining"]),
        "--eval-freq", "50000", "--eval-episodes", "20",
    ] + [x for x in a.extra if x != "--"]
    if a.dry_run:
        print("python -m rl_zoo3.train " + " ".join(cmd))
        return
    _link_replay_buffer(plan)
    offsets = _load_offsets(a.run_dir)
    offsets[plan["next_exp_name"]] = plan["done"]
    _offsets_path(a.run_dir).write_text(json.dumps(offsets, indent=1))
    _patched_zoo_train(cmd)


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
