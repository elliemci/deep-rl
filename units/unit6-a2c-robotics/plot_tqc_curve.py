"""Learning curve of the TQC+HER PandaPickAndPlace-v3 run, rebuilt from the small files in results/.

    python plot_tqc_curve.py [results_dir] [out.png]      # default: ./results -> ../../assets/plots/tqc_eval_curve.png

results/ holds, per experiment folder of the run, RL-Zoo's `evaluations.npz` (20-episode evaluations every 50k steps,
with the step counter restarting at 0 after every resume), `offsets.json` (global step at which each resumed folder
started) and the final evaluation JSONs. The run was resumed from checkpoints, so evaluations that appear in two
folders (1.85M is in both run 3 and run 4: run 4 restarted from the 1.8M checkpoint) are taken from the LATER folder,
which is the one the final model continued from.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = "PandaPickAndPlace-v3_{}"


def load_curve(results_dir):
    """Returns dict(steps, success_pct, run, starts, superseded) with one point per 50k steps."""
    d = Path(results_dir)
    offsets = json.loads((d / "offsets.json").read_text())
    points, superseded, starts = {}, [], {}
    for run in range(1, 100):
        f = d / f"evaluations_run{run}.npz"
        if not f.exists():
            break
        off = 0 if run == 1 else int(offsets[EXP.format(run)])
        starts[run] = off
        z = np.load(f)
        for t, succ in zip(z["timesteps"], z["successes"].mean(axis=1) * 100):
            g = int(t) + off
            if g in points:
                superseded.append((g, points[g][0], points[g][1]))
            points[g] = (float(succ), run)
    steps = sorted(points)
    return dict(steps=steps, success_pct=[points[s][0] for s in steps], run=[points[s][1] for s in steps],
                starts=starts, superseded=superseded)


def plot(curve, results_dir, out=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    d = Path(results_dir)
    final = json.loads((d / "eval_1M_vs_2M_500ep.json").read_text())
    a2c = json.loads((d / "eval_results_100ep.json").read_text())["a2c"]

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=110)
    ax.plot(curve["steps"], curve["success_pct"], marker="o", ms=3.5, color="#2a6fdb", label="TQC+HER, 20 episodes per point")
    for g, v, run in curve["superseded"]:
        ax.plot([g], [v], marker="x", color="#9bb8e8", ls="none", label="replaced by the resumed run (same step)")
    for k, (run, off) in enumerate(r for r in curve["starts"].items() if r[1]):
        ax.axvline(off, color="grey", ls=":", lw=0.9)
        ax.text(off, 97 - 7 * (k % 2), f" resume {off / 1e6:g}M", fontsize=7, color="grey", va="top")
    ax.axhline(final["2M"]["success_rate"] * 100, color="#2a6fdb", ls="--", lw=1,
               label=f"2M steps, 500-episode eval: {final['2M']['success_rate']:.1%}")
    ax.axhline(final["1M"]["success_rate"] * 100, color="#7aa6ec", ls="--", lw=1,
               label=f"1M steps, 500-episode eval: {final['1M']['success_rate']:.1%}")
    ax.axhline(a2c["success_rate"] * 100, color="#c0392b", ls="--", lw=1, label=f"A2C (Unit 6), 100-episode eval: {a2c['success_rate']:.0%}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.set_xlabel("training steps"); ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("PandaPickAndPlace-v3: evaluation success during training", fontsize=10)
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    if out:
        fig.savefig(out)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here.parent.parent / "assets/plots/tqc_eval_curve.png"
    c = load_curve(res)
    plot(c, res, out=out)
    print(f"{len(c['steps'])} points, {c['steps'][0]:,} to {c['steps'][-1]:,} steps -> {out}")
