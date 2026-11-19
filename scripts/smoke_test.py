#!/usr/bin/env python3
"""Smoke-test a notebook locally with tiny budgets, without touching the repo or the Hub.

Works on a temporary copy of the notebook (the committed notebook is never modified):
  * comments out shell/magic lines (!pip, !apt, %%capture, ...): install dependencies beforehand in the venv
  * replaces top-level calls that publish (push_to_hub, package_to_hub, upload_*, create_repo, notebook_login)
    with `pass`
  * shrinks budgets: total_timesteps, n_training_episodes, n_eval_episodes, n_episodes, ...
  * runs with HF_HUB_OFFLINE=1 and an empty HF_HOME so nothing can be pushed even if a call slips through
Usage: smoke_test.py NOTEBOOK [--timeout 900] [--skip-cells 12,13] [--set name=value ...]
"""
import argparse, ast, os, re, sys, tempfile, time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

PUBLISH = {"push_to_hub", "package_to_hub", "upload_folder", "upload_file", "create_repo", "notebook_login", "login"}
BUDGETS = {
    "total_timesteps": "2048", "n_training_episodes": "20", "n_eval_episodes": "2", "n_evaluation_episodes": "2",
    "n_episodes": "5", "n_timesteps": "1000", "video_length": "50", "max_t": "100",
}


def call_name(node):
    f = getattr(node, "func", None)
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def neutralise(src):
    lines = [("# [smoke] " + l) if re.match(r"\s*[!%]", l) else l for l in src.split("\n")]
    src = "\n".join(lines)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.split("\n")
    for st in tree.body:                                  # top-level statements only (not inside defs)
        calls = [n for n in ast.walk(st) if isinstance(n, ast.Call)] if not isinstance(st, (ast.FunctionDef, ast.ClassDef)) else []
        colab = isinstance(st, (ast.Import, ast.ImportFrom)) and "google.colab" in ast.unparse(st) or \
            any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("drive", "files") for n in ast.walk(st))
        colab = colab or bool(re.search(r"os\.kill|/content/|pyvirtualdisplay|virtual_display|Display\(", ast.unparse(st)))
        if colab or any(call_name(c) in PUBLISH for c in calls):
            for i in range(st.lineno - 1, st.end_lineno):
                lines[i] = ("" if i != st.lineno - 1 else " " * st.col_offset) + ("pass  # [smoke] publish skipped" if i == st.lineno - 1 else "")
    return "\n".join(lines)


def shrink(src, extra):
    for name, val in {**BUDGETS, **extra}.items():
        src = re.sub(rf"(?m)^(\s*)({name}\s*=\s*)[^,)\n#]+", rf"\g<1>\g<2>{val}", src)
        src = re.sub(rf"([\"']{name}[\"']\s*:\s*)\d+", rf"\g<1>{val}", src)
        src = re.sub(rf"({name}\s*=\s*)(?:int\()?[\d_.e*]+\)?(?=\s*[,)])", rf"\g<1>{val}", src)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook"); ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--skip-cells", default=""); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--kernel", default="deeprl-smoke")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_cells.split(",") if x}
    extra = dict(s.split("=", 1) for s in a.set)
    nb = nbformat.read(a.notebook, 4)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code": continue
        c.source = "pass  # [smoke] skipped by --skip-cells" if i in skip else shrink(neutralise(c.source), extra)
    work = tempfile.mkdtemp(prefix="smoke_")
    for d in ("videos", "video", "hub", "logs", "models"):   # folders that exist on the author's Drive
        os.makedirs(os.path.join(work, d), exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=os.path.join(work, "hf"), HF_TOKEN="", WANDB_MODE="disabled",
                      MPLBACKEND="Agg", SDL_VIDEODRIVER="dummy", PYTHONWARNINGS="ignore")
    ep = ExecutePreprocessor(timeout=a.timeout, kernel_name=a.kernel, allow_errors=False)
    t = time.time()
    state = {"i": None}
    ep.on_cell_start = lambda cell, cell_index: (state.update(i=cell_index), print(f"[smoke] cell {cell_index}", flush=True))
    try:
        ep.preprocess(nb, {"metadata": {"path": work}})
        print(f"PASS {a.notebook}  ({time.time() - t:.0f}s, workdir {work})"); return 0
    except Exception as e:
        if not isinstance(e, CellExecutionError):
            print(f"FAIL {a.notebook}  cell {state['i']}  ({time.time() - t:.0f}s)  {type(e).__name__}: {e}"); return 1
        idx = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and any(o.output_type == "error" for o in c.get("outputs", []))), "?")
        print(f"FAIL {a.notebook}  cell {idx}  ({time.time() - t:.0f}s)\n{str(e)[-1500:]}"); return 1


if __name__ == "__main__":
    sys.exit(main())
