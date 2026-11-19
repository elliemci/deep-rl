#!/usr/bin/env python3
"""Notebook hygiene for the repo.

  strip_notebooks.py strip  [paths...]   strip outputs/metadata in place, then verify no cell source changed
  strip_notebooks.py verify [paths...] [--against REF]
                                         compare cell sources (all cell types) with the notebook at git REF (default HEAD)

What `strip` changes (outputs and metadata only, never cell sources):
  * drops notebook-level `widgets` metadata (it has no `state` key on GitHub and breaks rendering)
  * keeps only `kernelspec`, `language_info`, `accelerator`, and `colab.name/gpuType`
  * clears per-cell Colab metadata (outputId, base_uri, resources, ...)
  * drops stdout/stderr of install cells (pip/apt/git clone/wget logs)
  * drops outputs larger than 200 KB (embedded videos/images) and widget-view mimetypes
  * truncates stream outputs longer than 10 KB to head + tail
  * redacts wandb user names/URLs, /Users/<name> and any names in $REDACT_NAMES in outputs
  * resets execution_count to null
Exit status is non-zero if any cell source differs.
"""
import argparse, copy, json, os, re, subprocess, sys
from pathlib import Path
import nbformat

MAX_OUTPUT = 200_000
MAX_STREAM = 10_000
INSTALL = re.compile(r"(^|\n)\s*(%pip|!pip|!apt|!apt-get|%%capture|!git clone|!wget|!curl|!sudo|!bash|%%bash|!sh )", re.M)
KEEP_COLAB = ("name", "gpuType")


def sources(nb):
    return [(c.cell_type, c.source) for c in nb.cells]


REDACT = [
    (re.compile(r"wandb\.ai/(?!authorize)[A-Za-z0-9_.-]+"), "wandb.ai/<user>"),            # run/project URLs
    (re.compile(r"(logged in as:\s*(?:\x1b\[\d+m)?)[A-Za-z0-9_.@-]+"), r"\1<user>"),    # "Currently logged in as: name"
    (re.compile(r"/Users/[A-Za-z0-9_.-]+"), "/Users/<user>"),
]


# Extra identifiers to redact (user names, handles), comma-separated, e.g. REDACT_NAMES="myname,my_wandb_user"
_NAMES = [n for n in os.environ.get("REDACT_NAMES", "").split(",") if n]
if _NAMES:
    REDACT.append((re.compile("|".join(re.escape(n) for n in _NAMES)), "<user>"))


def redact(text):
    for rx, rep in REDACT:
        text = rx.sub(rep, text)
    return text


def redact_output(o):
    """Redact identifiers inside an output (stream text and every string in data/ename/evalue/traceback)."""
    def walk(x):
        if isinstance(x, str): return redact(x)
        if isinstance(x, list): return [walk(i) for i in x]
        if isinstance(x, dict): return {k: walk(v) for k, v in x.items()}
        return x
    for k in ("text", "data", "evalue", "traceback"):
        if k in o: o[k] = walk(o[k])


def strip_outputs(cell):
    install = bool(INSTALL.search(cell.source))
    kept = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            if install:
                continue
            text = "".join(o.text) if isinstance(o.text, list) else o.text
            if len(text) > MAX_STREAM:
                lines = text.splitlines()
                text = "\n".join(lines[:40] + [f"... [{len(lines) - 80} lines truncated] ..."] + lines[-40:]) + "\n"
            o.text = text
        elif o.output_type in ("display_data", "execute_result"):
            o.get("data", {}).pop("application/vnd.jupyter.widget-view+json", None)
            if not o.get("data"):
                continue
            if len(json.dumps(o)) > MAX_OUTPUT:
                continue
            if o.output_type == "execute_result":
                o.execution_count = None
        redact_output(o)
        kept.append(o)
    cell.outputs = kept
    cell.execution_count = None


def strip_nb(nb):
    nb.metadata.pop("widgets", None)
    colab = nb.metadata.get("colab", {})
    nb.metadata["colab"] = {k: v for k, v in colab.items() if k in KEEP_COLAB}
    if not nb.metadata["colab"]:
        nb.metadata.pop("colab")
    for c in nb.cells:
        c.metadata = {k: v for k, v in c.metadata.items() if k in ("tags",)}
        if c.cell_type == "code":
            strip_outputs(c)


def notebooks(paths):
    out = []
    for p in paths or ["units"]:
        p = Path(p)
        out += [p] if p.suffix == ".ipynb" else sorted(p.rglob("*.ipynb"))
    return [p for p in out if ".ipynb_checkpoints" not in p.parts]


def diff_sources(a, b):
    bad = []
    if len(a) != len(b):
        bad.append(f"cell count {len(a)} -> {len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            bad.append(f"cell {i} ({x[0]}) differs")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["strip", "verify"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--against", default="HEAD")
    args = ap.parse_args()
    failed = False
    for p in notebooks(args.paths):
        if args.mode == "strip":
            before = nbformat.read(p, 4)
            snap = sources(before)
            nb = copy.deepcopy(before)
            strip_nb(nb)
            nbformat.write(nb, p)
            bad = diff_sources(snap, sources(nbformat.read(p, 4)))
        else:
            old = nbformat.reads(subprocess.check_output(["git", "show", f"{args.against}:{p}"], text=True), 4)
            bad = diff_sources(sources(old), sources(nbformat.read(p, 4)))
        print(("FAIL " if bad else "ok   ") + str(p), *bad)
        failed |= bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
