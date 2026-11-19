"""Fast repository hygiene checks: sizes, secrets/personal info, notebooks, README links.

Personal strings are NOT hard-coded here. Set PRIVATE_PATTERNS (a GitHub Actions secret in CI) to a list of
case-insensitive regular expressions separated by newlines or commas, e.g.
    PRIVATE_PATTERNS=$'my_handle\nmy@mail\\.com'
When it is unset the personal-info tests are skipped locally and fail in CI on this repository's own runs.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE = 5 * 1024 * 1024
MAX_REPO = 30 * 1024 * 1024
MAX_GIF = int(1.5 * 1024 * 1024)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".ipynb", ".sh", ".cfg", ".ini", ".toml", ".tsv", ".csv"}


def tracked():
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
    return [Path(p) for p in out.split("\0") if p and (ROOT / p).exists()]


def text_files():
    return [p for p in tracked() if p.suffix in TEXT_SUFFIXES or p.name in {".gitignore"}]


def read(p):
    return (ROOT / p).read_text(errors="ignore")


def private_patterns():
    raw = os.environ.get("PRIVATE_PATTERNS", "")
    pats = [x.strip() for x in re.split(r"[\n,]", raw) if x.strip()]
    return [re.compile(x, re.I) for x in pats]


# ---------------------------------------------------------------- sizes
def test_no_tracked_file_over_5mb():
    big = [(str(p), (ROOT / p).stat().st_size) for p in tracked() if (ROOT / p).stat().st_size > MAX_FILE]
    assert not big, f"files over 5 MB: {big}"


def test_repo_under_30mb():
    total = sum((ROOT / p).stat().st_size for p in tracked())
    assert total <= MAX_REPO, f"tracked files total {total / 1e6:.1f} MB (limit 30 MB)"


def test_gifs_under_1_5mb():
    big = [(str(p), round((ROOT / p).stat().st_size / 1e6, 2)) for p in tracked()
           if p.suffix == ".gif" and (ROOT / p).stat().st_size > MAX_GIF]
    assert not big, f"GIFs over 1.5 MB (MB shown): {big}"


def test_gifs_only_in_assets_gifs():
    stray = [str(p) for p in tracked() if p.suffix == ".gif" and p.parent != Path("assets/gifs")]
    assert not stray, f"GIFs outside assets/gifs: {stray}"


# ---------------------------------------------------------------- secrets and personal info
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "wandb API key value": re.compile(r"WANDB_API_KEY[\"']?\s*[=:]\s*[\"']?[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{30,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "home directory path": re.compile(r"/Users/(?!<user>|\[)[A-Za-z][A-Za-z0-9_.-]*"),
}


@pytest.mark.parametrize("name", sorted(SECRET_PATTERNS))
def test_no_secrets(name):
    rx = SECRET_PATTERNS[name]
    hits = []
    for p in text_files():
        for m in rx.finditer(read(p)):
            hits.append(f"{p}: {m.group(0)[:12]}...")
    assert not hits, f"{name} found in: {hits[:10]}"


def _need_patterns():
    pats = private_patterns()
    if not pats:
        if os.environ.get("PRIVATE_PATTERNS_REQUIRED") == "1":
            pytest.fail("PRIVATE_PATTERNS is not set but this run requires it")
        pytest.skip("PRIVATE_PATTERNS not set")
    return pats


def test_no_personal_info_in_files():
    pats = _need_patterns()
    hits = [f"{p}: /{rx.pattern}/" for p in text_files() for rx in pats if rx.search(read(p))]
    assert not hits, f"personal-info patterns found in: {hits[:10]}"


def test_no_personal_info_in_commit_metadata():
    pats = _need_patterns()
    log = subprocess.run(["git", "log", "HEAD", "--format=%an <%ae> | %cn <%ce> | %s%n%b"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    hits = [f"/{rx.pattern}/" for rx in pats if rx.search(log)]
    assert not hits, f"personal-info patterns found in commit author/committer/message: {hits}"


# ---------------------------------------------------------------- notebooks
def notebooks():
    return [p for p in tracked() if p.suffix == ".ipynb"]


@pytest.mark.parametrize("path", notebooks(), ids=str)
def test_notebook_parses(path):
    nb = nbformat.read(ROOT / path, as_version=4)
    nbformat.validate(nb)


# ---------------------------------------------------------------- README links
LINK = re.compile(r"""(?:\]\(\s*<?([^)\s>]+)>?(?:\s+["'][^)]*["'])?\s*\)|(?:src|href)\s*=\s*["']([^"']+)["'])""", re.I)


def readmes():
    return [p for p in tracked() if p.name.lower().startswith("readme") and p.suffix.lower() in {".md", ".markdown"}]


def local_targets(readme):
    fence = re.compile(r"```.*?```", re.S)
    text = fence.sub("", read(readme))
    for m in LINK.finditer(text):
        target = (m.group(1) or m.group(2)).split("#")[0].split("?")[0]
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "//")):
            continue
        yield target


def test_readme_local_paths_exist():
    missing = []
    for r in readmes():
        for t in local_targets(r):
            base = ROOT if t.startswith("/") else (ROOT / r).parent
            if not (base / t.lstrip("/")).exists():
                missing.append(f"{r}: {t}")
    assert not missing, f"README paths that do not exist: {missing}"
