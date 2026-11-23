"""Colab side of the Modal runs: upload code and data, start jobs, download results, clean up. Needs no GPU.

Uses MODAL_TOKEN_ID, MODAL_TOKEN_SECRET and HF_TOKEN from the environment (Colab Secrets) and never prints them. HF_TOKEN
goes to Modal only with each run; nothing is stored in the workspace.

  python harvest/modal_launch.py check                 # secrets set, Modal login, workspace name; spends nothing
  python harvest/modal_launch.py upload                # code, pairs table, Mistral C1 folder -> Volume
  python harvest/modal_launch.py run 1                 # SPENDS CREDITS. 1-10, minimum (2-4), recommended (5-6), bf16 (7-8)
  python harvest/modal_launch.py list                  # running and recent apps
  python harvest/modal_launch.py stop <app id>         # stop a run and its charges
  python harvest/modal_launch.py download              # results -> Drive
  python harvest/modal_launch.py cleanup --yes         # deletes harmful text and model weights from the Volume

A run keeps going if Colab disconnects, so stop it with `stop` if it is no longer wanted.
The Gemma C1 and Qwen bf16 folders download to ws3_modal_pending/, NOT ws3_harvest/: moving them changes paper numbers.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_FILE = "harvest/modal_app.py"
VOLUME = "a5-ws3"
TOKEN_NAMES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN")
JOBS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "minimum", "recommended", "bf16")
REMOTE_WORK = "/a5-work"
REMOTE_REPO = f"{REMOTE_WORK}/a5-adversarial-internals"
PAIRS = "data/pairs_table_v1_final-2.csv"
MISTRAL_C1 = "mistral-7b__wj-sorry__hf-bf16-bs1__sub4000"
GEMMA_C1 = "gemma2-9b__wj-sorry__hf-bf16-bs1__sub4000"
QWEN_C1 = ("qwen25-7b__wj-sorry__hf-bf16-bs1", "qwen25-1p5b__wj-sorry__hf-bf16-bs1")
FULL_C1 = ("gemma2-9b__wj-sorry__hf-bf16-bs1", "mistral-7b__wj-sorry__hf-bf16-bs1")   # job 10
CODE_GLOBS = ("analysis/*.py", "harvest/*.py", "harvest/*.sh", "seeds.py", "requirements.txt")
CLEANUP_PATHS = (f"{REMOTE_REPO}/data", "/hf-cache")


# ------------------------------------------------------------------------------------------------ pure helpers
def missing_tokens(env):
    return [n for n in TOKEN_NAMES if not env.get(n)]


def stage_repo(repo, stage):
    """Copy only what the jobs need into stage. Nothing else in the repo or in data/ leaves Drive."""
    repo, stage = Path(repo), Path(stage)
    files = [p for pattern in CODE_GLOBS for p in repo.glob(pattern)] + [repo / PAIRS]
    if not (repo / PAIRS).exists():
        raise FileNotFoundError(f"{PAIRS} is missing")
    for p in files:
        rel = p.relative_to(repo)
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, stage / rel)
    return sorted(str(p.relative_to(repo).as_posix()) for p in files)


def download_plan(repo):
    """(remote path, local parent folder). `modal volume get` writes <parent>/<last part of the remote path>."""
    repo = Path(repo)
    work = repo.parent
    return [(f"{REMOTE_REPO}/data/ws3_replies", repo / "data"),
            (f"{REMOTE_WORK}/ws3_judges", work),
            (f"{REMOTE_WORK}/ws3_oom_fill", work),
            (f"{REMOTE_WORK}/logs", work),
            (f"{REMOTE_WORK}/ws3_harvest/{GEMMA_C1}", work / "ws3_modal_pending")] + [
        (f"{REMOTE_WORK}/ws3_harvest/{q}", work / "ws3_modal_pending") for q in QWEN_C1 + FULL_C1]


def modal_cmd(*args):
    return [sys.executable, "-m", "modal", *args]


def run_command(job):
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; choose from {JOBS}")
    return modal_cmd("run", "--detach", f"{APP_FILE}::run_job", "--job", job)


def call(cmd, check=True, capture=False):
    """Run a Modal CLI command from the repo folder. Tokens stay in the environment, never in the command."""
    return subprocess.run(cmd, cwd=REPO, check=check, text=True, capture_output=capture)


# Modal's progress spinner and model-loading bars repeat many times a second; streamed into a Colab cell over a long run
# they froze the tab for 25 min on 10 Nov. Job lines (=== step, speed, DONE, errors) never match these.
_NOISE = re.compile(r"(Running \(|Loading images|View app at|Creating objects|Creating mount|Capturing CUDA graphs|"
                    r"Loading safetensors|it/s\]|s/it\]|^\s*[⠀-⣿]\s*$|^\s*[│├└])")


def keep_line(line):
    """True for lines worth showing while a job runs."""
    return bool(line.strip()) and not _NOISE.search(line)


def run_filtered(cmd):
    """Run a job and stream only useful lines; the full log is still in ../logs/gpu_*.log on the volume."""
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    hidden = 0
    for raw in proc.stdout:
        for line in raw.replace("\r", "\n").split("\n"):
            if keep_line(line):
                print(line, flush=True)
            elif line.strip():
                hidden += 1
    code = proc.wait()
    print(f"({hidden} progress lines hidden)")
    if code:
        sys.exit(f"STOP: the job ended with exit code {code}; read the lines above")


def require_tokens():
    missing = missing_tokens(os.environ)
    if missing:
        sys.exit(f"STOP: {', '.join(missing)} not set. Add them to Colab Secrets and run the notebook's first cell.")


def workspace_line(token_info_output):
    """Only the workspace line of `modal token info`, so the token id is never shown."""
    lines = [l.strip() for l in token_info_output.splitlines() if l.strip().startswith("Workspace:")]
    return lines[0] if lines else None


# ------------------------------------------------------------------------------------------------ commands
def check():
    require_tokens()
    print("Modal tokens and HF_TOKEN are set (values not shown)")
    info = call(modal_cmd("token", "info"), capture=True, check=False)
    if info.returncode != 0:
        sys.exit("STOP: Modal rejected the login or could not be reached; check the two Modal tokens")
    print(f"Modal login works | {workspace_line(info.stdout) or 'workspace name not shown by this Modal version'}")
    if not (REPO / PAIRS).exists():
        sys.exit(f"STOP: {PAIRS} is missing")
    print("ready: pairs table found. Nothing was spent.")


def upload():
    require_tokens()
    call(modal_cmd("volume", "create", VOLUME), check=False, capture=True)   # fails harmlessly if it already exists
    with tempfile.TemporaryDirectory() as td:
        files = stage_repo(REPO, td)
        call(modal_cmd("volume", "put", VOLUME, td, REMOTE_REPO, "--force"))
    print(f"uploaded {len(files)} files to {REMOTE_REPO}")
    harvest = REPO.parent / "ws3_harvest" / MISTRAL_C1
    if (harvest / "meta.json").exists():
        call(modal_cmd("volume", "put", VOLUME, str(harvest), f"{REMOTE_WORK}/ws3_harvest/{MISTRAL_C1}", "--force"))
        print(f"uploaded {MISTRAL_C1}")
    else:
        print(f"WARNING: {harvest} not found, so job 5 cannot run")
    call(modal_cmd("volume", "ls", VOLUME, f"{REMOTE_REPO}/harvest"))


def already_on_drive(remote, repo):
    """True for a harvest folder that is already finished on Drive (in ws3_harvest/ or ws3_modal_pending/), so a
    download does not fetch it again: the full Qwen folders are 1.3 GB and Drive is nearly full."""
    if "/ws3_harvest/" not in remote:
        return False
    name, work = remote.rsplit("/", 1)[1], Path(repo).parent
    return any((work / sub / name / "meta.json").exists() for sub in ("ws3_harvest", "ws3_modal_pending"))


def download():
    require_tokens()
    failed = []
    for remote, parent in download_plan(REPO):
        if already_on_drive(remote, REPO):
            print(f"already on Drive, skipped: {remote}")
            continue
        Path(parent).mkdir(parents=True, exist_ok=True)
        got = call(modal_cmd("volume", "get", VOLUME, remote, str(parent), "--force"), check=False, capture=True)
        print(f"{'ok' if got.returncode == 0 else 'not downloaded'}: {remote}")
        if got.returncode != 0:
            failed.append(remote)
    print(f"{len(failed)} paths not downloaded; expected only for jobs that have not run: {failed}")


def cleanup(confirmed):
    require_tokens()
    if not confirmed:
        sys.exit("STOP: this permanently deletes data from the Volume; rerun with --yes")
    if not (REPO / "data" / "ws3_replies").exists():
        sys.exit("STOP: no replies on Drive yet. Run download first, or the replies would be lost")
    for path in CLEANUP_PATHS:
        got = call(modal_cmd("volume", "rm", VOLUME, path, "--recursive"), check=False, capture=True)
        print(f"{'deleted' if got.returncode == 0 else 'not deleted (may already be gone)'}: {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("check", "upload", "list", "download"):
        sub.add_parser(name)
    sub.add_parser("run").add_argument("job", choices=JOBS)
    sub.add_parser("stop").add_argument("app_id")
    sub.add_parser("cleanup").add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.command == "check":
        check()
    elif a.command == "upload":
        upload()
    elif a.command == "run":
        require_tokens()
        print(f"starting job {a.job} on an L40S: about $2.40/h until it ends (hard stop after 8 h per attempt)")
        run_filtered(run_command(a.job))
    elif a.command == "list":
        require_tokens()
        call(modal_cmd("app", "list"))
    elif a.command == "stop":
        require_tokens()
        call(modal_cmd("app", "stop", a.app_id, "--yes"))
    elif a.command == "download":
        download()
    elif a.command == "cleanup":
        cleanup(a.yes)


if __name__ == "__main__":
    main()
