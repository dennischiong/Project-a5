"""drive_archive.py - pack a directory of many small activation files into one
indexed pixz tarball, store it on Google Drive, and restore it in a later
Colab session instead of recomputing.

Usage in Colab:
    from drive_archive import setup, archive_name, already_archived, archive_to_drive, restore_from_drive
    setup()                                                  # once per session
    name = archive_name("llama31-8b", "wildjailbreak", "vllm-bf16-c64", spec_hash)
    if not already_archived(name):
        ...harvest into /content/harvest/llama31-8b__wildjailbreak ...
        archive_to_drive("/content/harvest/llama31-8b__wildjailbreak", name)
    # later / on another account with access to the folder:
    restore_from_drive(name, "/content/restore")
    # before the session ends:
    from google.colab import drive; drive.flush_and_unmount()
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/a5-activations")  # or a Shared Drive path
SCRATCH = Path("/content/archives")                         # local disk: much faster than Drive


def setup(mount=True):
    """Install pixz and mount Google Drive. Run once per Colab session."""
    if shutil.which("pixz") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y pixz", shell=True, check=True)
    if mount:
        from google.colab import drive
        drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def archive_name(model, dataset, config, spec_hash):
    """One archive per (model, dataset, serving config, spec); the spec hash
    means a changed spec never silently reuses old activations."""
    return f"{model}__{dataset}__{config}__{spec_hash[:8]}.tpxz"


def already_archived(name):
    """True only if the upload finished: the manifest is written last."""
    return (DRIVE_ROOT / name).exists() and (DRIVE_ROOT / f"{name}.json").exists()


def archive_to_drive(src_dir, name, level=1, keep_local=False):
    """tar + pixz (parallel xz on all cores, indexed) on local disk, verify,
    then copy to Drive with a sha256 manifest. Level 1 is the default: on
    bf16 data it is about 4x faster than level 6 for ~4% larger output."""
    src = Path(src_dir).resolve()
    if already_archived(name):
        print(f"skip: {name} is already on Drive")
        return DRIVE_ROOT / name
    files = [p for p in src.rglob("*") if p.is_file()]
    raw_bytes = sum(p.stat().st_size for p in files)
    local = SCRATCH / name
    t0 = time.time()

    # 1. One tar stream compressed by pixz. Many tiny files become one large
    #    file, which Drive handles far better than thousands of small ones.
    subprocess.run(["tar", "-C", str(src.parent), f"--use-compress-program=pixz -{level}",
                    "-cf", str(local), src.name], check=True)

    # 2. Verify before uploading: the index must list every file.
    listed = subprocess.run(["pixz", "-l", str(local)], check=True,
                            capture_output=True, text=True).stdout.split()
    n_listed = sum(1 for p in listed if not p.endswith("/"))
    if n_listed != len(files):
        raise RuntimeError(f"archive lists {n_listed} files, expected {len(files)}")
    sha = _sha256(local)

    # 3. Copy to Drive under a temporary name, then rename, then write the
    #    manifest; a half-finished upload is never mistaken for a complete one.
    tmp = DRIVE_ROOT / f"{name}.partial"
    shutil.copyfile(local, tmp)
    os.replace(tmp, DRIVE_ROOT / name)
    manifest = {
        "name": name, "source_dir": str(src), "n_files": len(files),
        "raw_bytes": raw_bytes, "archive_bytes": local.stat().st_size,
        "sha256": sha, "pixz_level": level,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (DRIVE_ROOT / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    if not keep_local:
        local.unlink()
    print(f"{name}: {len(files)} files, {raw_bytes / 1e9:.2f} GB -> "
          f"{manifest['archive_bytes'] / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    return DRIVE_ROOT / name


def restore_from_drive(name, dest_dir, verify=True):
    """Copy an archive from Drive to local disk, check its hash, unpack it."""
    local = SCRATCH / name
    shutil.copyfile(DRIVE_ROOT / name, local)
    if verify:
        expected = json.loads((DRIVE_ROOT / f"{name}.json").read_text())["sha256"]
        if _sha256(local) != expected:
            raise RuntimeError(f"{name}: sha256 mismatch, re-copy from Drive")
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-C", str(dest_dir), "--use-compress-program=pixz",
                    "-xf", str(local)], check=True)
    local.unlink()
    return Path(dest_dir)


def extract_one(name, member, dest_dir):
    """Pull a single file out of an archive without unpacking the rest
    (uses the pixz index). `member` is the path as listed by `pixz -l`."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with open(DRIVE_ROOT / name, "rb") as src:
        pixz = subprocess.Popen(["pixz", "-x", member], stdin=src, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(dest_dir)], stdin=pixz.stdout, check=True)
        pixz.stdout.close()
        if pixz.wait() != 0:
            raise RuntimeError(f"pixz could not extract {member}")
