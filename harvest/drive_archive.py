
# SPDX-License-Identifier: Apache-2.0
"""Archive and restore Project A-5 harvest shards using indexed pixz tarballs."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


# Adapted from Appendix A to your actual project Drive folder.
DRIVE_ROOT = Path("/content/gdrive/MyDrive/MYSSP/a5-artifacts/archives")
SCRATCH = Path("/content/archives")


def setup(mount=True):
    """Install pixz, mount Drive if requested, and prepare directories."""
    if shutil.which("pixz") is None:
        subprocess.run(
            ["apt-get", "-qq", "update"],
            check=True
        )
        subprocess.run(
            ["apt-get", "-qq", "install", "-y", "pixz"],
            check=True
        )

    if mount:
        from google.colab import drive
        drive.mount("/content/drive")

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    print("Archive system ready.")
    print("Drive archive root:", DRIVE_ROOT)
    print("Local scratch:", SCRATCH)


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)

    return h.hexdigest()


def archive_name(model, dataset, config, spec_hash):
    """Generate a unique archive name tied to the extraction spec."""
    return (
        f"{model}__{dataset}__{config}__"
        f"{spec_hash[:8]}.tpxz"
    )


def already_archived(name):
    """True only when both archive and completion manifest exist."""
    return (
        (DRIVE_ROOT / name).is_file()
        and (DRIVE_ROOT / f"{name}.json").is_file()
    )


def archive_to_drive(src_dir, name, level=1, keep_local=False):
    """Create, verify and upload an indexed pixz archive."""

    src = Path(src_dir).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    if already_archived(name):
        print(f"Skip: {name} is already archived.")
        return DRIVE_ROOT / name

    files = sorted(p for p in src.rglob("*") if p.is_file())

    if not files:
        raise ValueError(f"Refusing to archive empty directory: {src}")

    raw_bytes = sum(p.stat().st_size for p in files)
    local = SCRATCH / name
    partial = DRIVE_ROOT / f"{name}.partial"
    t0 = time.time()

    try:
        # Create one indexed tar archive compressed with pixz.
        subprocess.run(
            [
                "tar",
                "-C", str(src.parent),
                f"--use-compress-program=pixz -{level}",
                "-cf", str(local),
                src.name,
            ],
            check=True,
        )

        if not local.is_file() or local.stat().st_size == 0:
            raise RuntimeError("Archive creation failed or produced an empty file.")

        # Verify that the pixz index contains every source file.
        listed = subprocess.run(
            ["pixz", "-l", str(local)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()

        n_listed = sum(
            1 for item in listed
            if item and not item.endswith("/")
        )

        if n_listed != len(files):
            raise RuntimeError(
                f"Archive index lists {n_listed} files; "
                f"expected {len(files)}."
            )

        sha = _sha256(local)

        # Copy to Drive under a temporary name.
        shutil.copyfile(local, partial)

        # Verify the copied archive before making it final.
        if _sha256(partial) != sha:
            raise RuntimeError("SHA-256 mismatch after copying to Drive.")

        os.replace(partial, DRIVE_ROOT / name)

        manifest = {
            "name": name,
            "source_dir": str(src),
            "n_files": len(files),
            "raw_bytes": raw_bytes,
            "archive_bytes": local.stat().st_size,
            "sha256": sha,
            "pixz_level": level,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Manifest is written last: its presence marks completion.
        manifest_tmp = DRIVE_ROOT / f"{name}.json.partial"
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(manifest_tmp, DRIVE_ROOT / f"{name}.json")

        if not keep_local:
            local.unlink(missing_ok=True)

        print(
            f"Archived {name}: {len(files)} files, "
            f"{raw_bytes / 1e9:.3f} GB -> "
            f"{manifest['archive_bytes'] / 1e9:.3f} GB "
            f"in {time.time() - t0:.1f} seconds."
        )

        return DRIVE_ROOT / name

    finally:
        partial.unlink(missing_ok=True)


def restore_from_drive(name, dest_dir, verify=True):
    """Restore an archive locally, checking its SHA-256 manifest."""

    archive = DRIVE_ROOT / name
    manifest_path = DRIVE_ROOT / f"{name}.json"

    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Archive or completion manifest missing for {name}."
        )

    local = SCRATCH / name
    dest = Path(dest_dir)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(archive, local)

    try:
        if verify:
            manifest = json.loads(manifest_path.read_text())
            expected = manifest["sha256"]

            if _sha256(local) != expected:
                raise RuntimeError(
                    f"{name}: SHA-256 mismatch. "
                    "The restored archive may be corrupted."
                )

        subprocess.run(
            [
                "tar",
                "-C", str(dest),
                "--use-compress-program=pixz",
                "-xf", str(local),
            ],
            check=True,
        )

    finally:
        local.unlink(missing_ok=True)

    print(f"Restored {name} into {dest}")
    return dest


def extract_one(name, member, dest_dir):
    """Extract one indexed archive member without unpacking everything."""

    archive = DRIVE_ROOT / name
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not archive.is_file():
        raise FileNotFoundError(archive)

    with open(archive, "rb") as src:
        pixz = subprocess.Popen(
            ["pixz", "-x", member],
            stdin=src,
            stdout=subprocess.PIPE,
        )

        try:
            subprocess.run(
                ["tar", "-x", "-C", str(dest)],
                stdin=pixz.stdout,
                check=True,
            )
        finally:
            if pixz.stdout:
                pixz.stdout.close()

        if pixz.wait() != 0:
            raise RuntimeError(
                f"pixz could not extract archive member: {member}"
            )

    return dest
