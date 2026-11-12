"""Runs harvest/ws3_gpu_jobs.sh on one Modal L40S. Spends credits: started only from harvest/modal_launch.py.

The code and the pairs table come from a Modal Volume (uploaded by modal_launch.py), and results stay there until they
are downloaded, so an interrupted job resumes from its saved chunks. Modal may stop a GPU job and rerun it with the same
input; the scripts skip finished work. A failed job is not retried, so a bug cannot keep spending credits.

HF_TOKEN is sent with each run from the launching environment (Colab Secrets); nothing is stored as a named secret in
the workspace, and no secret already there is attached. Before any model loads, the job checks that this token belongs to
EXPECTED_HF_USER and can read every model, and stops if not. It prints the username and the model names, never the token.
TIMEOUT_HOURS is a hard stop on one attempt.
"""
import os
import subprocess

import modal

APP_NAME = "a5-ws3"
VOLUME_NAME = "a5-ws3"
EXPECTED_HF_USER = "hasifhammani"
HF_REPOS = ("google/gemma-2-9b-it", "mistralai/Mistral-7B-Instruct-v0.3", "allenai/wildguard")
GPU = "L40S"
TIMEOUT_HOURS = 8   # about $20 at $1.95/h GPU + 4 cores + 32 GiB
MOUNT = "/vol"
WORK = f"{MOUNT}/a5-work"
RUNNER = f"{WORK}/a5-adversarial-internals/harvest/ws3_gpu_jobs.sh"
JOBS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "minimum", "recommended", "bf16")
# accelerate: ws3_harvest.py and ws3_fill_lost.py load models with device_map, which transformers refuses without it
# (job 1 failed on 10 Nov at the Gemma C1 step)
PACKAGES = ["vllm", "transformers", "accelerate", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib", "safetensors",
            "huggingface_hub"]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PACKAGES)
         .env({"A5_WORK": WORK, "HF_HOME": f"{MOUNT}/hf-cache", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
               # vLLM's FlashInfer sampler compiles CUDA code at startup and this image has no nvcc (job 1 failed on
               # 10 Nov); greedy decoding does not need it, so vLLM's own PyTorch sampler is used instead
               "VLLM_USE_FLASHINFER_SAMPLER": "0",
               "VLLM_LOGGING_LEVEL": "WARNING"}))   # fewer log lines streamed into Colab


def check_hf_access(api, expected=EXPECTED_HF_USER, repos=HF_REPOS):
    """Stop unless the token belongs to the expected account and can read every model. Library errors are not repeated,
    only their type, so a token can never reach the logs through an error message."""
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise RuntimeError(f"STOP: Hugging Face did not accept HF_TOKEN ({type(e).__name__})") from None
    if user != expected:
        raise RuntimeError(f"STOP: HF_TOKEN belongs to {user}, not {expected}")
    blocked = []
    for repo in repos:
        try:
            api.auth_check(repo)
        except Exception as e:
            blocked.append(f"{repo} ({type(e).__name__})")
    if blocked:
        raise RuntimeError(f"STOP: {user}'s token cannot read: {', '.join(blocked)}")
    print(f"Hugging Face account: {user} | can read {len(repos)} models: {', '.join(repos)}")
    return user


@app.function(image=image, gpu=GPU, cpu=4.0, memory=32768, volumes={MOUNT: volume},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN")})],
              timeout=TIMEOUT_HOURS * 3600, scaledown_window=2, retries=0)
def run_job(job: str):
    """One job or job group of harvest/ws3_gpu_jobs.sh."""
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; choose from {JOBS}")
    from huggingface_hub import HfApi
    check_hf_access(HfApi())
    try:
        subprocess.run(["bash", RUNNER, job], check=True)
    finally:
        volume.commit()
