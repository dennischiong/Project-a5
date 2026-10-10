# Project A-5: Adversarial prompts, LLM internals and activation guard probes

MYSSP Project A-5 (Program Weeks 3–8, 5 Oct – 15 Nov 2026).

Working paper title: *Twins apart: how adversarial rewrites move LLM internal states,
and what that means for activation guard probes.*

- **Track 1** – algebraic, numerical and geometric drift of adversarial rewrites (A1–A9)
- **Track 2** – activation guard probes for jailbreak detection (G1–G4)
- **Track 3** – probe transfer across serving configs: backend, batch, precision, quantization (X1–X6)

**Mentor:** Alizishaan Khatri · **Researchers:** Dennis Chiong (WS1), Kayson Goh (WS2), Hasif Hammani (WS3)

## Layout

```
data/              # pairs tables (parquet) with split columns; raw downloads git-ignored (data/raw/)
harvest/
  specs/           # one Undercurrent YAML spec per model
  configs/         # Track 3 serving configs (backend, dtype, quantization)
  collector.py     # vendored ActivationCollectorProbe (Apache-2.0 header kept)
  run.py           # render template -> ProbedModel.generate -> shards; resumable
  drive_archive.py # pixz archive / restore on Google Drive
  judge.py         # WildGuard / Llama Guard 3 labelling
colab/             # thin launch notebooks, one per researcher's runs
analysis/          # A1-A9 and X4-X6 notebooks; shared utils
probes/            # G1-G4 and X1-X3; baselines
demo/              # inline Undercurrent guard-probe demo
release/           # HF Hub conversion and dataset card
figures/  paper/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Conventions

- Every PR is reviewed by one other researcher (reviewers rotate weekly).
- Seeds are fixed; every figure regenerates from one script.
- Every result states its serving config (backend, concurrency, dtype, quantization, GPU, Undercurrent version).
- **No raw harmful prompt/reply text in git or Drive archives** – activations and ids only.
- Activations never go in git: they live as `.tpxz` archives in the `a5-activations` Drive folder.
