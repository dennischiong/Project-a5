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
tests/             # tests on synthetic data (no model, no GPU, no real prompts)
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

## WS3: algebraic and geometric analyses (Hasif)

A2 (harmfulness direction r̂ and Δp, figure F2), A5 (tactic geometry), A7 (style vs content, H3), A8 (low-rank
structure) and X5 (direction stability under 4-bit weights), on the hidden state at `prompt[-1]` of the last decoder
layer before the final norm. r̂ is fitted on vanilla TRAIN pairs only; splits are 70/15/15 by sha256(pair_id); CIs are
95% bootstrap over pairs (2,000 resamples, seed 0). The hypotheses and decision rules are in `prereg.md` (locked
17 Oct 2026; sha256 in `prereg_lock.json`).

| Path | What it does |
|---|---|
| `analysis/lib.py`, `geometry.py` | Loading, splits, r̂, projections, bootstrap CIs, effect sizes; the A2/A5/A7/A8/X5 maths |
| `analysis/ws3_pilot.py` | `run_all(folder, view)`: every WS3 analysis and figure for one extraction folder |
| `analysis/regenerate_all.py` | Every WS3 figure (including paper figure F2) and `results/*.json` from one command, over all folders under `$A5_WORK_ROOT` |
| `analysis/figures.py`, `plotstyle.py`, `diagrams.py` | Figures (shared style: harmful orange, benign blue, adversarial hatched) |
| `analysis/pilot_*.py`, `harvest_figures.py`, `prereg_tools.py`, `team_inputs.py`, `notebook_maps.py`, `contact_sheet.py` | Week 3 pilot (A1-A3), shard checks, pre-registration lock, Colab helpers |
| `harvest/ws3_harvest.py` | Gemma-2-9B / Mistral-7B / Qwen prompt-position harvest (HF, batch 1; `--config c1` bf16 or `c6` NF4) |
| `harvest/ws3_pilot_extract.py`, `pilot_extract.py`, `token_check.py` | Qwen2.5-1.5B pilots and the chat-template token-position check |
| `harvest/dennis_harvest_convert.py` | Converts Dennis's Qwen2.5-7B `.tpxz` harvest + pair map into the same folder format |
| `tests/` | Tests on synthetic data with planted answers (no model, no GPU, no real prompts) |

Run (Colab, CPU):

```bash
pip install -r requirements.txt
python -m pytest tests -q
A5_WORK_ROOT=/content/drive/MyDrive/a5-work python analysis/regenerate_all.py
```

Settings: `A5_WORK_ROOT` = the Drive folder holding `ws3_pilot/` and `ws3_harvest/` (default
`/content/drive/MyDrive/a5-work`); `A5_TEST_FOLDER` = a private folder for archive tests (default
`/content/drive/MyDrive/a5-test`). The pairs table goes in `data/` and is git-ignored. Nothing in WS3 prints, plots or
saves prompt text: outputs are ids, counts and numbers only.
