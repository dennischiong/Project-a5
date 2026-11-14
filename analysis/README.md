# analysis/ — WS2 numerical analyses

Scripts for the Track 1 numerical analyses owned by Workstream 2. Each one is
written to be pasted into a Colab cell or run with `python`, and each resolves
the project root from the mounted Drive rather than taking a path argument, so
the same file works in both places.

Every script writes its outputs to Drive and records, alongside the numbers, the
configuration that produced them — model, backend, precision, quantization, GPU,
Undercurrent version and spec hash — as §13 of the project plan requires.

| Script | Analysis | Hardware | Runtime |
| --- | --- | --- | --- |
| `a9_noise_floor.py` | A9 numerical noise floor: run-to-run, fp32 vs bf16 | any CUDA GPU | 8–12 min |
| `nf4_smoke_test.py` | Week 3 bitsandbytes smoke test via Undercurrent, plus the NF4-vs-bf16 displacement | any CUDA GPU | ~4 min |
| `a1_a6_displacement.py` | A1 displacement per tactic, A6 layer sweep | CPU only | 6–10 min |

## A9 — `a9_noise_floor.py`

Measures how far an activation moves when the prompt does not change and only
the numerics do. Two legs on the Hugging Face backend: the same bf16 run twice,
and fp32 against bf16, over 32 prompts drawn from the pairs table by a salted
hash of `prompt_id`.

It uses `harvest/collector_fullprec.py` rather than the production collector,
because the production collector casts to bf16 on capture and so cannot see a
precision difference by construction.

Outputs to `harvest/qa/a9/`: `a9_per_activation.csv`, `a9_per_layer.csv`,
`a9_summary.csv`, `a9_meta.json`.

Two legs of A9 are **not** measurable on a Tesla T4 and are not attempted here.
A Hugging Face against vLLM comparison needs native bfloat16, which sm_75 does
not have — vLLM silently casts bf16 to fp16 on that card, so the comparison
would confound backend with precision. Concurrency 1 against 64 is vLLM-only and
blocked the same way.

## NF4 smoke test — `nf4_smoke_test.py`

Answers the Week 3 question that the earlier notebook did not: whether
Undercurrent's hooks fire, and return finite correctly-shaped tensors, when the
weights underneath them are 4-bit. Loading a quantized model through plain
transformers and reading `hidden_states` is a different code path and does not
test this.

Also measures NF4 against bf16 on the same prompts, which is the quantization
displacement figure Track 3 needs.

Prompts over 1,000 tokens are excluded. NF4 prefill on a T4 exceeded 13 minutes
on a 3,257-token prompt and was interrupted, while the same prompt completes in
seconds in bf16; the exclusion and its reason are recorded in the output JSON.

Outputs: `harvest/qa/undercurrent_nf4_smoke.json`,
`harvest/qa/a9/nf4_vs_bf16_per_activation.csv`.

## A1 and A6 — `a1_a6_displacement.py`

Computes δ = h_adv − h_van per pair from a cached multilayer activation file,
reports its size per tactic against the paraphrase, length-filler and numerical
floors, and sweeps the same measurement across layers. CPU only — it reads
cached activations rather than running a model.

Outputs to `harvest/results/`: `a1_per_pair_qwen25-7b.csv`,
`a1_summary_by_tactic_qwen25-7b.csv`, `a6_layer_sweep_qwen25-7b.csv`,
`a1_a6_meta.json`.

### Two things a reviewer should check

**The degenerate controls.** Twenty-one of the 56 source/tactic groups are
unmodified controls whose two halves are the same text. All 1,403 of those pairs
return a displacement of exactly zero. That is intended, it is the cleanest
available check that the pairing and the metric are exact, and those groups are
excluded from the figures.

**The deviations.** The activations this was run on are Qwen2.5-7B in 4-bit NF4,
harvested with plain HF hooks rather than Undercurrent, on a T4 — not the
Llama-3.1-8B bf16 reference that WS2 owns. All four deviations are listed in
`a1_a6_meta.json` under `deviations_to_state_in_the_paper`, and they appear in
the paper rather than only here.

## Conventions

No prompt text and no model replies appear in any output. The index and result
files carry `prompt_id`, `pair_id`, `source`, `tactic` and `split` only, so raw
text stays with its original dataset under that dataset's own terms.
