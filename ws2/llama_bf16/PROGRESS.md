# WS2 progress report - 2026-11-14 14:58 UTC

Owner: Kayson Goh (WS2). Mentor: Alizishaan Khatri.

## Progress against the WS2 checklist

Done: harvest (18,500 of 18,500 prompts across 37 shards; archives verified); first A1 on Llama; A6 layer sweep with position split; judges generation (2,742 replies).

Partly done: decode subset and spec built; decode run relaunched after a fix; NaN/inf QA not yet run.

Not started: judge classifiers and agreement; A4; Track 3 configs C2 to C7; X4 and X6; replication across models; release package.

A9 precision floor status needs verification: the checklist says it is done, but the key numbers section says it has not yet been run.

## Harvest
- 18,500 of 18,500 prompts harvested across 37 shards
- 16,515 complete vanilla/adversarial pairs
- Provenance confirmed from shard index: {'model_id': ['meta-llama/Llama-3.1-8B-Instruct'], 'undercurrent_version': ['0.1.0'], 'inference_dtype': ['torch.bfloat16'], 'storage_dtype': ['torch.bfloat16'], 'quantization': ['none'], 'backend': ['hf'], 'extraction_point': ['last_instruction', 'last_prefill']}

No raw prompts, model replies, or activation files are included. Results are on the project Google Drive, under harvest/results.
