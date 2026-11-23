# WS2 progress report - 2026-11-14 14:21 UTC

Owner: Kayson Goh (WS2). Mentor: Alizishaan Khatri.

## Four corrections (status)
- Undercurrent (not HF hooks): done, harvest used undercurrent==0.1.0
- Llama-3.1-8B-Instruct (not Qwen) for A1/A6/A9: done
- bf16 weights (not NF4): done
- A6 layer sweep with position split: done

## Harvest
- 18,500 of 18,500 prompts harvested across 37 shards
- 16,515 complete vanilla/adversarial pairs
- provenance confirmed from shard index: {'model_id': ['meta-llama/Llama-3.1-8B-Instruct'], 'undercurrent_version': ['0.1.0'], 'inference_dtype': ['torch.bfloat16'], 'storage_dtype': ['torch.bfloat16'], 'quantization': ['none'], 'backend': ['hf'], 'extraction_point': ['last_instruction', 'last_prefill']}

## A9 precision floor
- A9 noise floor on Llama: not yet run.

## Still pending (not in this push)
- Judges (WildGuard / Llama Guard 3) + Cohen's kappa
- A4 from judge labels
- Track 3 decode configs, X4, X6
- Replication across Qwen2.5-7B / Gemma-2-9B / Mistral
- Methods and numerical-results rewrite
- Release package and licence table (mentor's decision)

No raw prompts, model replies, or activation files are included in this commit. Results are the CSV/JSON summaries already on the project's Google Drive, not reproduced verbatim here.