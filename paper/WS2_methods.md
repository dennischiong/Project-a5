# Methods (WS2: harvest and numerical analyses)

This section covers the harvest of Llama-3.1-8B-Instruct, the judges, the noise floor, the numerical analyses A1, A4, A6, A9, X4 and X6, and the replication of A1, A4 and A6 on Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3.

## 1. Data

Each experimental unit is a pair: a vanilla prompt and an adversarial rewrite of the same request. Pairs come from WildJailbreak, SORRY-Bench, StrongREJECT and JailbreakBench. Three kinds of control pair are added: paraphrases (same meaning, different words), length-matched filler (the vanilla prompt padded to the adversarial prompt's length) and XSTest safe prompts. Each pair has a tactic, an intent (harmful or benign) and a split. The Llama and Qwen harvests contain 18,500 unique prompts and 16,515 complete pairs. Four SORRY-Bench rows with no adversarial version are excluded.

*(Pairs-table construction, controls and splits: WS1 section, to be inserted here.)*

## 2. Models

| Model | Hugging Face ID | Layers | Hidden size | Role |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4,096 | Primary model |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3,584 | Replication |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4,096 | Replication |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 28 | 1,536 | Pilot only |

Gemma-2-9B-it was harvested by WS3 but is not analysed in this section.

## 3. Harvest

Activations were extracted with Undercurrent 0.1.0 on the Hugging Face backend: bf16 weights, batch size 1, the model's own chat template. Before any GPU run, the rendered template was checked token by token to catch a duplicated beginning-of-sequence token. No prompt was truncated.

The stored quantity is the residual stream at the output of each decoder block, before the final norm. Two positions are kept:

- **Last prefill token:** the final prompt token, all 32 layers for Llama.
- **Last instruction token:** the last token of the user's text, 16 odd layers for Llama. It sits at prompt[-6] for Llama and Qwen and at prompt[-2] for Mistral. Each offset was confirmed by decoding the token at that index.

A separate decode harvest stored the first 20 generated tokens at layers 8, 16, 24 and 31 for a 500-prompt subset.

**Storage.** Prompts were harvested in 37 shards of 500. Each shard was checked for NaN and infinite values, packed into a pixz tar, verified by file count and SHA-256, and copied to Google Drive. A run that restarts skips shards whose archive already verifies. Prompt text and replies are never stored with activations.

**Compute.** The plan assumed four Colab L4 GPUs. The Colab tier available only had T4s, which do not have enough memory for full-scale bf16 harvests of an 8B model. All Llama harvesting, reply generation, judging and analysis therefore ran on Modal, on NVIDIA A100-40GB GPUs. One L4 was used to test the vLLM backend.

**Code:** `harvest/a5_modal_harvest_v2.py` (harvest), `harvest/specs/llama31-8b-decode.yaml` (decode spec), `harvest/a5_modal_qa.py` (token counts, truncation, NaN/inf checks).

## 4. Serving configurations (Track 3)

Each configuration changes one factor relative to the reference C1 (HF backend, bf16, batch 1):

| Config | Change from C1 | Run? |
|---|---|---|
| C2 | vLLM backend, concurrency 1 | Not run |
| C3 | fp16 compute | Run |
| C5 | 8-bit weights (LLM.int8) | Not run |
| C6 | 4-bit NF4 weights (bitsandbytes) | Run |
| C7 | 4-bit AWQ checkpoint | Not run |

C3 and C6 were run on a 500-prompt subset (the plan specified 4,000), including the decode positions. Every result states the configuration it was measured under.

**Code:** `harvest/a5_modal_pack2.py`.

## 5. Judges

Llama produced greedy replies of up to 256 tokens for a 2,742-prompt judge set. WildGuard (`allenai/wildguard`) labelled every reply for prompt harm, refusal and response harm. Llama Guard 3 (`meta-llama/Llama-Guard-3-8B`) labelled a fixed 2,000-reply subset, chosen by hash, as safe or unsafe. Agreement between WildGuard's harmful-response label and Llama Guard 3's unsafe label is reported as Cohen's κ.

**Code:** `harvest/a5_modal_replies.py`, `harvest/judge.py`.

## 6. Analyses

**Statistics (all analyses).** The point estimate is the median. Intervals are 95% percentile bootstrap intervals with 2,000 resamples and seed 0, resampled over pairs, or over prompts for decode positions. Proportions near 0 or 1 use exact Clopper-Pearson intervals. AUROC is computed from ranks. Logistic fits use standardised predictors, so coefficients are in log-odds per standard deviation.

**A9: numerical noise floor.** 32 prompts were run three times: twice in bf16 on identical input, and once in fp32. The floor is the 95th percentile of relative ℓ2 between bf16 and fp32 over all 1,536 activations. It is the denominator of the signal-to-noise ratio (SNR) in A1 and A6.
**Code:** `analysis/a5_modal_a9_llama.py`.

**A1: displacement per tactic.** For each pair, δ = h_adv − h_van at the last prefill token of the last layer, reported as:
- relative ℓ2 = ‖δ‖ / ‖h_van‖
- cosine distance between h_van and h_adv

Each tactic group is compared with the paraphrase baseline, the length-filler baseline and the A9 floor. Groups whose vanilla and adversarial text are identical have zero displacement by construction and are listed separately. F1 plots the per-pair distributions with the three baselines drawn as reference lines.
**Code:** `analysis/a5_modal_analysis.py`.

**A6: layer sweep and position split.** The A1 measurement is repeated at every stored layer and at both positions. At each layer that has both positions, the confidence intervals of the two positions are compared. The plan defines A6 as a sweep of the A2/A3 separation scores (WS1 and WS3 analyses); this section reports the displacement version.
**Code:** `analysis/a5_modal_analysis.py`.

**A4: link to jailbreak success.** Steps:
1. The harmfulness direction r̂ is the normalised difference between the mean harmful and mean benign vanilla activations. It is fitted on vanilla prompts from the training split only.
2. For each harmful pair, Δp = δ · r̂.
3. First-token refusal mass is the probability the model starts its reply with "I" or "Sorry". It is computed offline by applying the model's final norm and output head to the stored activation.
4. A jailbreak counts as successful when WildGuard labels the adversarial reply as a harmful response. Prompts flagged as possibly not understood (for example heavy encodings) are excluded from the success comparison.

Δp and refusal mass are compared between successful and failed jailbreaks using median differences, AUROC and a joint logistic fit. F4 plots Δp against refusal mass, marked by outcome. As a check, the offline first-token prediction is compared with the first token of each stored reply.
**Code:** `analysis/a5_modal_a4.py`.

**X4: drift under serving changes.** For C3 and C6, activations at layer 31, last prefill token, are compared with C1 for the same prompts. Drift is reported as relative ℓ2, cosine distance and linear CKA, and as a fraction of the A1 headline displacement.
**Code:** `analysis/a5_modal_x4.py`.

**X6: decode-position probe flips.** On the decode subset (369 training prompts, 131 held out):
1. A logistic probe for harmful versus benign intent is trained on C1 activations. Its layer is chosen by validation AUROC.
2. The probe is applied unchanged to C3 and C6 activations of the same prompts.
3. A flip is a changed probe decision at a generated position.

The generated token at each position is recovered offline with the final norm and output head. Rows are split into same-token (the configuration still produces the C1 token) and diverged, because a diverged row has a different input.
**Code:** `analysis/a5_modal_x6.py`, `analysis/x6_followup.py`.

## 7. Replication on Qwen and Mistral

A1, A4 and A6 use the same code logic and statistics on the other models. The measurements are taken at each model's last layer (27 for Qwen, 31 for Mistral). Neither replication model has an A9 floor, so no SNR is reported for them. The floor measured on the 1.5B pilot model is not substituted.

**Qwen2.5-7B-Instruct.** WS1 harvested Qwen (Undercurrent 0.1.0, HF backend, bf16, batch 1), with the same prompts and pairs as Llama. The A6 sweep covers all 28 layers at the last prefill token and the 14 odd layers at the last instruction token. A4 was not run on Qwen.
**Code:** `analysis/a5_modal_qwen_analysis.py`.

**Mistral-7B-Instruct-v0.3.** WS3 harvested Mistral: HF backend, bf16, batch 1, on an NVIDIA L40S. The harvest has 13,229 prompts (SORRY-Bench and WildJailbreak only) and 10,796 pairs. It stores layer 31 at the last prefill token only and has no paraphrase or length controls, so A1 is reported without baselines. WS3 also generated replies and WildGuard labels for a 4,000-prompt subset with vLLM in bf16. The A4 activations (HF) and labels (vLLM replies) therefore come from different backends; the first-token check above measures how far they agree.

**Mistral A6 re-capture.** The stored harvest has one layer only, so A6 needs a new forward pass:
1. Sample 30 pairs per tactic group (21 groups, seed 0).
2. Run each prompt once with Hugging Face transformers (bf16, batch 1) on a Colab GPU. Forward hooks save the output of all 32 decoder blocks at the last prefill token and the last instruction token (prompt[-2]).
3. Skip prompts longer than 1,536 tokens to fit GPU memory, and drop a pair if either of its prompts was skipped.
4. Check the re-captured layer-31 vector against the stored harvest vector for the same prompt. The run stops if the mean cosine similarity is below 0.99.
5. Compute the sweep and position split as for Llama.

**Code:** `analysis/mistral_m0_setup.py` (load and check the WS3 files), `analysis/mistral_m1_a1.py` (A1), `analysis/mistral_m2_a4.py` (A4), `analysis/mistral_m3_a6.py` (A6 re-capture and analysis).

*(Representation analyses A2, A3, A5, A7, A8 and guard probes: WS1 and WS3 sections, to be inserted here.)*

## 8. Deviations from the plan

- Llama runs used Modal A100s (and one L4 test) instead of four Colab L4s. The harvest code is unchanged. The Mistral A6 re-capture ran on a Colab GPU.
- The reference for X4 and X6 is C1 (HF, batch 1), not R (vLLM, concurrency 64), because no vLLM configuration was run.
- C3 and C6 were run on 500 prompts, not 4,000. C2, C5 and C7 were not run.
- The A9 floors for HF vs vLLM and concurrency 1 vs 64 were not measured. Only bf16 vs fp32 on Llama was measured.
- A6 is a displacement sweep, not the planned sweep of A2/A3 separation.
- A4 was run on Llama and Mistral only. On Mistral, activations and replies come from different backends.
- X6 reports generated positions 0, 4 and 9. Position 19 is absent from the decode output; the cause has not been checked.
- Llama Guard 3 labels exist only for Llama replies.
