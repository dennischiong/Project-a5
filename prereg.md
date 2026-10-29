# Pre-registration (written BEFORE any pilot analysis; final before any full-harvest result)

Status: FINAL (Thu 15 Oct 2026). Every threshold is the value proposed on 11 Oct, unchanged, plus two items fixed
on 15 Oct (the H1 primary/secondary split and the H3 equivalence margin). All were fixed before any full-harvest
result was seen. Anything not listed here is **exploratory**. Pilot data (~100 and 2,000 pairs, Qwen2.5-1.5B) is for
plumbing only and was not used to tune these criteria.

## Common setup
- Position: prompt[-1] (last prefill token) of the chat-templated prompt (`add_generation_prompt=True`).
- Layer for the representation hypotheses H1, H2, H3, H6: the **last layer**, fixed in advance. The kickoff
  says the headline is the last-layer state, and a weak last-layer signal is still reported as the
  pre-registered result. Norm convention: the last decoder-layer output BEFORE the final norm, matching
  Undercurrent's `residual_stream`, with the post-norm vector reported next to it. Layer sweeps and the
  last-instruction position (A6) are exploratory.
- Layer and position for the probe hypotheses H4, H5, H7: chosen on the **validation** split only (work plan
  section 7.2), by a rule fixed before test is touched: highest vanilla harmful-vs-benign AUROC on validation.
  State pre/post final norm every time.
- r_hat, probes, scalers: fitted on vanilla TRAIN rows only (and, for H4's "augmented" probe, adversarial
  TRAIN rows of the stated tactics). All metrics on validation (model/layer choice) and test (reported).
- Splits 70/15/15 by sha256 of pair_id. CIs: 95% percentile bootstrap over pairs, 2000 resamples, seed 0.
  Effect sizes: Cohen's d (paired/one-sample for per-pair quantities).
- Design 2x2 (harmful/benign x vanilla/adversarial). Every harmful-prompt effect is repeated on the same
  tactic's benign prompts; a harmful effect is only called "specific" if it is absent or clearly smaller
  in benign prompts (difference-in-differences CI excludes 0).
- Rule used throughout: "CI excludes X" means the 95% interval does not contain X.

## Design at a glance
Which cells of the 2x2 design each hypothesis compares (V = vanilla, A = adversarial; H = harmful, B = benign):

| | VH | VB | AH | AB | Other inputs | Position | Layer rule |
|---|---|---|---|---|---|---|---|
| H1 | fit | fit | **test** | control | - | prompt[-1] | last layer |
| H2 | fit | fit | **test** (success vs fail) | control | judge labels | prompt[-1] | last layer |
| H3 | fit | fit | content | **style** | G2 probe | prompt[-1] | last layer |
| H4 | train | train | **test** (LOTO) | **test** (LOTO) | - | chosen on validation | chosen on validation |
| H5 | train | train | **test** | **test** (FPR) | Llama Guard 3 | chosen on validation | chosen on validation |
| H6 | - | - | **pairs** | **pairs** | noise floor | prompt[-1] | last layer |
| H7 | all | all | all | all | serving configs | chosen on validation | chosen on validation |

```
  fit r_hat on VH vs VB (TRAIN)  -->  project VALIDATION/TEST pairs  -->  dp = (h_adv - h_van) . r_hat
           |                                                                   |
           +--> H1 share below midpoint      H2 success vs fail      H3 style (AB) vs content (AH - AB)
```

## Hypotheses
| # | Metric (exact) | Split / comparison | Supported if | Not supported if |
|---|---|---|---|---|
| H1 | Share of adversarial-harmful test prompts whose projection onto r_hat is below the midpoint of the train means of vanilla-harmful and vanilla-benign projections. One primary test (decides H1) and one secondary test (reported, labelled secondary) | Test; vs vanilla-harmful share below the same midpoint | **Primary (majority crosses the midpoint):** share's CI lower bound > 0.5 AND the share exceeds the vanilla-harmful share with CI of the difference excluding 0. **Secondary (shift vs vanilla):** share minus vanilla-harmful share > 0 with CI excluding 0 | **Primary:** CI upper bound <= 0.5, or difference CI includes 0. **Secondary:** difference CI includes 0 or is < 0 |
| H2 | dp = delta . r_hat per harmful pair, split by behaviour label (jailbreak succeeded vs failed; label from Llama Guard 3 / WildGuard on the generated reply) | Test; mean dp(success) - mean dp(fail) | Difference < 0 with CI excluding 0, and the difference-in-differences against the same tactics' benign prompts also has CI excluding 0 | Difference CI includes 0 or is > 0 |
| H3 | A7 (work plan section 7.1). For each tactic t that has both harmful and benign pairs, mean displacements d_harm(t) and d_ben(t). Style part = d_ben(t) . r_hat; content part = (d_harm(t) - d_ben(t)) . r_hat. Plus G2: a probe trained on adversarial vs vanilla form (TRAIN), scored on adversarial-benign vs vanilla-benign; and the vanilla-trained harmfulness probe's score (its predicted probability of "harmful") on adversarial-benign vs vanilla-benign | Test; per tactic and pooled over tactics (bootstrap over pairs within tactic) | (a) Pooled style part differs from 0 (CI excludes 0), so benign rewrites move along r_hat too; AND (b) G2 AUROC >= 0.8; AND (c) harmfulness-probe score difference (adversarial-benign minus vanilla-benign) has CI inside +/- m, where m = 10% of the vanilla gap (0.1 x (mean score on vanilla-harmful test prompts minus mean score on vanilla-benign test prompts)) | (a) style CI includes 0, or (b) G2 AUROC below 0.8, or (c) the score difference CI reaches outside +/- m (the harmfulness probe is fooled by style) |
| H4 | AUROC (harmful vs benign) of a logistic probe trained on vanilla only, evaluated on adversarial test prompts; vs AUROC on vanilla test prompts. Then a probe trained on vanilla + adversarial TRAIN prompts from other tactics, evaluated on a held-out tactic (leave-one-tactic-out) | Test; paired bootstrap over pairs | (a) AUROC(vanilla test) - AUROC(adversarial test) > 0 with CI excluding 0; (b) augmented probe recovers >= 50% of that drop on held-out tactics, with CI of the recovered fraction excluding 0 | (a) fails, or (b) recovered fraction CI includes values below 50% |
| H5 | True-positive rate at 1% false-positive rate (threshold set on validation negatives) on held-out tactics; compute = FLOPs or wall-clock per prompt | Test; best probe vs Llama Guard 3 | Non-inferiority: the CI of (probe TPR - LG3 TPR) lies entirely above -0.05 AND probe compute <= 5% of LG3 | CI lower bound at or below -0.05 (not shown non-inferior), or compute ratio above 5% |
| H6 | ||delta|| (relative L2 and cosine distance) of adversarial pairs vs the numerical noise floor (same prompt, different backend/batch shape/dtype) | Test pairs; ratio of median displacement to median noise floor | Median ratio >= 10 with bootstrap CI lower bound >= 10 | CI lower bound < 10 |
| H7 | For each serving config vs the reference R (work plan section 7.3: backend, batch shape, bf16 vs fp16, 4-bit weights): change in probe AUROC and TPR at 1% FPR, shift in the 1%-FPR threshold, and **per-example agreement** (section 13: robustness needs more than aggregate accuracy): flip rate of verdicts at the reference threshold and Cohen's kappa. Then refit only the standardisation and threshold on ~200 labelled prompts from the target config (X3) | Test subset; each config vs R, paired over the same prompts | (a) 16-bit/backend/batch: AUROC change CI within +/- 0.01 AND flip rate CI upper bound <= 1% AND kappa CI lower bound >= 0.9; (b) 4-bit: threshold shift CI excludes 0; (c) the refit recovers >= 50% of the TPR@1%FPR lost, CI excluding 0 | Any of (a)-(c) fails its stated criterion |

## Notes
- Paraphrase baselines (needed for fair A1 comparisons) do not exist in the pilot; pilot A1 is compared to a rough
  batch-size noise floor only and is labelled pilot-only.
- Multiple comparisons: Holm correction over the seven primary tests H1-H7. The H1 secondary test is reported
  separately with its own 95% CI, labelled secondary, and is not part of the Holm family.
- H1 primary and secondary (fixed Thu 15 Oct). The primary is the original rule, written before any pilot analysis.
  The secondary was added after the Qwen2.5-1.5B pilot A2 (2,000 WildJailbreak pairs, pre-norm, last layer: share
  0.36 vs 0.07 for vanilla-harmful, difference +0.30 [+0.20, +0.39]) showed the rule could be read as two different
  claims. Keeping the original as primary and reporting both means neither is chosen after seeing data.
- H3 equivalence margin (fixed Thu 15 Oct; the 11 Oct message left it open). It is set relative to the vanilla
  harmful-vs-benign score gap, so it does not depend on the model's score scale, and no pilot number was used.
- If anything here is changed later, the change is reported as a deviation.
- Deviations from this document must be listed in the final report as deviations.
