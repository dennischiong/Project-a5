"""Tests for the pure helpers of harvest/ws3_harvest.py (no model, no GPU, no real prompts)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harvest import ws3_harvest as wh  # noqa: E402


def fake_plan(n_wj=40, n_sorry=200):
    """A plan in the prompt_plan shape. SORRY-Bench mutations share base instructions (10 bases), as in v1.1."""
    rows = []
    for i in range(n_wj):
        rows.append(dict(pair_id=f"wj{i}", source="WildJailbreak", intent="harmful" if i % 2 else "benign",
                         tactic="unknown", split="train", holdout_group="", key_v=f"v{i}", key_a=f"a{i}"))
    for i in range(n_sorry):
        rows.append(dict(pair_id=f"s{i}", source="SORRY-Bench", intent="harmful", tactic=f"t{i % 20}",
                         split="test", holdout_group="", key_v=f"base{i % 10}", key_a=f"m{i}"))
    return pd.DataFrame(rows)


def n_unique_prompts(plan):
    return len(set(plan["key_v"]) | set(plan["key_a"]))


def test_sweep_layers_stay_below_the_last_layer():
    assert wh.sweep_layers(42, 8) == [0, 8, 16, 24, 32, 40]      # Gemma-2-9B
    assert wh.sweep_layers(32, 8) == [0, 8, 16, 24]              # Mistral-7B
    assert all(l < 28 for l in wh.sweep_layers(28, 4))


def test_subset_reaches_the_target_with_whole_pairs_and_every_stratum():
    plan = fake_plan()
    sub = wh.subset_plan(plan, 100)
    assert n_unique_prompts(sub) >= 100
    assert n_unique_prompts(sub) <= 100 + 1          # stops as soon as the target is reached (a pair adds <= 2)
    assert sub["pair_id"].is_unique
    strata = set(map(tuple, sub[["source", "intent"]].drop_duplicates().to_numpy()))
    assert strata == {("WildJailbreak", "benign"), ("WildJailbreak", "harmful"), ("SORRY-Bench", "harmful")}
    # the biggest stratum gets the most pairs (proportional allocation)
    counts = sub.groupby(["source", "intent"]).size()
    assert counts[("SORRY-Bench", "harmful")] > counts[("WildJailbreak", "benign")]


def test_subset_is_deterministic_and_independent_of_row_order():
    plan = fake_plan()
    a = wh.subset_plan(plan, 120)
    b = wh.subset_plan(plan.sample(frac=1, random_state=3), 120)
    assert sorted(a["pair_id"]) == sorted(b["pair_id"])


def test_subset_larger_than_the_table_returns_everything():
    plan = fake_plan(n_wj=4, n_sorry=6)
    assert len(wh.subset_plan(plan, 10_000)) == len(plan)


def test_run_labels_name_the_model_and_config():
    from analysis import ws3_pilot as wp
    assert wp.run_labels({"model": "Qwen/Qwen2.5-1.5B-Instruct"})[:2] == ("qwen25-1p5b", "PILOT")
    assert wp.run_labels(dict(model_tag="mistral-7b", config="c1"))[:2] == ("mistral-7b", "RESULT")
    assert wp.run_labels(dict(model_tag="gemma2-9b", config="c6", config_name="hf-nf4-bs1", subset=4000))[0] == \
        "gemma2-9b [hf-nf4-bs1, sub4000]"
    # the pilot model re-run through the harvest script stays PILOT and names its pairs table
    assert wp.run_labels(dict(model_tag="qwen25-1p5b", config="c1", pairs_table="/x/pairs_table_v1_final-2.csv"))[:2] == \
        ("qwen25-1p5b [pairs_table_v1_final-2]", "PILOT")


def test_prompt_plan_drops_pairs_with_identical_prompts():
    from analysis import ws3_pilot as wp
    rows = [dict(pair_id="s_base", source="SORRY-Bench", intent="harmful", tactic="base", split="train",
                 vanilla_prompt="base x", adversarial_prompt="base x"),
            dict(pair_id="s_slang", source="SORRY-Bench", intent="harmful", tactic="slang", split="train",
                 vanilla_prompt="base x", adversarial_prompt="slang x")]
    prompts, plan = wp.prompt_plan(pd.DataFrame(rows))
    assert list(plan["pair_id"]) == ["s_slang"]          # the base row has no displacement
    assert len(prompts) == 2                              # but its prompt is still the slang pair's vanilla


def test_size_estimate_and_names():
    # 12,576 prompts x 4,096 dims x (2 float32 + 4 float16 sweep layers) = 0.82 GB
    assert abs(wh.size_estimate_gb(12_576, 4096, 4) - 12_576 * 4096 * 16 / 1e9) < 1e-9
    assert wh.out_name("mistral-7b", "c6", 4000, 0) == "mistral-7b__wj-sorry__hf-nf4-bs1__sub4000"
    assert wh.out_name("gemma2-9b", "c1", 0, 20) == "gemma2-9b__wj-sorry__hf-bf16-bs1__smoke20"
    assert wh.out_name("qwen25-1p5b", "c1", 0, 0, "float16") == "qwen25-1p5b__wj-sorry__hf-fp16-bs1"
    assert set(wh.MODELS) >= {"gemma2-9b", "mistral-7b"}
    assert wh.MODELS["gemma2-9b"]["fp16_ok"] is False            # Gemma in fp16 overflows (work plan s.6.6)
