"""Tests for the pure helpers of harvest/ws3_generate.py, ws3_judge.py and ws3_fill_lost.py (no model, no GPU, no real prompts)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import ws3_pilot as wp  # noqa: E402
from harvest import ws3_fill_lost as fl  # noqa: E402
from harvest import ws3_generate as wg  # noqa: E402
from harvest import ws3_harvest as wh  # noqa: E402
from harvest import ws3_judge as wj  # noqa: E402


class FakeTok:
    """Tokenizer stand-in: a token it does not have maps to unk, as real tokenizers do."""
    unk_token_id = 3

    def __init__(self, eos, vocab):
        self.eos_token_id, self.vocab = eos, vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


def fake_pairs():
    rows = []
    for i in range(30):
        rows.append(dict(pair_id=f"wj{i}", source="WildJailbreak", intent="harmful" if i % 2 else "benign", tactic="unknown",
                         split="test", holdout_group="", vanilla_prompt=f"plain {i}", adversarial_prompt=f"rewrite {i}"))
    for i in range(20):
        rows.append(dict(pair_id=f"s{i}", source="SORRY-Bench", intent="harmful", tactic=f"t{i % 4}", split="test",
                         holdout_group="", vanilla_prompt=f"base {i % 5}", adversarial_prompt=f"mutation {i}"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ generation
def test_gemma_end_of_turn_is_a_stop_token_even_though_generation_config_omits_it():
    tok = FakeTok(eos=1, vocab={"<end_of_turn>": 107})
    assert wg.stop_token_ids(tok, generation_eos=1) == [1, 107]


def test_stop_tokens_ignore_tokens_the_model_does_not_have_and_accept_lists():
    tok = FakeTok(eos=2, vocab={})
    assert wg.stop_token_ids(tok, generation_eos=None) == [2]
    assert wg.stop_token_ids(tok, generation_eos=[2, 9]) == [2, 9]


def test_a_prompt_only_fits_with_room_for_a_full_reply():
    assert wg.fits(7936, 256, 8192)
    assert not wg.fits(7937, 256, 8192)


def test_max_model_len_ignores_prompts_that_will_be_skipped_and_never_passes_the_limit():
    assert wg.max_model_len([100, 4000], 256, 8192) == 4256
    assert wg.max_model_len([100, 9000], 256, 8192) == 356
    assert wg.max_model_len([7936], 256, 8192) == 8192
    assert wg.max_model_len([], 256, 8192) == 256


def test_chunk_count_rounds_up():
    assert [wg.n_chunks(n, 500) for n in (1, 500, 501, 4000)] == [1, 1, 2, 8]


def test_folder_name_says_vllm_so_it_cannot_be_mistaken_for_a_harvest():
    assert wg.out_name("gemma2-9b", subset=4000) == "gemma2-9b__wj-sorry__vllm-bf16__sub4000"
    assert wg.out_name("mistral-7b", limit=10) == "mistral-7b__wj-sorry__vllm-bf16__smoke10"


def test_settings_hash_changes_with_settings_but_not_key_order():
    assert wg.settings_hash(dict(a=1, b=2)) == wg.settings_hash(dict(b=2, a=1))
    assert wg.settings_hash(dict(a=1, b=2)) != wg.settings_hash(dict(a=1, b=3))


def test_speed_line_extrapolates_from_prompts_done():
    assert "about 2.00 h" in wg.speed_line(elapsed=36.0, n_prompts=20, n_prompt_tokens=100, n_reply_tokens=100, target=4000)
    assert wg.speed_line(0, 0, 0, 0) == "no prompts generated"


def test_replies_use_the_same_prompts_in_the_same_order_as_the_harvest():
    """Labels join to the activations by prompt_key, so the selection must match harvest/ws3_harvest.py's main()."""
    pairs = fake_pairs()
    prompts, plan = wp.prompt_plan(pairs)
    sub = wh.subset_plan(plan, 40)
    need = set(sub["key_v"]) | set(sub["key_a"])
    harvest_keys = [k for k, _ in prompts if k in need]
    got, got_plan = wg.select_prompts(pairs, subset=40)
    assert [k for k, _ in got] == harvest_keys
    assert got_plan["pair_id"].tolist() == sub["pair_id"].tolist()


def test_limit_takes_the_first_pairs_of_each_source():
    _, plan = wg.select_prompts(fake_pairs(), limit=3)
    assert plan["source"].value_counts().to_dict() == {"WildJailbreak": 3, "SORRY-Bench": 3}


# ------------------------------------------------------------------------------------------------ judge
def test_verdict_from_the_model_card_example():
    text = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
    assert wj.parse_verdict(text) == dict(harmful_request="yes", refusal="yes", harmful_response="no")


def test_verdict_tolerates_case_and_leading_space():
    text = "\n harmful request: NO\n RESPONSE REFUSAL: no \nharmful response: Yes\n"
    assert wj.parse_verdict(text) == dict(harmful_request="no", refusal="no", harmful_response="yes")


def test_incomplete_or_unclear_verdicts_are_not_guessed():
    assert wj.parse_verdict("Harmful request: yes\nResponse refusal: yes") is None
    assert wj.parse_verdict("Harmful request: yes\nResponse refusal: N/A\nHarmful response: no") is None
    assert wj.parse_verdict("") is None


def test_judge_input_keeps_braces_in_prompts_and_follows_the_card_layout():
    text = wj.judge_input("use {this} and {0}", "reply {x}")
    assert "Human user:\nuse {this} and {0}\n\nAI assistant:\nreply {x}\n\n---\n\nAnswers: [/INST]\n<|assistant|>\n" in text
    assert text.startswith("<s><|user|>\n[INST] You are a classifier")


def test_rows_with_nothing_to_judge_get_no_verdict():
    assert wj.pre_status("", "not_generated") == "no_reply"
    assert wj.pre_status("  \n", "stop") == "empty_reply"
    assert wj.pre_status("text", "length") is None


def test_label_row_holds_no_text_and_marks_truncated_replies():
    row = wj.label_row("k1", "ok", dict(harmful_request="yes", refusal="no", harmful_response="yes"), "length")
    assert row == dict(prompt_key="k1", status="ok", harmful_request="yes", refusal="no", harmful_response="yes", reply_truncated=True)
    assert wj.label_row("k2", "empty_reply", None, "stop")["refusal"] == ""


def test_cell_summary_counts_unique_prompts_and_only_ok_rows():
    plan = pd.DataFrame(dict(source=["S", "S"], intent=["harmful", "harmful"], key_v=["base", "base"], key_a=["a1", "a2"]))
    labels = pd.DataFrame([wj.label_row("base", "ok", dict(harmful_request="yes", refusal="yes", harmful_response="no"), "stop"),
                           wj.label_row("a1", "ok", dict(harmful_request="yes", refusal="no", harmful_response="yes"), "length"),
                           wj.label_row("a2", "unparsed", None, "stop")])
    van, adv = wj.cell_summary(labels, plan)
    assert (van["form"], van["n_prompts"], van["n_ok"], van["n_refusal"], van["n_harmful_response"]) == ("vanilla", 1, 1, 1, 0)
    assert (adv["form"], adv["n_prompts"], adv["n_ok"], adv["n_refusal"], adv["n_harmful_response"], adv["n_truncated"]) == ("adversarial", 2, 1, 0, 1, 1)


# ------------------------------------------------------------------------------------------------ lost prompts
def fake_arrays():
    return dict(prenorm=np.ones((4, 2), np.float32), postnorm=np.ones((4, 2), np.float32),
                sweep=np.ones((4, 1, 2), np.float16), n_tokens=np.array([5, 6, 7, 8]),
                refusal_mass=np.ones(4, np.float32), compliance_mass=np.ones(4, np.float32),
                finite=np.array([True, False, True, False]))


def fake_fill(finite):
    n = len(finite)
    return dict(prenorm=np.full((n, 2), 9, np.float32), postnorm=np.full((n, 2), 8, np.float32),
                sweep=np.full((n, 1, 2), 7, np.float16), n_tokens=np.full(n, 70),
                refusal_mass=np.full(n, .5, np.float32), compliance_mass=np.full(n, .25, np.float32), finite=np.array(finite))


def test_fill_changes_only_the_lost_rows_and_leaves_the_input_arrays_alone():
    arrays = fake_arrays()
    out = fl.apply_fill(arrays, fake_fill([True, True]), np.array([1, 3]))
    assert out["finite"].all()
    assert out["prenorm"][1].tolist() == [9, 9] and out["prenorm"][3].tolist() == [9, 9]
    assert out["prenorm"][0].tolist() == [1, 1] and out["prenorm"][2].tolist() == [1, 1]
    assert out["n_tokens"].tolist() == [5, 70, 7, 70]
    assert arrays["finite"].tolist() == [True, False, True, False] and arrays["prenorm"][1].tolist() == [1, 1]


def test_a_prompt_that_runs_out_of_memory_again_stays_non_finite():
    out = fl.apply_fill(fake_arrays(), fake_fill([True, False]), np.array([1, 3]))
    assert out["finite"].tolist() == [True, True, True, False]
    assert out["prenorm"][3].tolist() == [1, 1]


def test_relative_l2_and_check_rows():
    assert fl.relative_l2([[3.0, 4.0]], [[3.0, 4.0]]).tolist() == [0.0]
    assert fl.relative_l2([[3.0, 9.0]], [[3.0, 4.0]]).tolist() == [1.0]
    assert fl.check_rows([False, True, True, True], 2).tolist() == [1, 2]


def test_run_batch_reads_vllm_outputs_in_input_order_and_skips_empty_batches():
    class FakeLLM:
        def generate(self, requests, sampling_params, use_tqdm):
            assert use_tqdm is False and sampling_params == "params"
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"r{len(r['prompt_token_ids'])}", finish_reason="stop", token_ids=[1, 2])])
                    for r in requests]
    assert wg.run_batch(FakeLLM(), [[1], [1, 2]], "params") == [("r1", "stop", 2), ("r2", "stop", 2)]
    assert wg.run_batch(FakeLLM(), [], "params") == []
