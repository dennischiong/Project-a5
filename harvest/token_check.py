"""Task 3: token position check. Tokenizers only (no model weights), CPU.

Uses only fixed benign sentences (no dataset text). Writes figures/offsets.csv.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from seeds import set_seed  # noqa: E402

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
SENTENCES = [
    "What is the capital of France?",
    "How many days are there in a leap year?",
    "Explain photosynthesis in two sentences.",
    "Write a short poem about autumn leaves.",
    "Translate 'good morning' into Spanish.",
    "List three primary colors.",
    "What is the boiling point of water at sea level?",
    "Give me a simple recipe for pancakes.",
    "Who wrote the novel Pride and Prejudice?",
    "Summarize the plot of Cinderella in one paragraph.",
    "How do I convert Celsius to Fahrenheit?",
    "What causes the seasons on Earth?",
    "Name five common houseplants.",
    "What is the square root of 144?",
    "Describe how a bicycle works.",
    "Suggest a name for a friendly robot.",
    "What is the largest planet in the solar system?",
    "How do bees make honey?",
    "Explain what a prime number is.",
    "Tell me a fun fact about octopuses.",
]
PROBE = "What is the capital of France?"


def ids_of(x):
    """apply_chat_template(tokenize=True) may return a list or a BatchEncoding."""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if len(x) and isinstance(x[0], list):
        x = x[0]
    return list(x)


def main():
    set_seed(0)
    token = os.environ.get("HF_TOKEN")  # never printed
    if not token:
        print("NOTE: no HF_TOKEN in this session, so gated models are reported as 'no HF_TOKEN', which is not an access result.")
    rows = []
    for name in MODELS:
        row = dict(model=name, status="ok")
        try:
            tok = AutoTokenizer.from_pretrained(name, token=token)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403) or type(e).__name__ == "GatedRepoError" or any(
                    c in msg for c in ("401", "403", "gated", "Access to model")):
                # only call it an access result when we actually authenticated
                row["status"] = "access not granted" if token else "no HF_TOKEN (gated; not checked)"
            else:
                row["status"] = f"error: {type(e).__name__}"
            rows.append(row)
            print(f"{name}: {row['status']}")
            continue

        n_eq = 0
        n_dup = 0
        has_bos_in_template = None
        for s in SENTENCES:
            msgs = [{"role": "user", "content": s}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            a = tok(text, add_special_tokens=False)["input_ids"]
            b = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            assert a == b, f"{name}: render+tokenize != apply_chat_template(tokenize=True)"
            n_eq += 1
            bos = tok.bos_token
            has_bos_in_template = bool(bos) and text.startswith(bos)
            d = tok(text)["input_ids"]  # default tokenization (adds special tokens)
            if tok.bos_token_id is not None and len(d) > 1 and d[0] == d[1] == tok.bos_token_id:
                n_dup += 1
        row.update(
            sentences=len(SENTENCES), ids_equal=n_eq == len(SENTENCES),
            template_has_bos=has_bos_in_template, default_tokenization_duplicate_bos=f"{n_dup}/{len(SENTENCES)}",
        )

        # last tokens (up to 12) for the fixed benign probe sentence
        text = tok.apply_chat_template([{"role": "user", "content": PROBE}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        n = len(ids)
        start = text.rfind(PROBE)
        end = start + len(PROBE)
        last_instr = max(i for i, (s0, e0) in enumerate(offs) if e0 > s0 and s0 < end)  # last token starting inside the sentence
        row["n_tokens_probe"] = n
        # Llama-3.1's template inserts a date line (Today Date: ...), so token counts can change by day; record it.
        row["template_inserts_date"] = "Today Date:" in text
        row["run_date_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row["last_instruction_token_index_from_end"] = last_instr - n  # e.g. -6
        row["last_instruction_token_string"] = tok.convert_ids_to_tokens(ids[last_instr])
        print(f"\n== {name} | probe length {n} tokens | last instruction token at index {last_instr - n}")
        # up to the last 12 tokens: short templates (Mistral's probe is ~10 tokens) have fewer than 12
        for k in range(1, min(12, n) + 1):
            print(f"  {-k:>4}  {tok.convert_ids_to_tokens(ids[-k])!r}")
        if n < 12:
            print(f"  (the whole prompt is only {n} tokens)")
        rows.append(row)

    out = REPO / "figures" / "offsets.csv"
    out.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df["team_estimate"] = df["model"].map(TEAM_ESTIMATE)
    df.to_csv(out, index=False)
    print("\nsaved", out.relative_to(REPO))
    print("Team estimate: about -6 for Qwen/Llama/Gemma and -2 for Mistral. Compare with the column above.")
    plot_offsets(df)


# Work plan section 5 table, Last-instruction position (verify W3)
TEAM_ESTIMATE = {
    "Qwen/Qwen2.5-1.5B-Instruct": -6, "Qwen/Qwen2.5-7B-Instruct": -6, "meta-llama/Llama-3.1-8B-Instruct": -6,
    "google/gemma-2-9b-it": -6, "mistralai/Mistral-7B-Instruct-v0.3": -2,
}


def plot_offsets(df):
    """figures/offsets.png: measured last-instruction offset vs the work plan's estimate, one row per model."""
    from analysis import plotstyle as ps
    import matplotlib.pyplot as plt

    ps.apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = list(range(len(df)))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        est = r["team_estimate"]
        ax.scatter([est], [yi], s=90, facecolors="none", edgecolors=ps.MUTED, linewidths=1.5, zorder=3,
                   label="work plan estimate" if yi == y[0] else None)
        found = r.get("last_instruction_token_index_from_end")
        if r["status"] == "ok" and pd.notna(found):
            ax.scatter([found], [yi], s=50, color=ps.BLUE, zorder=4, label="measured" if yi == y[0] else None)
            ax.annotate(f"{int(found)}" + ("" if int(found) == est else "  (differs)"), xy=(found, yi), xytext=(7, -3),
                        textcoords="offset points", fontsize=7.5, color=ps.INK2)
        else:
            ax.annotate(r["status"], xy=(est, yi), xytext=(10, -3), textcoords="offset points", fontsize=7.5, color=ps.MUTED)
    ax.set_yticks(y, [m.split("/")[-1] for m in df["model"]])
    ax.set_xlim(-12.5, 0.5)
    ax.set_xticks(range(-12, 1))
    ax.set_xlabel("index of the last user-instruction token, counted from the end (prompt[-1] = last prefill token)")
    ax.legend(loc="lower left")
    ps.caption(fig, "Task 3 - where the last instruction token sits in each chat template",
               'probe sentence: "What is the capital of France?" | tokenizers only (no weights), CPU | '
               "add_generation_prompt=True, default template (incl. default system blocks)")
    ps.save(fig, REPO / "figures" / "offsets.png")
    print("saved figures/offsets.png")


if __name__ == "__main__":
    main()
