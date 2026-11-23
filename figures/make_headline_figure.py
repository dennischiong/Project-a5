"""Headline comparison figure (Dennis's cells 32/39), redrawn from results/headline_bf16.csv.
The same figure is also written on Modal (dennis-data:/ws1_results/baselines/<model>__headline_figure_bf16.png).

    python make_headline_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POS = ["WJ_test_van", "WJ_test_adv", "StrongREJECT", "JBB_harmful"]
LABELS = ["Vanilla (WJ)", "Adversarial (WJ)", "StrongREJECT", "JailbreakBench"]
NAMES = {"qwen25-7b": "Qwen2.5-7B-Instruct", "llama31-8b": "Llama-3.1-8B-Instruct"}
COLORS = ["#1d4ed8", "#ef4444", "#10b981", "#f59e0b"]

df = pd.read_csv(HERE / "results" / "headline_bf16.csv")
for model, d in df.groupby("model"):
    d = d[d.metric == "auroc"]
    methods = list(dict.fromkeys(d.method))
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    x, w = np.arange(len(POS)), 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = d[d.method == m][POS].to_numpy()[0]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=m, color=COLORS[i % 4])
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylabel("AUROC (negatives: WJ benign test, adversarial form)")
    ax.set_title(f"{NAMES[model]}: internal probe vs output refusal vs external guards (all bf16)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(methods), fontsize=8, frameon=False)
    fig.tight_layout()
    out = HERE / "results" / f"{model}__headline_figure_bf16.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)
