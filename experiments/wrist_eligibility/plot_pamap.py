"""Show which windows the rule removes and how much the KD comparison changes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


folder = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
counts = pd.read_csv(folder / "pamap_coverage.csv")
order = ["walking", "standing", "sitting", "upstairs", "downstairs"]
counts = counts[(counts.rule == "strict") & (counts.subject.isin([105, 106]))]
counts = counts.groupby("activity").sum(numeric_only=True).loc[order]
start = np.zeros(len(order))
for key, label, color in (("complete", "三处完整", "#28796c"),
                          ("wrist_ok_aux_missing", "腕可用、辅助不全", "#efb558"),
                          ("wrist_unusable", "腕不可用", "#d3d9de")):
    width = 100 * counts[key] / counts.candidate
    left.barh(range(len(order)), width, left=start, color=color, label=label, height=.62)
    start += width.to_numpy()
left.set_yticks(range(len(order)), ["走路", "站立", "坐", "上楼", "下楼"])
left.invert_yaxis()
left.set(xlim=(0, 100), xlabel="占该活动全部候选窗的比例（%）")
left.set_title("原验证组 · 105、106 号", loc="left", weight="bold")
left.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=3, frameon=False, fontsize=9)

scores = pd.read_csv(folder / "pamap_metrics.csv")
scores = scores[(scores.rule == "strict") & (scores.split == "loso") & (scores.subject != "pooled")]
people = sorted(scores.subject.unique())
positions = np.arange(len(people))
for model, label, color, offset, marker in (("linear", "线性学生", "#28658c", -.10, "o"),
                                            ("rbf", "非线性学生", "#a45128", .10, "s")):
    table = scores[scores.model == model].pivot(index="subject", columns=["scope", "method"], values="macro_f1")
    gain = table.xs("multi_kd", axis=1, level="method") - table.xs("wrist_ce", axis=1, level="method")
    effect = 100 * (gain.wrist_eligible - gain.complete)
    right.scatter(effect.loc[people], positions + offset, label=label, color=color, marker=marker, s=46)
right.axvline(0, color="#728089", lw=1, zorder=0)
right.set_yticks(positions, [f"{person} 号" for person in people])
right.invert_yaxis()
right.set(xlabel="扩大评估范围后，蒸馏收益的变化（百分点）", xlim=(-2.5, 1.0))
right.set_title("六位开发人员 · 逐人留出", loc="left", weight="bold")
right.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=9)
fig.savefig(folder / "pamap_selection.png", dpi=160, facecolor="white")
