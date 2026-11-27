"""Plot paired UCI development results. Run after run.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "diversity.csv")
c = pd.read_csv(HERE / "calibration.csv")
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(10.3, 4.4), layout="constrained")
blue, orange, gray = "#166575", "#ad592b", "#acb8bb"

for _, trial in d.groupby("seed"):
    values = trial.groupby(["people", "windows"]).f1.mean().to_numpy() * 100
    left.plot(range(3), values, "o-", lw=1, ms=3.5, color=gray)
means = d.groupby(["people", "windows"]).f1.mean().to_numpy() * 100
left.plot(range(3), means, "o-", lw=2.6, color=blue)
for i, value in enumerate(means):
    left.annotate(f"{value:.2f}", (i, value), xytext=(0, 9), textcoords="offset points",
                  ha="center", color=blue, weight="bold")
left.set(xticks=range(3), xticklabels=["6 人\n1,296 窗", "12 人\n1,296 窗", "12 人\n2,592 窗"],
         ylabel="逐人 macro-F1 的均值（%）", ylim=(88,97))
left.set_title("同样多的窗口，覆盖更多训练者", loc="left", pad=14)
left.text(.03,.03,"灰线为六次配对抽样，蓝线为均值",transform=left.transAxes,color="#536266",fontsize=9)

for person, row in c[c.epoch==80].groupby("person")[["raw_nll", "scaled_nll"]].mean().iterrows():
    vals = row.to_numpy()
    color = blue if vals[1] < vals[0] else orange
    right.plot([0,1], vals, "o-", lw=1.4, color=color, ms=4)
    if vals[0] > 1:
        right.annotate(f"人员 {person}",(0,vals[0]),xytext=(6,5),textcoords="offset points",fontsize=9)
right.set(xticks=[0,1],xticklabels=["校准前","校准后"],yscale="log",xlim=(-.13,1.13),
          ylim=(.008,3.5),ylabel="逐人 NLL（对数刻度，越低越好）")
right.set_title("概率校准在不同人员上的效果",loc="left",pad=14)
right.text(.38,.03,"蓝线为改善，棕线为变差",transform=right.transAxes,color="#536266",fontsize=9)
for ax in (left,right):
    ax.grid(axis="y",alpha=.22)
    ax.spines[["left","bottom"]].set_color(gray)
fig.savefig(HERE / "uci_findings.png", dpi=170, bbox_inches="tight")
