"""Figure: how each outcome is written (from the full response texts, behav_rows.json from behav.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
OUT = "C:/Users/nehad/Desktop/llm lies/paper/figures"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold"})
CASE = {"C1": "#0fa3c9", "C2": "#e0308f", "C3": "#26a269", "C4": "#ee6c25"}
FAM = {"llama": ("Llama", "#1d3557", "s"), "qwen": ("Qwen-3B", "#c48a00", "o"), "gemma": ("Gemma", "#8d5a97", "^"), "qmath": ("Qwen-Math", "#6c757d", "D")}
rows = json.load(open(f"{S}/behav_rows.json"))
CS = ["C1", "C2", "C3", "C4"]

fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05), gridspec_kw=dict(width_ratios=[1, 1.1, 1, 1]))
# (a) fabrication
ax = axs[0]
for i, c in enumerate(CS):
    sel = [r for r in rows if r["case"] == c]
    ax.bar(i, 100 * np.mean([r["fab"] for r in sel]), width=0.62, color=CASE[c], alpha=0.85)
    for j, (m, (lab, col, mk)) in enumerate(FAM.items()):
        s2 = [r["fab"] for r in sel if r["model"] == m]
        if len(s2) >= 10:
            ax.scatter(i - 0.21 + 0.14 * j, 100 * np.mean(s2), marker=mk, s=13, color=col, edgecolor="white", linewidths=0.4, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(CS)
for t, c in zip(ax.get_xticklabels(), CS):
    t.set_color(CASE[c]); t.set_fontweight("bold")
ax.set_ylim(0, 105); ax.set_ylabel("lies with a fabricated number (%)")
ax.set_title("a  Is a lie ever computed?")
# (b) onset of the fabricated computation in successful lies
ax = axs[1]
for j, (m, (lab, col, mk)) in enumerate(FAM.items()):
    v = np.array([r["onset"] for r in rows if r["model"] == m and r["case"] == "C1" and r["onset"] is not None])
    mp = [r["meta_pos"] for r in rows if r["model"] == m and r["case"] == "C1" and r["meta_pos"] is not None]
    parts = ax.violinplot([v], positions=[3 - j], vert=False, widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(CASE["C1"]); b.set_alpha(0.35); b.set_edgecolor(CASE["C1"])
    ax.plot([np.median(v)] * 2, [3 - j - 0.32, 3 - j + 0.32], color="#0b5f78", lw=1.6)
    ax.scatter([np.median(mp)], [3 - j], marker="|", s=60, color="#222", zorder=4)
    ax.text(1.0, 3 - j + 0.28, f"{np.median(v):.2f}", fontsize=6.6, ha="right", color="#0b5f78")
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([v[0] for v in FAM.values()])
ax.set_xlim(0, 1); ax.set_xlabel("position in the lie text (0 = start)")
ax.set_title("b  Where the lie begins (C1)")
ax.text(0.04, -0.75, "| first word announcing an error", fontsize=6.2, color="#222")
ax.set_ylim(-0.95, 3.6)
# (c) arithmetic truth
ax = axs[2]
for i, c in enumerate(CS):
    sel = [r for r in rows if r["case"] == c]
    t = sum(r["teq_f"] for r in sel) / max(1, sum(r["teq_n"] for r in sel))
    l = sum(r["eq_f"] for r in sel) / max(1, sum(r["eq_n"] for r in sel))
    ax.bar(i - 0.17, 100 * t, width=0.32, color="white", edgecolor=CASE[c], hatch="////", lw=0.8)
    ax.bar(i + 0.17, 100 * l, width=0.32, color=CASE[c])
ax.set_xticks(range(4)); ax.set_xticklabels(CS)
for tt, c in zip(ax.get_xticklabels(), CS):
    tt.set_color(CASE[c]); tt.set_fontweight("bold")
ax.set_ylabel("equations that are false (%)")
from matplotlib.patches import Patch
ax.legend([Patch(fc="white", ec="#555", hatch="////"), Patch(fc="#555")], ["truth output", "lie output"], frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
ax.set_ylim(0, 33)
ax.set_title("c  Where arithmetic is false")
# (d) boldness
ax = axs[3]
for j, (m, (lab, col, mk)) in enumerate(FAM.items()):
    v = np.sort([r["bold"] for r in rows if r["model"] == m and r["case"] == "C1" and r["bold"] is not None])
    ax.plot(v, np.linspace(0, 1, len(v)), color=col, lw=1.4, label=lab)
ax.set_xscale("symlog", linthresh=0.05); ax.set_xlim(0, 3)
ax.set_xticks([0, 0.05, 0.3, 1, 3]); ax.set_xticklabels(["0", "0.05", "0.3", "1", "3"])
ax.set_xlabel(r"$|\log_{10}(\mathrm{lie}/\mathrm{gold})|$"); ax.set_ylabel("cumulative share of C1 lies")
ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
ax.set_title("d  How far the lie lands")
fig.tight_layout(w_pad=0.9)
fig.savefig(f"{OUT}/behaviour.pdf"); fig.savefig(f"{OUT}/behaviour.png", dpi=300)
