"""Physics figure: (a-d) schematic answer landscapes for the four outcomes, (e) paradox count vs rate by difficulty."""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold"})
CASE = {"C1": "#0fa3c9", "C2": "#e0308f", "C3": "#26a269", "C4": "#ee6c25"}
FAM = {"Llama": ("#1d3557", "s"), "Gemma": ("#8d5a97", "^"), "Qwen-3B": ("#c48a00", "o"), "Qwen-Math": ("#6c757d", "D")}
INK, MUTE = "#222", "#8a939c"

x = np.linspace(-3, 3, 600)
well = lambda c, d, w: -d * np.exp(-(x - c) ** 2 / (2 * w ** 2))
WRONG = well(1.55, 1.0, 0.32) + well(-1.75, 0.85, 0.3) + well(2.55, 0.6, 0.22) + well(-2.65, 0.5, 0.2)


def U(d_gold, push, wrong=WRONG):
    return well(0, d_gold, 0.38) + push * np.exp(-x ** 2 / (2 * 0.42 ** 2)) + wrong


def at(u, xv):
    return np.interp(xv, x, u)


def ball(ax, u, xv, col, filled, dy=0.13):
    ax.scatter([xv], [at(u, xv) + dy], s=34, color=col if filled else "white", edgecolor=col, linewidths=1.3, zorder=6)


def arrow(ax, p0, p1, col, rad=-0.35):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=7, lw=1.0, color=col, zorder=5))


fig = plt.figure(figsize=(7.0, 1.95))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.3, left=0.015, right=0.94, bottom=0.2, top=0.84)
panels = [
    ("C1", "d  Lie: valley lifted", 1.9, 2.4),
    ("C2", "e  Leak: valley too deep", 4.2, 2.4),
    ("C3", "f  Ignorance: no valley", 0.25, 0.25),
    ("C4", "g  Paradox: on the ridge", 1.15, 0.0),
]
for i, (c, title, d, push) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    col = CASE[c]
    if c == "C4":
        wrong = well(1.25, 1.3, 0.3) + WRONG - well(1.55, 1.0, 0.32)
        ut = U(d, 0, wrong); ul = U(d, 0.1, wrong)
    else:
        ut, ul = U(d, 0), U(d, push)
    ax.plot(x, ut, color=MUTE, lw=1.1, label="truth prompt")
    ax.plot(x, ul, color=col, lw=1.3, ls=(0, (3, 1.6)), label="lie prompt")
    ax.axvline(0, color="#cfd5db", lw=0.6, zorder=0)
    if c == "C1":
        ball(ax, ut, 0, col, False); ball(ax, ul, 1.55, col, True)
        arrow(ax, (0.05, at(ul, 0) + 0.3), (1.45, at(ul, 1.55) + 0.32), col)
    elif c == "C2":
        ball(ax, ut, 0, col, False, dy=0.13); ball(ax, ul, 0, col, True, dy=0.13)
    elif c == "C3":
        ball(ax, ut, -1.75, col, False); ball(ax, ul, 1.55, col, True)
        arrow(ax, (-1.65, at(ul, -1.75) + 0.32), (1.45, at(ul, 1.55) + 0.34), col, rad=-0.3)
        ax.text(0, -1.5, "new wrong answer (C3$\\neq$)\nor the same one (C3$=$)", ha="center", fontsize=6.4, color=INK)
    else:
        xr = 0.64
        ball(ax, ut, 1.25, col, False); ball(ax, ul, 0, col, True)
        ax.scatter([xr], [at(ut, xr) + 0.13], s=16, color=MUTE, zorder=6)
        arrow(ax, (xr + 0.05, at(ut, xr) + 0.2), (1.2, at(ut, 1.25) + 0.3), MUTE, rad=-0.4)
        arrow(ax, (xr - 0.05, at(ut, xr) + 0.2), (0.05, at(ul, 0) + 0.3), col, rad=0.4)
        ax.text(xr, at(ut, xr) + 0.55, "ridge", ha="center", fontsize=6.4, color=INK)
    ax.set_ylim(*{"C3": (-2.9, 1.0), "C4": (-2.3, 1.0)}.get(c, (-4.4, 1.1)))
    ax.set_xlim(-3, 3)
    ax.set_xticks([0]); ax.set_xticklabels(["gold"], fontsize=7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(title, color=col, fontsize=8.2)
    if i == 0:
        ax.text(-2.95, -4.25, "energy $=-\\log p$", fontsize=6.5, color=MUTE, rotation=0)
        ax.plot([], [], color="white", label=" ")
        h = [plt.Line2D([], [], color=MUTE, lw=1.1), plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (3, 1.6))),
             plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4.5), plt.Line2D([], [], marker="o", ls="", color=INK, ms=4.5)]
        fig.legend(h, ["truth-prompt landscape", "lie-prompt landscape", "answer under truth prompt", "answer under lie prompt"],
                   loc="lower left", bbox_to_anchor=(0.015, -0.02), ncol=4, frameon=False, fontsize=6.8, handlelength=1.8)

# (e) count vs rate of the paradox by difficulty
fam = lambda r: "Llama" if "Lamma" in r else "Gemma" if "GEMMA" in r else "Qwen-Math" if "QWEN-MATH" in r else "Qwen-3B"
dsi = lambda r: 0 if "synth" in r.lower() else 1 if "gsm" in r.lower() else 2
data = {}
for lp in glob.glob(f"{B}/**/labels.csv", recursive=True):
    p = lp.replace(os.sep, "/")
    if "paper" in p.split("/")[-3:-1] or "mist" in p.lower():
        continue
    run = os.path.relpath(os.path.dirname(lp), B).replace(os.sep, "/")
    case = np.array([r["case"] for r in csv.DictReader(open(lp, encoding="utf-8-sig"))])
    n3, n4 = (case == "C3").sum(), (case == "C4").sum()
    data.setdefault(fam(run), {})[dsi(run)] = (n4, n4 / (n3 + n4))
ax = fig.add_subplot(gs[0, 4])
ax2 = ax.twinx()
for f, (col, mk) in FAM.items():
    ks = sorted(data[f])
    ax.plot(ks, [data[f][k][0] for k in ks], color=col, marker=mk, ms=3.6, lw=1.3)
    ax2.plot(ks, [100 * data[f][k][1] for k in ks], color=col, marker=mk, ms=3.0, lw=0.9, ls=(0, (2, 1.4)), mfc="white")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Syn", "GSM", "MATH"])
ax.set_ylabel("C4 count (solid)"); ax2.set_ylabel("C4 rate, % (dashed)")
ax.set_ylim(0, 32); ax2.set_ylim(0, 60)
ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_linewidth(0.6)
ax.set_xlim(-0.25, 2.25)
ax.set_title("h  Paradox count vs. rate", fontsize=8.2)
hh = [plt.Line2D([], [], marker=m, color=c, ls="", ms=3.6) for c, m in FAM.values()]
ax.legend(hh, list(FAM), loc="upper left", fontsize=6.0, frameon=False, ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
fig.savefig(f"{OUT}/fig_physics.pdf"); fig.savefig(f"{OUT}/fig_physics.png", dpi=300)
for f in data:
    print(f, {k: (int(v[0]), round(float(v[1]), 3)) for k, v in sorted(data[f].items())})
