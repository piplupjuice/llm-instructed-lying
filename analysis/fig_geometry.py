import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

D = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold"})
G = {"C1": ("#0fa3c9", "C1 successful lie"), "C2": ("#e0308f", "C2 truth leakage"), "CH": ("#e09a10", "no lie, answer changed (C3≠, C4)"), "UN": ("#7b5cc4", "no lie, answer kept (C3=)")}
FAM = {"llama": ("#1d3557", "s"), "gemma": ("#8d5a97", "^"), "Qwen2.5": ("#c48a00", "o"), "qwen2.5-3b": ("#c48a00", "o"), "qwen15bmath": ("#6c757d", "D")}


def fam(name):
    for k in ("qwen15bmath", "llama", "gemma"):
        if name.lower().replace("_", "").startswith(k):
            return FAM[k]
    return FAM["Qwen2.5"]


pts = {k: [] for k in G}; wts = {k: [] for k in G}; per = []
for f in sorted(glob.glob(f"{D}/*.npz")):
    if f.endswith("_vec.npz"):
        continue
    z = np.load(f, allow_pickle=True); name = os.path.basename(f)[:-4]
    s = np.median(z["rel"]); par, orth, case, ch = z["par"] / s, z["orth"] / s, z["case"], z["changed"]
    sel = {"C1": case == "C1", "C2": case == "C2", "CH": ((case == "C3") & ch) | (case == "C4"), "UN": (case == "C3") & ~ch}
    for k, m in sel.items():
        if m.sum() >= 3:
            pts[k].append(np.stack([par[m], orth[m]])); wts[k].append(np.full(m.sum(), 1 / m.sum()))
    row = {k: (np.median(orth[m]) if m.sum() >= 3 else np.nan) for k, m in sel.items()}
    per.append((name, row))

xs = np.linspace(-0.3, 1.4, 140); ys = np.linspace(0, 2.6, 160); XX, YY = np.meshgrid(xs, ys); grid = np.vstack([XX.ravel(), YY.ravel()])
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35))
for ax, keys, title in ((axs[0], ("C1", "C2"), "a  Lie vs. leakage"), (axs[1], ("CH", "UN"), "b  The same split with no lie")):
    for k in keys:
        P = np.concatenate(pts[k], 1); W = np.concatenate(wts[k])
        Z = gaussian_kde(P, weights=W, bw_method=0.3)(grid).reshape(XX.shape); Z /= Z.max()
        col, lab = G[k]
        ax.contourf(XX, YY, Z, levels=[0.2, 0.45, 0.7, 1.01], colors=[col], alpha=0.12)
        ax.contour(XX, YY, Z, levels=[0.2, 0.45, 0.7], colors=[col], linewidths=0.9)
        cx, cy = np.mean([p[0].mean() for p in pts[k]]), np.mean([p[1].mean() for p in pts[k]])
        ax.scatter([cx], [cy], s=26, color=col, edgecolor="#222", linewidths=0.6, zorder=5)
        ax.text(cx + 0.06, cy + 0.06, f"({cx:.2f}, {cy:.2f})", fontsize=7, color="#222")
        ax.plot([], [], color=col, lw=2, label=lab)
    ax.set_xlabel(r"common lie push $\parallel$"); ax.set_ylabel(r"answer-specific push $\perp$")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, 2.6); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=6.6, frameon=False, handlelength=1.2)
ax = axs[2]
lo, hi = 0.4, 5.0
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot([lo, hi], [lo, hi], color="#9aa3ad", lw=0.8, ls="--")
for name, row in per:
    col, mk = fam(name)
    ax.scatter(row["C2"], row["C1"], marker=mk, s=24, color=G["C1"][0], edgecolor=col, linewidths=1.0, zorder=3)
    if not np.isnan(row["UN"]) and not np.isnan(row["CH"]):
        ax.scatter(row["UN"], row["CH"], marker=mk, s=24, color=G["CH"][0], edgecolor=col, linewidths=1.0, zorder=3)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r"median $\perp$, answer kept"); ax.set_ylabel(r"median $\perp$, answer changed")
ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"])
ax.minorticks_off()
npairs = sum(1 for _, r in per for a, b in (("C1", "C2"), ("CH", "UN")) if not np.isnan(r[a]) and not np.isnan(r[b]))
nabove = sum(1 for _, r in per for a, b in (("C1", "C2"), ("CH", "UN")) if not np.isnan(r[a]) and not np.isnan(r[b]) and r[a] > r[b])
ax.text(1.9, 1.15, f"{nabove} of {npairs} pairs lie\nabove the diagonal" if nabove < npairs else f"all {npairs} pairs lie\nabove the diagonal", fontsize=7, color="#333")
ax.scatter([], [], color=G["C1"][0], s=20, label="C1 vs C2 (lie prompt worked / failed)")
ax.scatter([], [], color=G["CH"][0], s=20, label="C3≠,C4 vs C3= (no lie)")
ax.legend(loc="lower right", fontsize=6.4, frameon=False, handletextpad=0.2)
ax.set_title(f"c  Per run ({len(per)} runs)")
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUT}/fig_geometry.pdf"); fig.savefig(f"{OUT}/fig_geometry.png", dpi=300)
for name, row in per:
    print(name, {k: round(float(v), 2) for k, v in row.items()})
