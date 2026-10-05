import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

D = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})

G = {
    "C1": dict(sel=lambda c, ch: c == "C1", color="#139fc2", text="#0b7c99", label="Successful lie (C1)", cmap=["#c9f1fa", "#7fdcf0", "#19b4d6", "#0a6f8a"]),
    "C2": dict(sel=lambda c, ch: c == "C2", color="#d92e8f", text="#b01f73", label="Truth leakage (C2)", cmap=["#fbd3ea", "#f59ccf", "#e0379c", "#8f1660"]),
    "CH": dict(sel=lambda c, ch: ((c == "C3") & ch) | (c == "C4"), color="#e09a10", text="#9a6a00", label="Answer changed, no lie\n(C3≠, C4)", cmap=["#fbe6b4", "#f6d27a", "#e8a317", "#8a5d00"]),
    "UN": dict(sel=lambda c, ch: (c == "C3") & ~ch, color="#7b5cc4", text="#5a3fa3", label="Answer unchanged, no lie\n(C3=)", cmap=["#e1d8f7", "#b9a6ec", "#8e70d6", "#4b2f99"]),
}

pts = {k: [] for k in G}; wts = {k: [] for k in G}
for f in sorted(glob.glob(f"{D}/*.npz")):
    if f.endswith("_vec.npz"):
        continue
    z = np.load(f, allow_pickle=True)
    scale = np.median(z["rel"])  # per-run normalisation: 1.0 = that run's typical total shift
    par, orth = z["par"] / scale, z["orth"] / scale
    for k, g in G.items():
        m = g["sel"](z["case"], z["changed"])
        if m.sum() >= 3:
            pts[k].append(np.stack([par[m], orth[m]]))
            wts[k].append(np.full(m.sum(), 1.0 / m.sum()))  # every run weighs the same

xs = np.linspace(-0.25, 1.35, 150)   # common lie push (parallel)
ys = np.linspace(0.0, 2.6, 200)      # answer-specific push (orthogonal)
XX, YY = np.meshgrid(xs, ys)
grid = np.vstack([XX.ravel(), YY.ravel()])
dens, cent = {}, {}
for k in G:
    P = np.concatenate(pts[k], axis=1); W = np.concatenate(wts[k])
    d = gaussian_kde(P, weights=W, bw_method=0.3)(grid).reshape(XX.shape)
    dens[k] = d / d.max()  # each landscape scaled to its own peak (shape comparison)
    cent[k] = (float(np.average(P[0], weights=W)), float(np.average(P[1], weights=W)), P.shape[1])


def style(ax):
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_zlim(0, 1.25)
    ax.set_xlabel("common lie push  ∥", labelpad=8, weight="bold")
    ax.set_ylabel("answer-specific push  ⊥", labelpad=10, weight="bold")
    ax.set_zticks([]); ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.view_init(elev=24, azim=-122)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.975, 0.98, 0.99, 1)); a.pane.set_edgecolor("#d5dbe2")
        a._axinfo["grid"]["color"] = (0.86, 0.88, 0.91, 1)
    ax.set_box_aspect((0.95, 1.6, 0.6))


def landscape(ax, keys):
    for j, k in enumerate(keys):
        g = G[k]; Z = dens[k]
        cmap = LinearSegmentedColormap.from_list(k, g["cmap"])
        ax.plot_surface(XX, YY, np.where(Z < 0.07, np.nan, Z), cmap=cmap, vmin=0.07, vmax=1.0, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.82)
        ax.contour(XX, YY, Z, zdir="z", offset=0, levels=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9], colors=[g["color"]], linewidths=0.9)
        cx, cy, n = cent[k]
        ax.scatter([cx], [cy], [0], marker="o", s=42, color=g["color"], edgecolor="#222", linewidths=0.7, depthshade=False, zorder=10)
        ax.text(cx + 0.06, cy - 0.05, 0.0, f"({cx:.2f}, {cy:.2f})", fontsize=8.5, color="#222", zorder=11)
        iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
        ly = ys[iy] + (0.75 if j == 0 else -0.55)
        ax.plot([xs[ix], xs[ix]], [ys[iy], ly], [1.0, 1.13], color=g["color"], lw=1.0)
        ax.text(xs[ix], ly, 1.13, f"{g['label']}\nn = {n:,}", fontsize=9.5, ha="center", va="bottom", color=g["text"], weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=g["color"], lw=1.3), zorder=12)
    (x0, y0, _), (x1, y1, _) = cent[keys[1]], cent[keys[0]]
    ax.quiver(x0, y0, 0.02, x1 - x0, y1 - y0, 0, color="#333", arrow_length_ratio=0.12, linewidth=1.3, linestyle="--")


fig = plt.figure(figsize=(8.4, 11.2))
ax1 = fig.add_subplot(2, 1, 1, projection="3d", computed_zorder=False)
landscape(ax1, ["C1", "C2"]); style(ax1)
ax1.text2D(0.03, 0.97, "a   Lie vs. leakage", transform=ax1.transAxes, fontsize=12.5, weight="bold", va="top")
ax2 = fig.add_subplot(2, 1, 2, projection="3d", computed_zorder=False)
landscape(ax2, ["CH", "UN"]); style(ax2)
ax2.text2D(0.03, 0.97, "b   Same split without any lie", transform=ax2.transAxes, fontsize=12.5, weight="bold", va="top")
fig.subplots_adjust(left=0.0, right=0.97, top=1.0, bottom=0.02, hspace=-0.02)
fig.savefig(f"{OUT}/fig1_lie_landscape.png", dpi=300)
fig.savefig(f"{OUT}/fig1_lie_landscape.pdf")
print({k: tuple(round(x, 3) for x in v[:2]) + (v[2],) for k, v in cent.items()})
