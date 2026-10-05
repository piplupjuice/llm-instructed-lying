import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

D = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
RUNS = {"gemma2b_gsm8k": "Gemma-2-2B · GSM8K", "llama_gsm8k": "Llama-3.1-8B · GSM8K", "llama_math500": "Llama-3.1-8B · MATH500", "Qwen2.5-3B_gsm8k": "Qwen2.5-3B · GSM8K"}
C1c, C2c = "#139fc2", "#d92e8f"

for name, title in RUNS.items():
    z = np.load(f"{D}/{name}_vec.npz", allow_pickle=True)
    S, case = z["S"], z["case"]
    m = (case == "C1") | (case == "C2")
    S, case = S[m], case[m]
    mag = np.linalg.norm(S, axis=1)
    dirs = S / mag[:, None]
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.35, metric="cosine", random_state=0).fit_transform(dirs)
    emb = (emb - emb.mean(0)) / emb.std(0)

    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    lo = 0.0
    is1 = case == "C1"
    # floor footprint: where each vector points (direction only)
    for sel, col in ((~is1, C2c), (is1, C1c)):
        ax.scatter(emb[sel, 0], emb[sel, 1], np.full(sel.sum(), lo), s=7, color=col, alpha=0.18, depthshade=False, zorder=1)
    # median magnitude planes
    xr = np.linspace(emb[:, 0].min() - 0.3, emb[:, 0].max() + 0.3, 2)
    yr = np.linspace(emb[:, 1].min() - 0.3, emb[:, 1].max() + 0.3, 2)
    PX, PY = np.meshgrid(xr, yr)
    for sel, col, lab in ((~is1, C2c, "Truth leakage (C2)"), (is1, C1c, "Successful lie (C1)")):
        med = np.median(mag[sel])
        ax.plot_surface(PX, PY, np.full_like(PX, med), color=col, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(xr[[0, 1, 1, 0, 0]], yr[[0, 0, 1, 1, 0]], np.full(5, med), color=col, lw=1.0, alpha=0.8, zorder=3)
        ax.text(xr[1], yr[0], med, f"  median ‖shift‖ = {med:.2f}", color=col, fontsize=9, weight="bold", zorder=9)
    # stems + points
    order = np.argsort(is1)  # draw C2 first, C1 on top
    for i in order:
        col = C1c if is1[i] else C2c
        ax.plot([emb[i, 0]] * 2, [emb[i, 1]] * 2, [lo, mag[i]], color=col, lw=0.35, alpha=0.25, zorder=4)
    for sel, col, lab in ((~is1, C2c, f"Truth leakage (C2), n={int((~is1).sum())}"), (is1, C1c, f"Successful lie (C1), n={int(is1.sum())}")):
        ax.scatter(emb[sel, 0], emb[sel, 1], mag[sel], s=16, color=col, edgecolor="white", linewidths=0.3, alpha=0.95, depthshade=False, zorder=6, label=lab)
    ax.set_xlabel("UMAP-1  (direction)", labelpad=6, weight="bold")
    ax.set_ylabel("UMAP-2  (direction)", labelpad=6, weight="bold")
    ax.set_zlabel("magnitude  ‖h_lie − h_truth‖ / ‖h_truth‖", labelpad=8, weight="bold")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_zlim(lo, np.percentile(mag, 99.5) * 1.05)
    ax.view_init(elev=14, azim=-50)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.975, 0.98, 0.99, 1)); a.pane.set_edgecolor("#d5dbe2")
        a._axinfo["grid"]["color"] = (0.86, 0.88, 0.91, 1)
    ax.set_box_aspect((1, 1, 0.9))
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, bbox_to_anchor=(0.0, 0.97))
    ax.set_title(f"Same direction, different length — {title}, P2, last layer", fontsize=11.5, weight="bold", pad=2)
    fig.subplots_adjust(left=0.0, right=0.95, top=0.95, bottom=0.02)
    fig.savefig(f"{OUT}/fig2_umap3d_{name}.png", dpi=300)
    fig.savefig(f"{OUT}/fig2_umap3d_{name}.pdf")
    print("ok", name, "median mag C1 %.2f C2 %.2f" % (np.median(mag[is1]), np.median(mag[~is1])))
