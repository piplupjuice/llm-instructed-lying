import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

D = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
SYM = "DejaVu Sans"  # has the check / cross glyphs
INK, MUTE = "#1d2430", "#5b6573"
OK, BAD = "#1f9d55", "#d64545"
COL = {"C1": "#139fc2", "C2": "#d92e8f", "C3": "#2e9e6a", "C4": "#e0702b"}
TINT = {"C1": "#eaf8fc", "C2": "#fdeef6", "C3": "#ebf7f0", "C4": "#fdf1e9"}

CARDS = {
    "C1": dict(title="Successful lie", model="Llama-3.1-8B", q="Ali: \\$21 + half of \\$100 = ?", truth=("71", True), lie=("221", False), tag="knows the answer, buries it", n="1,385"),
    "C2": dict(title="Truth leakage", model="Gemma-2-2B", q="Ali: \\$21 + half of \\$100 = ?", truth=("71", True), lie=("71", True), tag="tries to lie, the truth leaks out", n="2,515"),
    "C3": dict(title="Ignorance", model="Gemma-2-2B", q="2 people × 4 apples × 30 days = ?", truth=("240", False), lie=("120", False), tag="never knew it, nothing to hide", n="786"),
    "C4": dict(title="Paradox", model="Gemma-2-2B", q="3 eggs a day: dozens in 4 weeks?", truth=("21", False), lie=("7", True), tag="half-knew it, second try lands", n="155"),
}

fig = plt.figure(figsize=(7.2, 3.55))

# ---------------- left: the four outcomes ----------------
axL = fig.add_axes([0.0, 0.0, 0.5, 1.0]); axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
axL.text(0.035, 0.955, "a", fontsize=11, weight="bold", color=INK, va="center")
axL.text(0.085, 0.955, "Four outcomes of one instruction", fontsize=10.5, weight="bold", color=INK, va="center")
axL.text(0.085, 0.893, "Asked twice:", fontsize=7.6, color=MUTE, va="center")
for x, label, fc, ec in ((0.255, "Solve it.", "#f1f4f8", "#c4ccd6"), (0.42, "Solve it incorrectly.", "#fff4e0", "#e3b25c")):
    axL.text(x, 0.893, label, fontsize=7.6, style="italic", color=INK, va="center", bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.8))

x0s, y0s, w, h = (0.13, 0.565), (0.445, 0.05), 0.40, 0.335
for ci, (lab, sym, sc) in enumerate((("lie answer wrong", "✗", BAD), ("lie answer right", "✓", OK))):
    cx = x0s[ci] + w / 2
    axL.text(cx + 0.012, 0.815, lab, fontsize=7.8, color=INK, ha="center", va="center", weight="bold")
    axL.text(cx - 0.125, 0.815, sym, fontsize=8.5, color=sc, ha="center", va="center", family=SYM)
for y0, lab, sym, sc in ((y0s[0], "truth answer right", "✓", OK), (y0s[1], "truth answer wrong", "✗", BAD)):
    axL.text(0.075, y0 + h / 2 - 0.012, lab, fontsize=7.8, color=INK, ha="center", va="center", rotation=90, weight="bold")
    axL.text(0.075, y0 + h / 2 + 0.165, sym, fontsize=8.5, color=sc, ha="center", va="center", family=SYM)

layout = {"C1": (0, 0), "C2": (1, 0), "C3": (0, 1), "C4": (1, 1)}
for k, (ci, ri) in layout.items():
    c, x0, y0 = CARDS[k], x0s[ci], y0s[ri]
    axL.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.018", fc=TINT[k], ec=COL[k], lw=1.1))
    axL.add_patch(FancyBboxPatch((x0, y0 + h - 0.07), w, 0.07, boxstyle="round,pad=0,rounding_size=0.018", fc=COL[k], ec=COL[k], lw=1.1))
    axL.add_patch(plt.Rectangle((x0, y0 + h - 0.07), w, 0.035, fc=COL[k], ec="none"))
    axL.text(x0 + 0.018, y0 + h - 0.035, k, fontsize=8.6, weight="bold", color="white", va="center")
    axL.text(x0 + 0.075, y0 + h - 0.035, c["title"], fontsize=8.6, weight="bold", color="white", va="center")
    axL.text(x0 + w - 0.015, y0 + h - 0.035, f"n = {c['n']}", fontsize=6.6, color="white", va="center", ha="right")
    axL.text(x0 + 0.018, y0 + h - 0.11, c["model"], fontsize=6.6, color=MUTE, va="center")
    axL.text(x0 + 0.018, y0 + h - 0.16, c["q"], fontsize=7.0, color=INK, va="center")
    for j, (lab, (val, ok)) in enumerate((("truth", c["truth"]), ("lie", c["lie"]))):
        cx = x0 + 0.018 + j * 0.2
        vx = cx + (0.072 if lab == "truth" else 0.045)
        axL.text(cx, y0 + 0.105, lab, fontsize=6.6, color=MUTE, va="center")
        axL.text(vx, y0 + 0.105, val, fontsize=9.5, weight="bold", color=INK, va="center", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=OK if ok else BAD, lw=0.9))
        axL.text(vx + 0.03 + 0.019 * len(val), y0 + 0.105, "✓" if ok else "✗", fontsize=8, family=SYM, color=OK if ok else BAD, va="center")
    axL.text(x0 + 0.018, y0 + 0.04, c["tag"], fontsize=7.2, style="italic", color=COL[k], va="center", weight="bold")

# ---------------- right: inside the model ----------------
rng = np.random.default_rng(3)
arrows = {"C1": [], "C2": []}
pool = {k: [] for k in ("C1", "C2", "CH", "UN")}
for f in sorted(glob.glob(f"{D}/*.npz")):
    if f.endswith("_vec.npz"):
        continue
    z = np.load(f, allow_pickle=True)
    s = np.median(z["rel"]); par, orth, az, case, ch = z["par"] / s, z["orth"] / s, z["az"], z["case"], z["changed"]
    sel = {"C1": case == "C1", "C2": case == "C2", "CH": ((case == "C3") & ch) | (case == "C4"), "UN": (case == "C3") & ~ch}
    for k, m in sel.items():
        if m.sum() >= 3:
            pool[k].append((par[m], orth[m]))
    for k in ("C1", "C2"):
        idx = np.where(sel[k] & (orth <= 2.0) & (par > -0.4) & (par < 1.15))[0]  # drop the few outliers that leave the frame
        if len(idx):
            pick = rng.choice(idx, size=min(9, len(idx)), replace=False)
            arrows[k] += [(orth[i], az[i], par[i]) for i in pick]

cen = {k: (np.mean([p.mean() for p, o in pool[k]]), np.mean([o.mean() for p, o in pool[k]])) for k in pool}  # runs weigh equally
axR = fig.add_axes([0.49, -0.07, 0.53, 1.07], projection="3d", computed_zorder=False)
axR.set_axis_off(); axR.set_facecolor((0, 0, 0, 0)); axR.patch.set_alpha(0)
axR.view_init(elev=17, azim=-62)
lim = 2.0
axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim); axR.set_zlim(-0.55, 1.3)
axR.set_box_aspect((1, 1, 0.78))
t = np.linspace(0, 2 * np.pi, 200)
floor = -0.5
for r in (0.5, 1.0, 1.5, 2.0):
    axR.plot(r * np.cos(t), r * np.sin(t), np.full_like(t, floor), color="#d9dee5", lw=0.6, zorder=0)
for a in np.linspace(0, np.pi, 4, endpoint=False):
    axR.plot([2 * np.cos(a), -2 * np.cos(a)], [2 * np.sin(a), -2 * np.sin(a)], [floor, floor], color="#e6e9ee", lw=0.5, zorder=0)
for k, col, ls in (("CH", "#e09a10", (0, (3, 2))), ("UN", "#7b5cc4", (0, (1, 1.5)))):
    zc, rc = cen[k]
    axR.plot(rc * np.cos(t), rc * np.sin(t), np.full_like(t, zc), color=col, lw=1.0, ls=ls, zorder=2)
for k, alpha in (("C1", 0.55), ("C2", 0.7)):
    col = COL[k]
    for r, a, zz in arrows[k]:
        r = min(r, 2.0)
        x, y = r * np.cos(a), r * np.sin(a)
        axR.plot([0, x], [0, y], [0, zz], color=col, lw=0.55, alpha=alpha * 0.6, zorder=3)
        axR.plot([x, x], [y, y], [floor, zz], color=col, lw=0.25, alpha=0.12, zorder=1)
        axR.scatter([x], [y], [zz], s=6, color=col, alpha=alpha, depthshade=False, zorder=4, edgecolor="none")
        axR.scatter([x], [y], [floor], s=3, color=col, alpha=0.18, depthshade=False, zorder=1, edgecolor="none")
for k in ("C1", "C2"):
    zc, rc = cen[k]; col = COL[k]
    TT, RR = np.meshgrid(t, np.linspace(0, 1, 12))
    axR.plot_surface(RR * rc * np.cos(TT), RR * rc * np.sin(TT), RR * zc, color=col, alpha=0.07, linewidth=0, zorder=2)
    axR.plot(rc * np.cos(t), rc * np.sin(t), np.full_like(t, zc), color=col, lw=2.2, zorder=6)
    axR.plot(rc * np.cos(t), rc * np.sin(t), np.full_like(t, floor), color=col, lw=1.0, alpha=0.45, zorder=1)
axR.quiver(0, 0, 0, 0, 0, 1.18, color=INK, lw=1.8, arrow_length_ratio=0.1, zorder=8)
axR.scatter([0], [0], [0], s=26, color="white", edgecolor=INK, linewidths=1.2, depthshade=False, zorder=9)
axR.text(0.12, -0.05, -0.16, r"$h_{\mathrm{truth}}$", fontsize=9, color=INK, zorder=10)
axR.text(0.0, 0.0, 1.32, "common lie push  ∥", fontsize=8, color=INK, weight="bold", ha="center", zorder=10)
z1, r1 = cen["C1"]; z2, r2 = cen["C2"]
a1, a2 = np.deg2rad(200), np.deg2rad(30)
axR.plot([r1 * np.cos(a1), (r1 + 0.35) * np.cos(a1)], [r1 * np.sin(a1), (r1 + 0.35) * np.sin(a1)], [z1, z1 + 0.25], color=COL["C1"], lw=0.9, zorder=10)
axR.text((r1 + 0.38) * np.cos(a1), (r1 + 0.38) * np.sin(a1), z1 + 0.27, f"C1 spreads wide\n⊥ = {r1:.2f}", fontsize=7.8, color=COL["C1"], weight="bold", ha="right", va="bottom", zorder=10)
axR.plot([r2 * np.cos(a2), r2 * np.cos(a2) + 0.1], [r2 * np.sin(a2), r2 * np.sin(a2)], [z2, z2 + 0.42], color=COL["C2"], lw=0.9, zorder=10)
axR.text(r2 * np.cos(a2) + 0.12, r2 * np.sin(a2), z2 + 0.44, f"C2 stays tight\n⊥ = {r2:.2f}", fontsize=7.8, color=COL["C2"], weight="bold", ha="left", va="bottom", zorder=10)
axR.text2D(0.03, 0.935, "b", transform=axR.transAxes, fontsize=11, weight="bold", color=INK)
axR.text2D(0.085, 0.935, "Inside the model: one lie push, two spreads", transform=axR.transAxes, fontsize=10.5, weight="bold", color=INK)
axR.text2D(0.085, 0.88, f"same height (∥ ≈ {z1:.2f} vs {z2:.2f}); the answer-specific push ⊥ sets them apart", transform=axR.transAxes, fontsize=7.4, color=MUTE)
axR.text2D(0.085, 0.125, "- - -", transform=axR.transAxes, fontsize=8, color="#e09a10", weight="bold")
axR.text2D(0.15, 0.125, "answer changed, no lie (C3≠, C4)", transform=axR.transAxes, fontsize=7, color=MUTE)
axR.text2D(0.085, 0.085, "· · ·", transform=axR.transAxes, fontsize=8, color="#7b5cc4", weight="bold")
axR.text2D(0.15, 0.085, "answer unchanged, no lie (C3=)", transform=axR.transAxes, fontsize=7, color=MUTE)

fig.savefig(f"{OUT}/fig0_teaser.png", dpi=400)
fig.savefig(f"{OUT}/fig0_teaser.pdf")
print({k: tuple(round(float(x), 2) for x in v) for k, v in cen.items()}, {k: len(v) for k, v in arrows.items()})
