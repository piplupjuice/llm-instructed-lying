import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

D = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
SYM = "DejaVu Sans"
PAR, PERP = r"$\parallel$", r"$\perp$"
INK, MUTE = "#16202c", "#5b6573"
OK, BAD = "#1f9d55", "#d64545"
COL = {"C1": "#0fa3c9", "C2": "#e0308f", "C3": "#26a269", "C4": "#ee6c25"}
TINT = {"C1": "#e3f6fb", "C2": "#fde9f3", "C3": "#e6f6ed", "C4": "#fdeee4"}
SECTOR = {"C1": 135, "C2": 45, "C3": 225, "C4": 315}  # floor = the 2x2 table: x = lie wrong | right, y = truth wrong | right
QUAD = {"C1": (-1, 1), "C2": (1, 1), "C3": (-1, -1), "C4": (1, -1)}

# ---------- data ----------
rng = np.random.default_rng(7)
pool = {k: [] for k in ("C1", "C2", "C3", "C4", "C3s", "C3d")}
arrows = {k: [] for k in ("C1", "C2", "C3", "C4")}
for f in sorted(glob.glob(f"{D}/*.npz")):
    if f.endswith("_vec.npz"):
        continue
    z = np.load(f, allow_pickle=True)
    s = np.median(z["rel"]); par, orth, az, case, ch = z["par"] / s, z["orth"] / s, z["az"], z["case"], z["changed"]
    sel = {"C1": case == "C1", "C2": case == "C2", "C3": case == "C3", "C4": case == "C4", "C3s": (case == "C3") & ~ch, "C3d": (case == "C3") & ch}
    for k, m in sel.items():
        if m.sum() >= 3:
            pool[k].append((par[m], orth[m]))
    for k in arrows:
        idx = np.where(sel[k] & (orth <= 2.0) & (par > -0.4) & (par < 1.15))[0]  # drop the few outliers that leave the frame
        if len(idx):
            pick = rng.choice(idx, size=min(10 if k == "C4" else 7, len(idx)), replace=False)
            arrows[k] += [(orth[i], az[i], par[i]) for i in pick]
cen = {k: (np.mean([p.mean() for p, o in v]), np.mean([o.mean() for p, o in v])) for k, v in pool.items()}  # runs weigh equally

# ---------- figure ----------
fig = plt.figure(figsize=(7.2, 4.3))
bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
bg.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", extent=(0, 1, 0, 1), zorder=0,
          cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", ["#f4f8fc", "#ffffff"]))

ax = fig.add_axes([0.16, -0.03, 0.68, 1.0], projection="3d", computed_zorder=False)
ax.set_axis_off(); ax.set_facecolor((0, 0, 0, 0)); ax.patch.set_alpha(0)
ax.view_init(elev=30, azim=-90)
L = 2.3; floor = -0.45
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(floor, 1.35)
ax.set_box_aspect((1.35, 1.0, 0.8))

for k, (sx, sy) in QUAD.items():
    X, Y = np.meshgrid(np.linspace(0, sx * L, 2), np.linspace(0, sy * L, 2))
    ax.plot_surface(X, Y, np.full_like(X, floor), color=TINT[k], alpha=0.95, linewidth=0, zorder=0, shade=False)
    ax.plot([0, sx * L, sx * L, 0, 0], [0, 0, sy * L, sy * L, 0], np.full(5, floor), color=COL[k], lw=0.6, alpha=0.5, zorder=1)
    ax.text(sx * L * 0.8, sy * L * 0.8, floor, k, fontsize=16, weight="bold", color=COL[k], alpha=0.3, ha="center", va="center", zorder=1)
t = np.linspace(0, 2 * np.pi, 240)
for r in (0.5, 1.0, 1.5, 2.0):
    ax.plot(r * np.cos(t), r * np.sin(t), np.full_like(t, floor), color="#ffffff", lw=0.9, alpha=0.9, zorder=1)
ax.plot([-L, L], [0, 0], [floor, floor], color="#ffffff", lw=2.2, zorder=1)
ax.plot([0, 0], [-L, L], [floor, floor], color="#ffffff", lw=2.2, zorder=1)
ax.text(-L * 0.97, 0.1, floor, "lie wrong", fontsize=7.4, color=MUTE, ha="left", weight="bold", zorder=2)
ax.text(L * 0.97, 0.1, floor, "lie right", fontsize=7.4, color=MUTE, ha="right", weight="bold", zorder=2)


def place(k, r, a):
    th = np.deg2rad(SECTOR[k] + np.degrees(a) / 180 * 36)  # azimuth inside the case's quadrant (visual only)
    return r * np.cos(th), r * np.sin(th)


for k in ("C1", "C2", "C3", "C4"):
    col = COL[k]
    for r, a, zz in arrows[k]:
        x, y = place(k, r, a)
        ax.plot([0, x], [0, y], [0, zz], color=col, lw=2.6, alpha=0.06, zorder=3, solid_capstyle="round")
        ax.plot([0, x], [0, y], [0, zz], color=col, lw=0.7, alpha=0.75, zorder=4)
        ax.plot([x, x], [y, y], [floor, zz], color=col, lw=0.3, alpha=0.18, zorder=2)
        ax.scatter([x], [y], [zz], s=9, color=col, edgecolor="white", linewidths=0.3, depthshade=False, zorder=5)
        ax.scatter([x], [y], [floor], s=5, color=col, alpha=0.35, depthshade=False, edgecolor="none", zorder=2)
for k, key, ls, lw in (("C1", "C1", "-", 2.6), ("C2", "C2", "-", 2.6), ("C4", "C4", "-", 2.6), ("C3", "C3s", (0, (1, 1.2)), 2.2), ("C3", "C3d", "-", 2.6)):
    zc, rc = cen[key]
    th = np.deg2rad(np.linspace(SECTOR[k] - 40, SECTOR[k] + 40, 60))
    ax.plot(rc * np.cos(th), rc * np.sin(th), np.full_like(th, zc), color=COL[k], lw=lw, ls=ls, zorder=7, solid_capstyle="round")
    ax.plot(rc * np.cos(th), rc * np.sin(th), np.full_like(th, floor), color=COL[k], lw=1.4, ls=ls, alpha=0.7, zorder=2)
ax.quiver(0, 0, 0, 0, 0, 1.2, color=INK, lw=2.0, arrow_length_ratio=0.09, zorder=9)
ax.scatter([0], [0], [0], s=34, color="white", edgecolor=INK, linewidths=1.4, depthshade=False, zorder=10)
ax.text(0.0, 0.0, 1.32, f"common lie push {PAR}", fontsize=8.2, color=INK, weight="bold", ha="center", zorder=11)
ax.text(0.15, -0.25, -0.12, r"$h_{\mathrm{truth}}$", fontsize=9.5, color=INK, zorder=11)

# ---------- corner cards ----------
CARDS = {
    "C1": dict(pos=(0.012, 0.585), title="Successful lie", ex=[("71", True), ("221", False)], tag="knows the answer, buries it", model="Llama-3.1-8B"),
    "C2": dict(pos=(0.788, 0.585), title="Truth leakage", ex=[("71", True), ("71", True)], tag="tries to lie, the truth leaks out", model="Gemma-2-2B"),
    "C3": dict(pos=(0.012, 0.045), title="Ignorance", ex=[("240", False), ("120", False)], tag="never knew it, nothing to hide", model="Gemma-2-2B"),
    "C4": dict(pos=(0.788, 0.045), title="Paradox", ex=[("21", False), ("7", True)], tag="half-knew it, second try lands", model="Gemma-2-2B"),
}
n_items = {"C1": "1,385", "C2": "2,515", "C3": "786", "C4": "155"}
stat = {"C1": f"{PERP} = {cen['C1'][1]:.2f}", "C2": f"{PERP} = {cen['C2'][1]:.2f}", "C4": f"{PERP} = {cen['C4'][1]:.2f}",
        "C3": f"{PERP} = {cen['C3s'][1]:.2f} kept / {cen['C3d'][1]:.2f} changed"}
cw, chh = 0.2, 0.31
fig.canvas.draw()
for k, c in CARDS.items():
    x0, y0 = c["pos"]
    fig.add_artist(FancyBboxPatch((x0, y0), cw, chh, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure, fc="white", ec=COL[k], lw=1.3, zorder=20))
    fig.add_artist(FancyBboxPatch((x0, y0 + chh - 0.068), cw, 0.068, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure, fc=COL[k], ec=COL[k], lw=1.3, zorder=21))
    fig.add_artist(plt.Rectangle((x0, y0 + chh - 0.068), cw, 0.03, transform=fig.transFigure, fc=COL[k], ec="none", zorder=21))
    fig.text(x0 + 0.01, y0 + chh - 0.034, f"{k}  {c['title']}", fontsize=8.8, weight="bold", color="white", va="center", zorder=22)
    fig.text(x0 + cw - 0.008, y0 + chh - 0.034, f"n={n_items[k]}", fontsize=6.4, color="white", va="center", ha="right", zorder=22)
    (tv, tok), (lv, lok) = c["ex"]
    yy = y0 + chh - 0.125
    fig.text(x0 + 0.01, yy, "truth", fontsize=6.6, color=MUTE, va="center", zorder=22)
    fig.text(x0 + 0.045, yy, tv, fontsize=11, weight="bold", color=INK, va="center", zorder=22, bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=OK if tok else BAD, lw=1.0))
    xc = x0 + 0.058 + 0.0095 * len(tv)
    fig.text(xc, yy, "✓" if tok else "✗", fontsize=8, family=SYM, color=OK if tok else BAD, va="center", zorder=22)
    fig.text(xc + 0.015, yy, "→", fontsize=9, color=MUTE, va="center", zorder=22, family=SYM)
    fig.text(xc + 0.034, yy, "lie", fontsize=6.6, color=MUTE, va="center", zorder=22)
    fig.text(xc + 0.052, yy, lv, fontsize=11, weight="bold", color=INK, va="center", zorder=22, bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=OK if lok else BAD, lw=1.0))
    fig.text(xc + 0.064 + 0.0095 * len(lv), yy, "✓" if lok else "✗", fontsize=8, family=SYM, color=OK if lok else BAD, va="center", zorder=22)
    fig.text(x0 + 0.01, y0 + chh - 0.195, c["tag"], fontsize=7.4, style="italic", weight="bold", color=COL[k], va="center", zorder=22)
    fig.text(x0 + 0.01, y0 + 0.068, stat[k], fontsize=7.0, color=INK, va="center", zorder=22)
    fig.text(x0 + 0.01, y0 + 0.03, c["model"], fontsize=6.4, color=MUTE, va="center", zorder=22)
    zc, rc = cen["C3d" if k == "C3" else k]
    qx, qy = rc * np.cos(np.deg2rad(SECTOR[k])), rc * np.sin(np.deg2rad(SECTOR[k]))
    px, py, _ = proj3d.proj_transform(qx, qy, zc, ax.get_proj())
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((px, py)))
    sx = x0 + cw if k in ("C1", "C3") else x0
    sy = y0 + chh * (0.35 if k in ("C1", "C2") else 0.65)
    fig.add_artist(Line2D([sx, fx], [sy, fy], transform=fig.transFigure, color=COL[k], lw=0.9, alpha=0.85, zorder=19))
    fig.add_artist(Line2D([fx], [fy], transform=fig.transFigure, marker="o", ms=3.5, color=COL[k], zorder=19))

# row labels of the floor table, placed beside the right edge of the floor
for yv, lab in ((L / 2, "truth\nright"), (-L / 2, "truth\nwrong")):
    px, py, _ = proj3d.proj_transform(L * 1.02, yv, floor, ax.get_proj())
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((px, py)))
    fig.text(fx + 0.004, fy, lab, fontsize=7.4, color=MUTE, weight="bold", ha="left", va="center", linespacing=1.0, zorder=18)
fig.text(0.5, 0.965, "Same instruction, four outcomes, one geometric rule", fontsize=11.5, weight="bold", color=INK, ha="center", va="center")
fig.text(0.5, 0.925, f"every lie gets the same push {PAR};  only answers that change fly far out {PERP}", fontsize=8, color=MUTE, ha="center", va="center", style="italic")

fig.savefig(f"{OUT}/fig0_teaser_v2.png", dpi=400)
fig.savefig(f"{OUT}/fig0_teaser_v2.pdf")
print({k: tuple(round(float(x), 2) for x in v) for k, v in cen.items()}, {k: len(v) for k, v in arrows.items()})
