import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
S = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
OUT = f"{S}/figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold"})
CASE = {"C1": "#0fa3c9", "C2": "#e0308f", "C3": "#26a269", "C4": "#ee6c25"}
FAM = {"Llama": ("#1d3557", "s"), "Gemma": ("#8d5a97", "^"), "Qwen-3B": ("#c48a00", "o"), "Qwen-Math": ("#6c757d", "D")}
fam = lambda r: "Llama" if "Lamma" in r else "Gemma" if "GEMMA" in r else "Qwen-Math" if "QWEN-MATH" in r else "Qwen-3B"

runs = []
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d = os.path.dirname(lp); run = os.path.relpath(d, B).replace(os.sep, "/")
    if "mist" in run.lower():
        continue
    L = list(csv.DictReader(open(lp, encoding="utf-8-sig")))
    case = np.array([r["case"] for r in L])
    luck = np.array([int(r["luck_correct"]) / int(r["luck_n"]) if r["luck_n"] not in ("", "0") else np.nan for r in L])
    lc = np.array([int(r["luck_correct"]) if r["luck_n"] not in ("", "0") else -1 for r in L])
    pg = [json.loads(x) for x in open(f"{d}/softmax.jsonl", encoding="utf-8")]
    pg = np.array([s["p_gold_first"] for s in pg if s.get("valid") and s["cond"] == "lie"])
    runs.append(dict(run=run, fam=fam(run), case=case, luck=luck, lc=lc, pg=pg))

# ---------------- Figure: knowledge strength governs the outcome ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
ax = axs[0]
for ci, c in enumerate(["C2", "C1", "C4", "C3"]):
    for r in runs:
        m = r["case"] == c
        if m.sum() >= 5:
            col, mk = FAM[r["fam"]]
            ax.scatter(ci + np.random.default_rng(hash(r["run"]) % 99).uniform(-0.17, 0.17), np.nanmean(r["luck"][m]), s=16, marker=mk, color=col, edgecolor="white", linewidths=0.4, zorder=3)
    vals = [np.nanmean(r["luck"][r["case"] == c]) for r in runs if (r["case"] == c).sum() >= 5]
    ax.hlines(np.mean(vals), ci - 0.3, ci + 0.3, color=CASE[c], lw=2.2, zorder=2)
ax.set_xticks(range(4)); ax.set_xticklabels(["C2\nleakage", "C1\nlie", "C4\nparadox", "C3\nignorance"])
for t, c in zip(ax.get_xticklabels(), ["C2", "C1", "C4", "C3"]):
    t.set_color(CASE[c]); t.set_fontweight("bold")
ax.set_ylabel("truth-prompt re-sample accuracy")
ax.set_ylim(0, 1.05); ax.set_title("a  Knowledge strength by outcome")

ax = axs[1]
xs, ys = [], []
for r in runs:
    c = r["case"]; n1, n2, n3, n4 = [(c == k).sum() for k in ("C1", "C2", "C3", "C4")]
    w = (c == "C3") | (c == "C4")
    pred = np.nanmean(r["luck"][w]) * n2 / (n1 + n2); obs = n4 / (n3 + n4)
    col, mk = FAM[r["fam"]]
    ax.scatter(pred, obs, s=22, marker=mk, color=col, edgecolor="white", linewidths=0.4, zorder=3)
    xs.append(pred); ys.append(obs)
rr = np.corrcoef(xs, ys)[0, 1]
ax.plot([0, 0.5], [0, 0.5], color="#9aa3ad", lw=0.8, ls="--", zorder=1)
ax.set_xlim(0, 0.48); ax.set_ylim(0, 0.48)
ax.set_xlabel("predicted: knowledge × leakage"); ax.set_ylabel("observed C4 / (C3 + C4)")
ax.text(0.03, 0.43, f"r = {rr:.2f}, {len(runs)} runs", fontsize=7.5, color="#333")
ax.set_title("b  A product rule for the paradox")

ax = axs[2]
allk = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
for r in runs:
    w = (r["case"] == "C3") | (r["case"] == "C4")
    pts = []
    for k in (0, 1, 2):
        m = w & (r["lc"] == k)
        allk[k][0] += (r["case"][m] == "C4").sum(); allk[k][1] += m.sum()
        pts.append((r["case"][m] == "C4").mean() if m.sum() >= 4 else np.nan)
    col, mk = FAM[r["fam"]]
    ax.plot([0, 1, 2], pts, color=col, lw=0.6, alpha=0.45, marker=mk, ms=2.6)
pooled = [allk[k][0] / allk[k][1] for k in (0, 1, 2)]
ax.plot([0, 1, 2], pooled, color=CASE["C4"], lw=2.4, marker="o", ms=4.5, zorder=5)
for k, p in enumerate(pooled):
    ax.text(k + 0.08, p - 0.075, f"{100 * p:.0f}%", ha="left", fontsize=7.5, color=CASE["C4"], weight="bold")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0 / 2", "1 / 2", "2 / 2"]); ax.set_xlim(-0.2, 2.35)
ax.set_xlabel("truth-prompt re-samples correct"); ax.set_ylabel("P(paradox | truth wrong)")
ax.set_ylim(0, 1.0); ax.set_title("c  Paradoxes need partial knowledge")
handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=4.5, label=f) for f, (c, m) in FAM.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
fig.savefig(f"{OUT}/fig_knowledge.pdf"); fig.savefig(f"{OUT}/fig_knowledge.png", dpi=300)
print("pooled P(C4|k)", [round(p, 3) for p in pooled], "r", round(rr, 3))

# ---------------- Figure: hearing vs doing, and bistability ----------------
cross = {c["run"]: c for c in json.load(open(f"{S}/geom2.json"))}
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
for ax, xk, yk, xl, yl, tt in ((axs[0], "P1_0.5_rel_all", "meta%", "lie shift at P1 (mid depth)", "outputs naming an error (%)", "a  Hearing the instruction"),
                               (axs[1], "P2_1.0_orth_all", "lie_success%", r"$\perp$ push at P2 (last layer)", "lie success (%)", "b  Carrying it out")):
    X, Y = [], []
    for run, c in cross.items():
        col, mk = FAM[fam(run)]
        ax.scatter(c[xk], c[yk], s=22, marker=mk, color=col, edgecolor="white", linewidths=0.4, zorder=3)
        X.append(c[xk]); Y.append(c[yk])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(tt)
    ax.text(0.04, 0.9, f"r = {np.corrcoef(X, Y)[0, 1]:.2f}", transform=ax.transAxes, fontsize=7.5)
ax = axs[2]
pg = np.concatenate([r["pg"] for r in runs])
bins = np.linspace(0, 1, 21)
h, _ = np.histogram(pg, bins=bins)
ax.bar((bins[:-1] + bins[1:]) / 2, h / h.sum() * 100, width=0.046, color=["#e0308f" if b >= 0.9 else "#0fa3c9" if b < 0.1 else "#b8c0c9" for b in bins[:-1]], edgecolor="none")
mid = np.mean((pg > 0.1) & (pg < 0.9))
ax.set_xlabel("P(gold first token), lie prompt"); ax.set_ylabel("share of items (%)")
ax.text(0.5, 0.55, f"only {100 * mid:.1f}% between\n0.1 and 0.9", transform=ax.transAxes, ha="center", fontsize=7.5, color="#333")
ax.set_title("c  The decision is all-or-nothing")
handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=4.5, label=f) for f, (c, m) in FAM.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.36, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
fig.savefig(f"{OUT}/fig_two_stage.pdf"); fig.savefig(f"{OUT}/fig_two_stage.png", dpi=300)
print("bistable middle mass", round(mid, 4), "n", len(pg))
