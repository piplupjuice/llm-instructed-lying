"""Cross-metric structure over the 12 runs: Spearman matrix (clustered) and PCA, saved for the paper."""
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
S = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
OUT = "C:/Users/nehad/Desktop/llm lies/paper/figures"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})
SHORT = {"llama_synthetic": "llama_syn", "llama_gsm8k": "llama_gsm", "llama_math500": "llama_math", "Qwen2.5-3b-Syntehtic": "qwen_syn",
         "Qwen2.5-3B_gsm8k": "qwen_gsm", "qwen2.5-3b_instruct_math500": "qwen_math", "gemma2b_synthetic": "gemma_syn", "gemma2b_gsm8k": "gemma_gsm",
         "gemma2b_Math500": "gemma_math", "qwen15b_MATH_SYNTHETIC": "qmath_syn", "qwen15bmath_gsm8k": "qmath_gsm", "qwen15bmath_math500": "qmath_math"}
G = json.load(open(f"{S}/geom.json")); G2 = {c["run"]: c for c in json.load(open(f"{S}/geom2.json"))}
PA = json.load(open(f"{S}/paper_analysis.json")); rows = json.load(open(f"{S}/behav_rows.json"))
M = {}
for lp in glob.glob(f"{B}/**/labels.csv", recursive=True):
    p = lp.replace(os.sep, "/")
    if "paper" in p.split("/")[-3:-1] or "mist" in p.lower():
        continue
    d = os.path.dirname(lp); run = os.path.relpath(d, B).replace(os.sep, "/"); name = os.path.basename(d); sh = SHORT[name]
    L = list(csv.DictReader(open(lp, encoding="utf-8-sig")))
    c = np.array([r["case"] for r in L]); n = {k: (c == k).sum() for k in ("C1", "C2", "C3", "C4")}
    lc = np.array([int(r["luck_correct"]) if r["luck_n"] not in ("", "0") else -1 for r in L])
    w = ((c == "C3") | (c == "C4")) & (lc >= 0)
    z = np.load(f"{S}/figdata/{name}.npz", allow_pickle=True)
    R = [r for r in rows if r["run"] == sh]
    c1 = [r for r in R if r["case"] == "C1"]
    beh = G[run]["behaviour"]
    M[sh] = {
        "Truth accuracy": (n["C1"] + n["C2"]) / sum(n.values()),
        "Lie success": n["C1"] / (n["C1"] + n["C2"]),
        "Strict lie success": PA[name]["strict"]["strict_lie_success"],
        "Paradox rate": n["C4"] / (n["C3"] + n["C4"]),
        "Re-sample baseline": float(np.mean(lc[w] / 2)),
        "Hearing (P1 shift)": G2[run]["P1_0.5_rel_all"],
        "Shared push (P2 \u2225)": float(np.median(z["par"])),
        "Turn (P2 \u22a5)": G2[run]["P2_1.0_orth_all"],
        "Acknowledges error": G2[run]["meta%"] / 100,
        "Ignores instruction": G2[run]["ignored%"] / 100,
        "Lie/truth length": beh["lie_chars_median"] / beh["truth_chars_median"],
        "Fabricates a number": float(np.mean([r["fab"] for r in R])),
        "Early lie onset": -float(np.median([r["onset"] for r in c1 if r["onset"] is not None])),
        "Lie boldness": float(np.median([r["bold"] for r in c1 if r["bold"] is not None])),
        "Bistable mass": PA[name]["bistable"]["frac_middle_all"],
    }
runs = list(M); keys = list(M[runs[0]])
X = np.array([[M[r][k] for k in keys] for r in runs])
rho = spearmanr(X).correlation
order = leaves_list(linkage(X.T if False else (1 - rho)[np.triu_indices(len(keys), 1)], "average"))
rk = rho[np.ix_(order, order)]; kk = [keys[i] for i in order]
fig, ax = plt.subplots(figsize=(3.3, 3.0))
im = ax.imshow(rk, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(kk))); ax.set_xticklabels(kk, rotation=60, ha="right", fontsize=6.2)
ax.set_yticks(range(len(kk))); ax.set_yticklabels(kk, fontsize=6.2)
for i in range(len(kk)):
    for j in range(len(kk)):
        if i != j:
            ax.text(j, i, f"{rk[i, j]:.1f}".replace("0.", ".").replace("-.", "\u2212."), ha="center", va="center", fontsize=4.6,
                    color="white" if abs(rk[i, j]) > 0.6 else "#333")
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=5.5); cb.outline.set_linewidth(0.3)
cb.set_label("Spearman $\\rho$ over 12 runs", fontsize=6)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/corr.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/corr.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
# PCA on standardised metrics
Z = (X - X.mean(0)) / X.std(0)
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
ev = s ** 2 / (s ** 2).sum()
load = (Vt[:3].T * s[:3] / np.sqrt(len(runs)))
json.dump(dict(keys=keys, runs=runs, X=X.tolist(), rho=rho.tolist(), order=[int(i) for i in order], ev=ev.tolist(), load=load.tolist()), open(f"{S}/corr.json", "w"), indent=1)
print("order", kk)
print("explained", np.round(ev[:4], 3), "cum3", round(float(ev[:3].sum()), 3))
for k, l in zip(keys, load):
    print(f"{k:24s}", np.round(l, 2))
for a, b in [("Hearing (P1 shift)", "Acknowledges error"), ("Turn (P2 \u22a5)", "Lie success"), ("Hearing (P1 shift)", "Lie success"), ("Truth accuracy", "Lie success"),
             ("Paradox rate", "Re-sample baseline"), ("Fabricates a number", "Lie success"), ("Lie/truth length", "Acknowledges error")]:
    print(a, "|", b, round(float(rho[keys.index(a), keys.index(b)]), 2))
