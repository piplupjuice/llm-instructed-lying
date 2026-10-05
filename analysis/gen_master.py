"""Generate the colour-coded master results table (tables/master.tex) from the run outputs."""
import csv, json, os, re
import numpy as np
from scipy.stats import binom

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
S = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
P = "C:/Users/nehad/Desktop/llm lies/paper"
RUNS = [("llama_syn", "Lamma/Synthetic/llama_synthetic"), ("llama_gsm", "Lamma/gsm8k/llama_gsm8k"), ("llama_math", "Lamma/Math500/llama_math500"),
        ("qwen_syn", "Qwen/Synthetic/Qwen2.5-3b-Syntehtic"), ("qwen_gsm", "Qwen/GSM-8K/Qwen2.5-3B_gsm8k"), ("qwen_math", "Qwen/Math 500/qwen2.5-3b_instruct_math500"),
        ("gemma_syn", "GEMMA-2-9B-IT/Synthetic/gemma2b_synthetic"), ("gemma_gsm", "GEMMA-2-9B-IT/gsm8k/gemma2b_gsm8k"), ("gemma_math", "GEMMA-2-9B-IT/Math500/gemma2b_Math500"),
        ("qmath_syn", "QWEN-MATH/synthetic/qwen15b_MATH_SYNTHETIC_results/qwen15b_MATH_SYNTHETIC"), ("qmath_gsm", "QWEN-MATH/GSM8K/qwen15bmath_gsm8k"), ("qmath_math", "QWEN-MATH/Math500/qwen15bmath_math500")]
G = json.load(open(f"{S}/geom.json")); G2 = {c["run"]: c for c in json.load(open(f"{S}/geom2.json"))}
PA = json.load(open(f"{S}/paper_analysis.json")); PC = json.load(open(f"{S}/percase.json")); BS = json.load(open(f"{S}/behav_summary.json"))
rows = json.load(open(f"{S}/behav_rows.json"))
V = {}
for sh, rp in RUNS:
    d = f"{B}/{rp}"; name = os.path.basename(rp)
    L = list(csv.DictReader(open(f"{d}/labels.csv", encoding="utf-8-sig")))
    c = np.array([r["case"] for r in L]); n = {k: int((c == k).sum()) for k in ("C1", "C2", "C3", "C4")}
    lc = np.array([int(r["luck_correct"]) if r["luck_n"] not in ("", "0") else -1 for r in L])
    w = ((c == "C3") | (c == "C4")) & (lc >= 0)
    kb = float(np.mean(lc[w] / 2)); pr = n["C4"] / (n["C3"] + n["C4"])
    below = binom.cdf(n["C4"], n["C3"] + n["C4"], kb) < 0.05 / 12
    st = G[rp]["behaviour"]["strategy_all"]; tot = sum(st.values())
    geo = G[rp]["geometry"]["P2"][-1]
    summ = open(f"{d}/summary.txt", encoding="utf-8").read()
    m = re.search(r"probe C1 vs C2 at P2: best AUROC ([\d.]+) .*?truth-prompt control ([\d.]+)", summ)
    R = [r for r in rows if r["run"] == sh]
    c1 = [r for r in R if r["case"] == "C1"]
    V[sh] = {
        "acc": float(re.search(r"truth_accuracy \(answered items only\)\s+([\d.]+)%", summ).group(1)),
        "ls": 100 * n["C1"] / (n["C1"] + n["C2"]), "sls": 100 * PA[name]["strict"]["strict_lie_success"],
        "s1": 100 * st.get("S1_committed_lie", 0) / tot, "s3": 100 * st.get("S3_narrated_unapplied", 0) / tot, "s4": 100 * st.get("S4_ignored", 0) / tot,
        "fab": 100 * np.mean([r["fab"] for r in R]), "onset": float(np.median([r["onset"] for r in c1 if r["onset"] is not None])),
        "len": G[rp]["behaviour"]["lie_chars_median"] / G[rp]["behaviour"]["truth_chars_median"],
        "pr": 100 * pr, "kb": 100 * kb, "below": below, "n4": n["C4"],
        "k1": PC[rp]["C1"].get("luck"), "k2": PC[rp]["C2"].get("luck"),
        "hear": G2[rp]["P1_0.5_rel_all"],
        "par1": geo["C1"]["par_c"], "par2": geo["C2"]["par_c"], "orth1": geo["C1"]["orth_c"], "orth2": geo["C2"]["orth_c"],
        "cos": geo.get("cos_meanshift_C1_C2"), "probe": (float(m.group(1)), float(m.group(2))) if m else None,
        "bi": 100 * PA[name]["bistable"]["frac_middle_all"],
    }
json.dump({k: {a: (b if not isinstance(b, (np.bool_,)) else bool(b)) for a, b in v.items()} for k, v in V.items()}, open(f"{S}/master.json", "w"), indent=1, default=float)

runs = [r for r, _ in RUNS]
PINK, CYAN, YEL, LAV = "pinkx", "cyanx", "yelx", "lavx"


def row(label, key, fmt, hi="max", flag=None, extra=None, sym=""):
    """hi: 'max' marks max (pink) and min (cyan); 'top' marks max only; None marks nothing."""
    vals = [V[r][key] for r in runs]
    num = [v for v in vals if v is not None]
    mx, mn = (max(num), min(num)) if num else (None, None)
    cells = []
    for r, v in zip(runs, vals):
        if v is None:
            cells.append("--"); continue
        s = fmt(v) if extra is None else extra(r, v)
        col = None
        if flag and flag(r):
            col = flag(r)
        elif hi and v == mx:
            col = PINK
        elif hi == "max" and v == mn:
            col = CYAN
        cells.append(f"\\cellcolor{{{col}}}{s}" if col else s)
    return f"{label} & " + " & ".join(cells) + " \\\\"


f0 = lambda v: f"{v:.0f}"
f2 = lambda v: f"{v:.2f}"
lines = [
    r"\multirow{8}{*}{\rotatebox{90}{\textbf{Behaviour}}}",
    "& " + row("Truth accuracy (\\%)", "acc", f0),
    "& " + row("Lie success, LS (\\%)", "ls", f0),
    "& " + row("Strict LS (\\%)", "sls", lambda v: f"{v:.0f}", flag=lambda r: LAV if V[r]["sls"] < V[r]["ls"] / 3 else None),
    "& " + row("S1 commit (\\%)", "s1", f0, hi="top"),
    "& " + row("S3 narrate (\\%)", "s3", f0, hi="top"),
    "& " + row("S4 ignore (\\%)", "s4", f0, hi="top"),
    "& " + row("Fabricates a number (\\%)", "fab", f0),
    "& " + row("Lie onset, C1 (rel.\\ pos.)", "onset", f2),
    r"\midrule",
    r"\multirow{4}{*}{\rotatebox{90}{\textbf{Knowl.}}}",
    "& " + row("Paradox rate (\\%)", "pr", f0, flag=lambda r: YEL if V[r]["below"] else None),
    "& " + row("Re-sample baseline (\\%)", "kb", f0),
    "& " + row("$\\bar k$ of \\lie", "k1", f2),
    "& " + row("$\\bar k$ of \\leak", "k2", f2),
    r"\midrule",
    r"\multirow{6}{*}{\rotatebox{90}{\textbf{Geometry}}}",
    "& " + row("Hearing: P1 shift", "hear", f2),
    "& " + row("Shared push \\pll, \\lie", "par1", f2, hi=None),
    "& " + row("Shared push \\pll, \\leak", "par2", f2, hi=None),
    "& " + row("Turn \\prp, \\lie", "orth1", f2),
    "& " + row("Turn \\prp, \\leak", "orth2", f2),
    "& " + row("cos(mean \\lie, mean \\leak)", "cos", f2, flag=lambda r: None),
    r"\midrule",
    r"\multirow{2}{*}{\rotatebox{90}{\textbf{Dec.}}}",
    "& " + row("Probe AUROC, P2 (control)", "probe", None, hi=None, extra=lambda r, v: f"{v[0]:.2f} ({v[1]:.2f})"),
    "& " + row("Bistable mass (\\%)", "bi", lambda v: f"{v:.1f}"),
]
os.makedirs(f"{P}/tables", exist_ok=True)
open(f"{P}/tables/master.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
