"""Appendix tables: behaviour by model and outcome, PCA loadings, and the items with four distinct outcomes."""
import csv, json
import numpy as np

S = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
P = "C:/Users/nehad/Desktop/llm lies/paper/tables"
B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
BS = json.load(open(f"{S}/behav_summary.json"))
NAME = {"llama": "Llama-8B", "qwen": "Qwen-3B", "gemma": "Gemma-2B", "qmath": "Qwen-Math", "all": "All"}
CM = {"C1": "\\lie", "C2": "\\leak", "C3": "\\ign", "C4": "\\para"}
pct = lambda v: "--" if v is None else f"{100 * v:.0f}"
f2 = lambda v: "--" if v is None else f"{v:.2f}"
out = []
for m in ("llama", "qwen", "gemma", "qmath", "all"):
    for i, c in enumerate(("C1", "C2", "C3", "C4")):
        x = BS[f"{m}|{c}"]
        lead = f"\\multirow{{4}}{{*}}{{{NAME[m]}}}" if i == 0 else ""
        out.append(f"{lead} & {CM[c]} & {x['n']} & {pct(x['fab'])} & {f2(x['onset_med'])} & {pct(x['truth_false_eq'])} / {pct(x['false_eq'])} & "
                   f"{f2(x['bold_med'])} & {pct(x['premise'])} & {pct(x['opswap'])} & {pct(x['omit'])} & {pct(x['meta'])} & {pct(x['expose'])} & "
                   f"{pct(x['corr'])} & {f2(x['sim_med'])} & {x['len_ratio']:.2f} \\\\")
    out.append("\\midrule" if m != "all" else "")
open(f"{P}/behav.tex", "w", encoding="utf-8").write("\n".join(out) + "\n")

C = json.load(open(f"{S}/corr.json"))
ev = C["ev"]; lines = []
for k, l in zip(C["keys"], C["load"]):
    j = int(np.argmax(np.abs(l)))
    cells = [(f"\\textbf{{{v:.2f}}}" if i == j else f"{v:.2f}").replace("-", "$-$") for i, v in enumerate(l)]
    lines.append(f"{k.replace('\u2225', '$\\parallel$').replace('\u22a5', '$\\perp$')} & " + " & ".join(cells) + f" & {sum(v * v for v in l):.2f} \\\\")
lines.append("\\midrule")
lines.append("Explained variance & " + " & ".join(f"{100 * v:.1f}\\%" for v in ev[:3]) + f" & {100 * sum(ev[:3]):.1f}\\% \\\\")
open(f"{P}/pca.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

R = {"GSM8K": {"Llama": "Lamma/gsm8k/llama_gsm8k", "Qwen-3B": "Qwen/GSM-8K/Qwen2.5-3B_gsm8k", "Gemma": "GEMMA-2-9B-IT/gsm8k/gemma2b_gsm8k", "Qwen-Math": "QWEN-MATH/GSM8K/qwen15bmath_gsm8k"},
     "MATH500": {"Llama": "Lamma/Math500/llama_math500", "Qwen-3B": "Qwen/Math 500/qwen2.5-3b_instruct_math500", "Gemma": "GEMMA-2-9B-IT/Math500/gemma2b_Math500", "Qwen-Math": "QWEN-MATH/Math500/qwen15bmath_math500"},
     "Synthetic": {"Llama": "Lamma/Synthetic/llama_synthetic", "Qwen-3B": "Qwen/Synthetic/Qwen2.5-3b-Syntehtic", "Gemma": "GEMMA-2-9B-IT/Synthetic/gemma2b_synthetic", "Qwen-Math": "QWEN-MATH/synthetic/qwen15b_MATH_SYNTHETIC_results/qwen15b_MATH_SYNTHETIC"}}
items = []
for ds, rr in R.items():
    L = {m: {r["qid"]: r for r in csv.DictReader(open(f"{B}/{p}/labels.csv", encoding="utf-8-sig"))} for m, p in rr.items()}
    Q = {json.loads(l)["qid"]: json.loads(l)["question"] for l in open(f"{B}/{rr['Llama']}/items.jsonl", encoding="utf-8")}
    for q in L["Llama"]:
        cs = [L[m][q]["case"] for m in rr]
        if len(set(cs)) == 4 and all(c in CM for c in cs):
            txt = Q[q].replace("\n", " ")
            txt = (txt[:78] + "\\ldots") if len(txt) > 80 else txt
            for a, b in (("\\", "\\textbackslash "), ("$", "\\$"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("^", "\\^{}"), ("{", "\\{"), ("}", "\\}")):
                if a == "\\":
                    txt = txt.replace("\\ldots", "@@L@@").replace(a, b).replace("@@L@@", "\\ldots")
                else:
                    txt = txt.replace(a, b)
            items.append(f"{ds} {q} & {txt} & " + " & ".join(CM[c] for c in cs) + " \\\\")
open(f"{P}/fouritems.tex", "w", encoding="utf-8").write("\n".join(items) + "\n")
print(len(items), "four-outcome items"); print("\n".join(lines))
