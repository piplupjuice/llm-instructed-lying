"""Robustness analyses for the paper: probe confound test, bistability, strict re-scoring."""
import csv, glob, json, os, re
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/paper_analysis.json"


def to_num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def gold_in_text(gold, text):
    g = to_num(gold)
    if g is not None:
        if g == int(g) and abs(g) < 10:
            return None
        nums = {to_num(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))}
        return any(n is not None and abs(n - g) < 1e-9 for n in nums)
    gs = re.sub(r"\s+", "", str(gold))
    return None if len(gs) < 2 else gs in re.sub(r"\s+", "", text)


def probe(n_train=10**6):
    k = int(max(2, min(48, 0.7 * n_train - 2)))
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))


res = {}
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d = os.path.dirname(lp); run = os.path.relpath(d, B).replace(os.sep, "/")
    if "mist" in run.lower():
        continue
    name = run.split("/")[-1]
    L = {r["qid"]: r for r in csv.DictReader(open(lp, encoding="utf-8-sig"))}
    gens = {}
    for line in open(f"{d}/generations.jsonl", encoding="utf-8"):
        g = json.loads(line)
        if g["sample"] == -1:
            gens.setdefault(g["qid"], {})[g["cond"]] = g["text"]
    idx = list(csv.DictReader(open(f"{d}/extract_index.csv")))
    qids = [x["qid"] for x in idx]
    case = np.array([x["case"] for x in idx])
    chg = np.array([L[q]["truth_pred"].strip() != L[q]["lie_pred"].strip() for q in qids])
    gin = np.array([gold_in_text(L[q]["gold"], gens[q]["lie"]) for q in qids], dtype=object)
    R = {}

    # ---- strict re-scoring: a lie only counts if the gold answer never appears in the lie output
    allc = np.array([r["case"] for r in L.values()])
    n1, n2 = (allc == "C1").sum(), (allc == "C2").sum()
    c1_strict = ((case == "C1") & (gin == False)).sum()
    c1_reveal = ((case == "C1") & (gin == True)).sum()
    c1_unknown = ((case == "C1") & np.array([g is None for g in gin])).sum()
    R["strict"] = dict(lie_success=float(n1 / (n1 + n2)), strict_lie_success=float(c1_strict / (n1 + n2)),
                       c1=int(n1), c1_strict=int(c1_strict), c1_reveal=int(c1_reveal), c1_unknown=int(c1_unknown))

    # ---- bistability: P(gold first token) under the lie prompt
    sm = {}
    for line in open(f"{d}/softmax.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("valid") and s["cond"] == "lie":
            sm.setdefault(s["case"], []).append(s["p_gold_first"])
    allp = np.concatenate([np.array(v) for v in sm.values()])
    R["bistable"] = dict(frac_middle_all=float(np.mean((allp > 0.1) & (allp < 0.9))), n=int(len(allp)),
                         by_case={k: dict(n=len(v), frac_lt_0_1=float(np.mean(np.array(v) < 0.1)), frac_gt_0_9=float(np.mean(np.array(v) > 0.9))) for k, v in sm.items()})

    # ---- probes: C1 vs C2 on lie-prompt states, truth-prompt control, and transfer to C3/C4
    h = np.load(f"{d}/hidden.npy", mmap_mode="r"); N, _, _, NL, D = h.shape
    m12 = (case == "C1") | (case == "C2")
    y = (case[m12] == "C1").astype(int)
    R["probe"] = {}
    if min(y.sum(), (1 - y).sum()) >= 15:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for pos, pname in ((0, "P1"), (1, "P2")):
            for frac in (0.5, 1.0):
                l = max(1, round(frac * (NL - 1)))
                Xl = np.asarray(h[:, 1, pos, l, :], dtype=np.float32)
                Xt = np.asarray(h[:, 0, pos, l, :], dtype=np.float32)
                p_cv = cross_val_predict(probe(), Xl[m12], y, cv=skf, method="predict_proba")[:, 1]
                p_ctrl = cross_val_predict(probe(), Xt[m12], y, cv=skf, method="predict_proba")[:, 1]
                ent = dict(auroc=float(roc_auc_score(y, p_cv)), auroc_truth_control=float(roc_auc_score(y, p_ctrl)))
                # transfer: fit on all C1/C2, score the items it never saw
                clf = probe().fit(Xl[m12], y)
                for gname, gm in (("C3_same", (case == "C3") & ~chg), ("C3_changed", (case == "C3") & chg), ("C4", case == "C4")):
                    if gm.sum() >= 5:
                        ent[f"pC1_{gname}"] = float(clf.predict_proba(Xl[gm])[:, 1].mean())
                        ent[f"n_{gname}"] = int(gm.sum())
                ent["pC1_C1_cv"] = float(p_cv[y == 1].mean()); ent["pC1_C2_cv"] = float(p_cv[y == 0].mean())
                # answer-changed vs unchanged among C3 only (no lying involved at all)
                g3 = case == "C3"
                if g3.sum() >= 20 and min(chg[g3].sum(), (~chg[g3]).sum()) >= 8:
                    k = min(5, int(min(chg[g3].sum(), (~chg[g3]).sum())))
                    p3 = cross_val_predict(probe(int(g3.sum() * (k - 1) / k)), Xl[g3], chg[g3].astype(int), cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=0), method="predict_proba")[:, 1]
                    ent["auroc_C3_changed_vs_same"] = float(roc_auc_score(chg[g3].astype(int), p3))
                R["probe"][f"{pname}_{frac}"] = ent
    res[name] = R
    print("ok", name, json.dumps(R["strict"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
