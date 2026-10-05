import csv, json, glob, os, re, difflib
import numpy as np

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
G = json.load(open("C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/geom.json"))
META = re.compile(r"incorrect|wrong|mistake|error|flaw", re.I)


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta[1:], (beta / se)[1:]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


cross = []
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d = os.path.dirname(lp)
    run = os.path.relpath(d, B).replace(os.sep, "/")
    if "mist" in run.lower():
        continue
    L = {r["qid"]: r for r in csv.DictReader(open(lp, encoding="utf-8-sig"))}
    gens = {}
    for line in open(f"{d}/generations.jsonl", encoding="utf-8"):
        g = json.loads(line)
        if g["sample"] == -1:
            gens.setdefault(g["qid"], {})[g["cond"]] = g["text"]
    idx = list(csv.DictReader(open(f"{d}/extract_index.csv")))
    qids = [x["qid"] for x in idx]
    case = np.array([x["case"] for x in idx])
    dis = np.array([1 - difflib.SequenceMatcher(None, gens[q]["truth"].split(), gens[q]["lie"].split(), autojunk=False).ratio() for q in qids])
    chg = np.array([L[q]["truth_pred"].strip() != L[q]["lie_pred"].strip() for q in qids])

    h = np.load(f"{d}/hidden.npy", mmap_mode="r")
    N, _, _, NL, D = h.shape
    out = {"run": run}
    for p, pn in ((0, "P1"), (1, "P2")):
        for frac in (0.5, 0.75, 1.0):
            l = max(1, round(frac * (NL - 1)))
            X = np.asarray(h[:, :, p, l, :], dtype=np.float32)
            mu = X.reshape(-1, D).mean(0)
            S = X[:, 1] - X[:, 0]
            nS = np.linalg.norm(S, axis=1)
            rel = nS / np.linalg.norm(X[:, 0] - mu, axis=1)
            tot = S.sum(0)
            U = (tot[None] - S) / (N - 1); U /= np.linalg.norm(U, axis=1, keepdims=True)
            par = (S * U).sum(1)
            orth = np.sqrt(np.maximum(nS ** 2 - par ** 2, 0)) / np.linalg.norm(X[:, 0] - mu, axis=1)
            key = f"{pn}_{frac}"
            out[f"{key}_rel_all"] = float(np.median(rel))
            out[f"{key}_orth_all"] = float(np.median(orth))
            out[f"{key}_rho_rel_dis"] = round(spearman(rel, dis), 3)
            m = (case == "C1") | (case == "C2")
            z = lambda v: (v - v.mean()) / v.std()
            if (case == "C1").sum() >= 5 and (case == "C2").sum() >= 5:
                b, t = ols(z(rel[m]), [z(dis[m]), (case[m] == "C1").astype(float)])
                out[f"{key}_C1C2_reg"] = {"beta_dis": round(b[0], 3), "t_dis": round(t[0], 1), "beta_isC1": round(b[1], 3), "t_isC1": round(t[1], 1)}
                b, t = ols(z(orth[m]), [z(dis[m]), (case[m] == "C1").astype(float)])
                out[f"{key}_C1C2_reg_orth"] = {"beta_isC1": round(b[1], 3), "t_isC1": round(t[1], 1), "t_dis": round(t[0], 1)}
            b, t = ols(z(rel), [z(dis), chg.astype(float)])
            out[f"{key}_all_reg"] = {"beta_dis": round(b[0], 3), "t_dis": round(t[0], 1), "beta_chg": round(b[1], 3), "t_chg": round(t[1], 1)}
    # behaviour summary for cross-run comparison
    lie_txt = [gens[q]["lie"] for q in L]
    out["meta%"] = 100 * np.mean([bool(META.search(t)) for t in lie_txt])
    beh = G[run]["behaviour"]
    out["ignored%"] = 100 * beh["strategy_all"].get("S4_ignored", 0) / 500
    n1 = (np.array([r["case"] for r in L.values()]) == "C1").sum(); n2 = (np.array([r["case"] for r in L.values()]) == "C2").sum()
    out["lie_success%"] = 100 * n1 / (n1 + n2)
    out["dis_median_C1"] = float(np.median(dis[case == "C1"])) if (case == "C1").any() else None
    out["dis_median_C2"] = float(np.median(dis[case == "C2"])) if (case == "C2").any() else None
    cross.append(out)
    print(json.dumps({k: v for k, v in out.items()}, default=float))

json.dump(cross, open("C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/geom2.json", "w"), indent=1, default=float)

# cross-run correlations
def col(k): return np.array([c[k] for c in cross], float)
for a, bk in (("P1_0.5_rel_all", "meta%"), ("P1_0.5_rel_all", "ignored%"), ("P1_0.5_rel_all", "lie_success%"), ("P2_1.0_orth_all", "lie_success%"), ("P2_1.0_rel_all", "lie_success%")):
    print("CROSS", a, "vs", bk, "pearson", round(float(np.corrcoef(col(a), col(bk))[0, 1]), 3), "spearman", round(spearman(col(a), col(bk)), 3))
