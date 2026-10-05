import csv, json, glob, os, re, math
import numpy as np

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/geom.json"
rng = np.random.default_rng(0)


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def mwu(x, y):
    """two-sided Mann-Whitney (normal approx). returns (rank-biserial r: >0 means x>y, p)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return None, None
    r = rankdata(np.concatenate([x, y]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sd if sd else 0
    p = math.erfc(abs(z) / math.sqrt(2))
    return round(2 * u1 / (n1 * n2) - 1, 3), p


def to_num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def gold_in_text(gold, text):
    g = to_num(gold)
    if g is not None:
        if g == int(g) and abs(g) < 10:
            return None  # single digit: too likely by chance
        nums = {to_num(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))}
        return any(n is not None and abs(n - g) < 1e-9 for n in nums)
    gs = re.sub(r"\s+", "", str(gold))
    if len(gs) < 2:
        return None
    return gs in re.sub(r"\s+", "", text)


META = re.compile(r"incorrect|wrong|mistake|error|flaw", re.I)


def strategy(final_is_gold, gin, text):
    if final_is_gold:
        return "S3_narrated_unapplied" if META.search(text) else "S4_ignored"
    if gin is None:
        return "S?_wrong_unknown"
    return "S2_wrong_but_reveals_gold" if gin else "S1_committed_lie"


res = {}
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

    # ---------- behaviour, all 500 items ----------
    beh = {"openings": {}, "strategy_by_case": {}, "strategy_all": {}}
    opens = {}
    for q, r in L.items():
        t = gens.get(q, {}).get("lie", "")
        o = " ".join(re.sub(r"[#*]", "", t).split()[:6])
        opens[o] = opens.get(o, 0) + 1
    beh["openings"] = sorted(opens.items(), key=lambda kv: -kv[1])[:3]
    lie_texts = [gens.get(q, {}).get("lie", "") for q in L]
    beh["has_correct_approach_header%"] = round(100 * np.mean([bool(re.search(r"correct approach|correct solution", t, re.I)) for t in lie_texts]), 1)
    beh["has_post_disclaimer%"] = round(100 * np.mean([bool(re.search(r"why this is (incorrect|wrong)|this is (not|incorrect)|not the correct", t, re.I)) for t in lie_texts]), 1)
    beh["boxed>=2%"] = round(100 * np.mean([t.count("boxed") >= 2 for t in lie_texts]), 1)
    beh["lie_truncated%"] = round(100 * np.mean([r["lie_truncated"] == "True" for r in L.values()]), 1)
    beh["lie_chars_median"] = int(np.median([len(t) for t in lie_texts]))
    beh["truth_chars_median"] = int(np.median([len(gens.get(q, {}).get("truth", "")) for q in L]))

    # per valid item
    strat, gin_flag, ans_changed = [], [], []
    for q in qids:
        r = L[q]
        t = gens[q]["lie"]
        lie_ok = r["lie_correct"] == "True"
        gi = gold_in_text(r["gold"], t)
        strat.append(strategy(lie_ok, gi, t))
        gin_flag.append(gi)
        ans_changed.append(r["truth_pred"].strip() != r["lie_pred"].strip())
    strat = np.array(strat)
    ans_changed = np.array(ans_changed)
    for c in ["C1", "C2", "C3", "C4"]:
        m = case == c
        if m.sum():
            u, n = np.unique(strat[m], return_counts=True)
            beh["strategy_by_case"][c] = {k: int(v) for k, v in zip(u, n)}
    u, n = np.unique(strat, return_counts=True)
    beh["strategy_all"] = {k: int(v) for k, v in zip(u, n)}

    groups = {
        "C1": case == "C1",
        "C2": case == "C2",
        "C3_same_ans": (case == "C3") & ~ans_changed,
        "C3_diff_ans": (case == "C3") & ans_changed,
        "C4": case == "C4",
        "C1_committed": (case == "C1") & (strat == "S1_committed_lie"),
        "C1_reveals": (case == "C1") & (strat == "S2_wrong_but_reveals_gold"),
        "C2_narrated": (case == "C2") & (strat == "S3_narrated_unapplied"),
        "C2_ignored": (case == "C2") & (strat == "S4_ignored"),
        "changed": ans_changed,
        "unchanged": ~ans_changed,
    }
    beh["group_n"] = {k: int(v.sum()) for k, v in groups.items()}

    # ---------- geometry ----------
    h = np.load(f"{d}/hidden.npy", mmap_mode="r")  # N, cond(truth,lie), pos(P1,P2), layer, D
    N, _, _, NL, D = h.shape
    geo = {}
    for p, pname in ((0, "P1"), (1, "P2")):
        per = []
        for l in range(1, NL):
            X = np.asarray(h[:, :, p, l, :], dtype=np.float32)
            T, Lh = X[:, 0], X[:, 1]
            mu = X.reshape(-1, D).mean(0)
            S = Lh - T
            nTc = np.linalg.norm(T - mu, axis=1)
            nT = np.linalg.norm(T, axis=1)
            nS = np.linalg.norm(S, axis=1)
            tot = S.sum(0)
            U = (tot[None, :] - S) / (N - 1)  # leave-one-out mean shift
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            par = (S * U).sum(1)
            orth = np.sqrt(np.maximum(nS ** 2 - par ** 2, 0))
            cosu = par / nS
            cos_tl = ((T - mu) * (Lh - mu)).sum(1) / (nTc * np.linalg.norm(Lh - mu, axis=1))
            row = {"depth": round(l / (NL - 1), 3)}
            met = {"rel_c": nS / nTc, "rel_raw": nS / nT, "par_c": par / nTc, "orth_c": orth / nTc, "cos_lie_dir": cosu, "cos_truth_lie": cos_tl}
            for gname, m in groups.items():
                if m.sum() >= 1:
                    row[gname] = {k: round(float(np.median(v[m])), 4) for k, v in met.items()}
            for k in ("rel_c", "par_c", "orth_c", "cos_lie_dir"):
                row[f"C1vsC2_{k}"] = mwu(met[k][groups["C1"]], met[k][groups["C2"]])
                row[f"chg_vs_unchg_{k}"] = mwu(met[k][groups["changed"]], met[k][groups["unchanged"]])
                row[f"C3diff_vs_C3same_{k}"] = mwu(met[k][groups["C3_diff_ans"]], met[k][groups["C3_same_ans"]])
                row[f"C1commit_vs_C1reveal_{k}"] = mwu(met[k][groups["C1_committed"]], met[k][groups["C1_reveals"]])
            # direction similarity of mean shifts, with split-half ceiling
            m1, m2 = groups["C1"], groups["C2"]
            if m1.sum() >= 4 and m2.sum() >= 4:
                a, b = S[m1].mean(0), S[m2].mean(0)
                row["cos_meanshift_C1_C2"] = round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 4)
                ceil = []
                for big in (m1, m2):
                    ix = np.where(big)[0]
                    vals = []
                    for _ in range(20):
                        pp = rng.permutation(ix)
                        h1, h2 = S[pp[: len(pp) // 2]].mean(0), S[pp[len(pp) // 2:]].mean(0)
                        vals.append(h1 @ h2 / np.linalg.norm(h1) / np.linalg.norm(h2))
                    ceil.append(float(np.mean(vals)))
                row["splithalf_C1"], row["splithalf_C2"] = round(ceil[0], 4), round(ceil[1], 4)
                # magnitude of mean shifts
                row["norm_meanshift_C1_over_C2"] = round(float(np.linalg.norm(a) / np.linalg.norm(b)), 3)
            per.append(row)
        geo[pname] = per
    res[run] = {"behaviour": beh, "geometry": geo, "n_layers": NL - 1}
    print("done", run, flush=True)

json.dump(res, open(OUT, "w"), indent=1)
