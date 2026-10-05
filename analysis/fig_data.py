import csv, glob, os, json
import numpy as np

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
os.makedirs(OUT, exist_ok=True)
KEEP_VECTORS = {"GEMMA-2-9B-IT/gsm8k/gemma2b_gsm8k", "Lamma/gsm8k/llama_gsm8k", "Lamma/Math500/llama_math500", "Qwen/GSM-8K/Qwen2.5-3B_gsm8k"}

meta = {}
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d = os.path.dirname(lp)
    run = os.path.relpath(d, B).replace(os.sep, "/")
    if "mist" in run.lower():
        continue
    L = {r["qid"]: r for r in csv.DictReader(open(lp, encoding="utf-8-sig"))}
    idx = list(csv.DictReader(open(f"{d}/extract_index.csv")))
    qids = [x["qid"] for x in idx]
    case = np.array([x["case"] for x in idx])
    changed = np.array([L[q]["truth_pred"].strip() != L[q]["lie_pred"].strip() for q in qids])
    h = np.load(f"{d}/hidden.npy", mmap_mode="r")
    N, _, _, NL, D = h.shape
    l = NL - 1  # last layer, P2
    X = np.asarray(h[:, :, 1, l, :], dtype=np.float32)
    mu = X.reshape(-1, D).mean(0)
    T, Lh = X[:, 0], X[:, 1]
    S = Lh - T
    nTc = np.linalg.norm(T - mu, axis=1)
    tot = S.sum(0)
    U = (tot[None] - S) / (N - 1)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    par = (S * U).sum(1) / nTc
    nS = np.linalg.norm(S, axis=1) / nTc
    orth = np.sqrt(np.maximum(nS ** 2 - par ** 2, 0))
    name = run.split("/")[-1]
    np.savez(f"{OUT}/{name}.npz", par=par, orth=orth, rel=nS, case=case, changed=changed)
    if run in KEEP_VECTORS:
        # vectors for 3D views: relative shift vectors, plus common direction
        Srel = S / nTc[:, None]
        u = tot / np.linalg.norm(tot)
        np.savez(f"{OUT}/{name}_vec.npz", S=Srel.astype(np.float32), u=u.astype(np.float32), case=case, changed=changed)
    meta[name] = {"run": run, "layer": int(l), "n_layers": int(NL - 1), "n": int(N)}
    print("ok", name, N, flush=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
