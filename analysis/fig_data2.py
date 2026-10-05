import csv, glob, os
import numpy as np
B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/figdata"
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d = os.path.dirname(lp); run = os.path.relpath(d, B).replace(os.sep, "/")
    if "mist" in run.lower(): continue
    name = run.split("/")[-1]
    h = np.load(f"{d}/hidden.npy", mmap_mode="r"); N, _, _, NL, D = h.shape
    X = np.asarray(h[:, :, 1, NL - 1, :], dtype=np.float32)
    mu = X.reshape(-1, D).mean(0)
    S = (X[:, 1] - X[:, 0]) / np.linalg.norm(X[:, 0] - mu, axis=1)[:, None]
    tot = S.sum(0); U = (tot[None] - S) / (N - 1); U /= np.linalg.norm(U, axis=1, keepdims=True)
    par = (S * U).sum(1)
    R = S - par[:, None] * U                      # orthogonal residual (exact |R| = orth)
    Rc = R - R.mean(0)
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    p = R @ Vt[:2].T                               # azimuth only from top-2 PCs of the residual
    az = np.arctan2(p[:, 1], p[:, 0])
    old = dict(np.load(f"{OUT}/{name}.npz", allow_pickle=True))
    old["az"] = az
    np.savez(f"{OUT}/{name}.npz", **old)
    print("ok", name)
