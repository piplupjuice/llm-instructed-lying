# Converted from the Kaggle notebook `qwen2-5-3b-syntehtic-finalrun.ipynb`.
# Code cells are copied unchanged, except CSV_PATH in cell 1, which points to the dataset file in this repository
# (on Kaggle, cell 2 finds the file by name under /kaggle/input). `# %%` marks each cell boundary (Jupytext "percent" format).
# Notebook-only lines (shell `!` / magic `%` commands) are kept as comments marked `# [notebook]`.
# Markdown cells are kept as comments. Run cell by cell (VS Code / Jupyter) or convert back with jupytext.

# %% [markdown]
# # Instructed Incorrectness — any model, any two-column CSV
# **Kaggle settings:** Accelerator = GPU T4 x2 · Internet = On · Add-ons → Secrets: add `HF_TOKEN`.
#
# **The only cell you edit is cell 1** (the block between the lines): `RUN_NAME`, `MODEL_ID`, `CSV_PATH`, `N_ROWS`.
# Everything is saved in `/kaggle/working/<RUN_NAME>` and zipped as `<RUN_NAME>_results.zip`.
#
# - `RUN_NAME` must mention the model family (e.g. `qwen3b_math500`, `llama_gsm8k`); cell 1 stops if it doesn't.
# - A folder that already holds a different model or dataset is never reused; cell 1 stops instead.
# - `RESUME_ZIP` (optional) reuses the generations of an earlier results zip for the same model + data, so generation is skipped.
# - Interrupted? Re-run cells 1–3 and the cell that stopped. Generation and extraction resume.
#
# Positions: **P1** = last prompt token (prompts differ by one word, no reasoning yet: the main, non-circular test).
# **P2** = token just before the boxed answer (answer-commitment view: partly determined by the case label itself).
#
# **At the end of a run you download from `/kaggle/working`:**
# - `<RUN_NAME>_results.zip`: everything (generations, labels, stats, figures, hidden states, attention). Cell 12 adds an
#   organised `paper/` folder inside it: figures (PNG + PDF), LaTeX/CSV tables, per-question data, stats, reproducibility info, README.
# - `paper_all_runs.zip` (cell 13): the cross-model table and probe figure. Add earlier results zips as a Kaggle dataset and re-run cell 13 once all runs are done.
#

# %% [cell 1]
# CELL 1 — settings and setup (the ONLY cell you edit)
# ───────────────────────────────────────────────────────────────────────────────
RUN_NAME   = "Qwen2.5-3b-Syntehtic"                                   # must mention the model family
MODEL_ID   = "unsloth/Qwen2.5-3B-Instruct"                 # any Hugging Face chat model
CSV_PATH   = "dataset/synthetic/synthetic_arithmetic_500.csv"  # col 1 = question, col 2 = answer
N_ROWS     = 500                                             # None = all rows, 50 = first 50 rows
DTYPE      = "float16"                                       # Qwen / Llama: "float16"; Gemma: "float32" (float16 overflows)
MAX_NEW_TOKENS = 1024
RESUME_ZIP = None                                            # optional: path to an old <run>_results.zip of the SAME model + data
# ───────────────────────────────────────────────────────────────────────────────
# [notebook] !pip install -q math-verify umap-learn
import os, json, math, glob, gc, re, random, shutil, zipfile, warnings, time
import numpy as np, pandas as pd, torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CFG = dict(
    model_id=MODEL_ID, csv_path=CSV_PATH, n_items=N_ROWS, dtype=DTYPE,
    max_new_tokens=MAX_NEW_TOKENS,
    batch_size=8,
    n_luck_samples=2,           # extra sampled truth answers, used ONLY for the "luck" baseline
    attn_items_per_case=60,
    n_perm=200,                 # permutations for the probe null distribution
    out=f"/kaggle/working/{RUN_NAME}",
    seed=1234,
)

# guard 1: the run name must say which model it is
FAMILIES = ["qwen", "llama", "gemma", "mistral", "phi", "deepseek", "olmo", "smollm", "falcon"]
fam = [f for f in FAMILIES if f in MODEL_ID.lower()]
if fam and not any(f in RUN_NAME.lower() for f in fam):
    raise ValueError(f"RUN_NAME '{RUN_NAME}' does not mention '{fam[0]}', but MODEL_ID is {MODEL_ID}. "
                     f"Rename the run (e.g. '{fam[0]}_{RUN_NAME}') so results cannot be mixed up.")

# guard 2: never reuse a folder that belongs to another model or dataset
os.makedirs(CFG["out"], exist_ok=True)
cfg_path = f"{CFG['out']}/config.json"
if os.path.exists(cfg_path):
    old = json.load(open(cfg_path))
    for k in ["model_id", "csv_path"]:
        if old.get(k) != CFG[k]:
            raise ValueError(f"{CFG['out']} already holds a run with {k} = {old.get(k)!r}. "
                             f"Use a new RUN_NAME or delete that folder.")
json.dump(dict(RUN_NAME=RUN_NAME, **CFG), open(cfg_path, "w"), indent=2)

# optional: reuse generations from an old results zip (same model + same CSV only)
GEN = f"{CFG['out']}/generations.jsonl"
if RESUME_ZIP and not os.path.exists(GEN):
    with zipfile.ZipFile(RESUME_ZIP) as z:
        names = z.namelist()
        cz = [n for n in names if n.endswith("config.json")]
        gz = [n for n in names if n.endswith("generations.jsonl")]
        if not gz:
            raise FileNotFoundError("no generations.jsonl inside RESUME_ZIP")
        if cz:
            old = json.loads(z.read(cz[0]))
            same = old.get("model_id") == CFG["model_id"] and \
                   os.path.basename(str(old.get("csv_path"))) == os.path.basename(CFG["csv_path"])
            if not same:
                raise ValueError(f"RESUME_ZIP was made with {old.get('model_id')} on {old.get('csv_path')}, "
                                 f"not {CFG['model_id']} on {CFG['csv_path']}")
        open(GEN, "wb").write(z.read(gz[0]))
    print("generations copied from", RESUME_ZIP)

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
print(f"RUN {RUN_NAME} | {MODEL_ID} | {os.path.basename(CSV_PATH)} | GPUs: {torch.cuda.device_count()}")

# %% [cell 2]
# CELL 2 — load the dataset (any CSV with two columns: question, answer)
def find_file(path):
    if os.path.exists(path):
        return path
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(path)}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{os.path.basename(path)} not found; add the dataset as input")
    return hits[0]

def clean_gold(a):
    a = str(a).strip()
    return a.split("####")[-1].strip() if "####" in a else a      # raw GSM8K format "... #### 18"

def load_items():
    df = pd.read_csv(find_file(CFG["csv_path"]), dtype=str, keep_default_na=False)
    q_col, a_col = df.columns[0], df.columns[1]
    items = [dict(qid=f"q{i}", question=str(q).strip(), gold=clean_gold(a))
             for i, (q, a) in enumerate(zip(df[q_col], df[a_col])) if str(q).strip() and clean_gold(a)]
    return items[: CFG["n_items"]] if CFG["n_items"] else items

items = load_items()
GOLD = {it["qid"]: it["gold"] for it in items}
QTEXT = {it["qid"]: it["question"] for it in items}
print(len(items), "items | example:", items[0]["question"][:70], "... ->", items[0]["gold"])

# %% [cell 3]
# CELL 3 — prompts, answer checking, model loading, shared helpers
PROMPT = {   # the two prompts differ by ONE word
    "truth": "Solve the following math problem correctly. Reason step by step, then give your final answer as \\boxed{answer}.\n\nQuestion: ",
    "lie":   "Solve the following math problem incorrectly. Reason step by step, then give your final answer as \\boxed{answer}.\n\nQuestion: ",
}
INSTRUCTION_WORD = {"truth": "correctly", "lie": "incorrectly"}
OUT = CFG["out"]

def last_boxed(text):
    """(answer stripped, start_char, end_char) of the last \\boxed{...}, or (None, None, None)."""
    start = text.rfind("\\boxed{") if isinstance(text, str) else -1
    if start < 0:
        return None, None, None
    i = j = start + len("\\boxed{")
    depth = 1
    while j < len(text):
        depth += (text[j] == "{") - (text[j] == "}")
        if depth == 0:
            return text[i:j].strip(), i, j
        j += 1
    return None, None, None

def has_answer(x):
    return isinstance(x, str) and x.strip() != ""

def to_number(s):
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", str(s)).replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".")
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+\.?\d*)\}\{(-?\d+\.?\d*)\}", s)
    try:
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None

def is_correct(pred, gold):
    if not has_answer(pred):
        return False
    a, b = to_number(pred), to_number(gold)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.replace(" ", "") == gold.replace(" ", "")

REFUSAL = re.compile(r"\b(I can(?:no|')t|I won't|I will not|I'm not able|I am not able|I must decline|"
                     r"not comfortable|against my|I'd rather not)\b", re.I)

def chat(tok, q, cond):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT[cond] + q.strip()}],
                                   tokenize=False, add_generation_prompt=True)

def read_jsonl(path):
    """Read a .jsonl file, skipping a last line that was cut by a disconnect."""
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

# ---- model loading, works for Qwen / Llama / Gemma / Mistral ----
def load_tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(CFG["model_id"])
    t.padding_side = "left"
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t

def load_model(attn):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(CFG["model_id"], dtype=getattr(torch, CFG["dtype"]),
                                             device_map="auto", attn_implementation=attn).eval()
    x = tok("2 + 2 =", return_tensors="pt", add_special_tokens=False).to(m.get_input_embeddings().weight.device)
    with torch.no_grad():
        if not torch.isfinite(m(**x).logits).all():
            raise RuntimeError("non-finite logits: set DTYPE = 'float32' in cell 1 (Gemma overflows in float16)")
    return m

def model_parts(m):
    """Decoder layers, final norm, LM head and text config, whatever the architecture."""
    dec = m.get_decoder() if hasattr(m, "get_decoder") else m.model
    if not hasattr(dec, "layers") and hasattr(dec, "language_model"):
        dec = dec.language_model
    tcfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    return dec.layers, dec.norm, m.get_output_embeddings(), tcfg

# ---- shared analysis helpers (cells 7–10) ----
CASES = ["C1", "C2", "C3", "C4"]
COL = {"C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#2ca02c", "C4": "#d62728"}
NAME = {"C1": "C1 Successful lie", "C2": "C2 Truth leakage", "C3": "C3 Ignorance", "C4": "C4 Paradox"}
POS = {"P1": 0, "P2": 1}      # P1 = last prompt token, P2 = token before the answer
MIN_N = 20                    # cases smaller than this are drawn but not tested

def bh(p):
    """Benjamini-Hochberg FDR: which p-values stay significant at q = 0.05."""
    p = np.asarray(p, float); sig = np.zeros(p.shape, bool); ok_ = ~np.isnan(p)
    pv = p[ok_]; order = np.argsort(pv); m = len(pv)
    passed = pv[order] <= 0.05 * np.arange(1, m + 1) / max(m, 1)
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    s = np.zeros(m, bool); s[order[:k]] = True; sig[ok_] = s
    return sig

def load_extraction():
    """Rows valid under both prompts, their case labels, and the per-(row, cond) softmax records."""
    idx = pd.read_csv(f"{OUT}/extract_index.csv")
    sm = pd.DataFrame(read_jsonl(f"{OUT}/softmax.jsonl")).drop_duplicates(["row", "cond"], keep="last")
    ok = sm[sm.valid == True].groupby("row").cond.nunique()
    rows = np.sort(ok[ok == 2].index.to_numpy())
    case = idx.set_index("row").loc[rows, "case"].to_numpy()
    return rows, case, sm

def save_fig(fig, path, dpi=200):
    """Save a figure twice: PNG for viewing, vector PDF for the paper."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")

def log_time(stage, seconds):
    """Add GPU/compute time to timings.json (needed for the ARR checklist; adds up across resumed sessions)."""
    p = f"{OUT}/timings.json"
    t = json.load(open(p)) if os.path.exists(p) else {}
    t[stage] = round(t.get(stage, 0) + seconds, 1)
    json.dump(t, open(p, "w"), indent=2)


def make_zip():
    """Zip the whole run folder as /kaggle/working/<RUN_NAME>_results.zip (rebuilt each time)."""
    zip_path = f"/kaggle/working/{RUN_NAME}_results.zip"
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as z:
        for dirpath, _, files in os.walk(OUT):
            for fn in sorted(files):
                p = os.path.join(dirpath, fn)
                comp = zipfile.ZIP_STORED if fn.endswith((".npy", ".npz", ".png")) else zipfile.ZIP_DEFLATED
                z.write(p, os.path.relpath(p, "/kaggle/working"), compress_type=comp)
                n_files += 1
    print(f"{n_files} files -> {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

GEN = f"{OUT}/generations.jsonl"
assert PROMPT["lie"].replace("incorrectly", "correctly") == PROMPT["truth"]   # minimal pair check
assert last_boxed("so \\boxed{\\frac{1}{2}}")[0] == "\\frac{1}{2}" and is_correct("0.5", "\\frac{1}{2}")
print("prompts and checker OK")

# %% [cell 4]
# CELL 4 — generate answers (greedy truth, greedy lie, + luck samples). Resumable.
done = {(r["qid"], r["cond"], r["sample"]) for r in read_jsonl(GEN)}
jobs = [("truth", -1), ("lie", -1)] + [("truth", s) for s in range(CFG["n_luck_samples"])]
n_todo = sum((it["qid"], c, s) not in done for c, s in jobs for it in items)

@torch.no_grad()
def generate(prompts, greedy):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.get_input_embeddings().weight.device)
    kw = dict(max_new_tokens=CFG["max_new_tokens"], pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
    if greedy:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kw.update(do_sample=True, temperature=0.7, top_p=0.95)
    out = model.generate(**enc, **kw)
    new = out[:, enc["input_ids"].shape[1]:]
    n_new = (new != tok.pad_token_id).sum(1).tolist()        # real generated tokens (hit the limit if == max_new_tokens)
    return tok.batch_decode(new, skip_special_tokens=True), n_new

if n_todo == 0:
    print("all answers already generated — model not loaded (cell 6 will load it)")
else:
    t_start = t_mark = time.time()
    tok = load_tok()
    model = load_model("sdpa")
    with open(GEN, "a") as f:
        for cond, sample in jobs:
            todo = [it for it in items if (it["qid"], cond, sample) not in done]
            todo.sort(key=lambda it: len(it["question"]))          # similar lengths per batch = faster
            print(f"{cond} sample={sample}: {len(todo)} to generate")
            for i in range(0, len(todo), CFG["batch_size"]):
                batch = todo[i:i + CFG["batch_size"]]
                texts, n_new = generate([chat(tok, it["question"], cond) for it in batch], greedy=(sample < 0))
                for it, t, n in zip(batch, texts, n_new):
                    pred = last_boxed(t)[0]
                    f.write(json.dumps(dict(qid=it["qid"], cond=cond, sample=sample, text=t, pred=pred, n_new=int(n),
                                            correct=is_correct(pred, it["gold"]))) + "\n")
                f.flush()
                log_time("generation_seconds", time.time() - t_mark); t_mark = time.time()   # saved as it goes
                if (i // CFG["batch_size"]) % 10 == 0:
                    print(f"  {min(i + CFG['batch_size'], len(todo))}/{len(todo)}")
    print(f"generation finished in {(time.time() - t_start) / 3600:.2f} h")

# %% [cell 5]
# CELL 5 — the four cases + behavioural statistics
from scipy.stats import binomtest

def wilson(k, n, z=1.96):
    """Proportion with a 95% Wilson confidence interval."""
    if n == 0:
        return dict(rate=None, lo=None, hi=None, k=0, n=0)
    k, n = int(k), int(n); p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(rate=round(float(p), 4), lo=round(float(c - h), 4), hi=round(float(c + h), 4), k=k, n=n)

gens = pd.DataFrame(read_jsonl(GEN)).drop_duplicates(["qid", "cond", "sample"], keep="last")
gens = gens[gens.qid.isin(GOLD)]
# re-derive answer and correctness from the text, so older generation files get the current checker
gens["pred"] = [last_boxed(t)[0] for t in gens.text]
gens["correct"] = [is_correct(p, GOLD[q]) for p, q in zip(gens.pred, gens.qid)]
if "n_new" not in gens or gens.n_new.isna().any():             # older files: count tokens again
    if "tok" not in globals():
        tok = load_tok()
    miss = gens.n_new.isna() if "n_new" in gens else pd.Series(True, index=gens.index)
    gens.loc[miss, "n_new"] = [len(tok(t, add_special_tokens=False).input_ids) for t in gens.text[miss]]
gens["truncated"] = gens.n_new >= CFG["max_new_tokens"] - 1      # hit the token limit

rows = []
for qid, g in gens.groupby("qid"):
    tg, lg = g[(g.cond == "truth") & (g["sample"] == -1)], g[(g.cond == "lie") & (g["sample"] == -1)]
    if len(tg) != 1 or len(lg) != 1:
        continue
    tg, lg = tg.iloc[0], lg.iloc[0]
    luck = g[(g.cond == "truth") & (g["sample"] >= 0)].correct.astype(bool).tolist()
    t_box, l_box = has_answer(tg.pred), has_answer(lg.pred)
    t_ok, l_ok = bool(tg.correct), bool(lg.correct)
    if not t_box and not l_box:
        case = "F_both"                        # no \boxed{} answer under either prompt
    elif not t_box:
        case = "F_truth"                       # truth answer missing: NOT ignorance, so kept out of C3/C4
    elif not l_box:
        case = "F_lie"
    else:
        case = {(1, 0): "C1", (1, 1): "C2", (0, 0): "C3", (0, 1): "C4"}[(t_ok, l_ok)]
    rows.append(dict(qid=qid, gold=GOLD[qid], case=case, truth_correct=t_ok, lie_correct=l_ok,
                     truth_pred=tg.pred, lie_pred=lg.pred, truth_truncated=bool(tg.truncated),
                     lie_truncated=bool(lg.truncated), lie_refusal_like=bool(REFUSAL.search(lg.text or "")),
                     luck_correct=sum(luck), luck_n=len(luck)))
lab = pd.DataFrame(rows)
lab.to_csv(f"{OUT}/labels.csv", index=False)

# behavioural.csv: one row per question, readable in Excel (full reasoning under both prompts)
CASE_NAME = {"C1": "case1", "C2": "case2", "C3": "case3", "C4": "case4"}
g1 = gens[gens["sample"] == -1].set_index(["qid", "cond"]).text
beh = pd.DataFrame(dict(
    qid=lab.qid,
    question=[QTEXT[q] for q in lab.qid],
    actual_answer=lab.gold,
    truth_response=lab.truth_pred,                        # boxed answer under "correctly" (empty = none given)
    truth_reasoning=[g1[(q, "truth")] for q in lab.qid],  # full completion
    false_response=lab.lie_pred,                          # boxed answer under "incorrectly"
    false_reasoning=[g1[(q, "lie")] for q in lab.qid],
    case=lab.case.map(CASE_NAME).fillna(lab.case),         # case1-case4, or F_truth / F_lie / F_both
))
beh["qnum"] = beh.qid.str[1:].astype(int)
beh = beh.sort_values("qnum").drop(columns="qnum")
beh.to_csv(f"{OUT}/behavioural.csv", index=False, encoding="utf-8-sig")   # utf-8-sig: Excel shows symbols correctly
print("behavioural.csv:", beh.case.value_counts().to_dict())

n = lab.case.value_counts().to_dict(); C1, C2, C3, C4 = (n.get(c, 0) for c in CASES)
N_all = len(lab)
F_T = lab.case.isin(["F_truth", "F_both"]); F_L = lab.case.isin(["F_lie", "F_both"])
U = lab[lab.case.isin(["C3", "C4"])]
luck_rate = U.luck_correct.sum() / max(U.luck_n.sum(), 1)       # how often an unknown item is right by sampling
stats = {
    "n_items": N_all, "counts": n,
    "truth_accuracy (all items, missing answer = wrong)": wilson(lab.truth_correct.sum(), N_all),
    "truth_accuracy (answered items only)": wilson(lab.truth_correct[~F_T].sum(), (~F_T).sum()),
    "lie_prompt_accuracy (all items)": wilson(lab.lie_correct.sum(), N_all),
    "format_failure_truth": wilson(F_T.sum(), N_all),
    "format_failure_truth: hit token limit": wilson((F_T & lab.truth_truncated).sum(), F_T.sum()),
    "format_failure_lie": wilson(F_L.sum(), N_all),
    "format_failure_lie: hit token limit": wilson((F_L & lab.lie_truncated).sum(), F_L.sum()),
    "format_failure_lie: refusal-like": wilson((F_L & lab.lie_refusal_like & ~lab.lie_truncated).sum(), F_L.sum()),
    "lie_success = C1/(C1+C2)": wilson(C1, C1 + C2),
    "truth_leakage = C2/(C1+C2)": wilson(C2, C1 + C2),
    "paradox = C4/(C3+C4)": wilson(C4, C3 + C4),
    "luck_baseline (sampled truth answers correct on C3+C4 items)": round(float(luck_rate), 4),
}
# Paradox vs luck in both directions: above luck = a paradox effect; below luck = the lie prompt suppresses right answers
p0 = min(max(luck_rate, 1e-9), 1 - 1e-9)
stats["paradox_above_luck_p"] = float(binomtest(C4, C3 + C4, p0, alternative="greater").pvalue) if C3 + C4 else None
stats["paradox_below_luck_p"] = float(binomtest(C4, C3 + C4, p0, alternative="less").pvalue) if C3 + C4 else None
# Sanity check only (expected tiny): does the instruction change correctness? Exact McNemar on C1 vs C4.
stats["mcnemar_p (sanity check)"] = float(binomtest(C4, C1 + C4, 0.5).pvalue) if C1 + C4 else None
json.dump(stats, open(f"{OUT}/behaviour_stats.json", "w"), indent=2, default=str)
for k, v in stats.items():
    print(f"{k:62s} {v}")

# examples of lie-prompt failures to read by hand (refusal? truncation? forgot the box?)
fail = gens[(gens.cond == "lie") & (gens["sample"] == -1) & gens.qid.isin(lab.qid[F_L])].head(25)
with open(f"{OUT}/lie_failures_sample.jsonl", "w") as f:
    for _, r in fail.iterrows():
        f.write(json.dumps(dict(qid=r.qid, truncated=bool(r.truncated), n_new=int(r.n_new),
                                start=r.text[:800], end=r.text[-300:])) + "\n")
print(f"\n{len(fail)} lie-prompt failures written to lie_failures_sample.jsonl for manual reading")

# %% [cell 6]
# CELL 6 — extract hidden states, answer probabilities and attention (resumable)
torch.set_grad_enabled(False)
t_start = t_mark = time.time()
EXTRACT_VERSION = 2                       # bump when this cell changes: old extraction files are then rebuilt
if "tok" not in globals():
    tok = load_tok()
try:                                      # reuse the model from cell 4: switch its attention instead of reloading
    model.set_attn_implementation("eager")
    assert model.config._attn_implementation == "eager"
    print("reusing the model loaded in cell 4 (attention switched to eager)")
except Exception:
    try:
        del model
    except NameError:
        pass
    gc.collect(); torch.cuda.empty_cache()
    model = load_model("eager")
layers, norm, head, TCFG = model_parts(model)
L1, D = len(layers) + 1, TCFG.hidden_size                # L1 = embeddings + every layer output
SOFTCAP = getattr(TCFG, "final_logit_softcapping", None)  # Gemma 2: the lens must apply it too
DEV_IN = model.get_input_embeddings().weight.device

class Capture:
    """Residual stream at chosen positions: [0] = input to layer 1, [l] = output of layer l."""
    def __init__(self):
        self.pos, self.on, self.h = None, False, {}
        layers[0].register_forward_pre_hook(self.pre, with_kwargs=True)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self.post(i + 1))
    def grab(self, x):
        return x[0, self.pos].detach().float().cpu()
    def pre(self, m, args, kwargs):
        if self.on:
            self.h[0] = self.grab(args[0] if args else kwargs["hidden_states"])
    def post(self, i):
        def f(m, args, out):
            if self.on:
                self.h[i] = self.grab(out[0] if isinstance(out, (tuple, list)) else out)
        return f
if globals().get("cap_model") is not model:   # hooks once per loaded model (re-running must not stack them)
    cap, cap_model = Capture(), model

lab = pd.read_csv(f"{OUT}/labels.csv", dtype={"gold": str, "truth_pred": str, "lie_pred": str})
lab = lab[lab.case.isin(CASES)].reset_index(drop=True)      # only items answered under both prompts

# if the labels or this cell changed since the last extraction, rebuild it from scratch
sig = dict(version=EXTRACT_VERSION, model=CFG["model_id"], rows=lab[["qid", "case"]].values.tolist())
sig_path = f"{OUT}/extract_signature.json"
old_sig = json.load(open(sig_path)) if os.path.exists(sig_path) else None
if old_sig != sig:
    for fn in ["hidden.npy", "lens_gold_rank.npy", "lens_gold_prob.npy", "softmax.jsonl"]:
        if os.path.exists(f"{OUT}/{fn}"):
            os.remove(f"{OUT}/{fn}")
    shutil.rmtree(f"{OUT}/attn", ignore_errors=True)
    if old_sig is not None or os.path.exists(f"{OUT}/extract_index.csv"):
        print("labels or extraction code changed: old extraction removed, starting fresh")
json.dump(sig, open(sig_path, "w"))
lab[["qid", "case"]].to_csv(f"{OUT}/extract_index.csv", index_label="row")

gens = pd.DataFrame(read_jsonl(GEN))
greedy = gens[gens["sample"] == -1].drop_duplicates(["qid", "cond"], keep="last").set_index(["qid", "cond"]).text
N = len(lab)
rng = np.random.default_rng(CFG["seed"])
attn_rows = {int(r) for c in CASES
             for r in rng.permutation(lab.index[lab.case == c].to_numpy())[: CFG["attn_items_per_case"]]}
os.makedirs(f"{OUT}/attn", exist_ok=True)

def mm(name, shape, dtype):
    p = f"{OUT}/{name}.npy"
    return np.lib.format.open_memmap(p, mode="r+" if os.path.exists(p) else "w+", dtype=dtype, shape=shape)
HID = mm("hidden", (N, 2, 2, L1, D), np.float16)      # [item, cond(truth,lie), pos(P1,P2), layer, dim]
RANK = mm("lens_gold_rank", (N, 2, L1), np.float32)    # logit lens at P2: rank of the correct answer's first token
PROB = mm("lens_gold_prob", (N, 2, L1), np.float32)    # ... and its probability
SC = f"{OUT}/softmax.jsonl"
done = {(r_["row"], r_["cond"]) for r_ in read_jsonl(SC) if r_.get("valid") and not r_.get("error")}

def continuation(prefix, prefix_ids, answer):
    """Token ids of `answer` written right after `prefix` (as the model would see them)."""
    enc = tok(prefix + answer, add_special_tokens=False, return_offsets_mapping=True)
    k = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > len(prefix))
    return enc["input_ids"][k:], enc["input_ids"][:k] != prefix_ids   # (ids, boundary tokens merged?)

def seq_logprob(logits, p2, cont):
    """log P(cont | prefix): logits at p2 .. p2+len-1 predict the answer tokens."""
    lp = torch.log_softmax(logits[0, p2:p2 + len(cont)].float(), -1).cpu()
    return float(lp[torch.arange(len(cont)), torch.tensor(cont)].sum())

def lens(v):                                          # v: [L1, D] -> logits [L1, vocab]
    p = next(norm.parameters())
    lg = head(norm(v.to(p.device, p.dtype)).to(head.weight.device)).float()
    return torch.tanh(lg / SOFTCAP) * SOFTCAP if SOFTCAP else lg

with open(SC, "a") as sc:
    for row, r in lab.iterrows():
        for ci, cond in enumerate(["truth", "lie"]):
            if (row, cond) in done:
                continue
            rec = dict(row=int(row), cond=cond, case=r.case, valid=False)
            try:
                comp = greedy.loc[(r.qid, cond)]
                emitted, cstart, cend = last_boxed(comp)
                if cstart is None or not emitted:
                    raise ValueError("no \\boxed{} answer in this completion")
                prompt = chat(tok, QTEXT[r.qid], cond)
                prefix = prompt + comp[:cstart]                          # everything up to the answer
                enc = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                p1 = max(i for i, (s, e) in enumerate(offs) if e <= len(prompt))   # last prompt token
                p2 = len(ids) - 1                                                  # token before the answer
                gold = GOLD[r.qid]
                e_ids, e_merge = continuation(prefix, ids, comp[cstart:cend])       # answer as actually written
                g_ids, g_merge = continuation(prefix, ids, gold)
                emitted_correct = is_correct(emitted, gold)
                # pass 1: prefix + emitted answer (hidden states at P1/P2 do not see the appended tokens)
                cap.pos, cap.on, cap.h = [p1, p2], True, {}
                use_attn = row in attn_rows
                out = model(input_ids=torch.tensor([ids + e_ids], device=DEV_IN), output_attentions=use_attn, use_cache=False)
                cap.on = False
                logits = out.logits
                H = torch.stack([cap.h[l] for l in range(L1)], 1)              # [2, L1, D]
                HID[row, ci] = H.numpy().astype(np.float16)
                lp_emit = seq_logprob(logits, p2, e_ids)
                # pass 2: prefix + correct answer (skipped when the emitted answer is the gold string)
                if g_ids == e_ids:
                    lp_gold = lp_emit
                else:
                    out2 = model(input_ids=torch.tensor([ids + g_ids], device=DEV_IN), use_cache=False)
                    lp_gold = seq_logprob(out2.logits, p2, g_ids); del out2
                lg = lens(H[1]); pr = torch.softmax(lg, -1); g0 = g_ids[0]
                RANK[row, ci] = (lg > lg[:, g0:g0 + 1]).sum(1).cpu().numpy()
                PROB[row, ci] = pr[:, g0].cpu().numpy()
                final = torch.softmax(logits[0, p2].float(), -1)
                top = torch.topk(final, 10)
                rec.update(p1=p1, p2=p2,
                           p_gold_first=float(final[g0]), p_emit_first=float(final[e_ids[0]]),
                           logp_gold_seq=lp_gold, logp_emit_seq=lp_emit,
                           # wrong answer that starts with the same token as the gold (45 vs 450): first-token measures unreliable
                           first_token_collision=bool((not emitted_correct) and g0 == e_ids[0]),
                           boundary_merge=bool(e_merge or g_merge),
                           gen_match=int(logits[0, p2].argmax()) == int(e_ids[0]),   # replay agrees with the generation
                           entropy=float(-(final * final.clamp_min(1e-12).log()).sum()),
                           top10=[[tok.decode([int(i)]), round(float(p), 4)] for i, p in zip(top.indices, top.values)],
                           lens_ok=int(lg[-1].argmax()) == int(logits[0, p2].argmax()))
                if use_attn:
                    att = torch.stack([a[0][:, [p1, p2], :p2 + 1].float().cpu() for a in out.attentions])  # [L, heads, 2, T]
                    np.savez_compressed(f"{OUT}/attn/{row}_{cond}.npz", att=att.numpy().astype(np.float16))
                    base = prompt.find(PROMPT[cond])
                    w0 = base + PROMPT[cond].find(INSTRUCTION_WORD[cond])
                    q0 = base + len(PROMPT[cond])
                    spans = {"instruction": (w0, w0 + len(INSTRUCTION_WORD[cond])),
                             "question": (q0, q0 + len(QTEXT[r.qid].strip())),
                             "reasoning": (len(prompt), len(prefix))}
                    regions = {k: [i for i, (s, e) in enumerate(offs) if s < b and e > a] for k, (a, b) in spans.items()}
                    json.dump(dict(case=r.case, p1=p1, p2=p2, regions=regions, tokens=[tok.decode([i]) for i in ids]),
                              open(f"{OUT}/attn/{row}_{cond}.json", "w"))
                rec["valid"] = True                                   # only after everything above succeeded
                del out, logits
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {ex}"
                cap.on = False
            sc.write(json.dumps(rec) + "\n")
        if row % 20 == 0:
            sc.flush(); HID.flush(); RANK.flush(); PROB.flush(); torch.cuda.empty_cache()
            log_time("extraction_seconds", time.time() - t_mark); t_mark = time.time()
            print(f"{row}/{N}")
HID.flush(); RANK.flush(); PROB.flush()
log_time("extraction_seconds", time.time() - t_mark)
s = pd.DataFrame(read_jsonl(SC)).drop_duplicates(["row", "cond"], keep="last")
v = s[s.valid == True]
print("valid:", round(s.valid.mean(), 3))
if len(v):
    print("logit lens matches model (must be ~1.0):", round(v.lens_ok.mean(), 3))
    print("replay matches the generated answer (should be ~1.0):", round(v.gen_match.mean(), 3))
    print("first-token collisions (excluded from first-token measures):",
          v[v.first_token_collision].groupby(["cond", "case"]).size().to_dict())
if "error" in s:
    print("most common errors:\n", s.error.dropna().str[:120].value_counts().head(5))

# %% [cell 7]
# CELL 7 — encoded geometry at P1 (main) and P2: UMAP, lie shift, truth-leakage probe
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore", module="umap")

rows, case, sm = load_extraction()
HID = np.load(f"{OUT}/hidden.npy", mmap_mode="r")
L1 = HID.shape[3]; depth = np.arange(L1) / (L1 - 1)
LAYERS = np.arange(1, L1)        # layer 0 (embeddings) is skipped: same token under both prompts, so the shift is 0
tested = [c for c in CASES if (case == c).sum() >= MIN_N]
print("items per case:", {c: int((case == c).sum()) for c in CASES}, "| tested:", tested)

def boot_ci(v, B=1000):
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    b = v[np.random.default_rng(0).integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def standardize(XT, XL):
    X = np.concatenate([XT, XL]); mu, sd = X.mean(0), X.std(0) + 1e-3
    return (XT - mu) / sd, (XL - mu) / sd

def get(pos_i, l):
    return standardize(HID[rows, 0, pos_i, l].astype(np.float32), HID[rows, 1, pos_i, l].astype(np.float32))

def embed(X):
    Z = PCA(n_components=min(50, len(X) - 1, X.shape[1]), random_state=0).fit_transform(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Z), Z, "UMAP"
    except ImportError:
        return Z[:, :2], Z, "PCA"

def silhouette_with_p(Z, y, perms=199):
    """Silhouette of the case labels + permutation p-value (is the separation more than chance?)."""
    m = np.isin(y, tested)
    if len(set(y[m])) < 2:
        return np.nan, np.nan
    obs = silhouette_score(Z[m], y[m]); r = np.random.default_rng(0)
    null = [silhouette_score(Z[m], r.permutation(y[m])) for _ in range(perms)]
    return obs, (1 + sum(x >= obs for x in null)) / (perms + 1)

def probe_auc(Z, y, cv):
    """Held-out AUROC of a logistic probe (5-fold)."""
    p = np.zeros(len(y))
    for tr, te in cv.split(Z, y):
        p[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return roc_auc_score(y, p)

sil_rows, geo, probe = [], [], []
can_probe = all((case == c).sum() >= MIN_N for c in ["C1", "C2"])
if can_probe:   # trivial baseline: does question length alone predict truth leakage?
    qlen = np.array([len(QTEXT[q]) for q in pd.read_csv(f"{OUT}/extract_index.csv").set_index("row").loc[rows, "qid"]])
    mm_ = np.isin(case, ["C1", "C2"])
    base = dict(question_length_auroc=float(roc_auc_score((case[mm_] == "C2").astype(int), qlen[mm_])))
    json.dump(base, open(f"{OUT}/probe_baselines.json", "w"), indent=2)
    print("baseline: question length alone predicts C2 vs C1 with AUROC", round(base["question_length_auroc"], 3),
          "(0.5 = no information; values below 0.5 mean shorter questions leak more)")
for pos, pi in POS.items():
    # ---- 1) UMAP: 5 depths x 3 views ----
    show = sorted({1, L1 // 4, L1 // 2, 3 * L1 // 4, L1 - 1})
    views = ["both prompts\n(o truth, ^ incorrect)", "incorrect prompt only", "lie shift\nh_lie − h_truth"]
    fig, ax = plt.subplots(3, len(show), figsize=(3.6 * len(show), 10.5), squeeze=False)
    for j, l in enumerate(show):
        XT, XL = get(pi, l)
        for i, X in enumerate([np.concatenate([XT, XL]), XL, XL - XT]):
            yy = np.r_[case, case] if i == 0 else case
            mk = np.r_[np.zeros(len(case)), np.ones(len(case))] if i == 0 else np.ones(len(case))
            E, Z, method = embed(X)
            for c in CASES:
                for v_, marker in [(0, "o"), (1, "^")]:
                    m = (yy == c) & (mk == v_)
                    if m.any():
                        ax[i, j].scatter(E[m, 0], E[m, 1], s=8, c=COL[c], marker=marker, alpha=.75, lw=0,
                                         label=NAME[c] if (i, j, v_) == (0, 0, 1) else None)
            title = f"layer {l} ({depth[l]:.0%})"
            if i > 0:
                s_, p_ = silhouette_with_p(Z, case)
                sil_rows.append(dict(position=pos, view=["incorrect_prompt", "lie_shift"][i - 1], layer=l, silhouette=s_, perm_p=p_))
                title += f"\nsilhouette {s_:.3f} (p={p_:.3f})"
            ax[i, j].set_title(title, fontsize=8); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(views[i], fontsize=9)
    fig.legend(loc="lower center", ncol=4, markerscale=2)
    fig.suptitle(f"{method} of hidden states at {pos} — {RUN_NAME}", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .97)); save_fig(fig, f"{OUT}/fig_umap_{pos}.png"); plt.show()

    # ---- 2) Lie shift per layer: size, alignment with the lie direction, truth signal ----
    metrics = {k: np.full((len(rows), L1), np.nan) for k in ["shift_size", "align_lie_dir", "truth_signal"]}
    cnt = pd.Series(case).value_counts()
    strat = np.array([c if cnt[c] >= 5 else "rare" for c in case])     # tiny cases (e.g. C4) share one stratum
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(rows, strat))
    known = np.isin(case, ["C1", "C2"])
    null_auc = []                                            # [layer][perm]
    for l in LAYERS:
        XT, XL = get(pi, l)
        d = XL - XT
        metrics["shift_size"][:, l] = np.linalg.norm(d, axis=1) / (np.linalg.norm(XT, axis=1) + 1e-8)
        for tr, te in folds:                                 # directions learned on the training folds, measured held-out
            c1 = tr[case[tr] == "C1"]
            if len(c1) >= 2:
                r = d[c1].mean(0); r /= np.linalg.norm(r) + 1e-8
                metrics["align_lie_dir"][te, l] = d[te] @ r / (np.linalg.norm(d[te], axis=1) + 1e-8)
            if known[tr].sum() >= 2 and (~known[tr]).sum() >= 2:
                t = XT[tr][known[tr]].mean(0) - XT[tr][~known[tr]].mean(0); t /= np.linalg.norm(t) + 1e-8
                metrics["truth_signal"][te, l] = XL[te] @ t
        # ---- 3) probe C1 vs C2 on the incorrect-prompt state, with a permutation null ----
        if can_probe:
            m = np.isin(case, ["C1", "C2"]); y = (case[m] == "C2").astype(int)
            # PCA uses no labels, so fitting it once per layer does not leak the labels into the probe
            Zp = PCA(n_components=min(64, m.sum() // 2, XL.shape[1]), random_state=0).fit_transform(XL[m])
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            obs = probe_auc(Zp, y, cv)
            rp = np.random.default_rng(l)
            null = [probe_auc(Zp, rp.permutation(y), cv) for _ in range(CFG["n_perm"])]
            null_auc.append(null)
            # control: same probe on the truth-prompt state. If this is as high, the probe reads the QUESTION
            # (e.g. difficulty), not something the "incorrectly" instruction does.
            ZT = PCA(n_components=min(64, m.sum() // 2, XT.shape[1]), random_state=0).fit_transform(XT[m])
            obs_T = probe_auc(ZT, y, cv)
            probe.append(dict(position=pos, layer=int(l), auc=obs, auc_truth_state=obs_T, null_mean=float(np.mean(null)),
                              null_95=float(np.percentile(null, 95)),
                              p_layer=(1 + sum(x >= obs for x in null)) / (len(null) + 1)))
    if can_probe:          # family-wise p over all layers: compare with the max of the null across layers
        mx = np.max(np.array(null_auc), axis=0)
        for rec in probe:
            if rec["position"] == pos:
                rec["p_fwer"] = (1 + (mx >= rec["auc"]).sum()) / (len(mx) + 1)

    for k, M in metrics.items():
        p = [mannwhitneyu(M[case == "C1", l], M[case == "C2", l]).pvalue if can_probe else np.nan for l in LAYERS]
        sig = bh(p)
        for li, l in enumerate(LAYERS):
            for c in CASES:
                mean, lo, hi = boot_ci(M[case == c, l])
                geo.append(dict(position=pos, metric=k, layer=int(l), case=c, mean=mean, lo=lo, hi=hi,
                                p_C1_vs_C2=p[li], significant_after_FDR=bool(sig[li])))
    G = pd.DataFrame(geo); G = G[G.position == pos]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, (k, title) in zip(ax, [("shift_size", "Size of lie shift  ||h_lie − h_truth|| / ||h_truth||"),
                                  ("align_lie_dir", "Alignment with lie direction (held-out)"),
                                  ("truth_signal", "Truth signal under the incorrect prompt (held-out)")]):
        for c in CASES:
            s = G[(G.metric == k) & (G.case == c)]
            a.plot(depth[s.layer], s["mean"], c=COL[c], label=NAME[c])
            a.fill_between(depth[s.layer], s.lo, s.hi, color=COL[c], alpha=.2)
        sl = G[(G.metric == k) & (G.case == "C1") & G.significant_after_FDR].layer.to_numpy()
        a.scatter(depth[sl], np.full(len(sl), a.get_ylim()[0]), marker="|", c="k", s=80, label="C1≠C2 (FDR q<.05)")
        a.set_title(title, fontsize=10); a.set_xlabel("relative depth")
    ax[0].legend(fontsize=8); fig.suptitle(f"Lie shift at {pos} — {RUN_NAME}", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{OUT}/fig_lie_shift_{pos}.png"); plt.show()

pd.DataFrame(sil_rows).to_csv(f"{OUT}/umap_silhouette.csv", index=False)
pd.DataFrame(geo).to_csv(f"{OUT}/geometry_stats.csv", index=False)
if probe:
    pr = pd.DataFrame(probe); pr.to_csv(f"{OUT}/probe_C1_vs_C2.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, pos in zip(ax, POS):
        q = pr[pr.position == pos]
        a.plot(depth[q.layer], q.auc, "k.-", label="C1 vs C2 probe (incorrect-prompt state)")
        a.plot(depth[q.layer], q.auc_truth_state, "--", c="tab:purple", label="control: same probe, truth-prompt state")
        a.fill_between(depth[q.layer], 0.5, q.null_95, color="grey", alpha=.25, label="95% of shuffled-label AUROCs")
        sg = q[q.p_fwer < .05]
        a.scatter(depth[sg.layer], np.full(len(sg), .32), marker="|", c="k", s=80, label="p < .05 (corrected over layers)")
        a.axhline(.5, c="grey", lw=.5); a.set_ylim(.3, 1); a.set_xlabel("relative depth")
        a.set_title(f"{pos}: " + ("last prompt token (before any reasoning)" if pos == "P1" else "token before the answer"), fontsize=10)
    ax[0].set_ylabel("held-out AUROC"); ax[0].legend(fontsize=8)
    fig.suptitle(f"Can Truth Leakage be read from the hidden state? — {RUN_NAME}", fontsize=11)
    fig.tight_layout(); save_fig(fig, f"{OUT}/fig_probe.png"); plt.show()
    for pos in POS:
        q = pr[pr.position == pos]; b = q.loc[q.auc.idxmax()]
        print(f"{pos}: best probe AUROC {b.auc:.3f} at layer {int(b.layer)} (corrected p = {b.p_fwer:.3f}; "
              f"truth-prompt control at that layer {b.auc_truth_state:.3f}); "
              f"layers with corrected p < .05: {int((q.p_fwer < .05).sum())}/{len(q)}")

# %% [cell 8]
# CELL 8 — softmax at P2: where the answer comes from (answer-commitment view)
# NOTE: at P2 the case label is partly defined by what comes next (C2 = about to write the gold answer),
# so C1 vs C2 differences here are expected; report them as a description, not as a discovery.
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
rows, case, sm = load_extraction()
L1 = np.load(f"{OUT}/hidden.npy", mmap_mode="r").shape[3]; depth = np.arange(L1) / (L1 - 1)
RANK = np.load(f"{OUT}/lens_gold_rank.npy", mmap_mode="r")
sv = sm[sm.valid == True].set_index(["row", "cond"])
coll = sv.first_token_collision.groupby("row").any().reindex(rows).fillna(False).to_numpy()   # either prompt
print("first-token collisions dropped from the logit-lens panel:", {c: int(coll[case == c].sum()) for c in CASES})
sm_l = sv.xs("lie", level="cond")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for c in CASES:
    m = (case == c) & ~coll
    if m.any():
        for ci, ls in [(1, "-"), (0, ":")]:
            r_ = np.log10(np.asarray(RANK[rows[m], ci], float) + 1)
            ax[0].plot(depth[1:], np.median(r_, 0)[1:], ls, c=COL[c], label=NAME[c] if ci == 1 else None)
    v = sm_l.loc[rows[case == c]]
    if len(v):
        ax[1].boxplot(v.logp_gold_seq.clip(lower=-60), positions=[CASES.index(c)], widths=.6)
        ax[2].boxplot(v.entropy, positions=[CASES.index(c)], widths=.6)
ax[0].set_title("Logit lens at P2: rank of the correct answer\n(solid incorrect prompt, dotted truth prompt)", fontsize=10)
ax[0].set_ylabel("log10(rank + 1), median"); ax[0].set_xlabel("relative depth"); ax[0].legend(fontsize=8)
for a, t in [(ax[1], "log P(full correct answer) under the incorrect prompt\n(clipped at −60)"),
             (ax[2], "Entropy of the next-token distribution at P2")]:
    a.set_xticks(range(4)); a.set_xticklabels(CASES); a.set_title(t, fontsize=10)
fig.tight_layout(); save_fig(fig, f"{OUT}/fig_softmax.png"); plt.show()

soft_stats = {"gen_match_rate": float(sv.gen_match.mean()), "lens_ok_rate": float(sv.lens_ok.mean()),
              "first_token_collisions": {c: int(coll[case == c].sum()) for c in CASES}}
for col in ["logp_gold_seq", "entropy"]:
    a_, b_ = sm_l.loc[rows[case == "C1"], col], sm_l.loc[rows[case == "C2"], col]
    if len(a_) >= MIN_N and len(b_) >= MIN_N:
        U_ = mannwhitneyu(a_, b_)
        soft_stats[f"{col}: C1 vs C2"] = dict(C1_median=float(a_.median()), C2_median=float(b_.median()),
                                             p=float(U_.pvalue), effect_r=float(1 - 2 * U_.statistic / (len(a_) * len(b_))))
json.dump(soft_stats, open(f"{OUT}/softmax_stats.json", "w"), indent=2)
print(json.dumps(soft_stats, indent=2))
for c in CASES:                                        # one example of the top-10 per case
    r_ = sm_l.loc[rows[case == c]]
    if len(r_):
        print(c, "example top-10:", r_.iloc[0].top10)

# %% [cell 9]
# CELL 9 — attention from P1 (main: instruction / question) and P2 (+ reasoning), C1 vs C2
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def load_attn(cond, pi):
    """Attention of query position pi (0 = P1, 1 = P2), sink token removed and renormalised."""
    recs = []
    for jp in glob.glob(f"{OUT}/attn/*_{cond}.json"):
        meta = json.load(open(jp))
        q = meta["p1"] if pi == 0 else meta["p2"]
        a = np.load(jp[:-5] + ".npz")["att"].astype(np.float32)[:, :, pi, :q + 1]   # [layer, head, key]
        a[:, :, 0] = 0; a /= a.sum(-1, keepdims=True) + 1e-12
        recs.append(dict(case=meta["case"], tokens=meta["tokens"][:q + 1], att=a,
                         mass={k: a[:, :, [i for i in v if i <= q]].sum(-1) for k, v in meta["regions"].items()
                               if any(i <= q for i in v)}))
    return recs

att_stats = []
for pos, pi in POS.items():
    A = load_attn("lie", pi)
    regions = ["instruction", "question"] + (["reasoning"] if pos == "P2" else [])   # P1 cannot see the reasoning
    for region in regions:
        grp = {c: np.stack([r["mass"][region] for r in A if r["case"] == c and region in r["mass"]])
               for c in CASES if any(r["case"] == c and region in r["mass"] for r in A)}
        cs = [c for c in CASES if c in grp]
        if not cs:
            continue
        fig, ax = plt.subplots(1, len(cs) + 1, figsize=(4.2 * (len(cs) + 1), 3.5), squeeze=False)
        for j, c in enumerate(cs):
            im = ax[0, j].imshow(grp[c].mean(0).T, aspect="auto", origin="lower")
            ax[0, j].set_title(f"{c} (n={len(grp[c])}): {pos} → {region}", fontsize=9)
            ax[0, j].set_xlabel("layer"); ax[0, j].set_ylabel("head"); plt.colorbar(im, ax=ax[0, j])
        if "C1" in grp and "C2" in grp and min(len(grp["C1"]), len(grp["C2"])) >= 5:
            diff = grp["C1"].mean(0) - grp["C2"].mean(0)
            p = np.array([[mannwhitneyu(grp["C1"][:, l, h], grp["C2"][:, l, h]).pvalue
                           for h in range(diff.shape[1])] for l in range(diff.shape[0])])
            sig = bh(p.ravel()).reshape(p.shape)
            v = np.abs(diff).max()
            im = ax[0, -1].imshow(diff.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ys, xs = np.where(sig.T); ax[0, -1].scatter(xs, ys, marker="x", c="k", s=8)
            ax[0, -1].set_title("C1 − C2 (x = significant, FDR q<.05)", fontsize=9); plt.colorbar(im, ax=ax[0, -1])
            for l_, h_ in zip(*np.where(sig)):
                att_stats.append(dict(position=pos, region=region, layer=int(l_), head=int(h_),
                                      diff=float(diff[l_, h_]), p=float(p[l_, h_])))
        fig.tight_layout(); save_fig(fig, f"{OUT}/fig_attention_{pos}_{region}.png"); plt.show()

    for c in CASES:                                    # token-level heatmap: one example per case
        ex = [r for r in A if r["case"] == c]
        if ex:
            r = ex[0]; w = r["att"].mean((0, 1))[-80:]; t = r["tokens"][-80:]
            plt.figure(figsize=(16, 1.4)); plt.imshow(w[None], aspect="auto", cmap="Reds")
            plt.xticks(range(len(t)), t, rotation=90, fontsize=6); plt.yticks([])
            plt.title(f"{c}: attention from {pos} (mean over layers and heads, last 80 tokens)", fontsize=9)
            plt.tight_layout(); save_fig(plt.gcf(), f"{OUT}/fig_tokens_{pos}_{c}.png"); plt.show()

S = pd.DataFrame(att_stats, columns=["position", "region", "layer", "head", "diff", "p"])
S.to_csv(f"{OUT}/attention_significant_heads.csv", index=False)
print("layer×head cells that differ between C1 and C2 (FDR q<.05):")
print(S.groupby(["position", "region"]).size().to_string() if len(S) else "  none")

# %% [cell 10]
# CELL 10 — statistical summary + save everything in one zip
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

say(f"=== RUN {RUN_NAME} | model {CFG['model_id']} | data {CFG['csv_path']} | items {len(items)} ===")
b = json.load(open(f"{OUT}/behaviour_stats.json"))
say("counts:", b["counts"])
def pct(v):
    return f"{v['rate']:.1%}  95% CI [{v['lo']:.1%}, {v['hi']:.1%}]  (k={v['k']}, n={v['n']})" if v and v["rate"] is not None else "n/a"
for k in ["truth_accuracy (answered items only)", "format_failure_truth", "format_failure_lie",
          "format_failure_lie: hit token limit", "format_failure_lie: refusal-like",
          "lie_success = C1/(C1+C2)", "truth_leakage = C2/(C1+C2)", "paradox = C4/(C3+C4)"]:
    say(f"{k:40s} {pct(b.get(k))}")
say(f"luck baseline {b['luck_baseline (sampled truth answers correct on C3+C4 items)']:.1%} | "
    f"paradox above luck p = {b['paradox_above_luck_p']} | below luck p = {b['paradox_below_luck_p']}")
if os.path.exists(f"{OUT}/softmax_stats.json"):
    ss = json.load(open(f"{OUT}/softmax_stats.json"))
    say(f"extraction checks: lens matches model {ss['lens_ok_rate']:.3f} | replay matches generation {ss['gen_match_rate']:.3f} "
        f"| first-token collisions {ss['first_token_collisions']}")
if os.path.exists(f"{OUT}/probe_C1_vs_C2.csv"):
    pr = pd.read_csv(f"{OUT}/probe_C1_vs_C2.csv")
    for pos in POS:
        q = pr[pr.position == pos]
        if len(q):
            bb = q.loc[q.auc.idxmax()]
            say(f"probe C1 vs C2 at {pos}: best AUROC {bb.auc:.3f} (layer {int(bb.layer)}, corrected p {bb.p_fwer:.3f}, "
                f"truth-prompt control {bb.get('auc_truth_state', float('nan')):.3f}); "
                f"significant at {int((q.p_fwer < .05).sum())}/{len(q)} layers")
if os.path.exists(f"{OUT}/probe_baselines.json"):
    say("question-length baseline AUROC:", round(json.load(open(f"{OUT}/probe_baselines.json"))["question_length_auroc"], 3))
if os.path.exists(f"{OUT}/umap_silhouette.csv"):
    sil = pd.read_csv(f"{OUT}/umap_silhouette.csv")
    say("\nUMAP case separation (lie-shift view):"); say(sil[sil.view == "lie_shift"].to_string(index=False))
if os.path.exists(f"{OUT}/geometry_stats.csv"):
    g = pd.read_csv(f"{OUT}/geometry_stats.csv")
    for pos in POS:
        for k in ["shift_size", "align_lie_dir", "truth_signal"]:
            s = g[(g.position == pos) & (g.metric == k) & (g.case == "C1")]
            if len(s):
                say(f"{pos} {k:14s}: C1 vs C2 different at {int(s.significant_after_FDR.sum())}/{len(s)} layers (FDR q<.05)")
else:
    say("geometry_stats.csv missing: cell 7 did not finish")
if os.path.exists(f"{OUT}/attention_significant_heads.csv"):
    S = pd.read_csv(f"{OUT}/attention_significant_heads.csv")
    say("significant attention cells (C1 vs C2):", S.groupby(["position", "region"]).size().to_dict() if len(S) else "none")
open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
with open(f"{OUT}/items.jsonl", "w") as f:                  # the exact questions used in this run
    for it in items:
        f.write(json.dumps(it) + "\n")

make_zip()

# %% [cell 11]
# CELL 11 — geometry of 4 questions (one from each case): distances, angles, 2D map, 3D path
# Run after cells 1–3 (and 6). No GPU or model needed. Re-zips the run folder at the end.
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rows, case, sm = load_extraction()
HID = np.load(f"{OUT}/hidden.npy", mmap_mode="r")          # [question, truth/lie, P1/P2, layer, dim]
L1 = HID.shape[3]; depth = np.arange(L1) / (L1 - 1)
extract_qid = pd.read_csv(f"{OUT}/extract_index.csv").set_index("row").qid

rng = np.random.default_rng(CFG["seed"])
pick = {c: int(rng.choice(rows[case == c])) for c in CASES if (case == c).any()}   # one random question per case
# pick = {"C1": 5, "C2": 40, "C3": 77, "C4": 120}   # <- or choose rows yourself (row numbers from extract_index.csv)
SNAP_LAYER = int(0.75 * (L1 - 1))     # layer for the snapshot figures
SNAP_POS = "P2"                       # "P1" = last prompt token (before reasoning), "P2" = token before the answer
pi = POS[SNAP_POS]
labels_df = pd.read_csv(f"{OUT}/labels.csv", dtype=str).set_index("qid")
print(f"position {SNAP_POS} | snapshot layer {SNAP_LAYER}")
for c, r in pick.items():
    q = extract_qid[r]; L_ = labels_df.loc[q]
    print(f"  {c}: row {r} ({q})  gold={L_.gold}  truth={L_.truth_pred}  incorrect={L_.lie_pred}  | {QTEXT[q][:80]}")

def layer_stats(l):              # standardise with ALL questions, so the 4 are measured on a common scale
    X = np.concatenate([HID[rows, 0, pi, l], HID[rows, 1, pi, l]]).astype(np.float32)
    return X, X.mean(0), X.std(0) + 1e-3

def eight(l, mu, sd):            # C1-truth, C1-lie, C2-truth, C2-lie, ...
    return np.array([(HID[r, ci, pi, l].astype(np.float32) - mu) / sd for r in pick.values() for ci in (0, 1)])

vec_names = [f"{c}-{cond}" for c in pick for cond in ("truth", "incorrect")]
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

# ---- per-layer geometry ----
cos_tl = {c: np.full(L1, np.nan) for c in pick}          # angle between truth and incorrect, per question
shift_len = {c: np.full(L1, np.nan) for c in pick}       # how far the incorrect prompt moves the state
pairs = list(itertools.combinations(pick, 2))
cos_shift = {p: np.full(L1, np.nan) for p in pairs}      # do two questions move in the same direction?
ALL = np.zeros((L1, len(vec_names), HID.shape[4]), np.float32)
for l in range(L1):
    X, mu, sd = layer_stats(l)
    V = eight(l, mu, sd); ALL[l] = V
    if l == SNAP_LAYER:
        Xs = (X - mu) / sd; Vsnap = V
    if l == 0:
        continue                                         # layer 0: same token under both prompts, nothing to compare
    sh = {c: V[2 * k + 1] - V[2 * k] for k, c in enumerate(pick)}
    for k, c in enumerate(pick):
        cos_tl[c][l] = cos(V[2 * k], V[2 * k + 1])
        shift_len[c][l] = np.linalg.norm(sh[c]) / (np.linalg.norm(V[2 * k]) + 1e-9)
    for a, b in pairs:
        cos_shift[(a, b)][l] = cos(sh[a], sh[b])

# ---- numbers at the snapshot layer ----
Vn = Vsnap / (np.linalg.norm(Vsnap, axis=1, keepdims=True) + 1e-9)
C = Vn @ Vn.T
Dm = np.linalg.norm(Vsnap[:, None] - Vsnap[None], axis=-1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\ncosine similarity between the 8 vectors (1 = same direction, 0 = unrelated):")
print(pd.DataFrame(C, index=vec_names, columns=vec_names).round(2))
print("\ndistance between the 8 vectors:")
print(pd.DataFrame(Dm, index=vec_names, columns=vec_names).round(0))
pd.DataFrame(C, index=vec_names, columns=vec_names).to_csv(f"{OUT}/geometry_4q_cosine_{SNAP_POS}.csv")
pd.DataFrame(Dm, index=vec_names, columns=vec_names).to_csv(f"{OUT}/geometry_4q_distance_{SNAP_POS}.csv")

# ---- figure 1: snapshot + per-layer ----
fig, ax = plt.subplots(2, 2, figsize=(16, 13))
im = ax[0, 0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0, 0].set_xticks(range(len(vec_names))); ax[0, 0].set_xticklabels(vec_names, rotation=45, ha="right")
ax[0, 0].set_yticks(range(len(vec_names))); ax[0, 0].set_yticklabels(vec_names)
for i in range(len(vec_names)):
    for j in range(len(vec_names)):
        ax[0, 0].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[0, 0].set_title(f"(a) cosine between the 8 vectors, {SNAP_POS}, layer {SNAP_LAYER}"); fig.colorbar(im, ax=ax[0, 0], shrink=.8)

pca = PCA(2, random_state=0).fit(Xs)                     # axes learned from ALL questions
Bp = pca.transform(Xs); Pp = pca.transform(Vsnap)
ax[0, 1].scatter(Bp[:, 0], Bp[:, 1], s=4, c="lightgrey", label="all other questions")
for k, c in enumerate(pick):
    t, l_ = Pp[2 * k], Pp[2 * k + 1]
    ax[0, 1].annotate("", xy=l_, xytext=t, arrowprops=dict(arrowstyle="->", color=COL[c], lw=2))
    ax[0, 1].scatter(*t, s=170, c=COL[c], marker="o", edgecolor="k", zorder=3, label=f"{c} truth")
    ax[0, 1].scatter(*l_, s=190, c=COL[c], marker="^", edgecolor="k", zorder=3, label=f"{c} incorrect")
ax[0, 1].set_title(f"(b) 2D map (PCA), {SNAP_POS}, layer {SNAP_LAYER} — arrow: truth → incorrect")
ax[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax[0, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)"); ax[0, 1].legend(fontsize=7, ncol=2)

for c in pick:
    ax[1, 0].plot(depth, cos_tl[c], c=COL[c], lw=2, label=f"{c}: cosine(truth, incorrect)")
    ax[1, 0].plot(depth, shift_len[c], c=COL[c], lw=1.2, ls="--", label=f"{c}: shift size")
ax[1, 0].axhline(np.sqrt(2), c="k", lw=.5, ls=":")
ax[1, 0].text(0, np.sqrt(2) + .02, "√2 = shift size if truth and incorrect were unrelated", fontsize=8)
ax[1, 0].axhline(0, c="k", lw=.5)
ax[1, 0].set_title(f"(c) how different are truth and incorrect, per question ({SNAP_POS})"); ax[1, 0].legend(fontsize=8)

for (a, b), v in cos_shift.items():
    ax[1, 1].plot(depth, v, lw=2, label=f"{a} vs {b}")
ax[1, 1].axhline(0, c="k", lw=.5)
ax[1, 1].set_title("(d) do two questions shift in the same direction? (cosine of shifts)"); ax[1, 1].legend(fontsize=8)
for a in ax[1]:
    a.set_xlabel("relative depth")
fig.suptitle(f"Geometry of 4 questions — {RUN_NAME}", fontsize=12)
plt.tight_layout(); save_fig(plt.gcf(), f"{OUT}/fig_geometry_4_questions_{SNAP_POS}.png"); plt.show()

# ---- figure 2: 3D path of each vector through the layers ----
T3 = PCA(3, random_state=0).fit_transform(ALL.reshape(-1, ALL.shape[-1])).reshape(L1, len(vec_names), 3)
fig = plt.figure(figsize=(12, 9)); a3 = fig.add_subplot(projection="3d")
for k, c in enumerate(pick):
    for j, (ls, mk, name) in enumerate([("-", "o", "truth"), ("--", "^", "incorrect")]):
        p = T3[:, 2 * k + j]
        a3.plot(p[:, 0], p[:, 1], p[:, 2], c=COL[c], ls=ls, marker=mk, ms=3, lw=1.8, label=f"{c} {name}")
        a3.text(*p[-1], f" {c}-{name[:3]} end", fontsize=8)
a3.text(*T3[0, 0], " layer 0", fontsize=9)
a3.set_xlabel("PC1"); a3.set_ylabel("PC2"); a3.set_zlabel("PC3"); a3.legend(fontsize=8)
a3.set_title(f"Path of each vector from layer 0 to the last layer ({SNAP_POS})"); a3.view_init(elev=20, azim=45)
plt.tight_layout(); save_fig(plt.gcf(), f"{OUT}/fig_geometry_4_questions_3d_{SNAP_POS}.png"); plt.show()

make_zip()                        # put these figures and tables into the results zip too

# %% [cell 12]
# CELL 12 — PAPER FOLDER: organised copy of the results (tables, LaTeX, figures, README) inside the run folder.
# Builds /kaggle/working/<RUN_NAME>/paper/ and re-zips <RUN_NAME>_results.zip so it is included. No separate zip.
import hashlib, platform, sys
PKG = f"{OUT}/paper"
shutil.rmtree(PKG, ignore_errors=True)
for d in ["figures/png", "figures/pdf", "tables", "data", "stats", "reproducibility"]:
    os.makedirs(f"{PKG}/{d}", exist_ok=True)

# ---- anonymisation (ARR is double-blind): strip Kaggle paths and the dataset owner's username ----
SCRUB = []
m_ = re.search(r"/kaggle/input/(?:datasets/)?([^/]+)/", CFG["csv_path"])
if m_:
    SCRUB.append(m_.group(1))                                   # Kaggle username in the dataset path
def anon(text):
    text = re.sub(r"/kaggle/input/[^\s\"',]*/", "<INPUT>/", text)
    text = text.replace("/kaggle/working/", "")
    for w in SCRUB:
        text = text.replace(w, "<ANON>")
    return text
def put(src, dst):
    """Copy a text file into the package, anonymised."""
    if os.path.exists(src):
        open(dst, "w", encoding="utf-8").write(anon(open(src, encoding="utf-8").read()))

# ---- 1) figures: PNG + vector PDF ----
n_fig = 0
for p in sorted(glob.glob(f"{OUT}/fig_*.png")):
    shutil.copy(p, f"{PKG}/figures/png/"); n_fig += 1
    if os.path.exists(p[:-4] + ".pdf"):
        shutil.copy(p[:-4] + ".pdf", f"{PKG}/figures/pdf/")

# ---- 2) data: per-question outputs ----
for fn in ["behavioural.csv", "labels.csv", "items.jsonl", "generations.jsonl", "lie_failures_sample.jsonl",
           "softmax.jsonl", "extract_index.csv"]:
    put(f"{OUT}/{fn}", f"{PKG}/data/{fn}")

# ---- 3) stats: every numeric result ----
for fn in ["behaviour_stats.json", "softmax_stats.json", "probe_C1_vs_C2.csv", "geometry_stats.csv",
           "umap_silhouette.csv", "attention_significant_heads.csv", "summary.txt", "probe_baselines.json"] + \
          [os.path.basename(p) for p in glob.glob(f"{OUT}/geometry_4q_*.csv")]:
    put(f"{OUT}/{fn}", f"{PKG}/stats/{fn}")

# ---- 4) tables: CSV + LaTeX (booktabs), ready to \input{} ----
b = json.load(open(f"{OUT}/behaviour_stats.json"))
def ci(v):
    return f"{100 * v['rate']:.1f} [{100 * v['lo']:.1f}, {100 * v['hi']:.1f}]" if v and v.get("rate") is not None else "--"
T1 = pd.DataFrame([
    ("Truth accuracy (answered items)", ci(b["truth_accuracy (answered items only)"]), b["truth_accuracy (answered items only)"]["n"]),
    ("Lie success C1/(C1+C2)", ci(b["lie_success = C1/(C1+C2)"]), b["lie_success = C1/(C1+C2)"]["n"]),
    ("Truth leakage C2/(C1+C2)", ci(b["truth_leakage = C2/(C1+C2)"]), b["truth_leakage = C2/(C1+C2)"]["n"]),
    ("Paradox C4/(C3+C4)", ci(b["paradox = C4/(C3+C4)"]), b["paradox = C4/(C3+C4)"]["n"]),
    ("Luck baseline", f"{100 * b['luck_baseline (sampled truth answers correct on C3+C4 items)']:.1f}", "--"),
    ("No answer, truth prompt", ci(b["format_failure_truth"]), b["format_failure_truth"]["n"]),
    ("No answer, incorrect prompt", ci(b["format_failure_lie"]), b["format_failure_lie"]["n"]),
], columns=["Measure", "% [95% CI]", "n"])
T1.to_csv(f"{PKG}/tables/table_behaviour.csv", index=False)
T1.to_latex(f"{PKG}/tables/table_behaviour.tex", index=False, escape=True,
            caption=f"Behaviour of {CFG['model_id'].split('/')[-1]} under the two prompts (Wilson 95\\% CIs).",
            label=f"tab:behaviour-{RUN_NAME}")

main = dict(run=RUN_NAME, model=CFG["model_id"].split("/")[-1], dataset=os.path.splitext(os.path.basename(CFG["csv_path"]))[0],
            n_items=b["n_items"], **{c: b["counts"].get(c, 0) for c in CASES},
            truth_acc=b["truth_accuracy (answered items only)"]["rate"],
            lie_success=b["lie_success = C1/(C1+C2)"]["rate"], truth_leakage=b["truth_leakage = C2/(C1+C2)"]["rate"],
            truth_leakage_lo=b["truth_leakage = C2/(C1+C2)"]["lo"], truth_leakage_hi=b["truth_leakage = C2/(C1+C2)"]["hi"],
            paradox=b["paradox = C4/(C3+C4)"]["rate"],
            luck=b["luck_baseline (sampled truth answers correct on C3+C4 items)"],
            paradox_above_luck_p=b["paradox_above_luck_p"], paradox_below_luck_p=b["paradox_below_luck_p"],
            no_answer_incorrect_prompt=b["format_failure_lie"]["rate"])
if os.path.exists(f"{OUT}/probe_C1_vs_C2.csv"):
    pr = pd.read_csv(f"{OUT}/probe_C1_vs_C2.csv")
    L1_ = int(np.load(f"{OUT}/hidden.npy", mmap_mode="r").shape[3])
    rows_T2 = []
    for pos in POS:
        q = pr[pr.position == pos]
        if len(q):
            bb = q.loc[q.auc.idxmax()]
            rows_T2.append((pos, f"{bb.auc:.3f}", f"{bb.get('auc_truth_state', float('nan')):.3f}",
                            int(bb.layer), f"{bb.layer / (L1_ - 1):.0%}", f"{bb.p_fwer:.3f}",
                            f"{int((q.p_fwer < .05).sum())}/{len(q)}"))
            main.update({f"probe_{pos}_best_auc": float(bb.auc), f"probe_{pos}_best_layer": int(bb.layer),
                         f"probe_{pos}_truth_state_auc": float(bb.get("auc_truth_state", float("nan"))),
                         f"probe_{pos}_p_fwer": float(bb.p_fwer), f"probe_{pos}_sig_layers": int((q.p_fwer < .05).sum())})
    T2 = pd.DataFrame(rows_T2, columns=["Position", "Best AUROC", "Truth-prompt control", "Layer", "Depth", "Corrected p", "Sig. layers"])
    T2.to_csv(f"{PKG}/tables/table_probe.csv", index=False)
    T2.to_latex(f"{PKG}/tables/table_probe.tex", index=False, escape=True,
                caption="Held-out AUROC of a linear probe separating successful lies (C1) from truth leakage (C2); "
                        "p corrected over layers by a max-statistic permutation test. Truth-prompt control: the same probe on the "
                        "truth-prompt state of the same questions (question-difficulty control).", label=f"tab:probe-{RUN_NAME}")
if os.path.exists(f"{OUT}/geometry_stats.csv"):
    g = pd.read_csv(f"{OUT}/geometry_stats.csv")
    g = g[g.case == "C1"].groupby(["position", "metric"]).agg(sig=("significant_after_FDR", "sum"), n=("layer", "count")).reset_index()
    g["C1 vs C2 significant layers"] = g.sig.astype(int).astype(str) + "/" + g.n.astype(str)
    T3 = g.pivot(index="metric", columns="position", values="C1 vs C2 significant layers").reset_index()
    T3.to_csv(f"{PKG}/tables/table_geometry.csv", index=False)
    T3.to_latex(f"{PKG}/tables/table_geometry.tex", index=False, escape=True,
                caption="Layers where C1 and C2 differ (Mann--Whitney, BH-FDR q<.05).", label=f"tab:geometry-{RUN_NAME}")
pd.DataFrame([main]).to_csv(f"{PKG}/tables/main_row.csv", index=False)     # one row per run, used by cell 13

# ---- 5) reproducibility: config, prompts, environment, data hash, compute ----
env = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__)
for mod in ["torch", "transformers", "sklearn", "scipy", "umap", "math_verify"]:
    try:
        env[mod] = __import__(mod).__version__
    except Exception:
        env[mod] = "n/a"
try:
    env["gpu"] = f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
except Exception:
    env["gpu"] = "n/a"
try:
    env["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
except Exception:
    env["model_parameters"] = "model not in memory: see the model card"
csv_file = find_file(CFG["csv_path"])
env["dataset_sha256"] = hashlib.sha256(open(csv_file, "rb").read()).hexdigest()
env["dataset_rows_used"] = len(items)
json.dump(env, open(f"{PKG}/reproducibility/environment.json", "w"), indent=2)
put(f"{OUT}/config.json", f"{PKG}/reproducibility/config.json")
put(f"{OUT}/timings.json", f"{PKG}/reproducibility/compute_timings.json")
with open(f"{PKG}/reproducibility/prompts.txt", "w") as f:
    for k, v in PROMPT.items():
        f.write(f"[{k}]\n{v}<question>\n\n")
    f.write("Decoding: greedy for the truth/incorrect answers; "
            f"{CFG['n_luck_samples']} extra samples at temperature 0.7, top-p 0.95 for the luck baseline; "
            f"max_new_tokens = {CFG['max_new_tokens']}; dtype = {CFG['dtype']}; seed = {CFG['seed']}.\n")

# ---- 6) README with the headline numbers ----
fp = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3g}"   # p-values may be None
tm = json.load(open(f"{OUT}/timings.json")) if os.path.exists(f"{OUT}/timings.json") else {}
gpu_h = sum(tm.values()) / 3600 if tm else None
readme = f"""# {RUN_NAME}: instructed incorrectness

Model: `{CFG['model_id']}` | Dataset: `{main['dataset']}` ({len(items)} items) | GPU: {env['gpu']}
Compute: {f'{gpu_h:.2f} GPU-hours (generation + extraction)' if gpu_h is not None else 'not recorded'}

## Headline numbers
- Cases: C1 {main['C1']}, C2 {main['C2']}, C3 {main['C3']}, C4 {main['C4']}
- Truth accuracy {ci(b['truth_accuracy (answered items only)'])}
- Lie success {ci(b['lie_success = C1/(C1+C2)'])}
- Truth leakage {ci(b['truth_leakage = C2/(C1+C2)'])}
- Paradox {ci(b['paradox = C4/(C3+C4)'])} vs luck {100 * main['luck']:.1f}
  (above luck p = {fp(main['paradox_above_luck_p'])}, below luck p = {fp(main['paradox_below_luck_p'])})
""" + "".join(f"- Probe C1 vs C2 at {p}: best AUROC {main[f'probe_{p}_best_auc']:.3f} "
              f"(layer {main[f'probe_{p}_best_layer']}, corrected p {main[f'probe_{p}_p_fwer']:.3f})\n"
              for p in POS if f"probe_{p}_best_auc" in main) + """
## Folders
- `figures/png`, `figures/pdf`: every figure (PDF = vector, use in the paper)
- `tables/`: CSV + LaTeX tables (`\\input{}` them; needs `\\usepackage{booktabs}`); `main_row.csv` = one-line summary of this run
- `data/`: per-question outputs (`behavioural.csv` = question, gold, both answers, both reasonings, case)
- `stats/`: all statistics behind the tables and figures
- `reproducibility/`: config, prompts and decoding, library versions, dataset SHA-256, compute time

## Positions
P1 = last prompt token (the two prompts differ by one word; no reasoning yet): main, non-circular test.
P2 = token before the boxed answer: answer-commitment view, partly determined by the case label.

## Reproduce
Run the notebook with RUN_NAME, MODEL_ID and CSV_PATH as in `reproducibility/config.json`.
"""
open(f"{PKG}/README.md", "w").write(anon(readme))

# ---- 7) anonymity check + zip ----
leaks = []
for dp, _, fs in os.walk(PKG):
    for fn in fs:
        if fn.endswith((".csv", ".json", ".jsonl", ".txt", ".md", ".tex")):
            t = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
            leaks += [f"{fn}: '{w}'" for w in SCRUB + ["/kaggle/input"] if w in t]
print("anonymity check:", "OK" if not leaks else "CHECK THESE -> " + "; ".join(leaks))
print(f"{n_fig} figures | tables: {sorted(os.listdir(f'{PKG}/tables'))}")
print(open(f"{PKG}/README.md").read())
make_zip()                        # the results zip now contains <RUN_NAME>/paper/ too


# %% [cell 13]
# CELL 13 — ALL RUNS TOGETHER: the cross-model table and figure for the paper
# Finds every run that has been through cell 12: run folders in /kaggle/working and <run>_results.zip files in
# /kaggle/working or /kaggle/input (add finished results zips as a Kaggle dataset). Nothing to edit.
import io
import matplotlib.pyplot as plt
ALLD = "/kaggle/working/paper_all_runs"
os.makedirs(ALLD, exist_ok=True)

def read_member(src, suffix):
    """Read paper/tables/main_row.csv or paper/stats/probe_C1_vs_C2.csv from a run folder or a results zip."""
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as z:                         # reads only that file, not the whole zip
            hit = [n for n in z.namelist() if n.endswith("paper/" + suffix)]
            return pd.read_csv(io.BytesIO(z.read(hit[0]))) if hit else None
    p = os.path.join(src, "paper", suffix)
    return pd.read_csv(p) if os.path.exists(p) else None

# this session's runs come first, so a fresh run replaces an older copy of the same run in the dataset
cands = sorted(p for p in glob.glob("/kaggle/working/*") if os.path.isdir(os.path.join(p, "paper"))) + \
        sorted(glob.glob("/kaggle/working/*_results.zip")) + \
        sorted(glob.glob("/kaggle/input/**/*_results.zip", recursive=True)) + \
        sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/paper", recursive=True))
mains, probes, seen = [], [], set()
for src in cands:
    m = read_member(src, "tables/main_row.csv")
    if m is None:
        continue                                                  # old run made before cell 12 existed
    run = str(m.run.iloc[0])
    if run in seen:
        continue
    seen.add(run); mains.append(m)
    q = read_member(src, "stats/probe_C1_vs_C2.csv")
    if q is not None:
        q["run"] = run; probes.append(q)
if not mains:
    raise FileNotFoundError("no finished runs found: run cell 12 first, or add results zips as a Kaggle dataset")
M = pd.concat(mains, ignore_index=True).drop_duplicates("run", keep="first")
M.to_csv(f"{ALLD}/all_runs_summary.csv", index=False)
print("runs:", M.run.tolist())

def fmt(r, k):
    return f"{100 * r[k]:.1f}" if pd.notna(r.get(k)) else "--"
tab = pd.DataFrame([{
    "Model": r.model, "Data": r.dataset, "n": int(r.n_items),
    "C1": int(r.C1), "C2": int(r.C2), "C3": int(r.C3), "C4": int(r.C4),
    "Truth acc.": fmt(r, "truth_acc"), "Lie succ.": fmt(r, "lie_success"),
    "Leakage [CI]": f"{fmt(r, 'truth_leakage')} [{fmt(r, 'truth_leakage_lo')}, {fmt(r, 'truth_leakage_hi')}]",
    "Paradox": fmt(r, "paradox"), "Luck": fmt(r, "luck"),
    "Probe P1": f"{r.probe_P1_best_auc:.2f}" if "probe_P1_best_auc" in r and pd.notna(r.probe_P1_best_auc) else "--",
    "Probe P2": f"{r.probe_P2_best_auc:.2f}" if "probe_P2_best_auc" in r and pd.notna(r.probe_P2_best_auc) else "--",
} for _, r in M.iterrows()])
tab.to_csv(f"{ALLD}/table_all_runs.csv", index=False)
tab.to_latex(f"{ALLD}/table_all_runs.tex", index=False, escape=True,
             caption="Behaviour and C1-vs-C2 probe (best held-out AUROC) for every model and dataset. "
                     "Rates in \\%; leakage with Wilson 95\\% CI.", label="tab:all-runs")
print(tab.to_string(index=False))

if probes:
    P_ = pd.concat(probes, ignore_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, pos in zip(ax, POS):
        for run, q in P_[P_.position == pos].groupby("run"):
            d_ = q.layer / q.layer.max()
            a.plot(d_, q.auc, ".-", label=run)
            sg = q.p_fwer < .05
            a.scatter(d_[sg], q.auc[sg], s=40, facecolors="none", edgecolors="k")
        a.axhline(.5, c="grey", lw=.5); a.set_ylim(.3, 1); a.set_xlabel("relative depth")
        a.set_title(f"{pos}: " + ("last prompt token" if pos == "P1" else "token before the answer"), fontsize=10)
    ax[0].set_ylabel("held-out AUROC, C1 vs C2"); ax[0].legend(fontsize=8)
    fig.suptitle("Truth-leakage probe across models (circles: p < .05 corrected over layers)", fontsize=11)
    plt.tight_layout(); save_fig(fig, f"{ALLD}/fig_probe_all_runs.png"); plt.show()

zp = "/kaggle/working/paper_all_runs.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(ALLD)):
        z.write(f"{ALLD}/{fn}", f"paper_all_runs/{fn}")
print(f"\n-> {zp}: {sorted(os.listdir(ALLD))}")
