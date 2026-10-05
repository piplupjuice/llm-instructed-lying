"""Behavioural analysis of the written responses, per run and per outcome.

For every item we compare the truth-prompt and lie-prompt reasoning and record:
  onset      relative position (0-1) in the lie text of the first fabricated number
             (a number that appears in neither the question nor the truth reasoning); None if none
  fab        whether the lie text contains any fabricated number at all
  eq_n/eq_f  parsable binary equations "a op b = c" and how many are arithmetically false
  bold       |log10(lie answer / gold)| for numeric answers
  sim        word-level similarity between truth and lie reasoning (difflib ratio)
  mech_*     keyword markers of how the error is introduced (premise, operation swap, omission)
  expose     self-exposure phrases ("the correct answer", "why this is incorrect", ...)
  corr       self-correction markers ("wait", "let me re-", "actually")
  meta_pos   relative position of the first error-announcing word
"""
import csv, difflib, glob, json, math, os, re
import numpy as np

B = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
OUT = "C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad"
SHORT = {"llama_synthetic": "llama_syn", "llama_gsm8k": "llama_gsm", "llama_math500": "llama_math", "Qwen2.5-3b-Syntehtic": "qwen_syn",
         "Qwen2.5-3B_gsm8k": "qwen_gsm", "qwen2.5-3b_instruct_math500": "qwen_math", "gemma2b_synthetic": "gemma_syn", "gemma2b_gsm8k": "gemma_gsm",
         "gemma2b_Math500": "gemma_math", "qwen15b_MATH_SYNTHETIC": "qmath_syn", "qwen15bmath_gsm8k": "qmath_gsm", "qwen15bmath_math500": "qmath_math"}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
STEP = re.compile(r"(?:step|Step|STEP)\s*\d+|^\s*\d+[.)]\s", re.M)
META = re.compile(r"incorrect|wrong|mistake|error|flaw|misinterpret|deliberately", re.I)
PREMISE = re.compile(r"\bassum|let'?s say|suppose|pretend|misread|misinterpret|mistakenly (think|believe|take)", re.I)
OPSWAP = re.compile(r"instead of (add|subtract|multipl|divid|sum)|(add|subtract|multiply|divide)\w* instead|wrong operation", re.I)
OMIT = re.compile(r"\b(ignore|forget|skip|overlook|neglect|leave out|omit)", re.I)
EXPOSE = re.compile(r"correct answer|actual answer|right answer|correct approach|correct solution|why this is (incorrect|wrong)|this is incorrect|the (mistake|error) (is|was|here|in)", re.I)
CORR = re.compile(r"\bwait\b|let me re|actually,|on second thought|let'?s re-?(check|evaluate|calculate)", re.I)
EQ = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def clean(t):
    t = t.replace("\\times", "×").replace("\\cdot", "×").replace("\\div", "÷").replace("$", "").replace("\\(", " ").replace("\\)", " ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    return t


def nums(t):
    t = STEP.sub(" ", t)
    return [(m.start(), float(m.group())) for m in NUM.finditer(t)]


def to_f(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def false_eqs(t):
    n = f = 0
    for a, op, b, c in EQ.findall(t):
        a, b, c = float(a), float(b), float(c)
        v = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "×": a * b, "/": a / b if b else None, "÷": a / b if b else None}[op]
        if v is None:
            continue
        n += 1
        if abs(v - c) > max(1e-6, 0.011 * abs(v)) and abs(round(v) - c) > 0.5:
            f += 1
    return n, f


rows = []
for lp in glob.glob(f"{B}/**/labels.csv", recursive=True):
    p = lp.replace(os.sep, "/")
    if "paper" in p.split("/")[-3:-1] or "mist" in p.lower():
        continue
    d = os.path.dirname(lp)
    run = SHORT[os.path.basename(d)]
    L = {r["qid"]: r for r in csv.DictReader(open(lp, encoding="utf-8-sig"))}
    for r in csv.DictReader(open(f"{d}/behavioural.csv", encoding="utf-8-sig")):
        lab = L[r["qid"]]
        case = lab["case"]
        if case not in ("C1", "C2", "C3", "C4"):
            continue
        q, tr, li = clean(r["question"]), clean(r["truth_reasoning"]), clean(r["false_reasoning"])
        known = {v for _, v in nums(q)} | {v for _, v in nums(tr)}
        ln = nums(li)
        onset = next((pos / max(1, len(li)) for pos, v in ln if v not in known and abs(v) >= 0), None)
        en, ef = false_eqs(li)
        tn, tf = false_eqs(tr)
        g, lv = to_f(lab["gold"]), to_f(lab["lie_pred"])
        bold = abs(math.log10(abs(lv) / abs(g))) if (g and lv and g != 0 and lv != 0) else None
        m = META.search(li)
        rows.append(dict(run=run, model=run.split("_")[0], data=run.split("_")[1], qid=r["qid"], case=case,
                         onset=onset, fab=onset is not None, eq_n=en, eq_f=ef, teq_n=tn, teq_f=tf, bold=bold,
                         sim=difflib.SequenceMatcher(None, tr.split(), li.split(), autojunk=False).ratio(),
                         mech_premise=bool(PREMISE.search(li)), mech_opswap=bool(OPSWAP.search(li)), mech_omit=bool(OMIT.search(li)),
                         expose=bool(EXPOSE.search(li)), corr=bool(CORR.search(li)), meta=m is not None,
                         meta_pos=(m.start() / max(1, len(li))) if m else None,
                         tlen=len(tr), llen=len(li), trunc=lab["lie_truncated"] == "True",
                         boxes=li.count("boxed"), changed=lab["truth_pred"].strip() != lab["lie_pred"].strip()))
    print("ok", run, flush=True)
json.dump(rows, open(f"{OUT}/behav_rows.json", "w"))


def summ(sel):
    def mean(k, cond=None):
        v = [x[k] for x in sel if x[k] is not None and (cond is None or cond(x))]
        return round(float(np.mean(v)), 3) if v else None

    def med(k):
        v = [x[k] for x in sel if x[k] is not None]
        return round(float(np.median(v)), 3) if v else None
    eqn = sum(x["eq_n"] for x in sel)
    return dict(n=len(sel), fab=mean("fab"), onset_med=med("onset"), false_eq=round(sum(x["eq_f"] for x in sel) / eqn, 3) if eqn else None,
                truth_false_eq=round(sum(x["teq_f"] for x in sel) / max(1, sum(x["teq_n"] for x in sel)), 3),
                bold_med=med("bold"), sim_med=med("sim"), premise=mean("mech_premise"), opswap=mean("mech_opswap"), omit=mean("mech_omit"),
                expose=mean("expose"), corr=mean("corr"), meta=mean("meta"), meta_pos_med=med("meta_pos"),
                len_ratio=round(float(np.median([x["llen"] / max(1, x["tlen"]) for x in sel])), 2) if sel else None)


S = {}
for model in ("llama", "qwen", "gemma", "qmath"):
    for c in ("C1", "C2", "C3", "C4"):
        S[f"{model}|{c}"] = summ([x for x in rows if x["model"] == model and x["case"] == c])
for c in ("C1", "C2", "C3", "C4"):
    S[f"all|{c}"] = summ([x for x in rows if x["case"] == c])
for run in sorted({x["run"] for x in rows}):
    for c in ("C1", "C2", "C3", "C4"):
        S[f"{run}|{c}"] = summ([x for x in rows if x["run"] == run and x["case"] == c])
json.dump(S, open(f"{OUT}/behav_summary.json", "w"), indent=1)
for k, v in S.items():
    if "_" not in k.split("|")[0]:
        print(k, v)
