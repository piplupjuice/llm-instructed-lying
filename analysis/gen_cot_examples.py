r"""Appendix: chain-of-thought responses, one GSM8K item per model x outcome.

Texts come from each run's behavioural.csv (full truth- and lie-prompt generations); outcome labels and
extracted answers come from labels.csv (the labels used throughout the paper). Items are drawn with
random.Random(1234) (unchanged selection). Wording and numbers are never changed. The models' own markup is
rendered instead of printed: math delimited by \[..\], \(..\) or $$..$$ is typeset as math, Markdown
**bold** and ### headings become bold, and line breaks are kept as line breaks. A math segment that uses
anything outside a small whitelist is printed as plain text instead, so the build cannot break. Long
responses are shortened at line boundaries to their opening and closing lines.
"""
import csv
import random
import re

csv.field_size_limit(10**9)
OUT = "C:/Users/nehad/Desktop/llm lies/Model Outputs"
RUNS = [("Llama-8B", "Lamma/gsm8k/llama_gsm8k"), ("Qwen-3B", "Qwen/GSM-8K/Qwen2.5-3B_gsm8k"),
        ("Gemma-2B", "GEMMA-2-9B-IT/gsm8k/gemma2b_gsm8k"), ("Qwen-Math", "QWEN-MATH/GSM8K/qwen15bmath_gsm8k")]
CASES = [("C1", "lie", "Successful lies"), ("C2", "leak", "Truth leakage"), ("C3", "ign", "Ignorance"), ("C4", "para", "Paradoxes")]
HEAD, TAIL, LIMIT, QLIMIT = 430, 300, 780, 260

UNI = {"÷": r"$\div$", "≈": r"$\approx$", "€": r"\texteuro{}", "√": r"$\surd$", "→": r"$\rightarrow$", "³": r"$^3$",
       "∞": r"$\infty$", "×": r"$\times$", "±": r"$\pm$", "²": r"$^2$", "é": r"\'e", "’": "'", "‘": "`", "“": "``",
       "”": "''", "–": "--", "—": "---", "…": r"\ldots{}", "\xa0": " ", "°": r"$^\circ$", "·": r"$\cdot$", "−": "-"}
UNI_M = {"÷": r"\div ", "≈": r"\approx ", "√": r"\surd ", "→": r"\rightarrow ", "³": "^3", "∞": r"\infty ", "×": r"\times ",
         "±": r"\pm ", "²": "^2", "°": r"^\circ ", "·": r"\cdot ", "−": "-", "\xa0": " "}
ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
       "_": r"\_", "^": r"\^{}", "~": r"\textasciitilde{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}
MATH_OK = {"times", "div", "cdot", "frac", "dfrac", "tfrac", "text", "textbf", "mathrm", "mathbf", "boxed", "sqrt", "left", "right",
           "approx", "le", "leq", "ge", "geq", "neq", "ne", "pm", "quad", "qquad", "%", "$", ",", ";", "!", " ", "ldots", "cdots",
           "dots", "infty", "pi", "circ", "implies", "Rightarrow", "rightarrow", "to", "lfloor", "rfloor", "lceil", "rceil", "mid",
           "equiv", "overline", "underline", "binom", "displaystyle", "max", "min", "log", "ln", "gcd", "{", "}", "_", "approx"}
GRAY_CUT = r"{\color{gray}[\ldots]}"


def typesettable(s):
    return all(ord(ch) < 127 or ch in UNI for ch in s)


def esc(s):
    s = "".join(ESC.get(ch, UNI.get(ch, ch)) for ch in s)
    return re.sub(r"([-`'!?])(?=[-`'])", r"\1{}", s)  # stop TeX ligatures so the text prints as written


def math_ok(m):
    if any(c in m for c in "&#") or "\\\\" in m or re.search(r"(?<!\\)\$", m):
        return False
    if any(cmd not in MATH_OK for cmd in re.findall(r"\\([A-Za-z]+|.)", m)):
        return False
    depth = 0
    for i, c in enumerate(m):
        if c == "{" and (i == 0 or m[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or m[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and m.count("\\left") == m.count("\\right")


def math(m):
    m = " ".join(m.split())
    if not m:
        return ""
    if not math_ok(m):
        return esc(m)  # cannot be typeset safely: show the content as plain text
    m = re.sub(r"(?<!\\)%", r"\\%", m)
    m = "".join(UNI_M.get(c, c) for c in m)
    # split at top-level "=" so a long equation wraps across lines instead of shrinking; each piece is
    # wrapped in \cotmath (defined at the top of cot_examples.tex), which shrinks a piece only if it alone is wider than the column
    pieces, depth, cur = [], 0, ""
    for c in m:
        depth += (c == "{") - (c == "}")
        if c == "=" and depth == 0:
            pieces.append(cur); cur = ""
        else:
            cur += c
    pieces.append(cur)
    return " = ".join(r"\cotmath{$" + p.strip() + "$}" if p.strip() else "" for p in pieces).strip()


def text_md(s):
    """Plain text with Markdown *italic* (asterisks hugging a word); spaced * stays (it is multiplication)."""
    res, pos = [], 0
    for mt in re.finditer(r"(?<![*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![*\w])", s):
        res.append(esc(s[pos:mt.start()]))
        res.append(r"\textit{" + esc(mt.group(1)) + "}")
        pos = mt.end()
    res.append(esc(s[pos:]))
    return "".join(res)


def segment(s):
    """Text without Markdown bold: inline math \\(..\\) or $\\cmd..$ (a $ pair with a command inside; plain $ is currency),
    text-mode \\boxed{..}, everything else escaped."""
    res, pos = [], 0
    for mt in re.finditer(r"\\\((.*?)\\\)|\$([^$]*\\[A-Za-z][^$]*)\$|\\boxed\{([^{}]*)\}", s):
        res.append(text_md(s[pos:mt.start()]))
        if mt.group(1) is not None:
            res.append(math(mt.group(1)))
        elif mt.group(2) is not None:
            res.append(math(mt.group(2)))
        else:
            res.append(r"\fbox{" + esc(mt.group(3)) + "}")
        pos = mt.end()
    res.append(text_md(s[pos:]))
    return "".join(res)


def inline(line):
    r"""One text line: Markdown headings, bullets and **bold** first, then math and boxes inside each piece."""
    head = re.match(r"\s*#{1,6}\s+(.*)$", line)
    if head:
        return r"\textbf{" + inline(head.group(1)) + "}"
    bullet = re.match(r"\s*[*-]\s+(.*)$", line)
    if bullet:
        return r"$\bullet$~" + inline(bullet.group(1))
    parts = line.split("**")
    if len(parts) % 2 == 0:  # unpaired ** : keep the last one literally
        parts = parts[:-2] + [parts[-2] + "**" + parts[-1]]
    return "".join((r"\textbf{" + segment(p) + "}" if i % 2 else segment(p)) for i, p in enumerate(parts) if p)


def units(text):
    """Split a response into display-math units and text-line units (blank lines dropped)."""
    text = text.replace("\r", "")
    us = []
    pos = 0
    for mt in re.finditer(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", text, flags=re.S):
        for ln in text[pos:mt.start()].split("\n"):
            if ln.strip():
                us.append(("t", ln.strip()))
        us.append(("m", mt.group(1) if mt.group(1) is not None else mt.group(2)))
        pos = mt.end()
    for ln in text[pos:].split("\n"):
        if ln.strip():
            us.append(("t", ln.strip()))
    return us


def size(u):
    return len(u[1])


def render(text):
    us = units(text.strip())
    if sum(map(size, us)) > LIMIT:
        a, n = [], 0
        for u in us:
            if n >= HEAD:
                break
            a.append(u); n += size(u)
        b, n = [], 0
        for u in reversed(us[len(a):]):
            if n >= TAIL:
                break
            b.insert(0, u); n += size(u)
        shown = a + [None] + b if len(a) + len(b) < len(us) else us
    else:
        shown = us
    out = []
    for u in shown:
        if u is None:
            out.append(GRAY_CUT)
        elif u[0] == "m":
            r = math(u[1])
            if r:
                out.append(r"\hspace*{1em}" + r)
        else:
            out.append(inline(u[1]))
    return r"\newline ".join(out)


lines = []
summary = []
for code, macro, title in CASES:
    lines.append(f"\\subsection{{{title} (\\{macro}{{}})}}")
    for model, path in RUNS:
        beh = {r["qid"]: r for r in csv.DictReader(open(f"{OUT}/{path}/behavioural.csv", encoding="utf-8-sig"))}
        lab = [r for r in csv.DictReader(open(f"{OUT}/{path}/labels.csv", encoding="utf-8-sig")) if r["case"] == code]
        lab.sort(key=lambda r: int(r["qid"][1:]))
        pool = [r for r in lab if all(typesettable(beh[r["qid"]][k]) for k in ("question", "truth_reasoning", "false_reasoning"))]
        r = random.Random(1234).choice(pool)
        b = beh[r["qid"]]
        assert b["case"] == "case" + code[1], (model, r["qid"])
        assert (r["truth_correct"] == "True") == (code in ("C1", "C2")) and (r["lie_correct"] == "True") == (code in ("C2", "C4"))
        summary.append((code, model, r["qid"], len(lab), len(lab) - len(pool)))
        ok = lambda c: "correct" if c == "True" else "wrong"
        qt = " ".join(b["question"].split())
        q = esc(qt[:qt.rfind(" ", 0, QLIMIT)]) + " " + GRAY_CUT if len(qt) > QLIMIT else esc(qt)
        lines.append(f"\\par\\medskip\\noindent\\textbf{{{model}, \\textsc{{GSM8K}} {r['qid']}.}}\\enspace\\textit{{Question:}} {q} "
                     f"\\textit{{Gold:}} {esc(r['gold'])}.\\par")
        for lab_name, key, pred, corr in (("Truth prompt", "truth_reasoning", "truth_pred", "truth_correct"),
                                          ("Lie prompt", "false_reasoning", "lie_pred", "lie_correct")):
            lines.append(f"\\noindent\\textbf{{{lab_name}}} (answer {esc(r[pred])}, {ok(r[corr])}):\\par")
            lines.append(f"{{\\leftskip=1em\\noindent {render(b[key])}\\par}}")
        lines.append("")
PRE = [r"% Generated by analysis/gen_cot_examples.py from the run files; do not edit by hand.",
       r"\newsavebox{\cotbox}",
       r"\newcommand{\cotmath}[1]{\sbox{\cotbox}{#1}\ifdim\wd\cotbox>\dimexpr\linewidth-2.4em\relax"
       r"\resizebox{\dimexpr\linewidth-2.4em\relax}{!}{\usebox{\cotbox}}\else\usebox{\cotbox}\fi}",
       # model text is set ragged-right (no stretched word gaps) with tight boxes around boxed answers; local to this file
       r"\begingroup\raggedright\setlength{\fboxsep}{1.2pt}"]
open("C:/Users/nehad/Desktop/llm lies/paper/cot_examples.tex", "w", encoding="utf-8").write("\n".join(PRE + lines + [r"\endgroup"]) + "\n")
for s in summary:
    print(s)
