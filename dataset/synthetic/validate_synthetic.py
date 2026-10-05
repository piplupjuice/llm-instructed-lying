"""Check a synthetic arithmetic CSV (columns: question, answer, n_ops, difficulty).

Every expression is parsed with a small grammar (no eval) and recomputed with exact rational arithmetic.
Checks: question format, answer correctness, n_ops = number of operators, difficulty label, exact division,
positive integer intermediate values, duplicate questions, and per-level operator balance. Prints a report and
summary statistics.

    python validate_synthetic.py synthetic_arithmetic_500.csv
    python validate_synthetic.py synthetic_arithmetic_500.csv --json stats.json

Items that are a single number (no operator) are listed as notes; their answers are still checked.
Exit code 0 if no problems are found, 1 otherwise.
"""
import argparse
import collections
import csv
import json
import re
import statistics
import sys
from fractions import Fraction

DIFFICULTY = {1: "easy", 2: "medium", 3: "hard", 4: "very hard"}
TOKEN = re.compile(r"\s*(\d+|[()+\-*/])")
QUESTION = re.compile(r"What is (.+)\?")


def parse(expr):
    """Parse a fully parenthesised expression whose outermost parentheses are omitted.
    Returns an int (bare number) or a tuple (op, left, right)."""
    toks = TOKEN.findall(expr)
    if "".join(toks) != expr.replace(" ", ""):
        raise ValueError("unexpected characters")
    pos = 0

    def operand():
        nonlocal pos
        if toks[pos] == "(":
            pos += 1
            left = operand()
            op = toks[pos]
            pos += 1
            right = operand()
            if toks[pos] != ")":
                raise ValueError("missing ')'")
            pos += 1
            return (op, left, right)
        if not toks[pos].isdigit():
            raise ValueError(f"expected a number, got {toks[pos]!r}")
        pos += 1
        return int(toks[pos - 1])

    first = operand()
    if pos == len(toks):
        return first
    op = toks[pos]
    pos += 1
    second = operand()
    if pos != len(toks):
        raise ValueError("trailing tokens")
    return (op, first, second)


def evaluate(tree, intermediates):
    """Exact value of the tree; appends every operator result to `intermediates`."""
    if isinstance(tree, int):
        return Fraction(tree)
    op, l, r = tree
    a, b = evaluate(l, intermediates), evaluate(r, intermediates)
    if op == "/" and b == 0:
        raise ZeroDivisionError
    v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
    intermediates.append((op, a, b, v))
    return v


def operators(tree):
    return [] if isinstance(tree, int) else [tree[0]] + operators(tree[1]) + operators(tree[2])


def depth(tree):
    return 0 if isinstance(tree, int) else 1 + max(depth(tree[1]), depth(tree[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--json", help="also write the statistics to this JSON file")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8", newline="")))
    problems = collections.defaultdict(list)          # kind -> [(csv line, question)]; any problem gives exit code 1
    notes = collections.defaultdict(list)             # listed for information only
    ops_by_level = collections.defaultdict(collections.Counter)
    by_level = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = collections.Counter(r["question"] for r in rows)

    for line, r in enumerate(rows, start=2):           # line 1 is the header
        q = r["question"]
        m = QUESTION.fullmatch(q)
        if not m:
            problems["question not of the form 'What is <expression>?'"].append((line, q))
            continue
        try:
            tree = parse(m.group(1))
            inter = []
            value = evaluate(tree, inter)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            problems[f"cannot parse or evaluate ({type(e).__name__})"].append((line, q))
            continue
        n_ops = int(r["n_ops"])
        ops = operators(tree)
        if value.denominator != 1 or value != int(r["answer"]):
            problems["answer does not equal the recomputed value"].append((line, q))
        if not ops and n_ops > 0:                       # a single number: listed, not an error (its answer is still checked)
            notes["single-number items (no operator; n_ops is the level the item belongs to)"].append((line, q))
        elif len(ops) != n_ops:
            problems["n_ops label does not equal the number of operators"].append((line, q))
        if DIFFICULTY.get(n_ops) != r["difficulty"]:
            problems["difficulty label does not match n_ops"].append((line, q))
        if any(op == "/" and v.denominator != 1 for op, _, _, v in inter):
            problems["inexact division"].append((line, q))
        if any(v.denominator != 1 or v < 1 for _, _, _, v in inter):
            problems["intermediate value that is not a positive integer"].append((line, q))
        if seen[q] > 1:
            problems["duplicate question"].append((line, q))
        ops_by_level[n_ops].update(ops)
        s = by_level[n_ops]
        s["answer"].append(int(r["answer"]))
        s["depth"].append(depth(tree))
        s["length"].append(len(q))
        s["intermediate"] += [int(v) for _, _, _, v in inter if v.denominator == 1]
        if isinstance(tree, tuple):                     # divisors written as a plain number, e.g. "... / 7"
            stack = [tree]
            while stack:
                t = stack.pop()
                if t[0] == "/" and isinstance(t[2], int):
                    s["plain_divisor"].append(t[2])
                stack += [c for c in t[1:] if isinstance(c, tuple)]

    # ---- report
    print(f"file: {args.csv}")
    print(f"rows: {len(rows)}   unique questions: {len(seen)}")
    stats = {"rows": len(rows), "levels": {}, "problems": {k: v for k, v in problems.items()}}
    print("\nper level (n_ops):")
    print(f"  {'n_ops':>5} {'rows':>5} {'+':>5} {'-':>5} {'*':>5} {'/':>5} {'answer min/median/max':>24} {'max depth':>12} {'plain divisors':>15}")
    for n in sorted(by_level):
        s = by_level[n]
        c = ops_by_level[n]
        a = s["answer"]
        dv = s["plain_divisor"]
        depths = dict(sorted(collections.Counter(s["depth"]).items()))
        print(f"  {n:>5} {len(a):>5} {c['+']:>5} {c['-']:>5} {c['*']:>5} {c['/']:>5} "
              f"{f'{min(a)} / {statistics.median(a):g} / {max(a)}':>24} {str(depths):>12} "
              f"{(f'{min(dv)}-{max(dv)}' if dv else '-'):>15}")
        stats["levels"][n] = {"rows": len(a), "operators": dict(c), "answer_min": min(a), "answer_median": statistics.median(a),
                              "answer_max": max(a), "depth_counts": depths, "intermediate_max": max(s["intermediate"], default=None),
                              "plain_divisor_range": [min(dv), max(dv)] if dv else None,
                              "question_length_range": [min(s["length"]), max(s["length"])]}
    print("\nchecks:")
    if not problems:
        print("  all checks passed")
    for kind, items in problems.items():
        print(f"  {kind}: {len(items)}")
        for line, q in items[:10]:
            print(f"      CSV line {line}: {q}")
    if notes:
        print("\nnotes:")
        for kind, items in notes.items():
            print(f"  {kind}: {len(items)}")
            for line, q in items[:10]:
                print(f"      CSV line {line}: {q}")
    stats["notes"] = {k: v for k, v in notes.items()}
    if args.json:
        json.dump(stats, open(args.json, "w"), indent=1, default=str)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
