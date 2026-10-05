"""Generate additional synthetic integer-arithmetic datasets with the design of synthetic_arithmetic_500.csv.

The evaluation set used in the paper is synthetic_arithmetic_500.csv. This script creates new items of the same
design (for example larger or held-out sets); it does not regenerate the 500 evaluation items.

Design:
  * question template "What is <expression>?", single spaces around operators, every sub-expression in
    parentheses except the outermost one;
  * difficulty = number of operators n_ops: 1 easy, 2 medium, 3 hard, 4 very hard; equal rows per level;
  * operators + - * / balanced within every level (drawn from a shuffled pool with equal counts);
  * random binary-tree shapes (the left subtree gets a uniformly chosen number of the operators);
  * exact integer division; a divisor written as a plain number is 2-12;
  * every intermediate value is a positive integer;
  * answers at least 2 and below 100 / 500 / 1,000 / 10,000 for levels 1-4;
  * some subtractions of two plain numbers are near-equal (e.g. 4079 - 4055);
  * no duplicate questions; rows shuffled.
Every row has exactly n_ops operators. Where n_ops * rows is not divisible by 4, the operator counts of a level
differ by at most 1.

    python generate_synthetic.py                       # 500 rows, seed 1234 -> synthetic_arithmetic_regenerated.csv
    python generate_synthetic.py --rows 1000 --seed 7 --out my_set.csv
    python validate_synthetic.py my_set.csv            # check the result
"""
import argparse
import csv
import random

OPS = ["+", "-", "*", "/"]
DIFFICULTY = {1: "easy", 2: "medium", 3: "hard", 4: "very hard"}
ANSWER_MAX = {1: 99, 2: 499, 3: 999, 4: 9999}          # inclusive
LEAF_MAX = {1: 99, 2: 999, 3: 999, 4: 9999}            # largest plain number in an expression of each level
INTERMEDIATE_MAX = {1: 99, 2: 5000, 3: 7000, 4: 110000}
DIVISOR = (2, 12)                                      # a divisor written as a plain number
MULTIPLIER = (2, 99)                                   # a factor written as a plain number on the right of *
NEAR_EQUAL = 0.4                                       # share of "number - number" made near-equal
NEAR_GAP = (1, 30)


def shape(n, rng):
    """Random binary tree with n operator nodes; leaves are None."""
    if n == 0:
        return None
    k = rng.randint(0, n - 1)
    return [None, shape(k, rng), shape(n - 1 - k, rng)]   # [op, left, right]


def assign_ops(tree, ops):
    """Fill operator slots in pre-order from the list `ops` (consumed)."""
    if tree is None:
        return
    tree[0] = ops.pop()
    assign_ops(tree[1], ops)
    assign_ops(tree[2], ops)


def build(tree, level, rng):
    """Choose numbers for the tree. Returns (text, value) or None if a constraint fails."""
    if tree is None:
        v = rng.randint(1, LEAF_MAX[level])
        return str(v), v
    op, lt, rt = tree
    if op == "/":
        if rt is None:
            r = rng.randint(*DIVISOR)
            rs = str(r)
        else:
            sub = build(rt, level, rng)
            if sub is None:
                return None
            rs, r = sub
        if lt is None:                                     # plain dividend: make it a multiple of the divisor
            hi = LEAF_MAX[level] // r
            if hi < 2:
                return None
            l = r * rng.randint(2, hi)
            ls = str(l)
        else:
            sub = build(lt, level, rng)
            if sub is None:
                return None
            ls, l = sub
            if l % r:
                return None
        v = l // r
    elif op == "*":
        sub = build(lt, level, rng)
        if sub is None:
            return None
        ls, l = sub
        if rt is None:
            r = rng.randint(*MULTIPLIER)
            rs = str(r)
        else:
            sub = build(rt, level, rng)
            if sub is None:
                return None
            rs, r = sub
        v = l * r
    else:
        sub = build(rt, level, rng)
        if sub is None:
            return None
        rs, r = sub
        if op == "-" and lt is None and rt is None and rng.random() < NEAR_EQUAL:
            l = r + rng.randint(*NEAR_GAP)                  # near-equal operands
            if l > LEAF_MAX[level]:
                return None
            ls = str(l)
        else:
            sub = build(lt, level, rng)
            if sub is None:
                return None
            ls, l = sub
        v = l + r if op == "+" else l - r
    if v < 1 or v > INTERMEDIATE_MAX[level]:
        return None
    return f"({ls} {op} {rs})", v


def make_item(n_ops, ops, rng, tries=4000):
    """One question with exactly the operators `ops` (in random positions)."""
    level = n_ops
    for _ in range(tries):
        tree = shape(n_ops, rng)
        pool = ops[:]
        rng.shuffle(pool)
        assign_ops(tree, pool)
        res = build(tree, level, rng)
        if res is None:
            continue
        text, v = res
        if not (2 <= v <= ANSWER_MAX[level]):
            continue
        return f"What is {text[1:-1]}?", v               # drop the outermost parentheses
    return None


def operator_pool(n_ops, rows, rng):
    """n_ops * rows operators, as balanced as possible over + - * /, shuffled."""
    total = n_ops * rows
    pool = OPS * (total // 4) + rng.sample(OPS, total % 4)
    rng.shuffle(pool)
    return pool


def generate(rows_total, seed):
    rng = random.Random(seed)
    per = rows_total // 4
    out, seen = [], set()
    for n_ops in (1, 2, 3, 4):
        count = per + (rows_total - 4 * per if n_ops == 4 else 0)
        pool = operator_pool(n_ops, count, rng)
        chunks = [pool[i * n_ops:(i + 1) * n_ops] for i in range(count)]
        i = 0
        while i < count:
            item = make_item(n_ops, chunks[i], rng)
            if item is None or item[0] in seen:
                j = rng.randrange(count)                    # swap operator sets with another item and retry
                chunks[i], chunks[j] = chunks[j], chunks[i]
                if j < i:                                   # keep finished items unchanged
                    chunks[i], chunks[j] = chunks[j], chunks[i]
                continue
            seen.add(item[0])
            out.append({"question": item[0], "answer": item[1], "n_ops": n_ops, "difficulty": DIFFICULTY[n_ops]})
            i += 1
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="synthetic_arithmetic_regenerated.csv")
    args = ap.parse_args()
    rows = generate(args.rows, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", "n_ops", "difficulty"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
