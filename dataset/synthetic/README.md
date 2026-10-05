# Synthetic arithmetic dataset

`synthetic_arithmetic_500.csv` is the Synthetic evaluation set used in the paper: 500 integer-arithmetic questions with exact answers, 125 at each of four difficulty levels. It is byte-identical to the file every Synthetic run read: SHA-256 `d6800ea135bc868cc217a5542703f9fc4b1e5d7a8dd73085846c09b36db7fdaa`, as recorded by each run.

| File | Purpose |
|---|---|
| `synthetic_arithmetic_500.csv` | The evaluation set used in the paper |
| `validate_synthetic.py` | Verifies any file in this format with exact arithmetic and prints its statistics |
| `generate_synthetic.py` | Generates **additional** datasets with the same design, e.g. larger or held-out sets. These are new items; the script does not regenerate the 500 evaluation items. |

```bash
python validate_synthetic.py synthetic_arithmetic_500.csv                 # verify the evaluation set
python generate_synthetic.py --rows 1000 --seed 7 --out new_set.csv       # create a new set of the same design
python validate_synthetic.py new_set.csv
```

Both scripts use only the Python standard library.

## Format

| Column | Content |
|---|---|
| `question` | `What is <expression>?`, e.g. `What is (265 - 203) / (18 / 9)?` |
| `answer` | the integer value of the expression |
| `n_ops` | the number of operators, 1–4 |
| `difficulty` | `easy` / `medium` / `hard` / `very hard` for `n_ops` = 1 / 2 / 3 / 4 |

The model pipeline reads only the first two columns (question and answer).

Expressions use `+ - * /` with explicit parentheses around every sub-expression except the outermost one, so operator precedence never has to be resolved. There is a single space around every operator.

## Verification

`validate_synthetic.py` parses every expression with a small grammar (no `eval`) and recomputes it with exact rational arithmetic. On the evaluation set it reports:

| n_ops | Rows | `+` | `-` | `*` | `/` | Answer min / median / max | Plain divisors |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 125 | 31 | 31 | 31 | 31 | 2 / 42 / 98 | 2–12 |
| 2 | 125 | 62 | 62 | 62 | 62 | 4 / 219 / 497 | 2–12 |
| 3 | 125 | 93 | 93 | 93 | 93 | 5 / 293 / 999 | 2–12 |
| 4 | 125 | 125 | 125 | 125 | 125 | 77 / 3120 / 9896 | 2–12 |

What the checks confirm:
- Every answer equals the recomputed value.
- Every division is exact.
- Every intermediate value is a positive integer.
- All 500 questions are distinct.
- The four operators appear equally often within every level.

Three items are a single number rather than an expression: line 135 `What is 303?`, line 385 `What is 67?` and line 386 `What is 440?`. These are the three zero-operator items reported in the paper's appendix. Their answers are correct, and their `n_ops` value is the level they belong to; the validator lists them under `notes`.

## Design

The evaluation set and `generate_synthetic.py` share these properties:
- the question template and full parenthesisation;
- an equal number of rows per level;
- the four operators balanced within each level;
- random binary-tree shapes;
- exact division, with a plain divisor of 2–12;
- every intermediate value a positive integer;
- answers of at least 2 and below 100 / 500 / 1,000 / 10,000 at the four levels;
- some subtractions of two near-equal numbers, such as `4079 - 4055`.

The generator is deterministic: the same `--rows` and `--seed` give the same file.
