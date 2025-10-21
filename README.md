# Lojban-Style Predicate Calculus (Python)

A tiny interpreter for a simplified Lojban-inspired predicate calculus with built-ins and user-defined predicates via `cmavo`.

## Run
```bash
# file
python lojban_calc.py program.txt
# pipe (prints 4)
echo "i lo .X. sumji 2 2" | python lojban_calc.py
```
> Note: statements that are **assertions** (e.g., `i lo .brook. fatci`) print nothing.  
> Output only comes from **variables in the final statement**.

## Syntax (very short)
- Program: one or more statements; each starts with `i`.
- Allowed chars: ASCII letters/digits/whitespace/periods.
- Arguments: numbers (`0` or no leading zeros), `lo .Name.`, lists: `lo steni` (empty) or `lo steko <elem> <tail>`.
- Variables: ALL-CAPS dot names (e.g., `.X.`). Others are constants.
- `se`: may appear before the 1st or 2nd argument (swaps first two args).
- Predicates (1–5 args). Built-ins: `fatci`, `sumji`, `vujni`, `dunli`, `steni`, `steko`, `cmavo`.

## Examples
**Sample (prints `.Brook.`):**
```
i lo .Brook. fatci i lo .coffee. fatci
i lo pinxe cmavo lo steko lo .Brook. lo steko lo .coffee. lo steni
i lo .X. pinxe lo .coffee.
```

**Arithmetic (prints `4`):**
```
i lo .X. sumji 2 2
```

## Output
Prints values bound to variables in the **final** statement (one per line).
