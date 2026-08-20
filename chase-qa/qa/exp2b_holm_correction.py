"""
exp2b_holm_correction.py

Reads the already-generated analysis/exp2b_significance_grid.tsv (from
exp2b_significance_sweep.py -- no rerun of the bootstrap sweep needed) and
applies Holm-Bonferroni correction within each (unsure_mode, pass_frac)
3-type family. This is the same family definition already used for the
v8/v9/v9b headline table ("Holm-Bonferroni corrected within each corpus's
three-type family"), just applied at every point of the pass_frac grid
instead of once at pf=0.75.

Family = the 3 qa_types at one (unsure_mode, pass_frac) cell. Step-down:
sort the 3 p-values ascending, reject smallest if p <= alpha/3, next if
p <= alpha/2, last if p <= alpha/1 -- stopping at the first failure (standard
Holm: once one step fails, nothing further in that family is rejected).

Hand-verified against the pasted exclude-mode grid before being written as
a script: type3 rejects at every pf from 0.50-0.80 inclusive, nothing
rejects at pf>=0.85 (type3's own p=0.0422 at pf=0.85 clears raw 0.05 but
fails Holm's first-step threshold of 0.05/3=0.01667).

Run from repo root, after exp2b_significance_sweep.py has already produced
its TSV:
    python3 exp2b_holm_correction.py

Input:
    analysis/exp2b_significance_grid.tsv

Output:
    analysis/exp2b_holm_grid.tsv  -- same rows as input, + holm_significant column
    stdout -- per unsure_mode, per pass_frac, which type(s) survive correction
"""

import os
import pandas as pd

ANALYSIS_DIR = "analysis"
IN_PATH = os.path.join(ANALYSIS_DIR, "exp2b_significance_grid.tsv")
OUT_PATH = os.path.join(ANALYSIS_DIR, "exp2b_holm_grid.tsv")
ALPHA = 0.05


def holm_reject(pvals, alpha=ALPHA):
    """pvals: list of p-values (any order). Returns list of booleans, same order,
    True where that test survives Holm step-down correction."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[idx] <= thresh:
            reject[idx] = True
        else:
            break  # Holm stops at first failure; nothing later in the sorted order is rejected
    return reject


def main():
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"ERROR: {IN_PATH} not found. Run exp2b_significance_sweep.py first.")

    df = pd.read_csv(IN_PATH, sep="\t")
    required = {"unsure_mode", "pass_frac", "qa_type", "mcnemar_p"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {IN_PATH} missing expected column(s): {missing}")

    df["holm_significant"] = False

    for (unsure_mode, pf), g in df.groupby(["unsure_mode", "pass_frac"]):
        if g["mcnemar_p"].isna().any():
            # n=0 cell in this family -- cannot form a valid 3-test family, skip
            # (leaves holm_significant=False for all rows in this cell, which is
            # correct: no test was actually performed, not "tested and failed")
            continue
        if len(g) != 3:
            raise SystemExit(
                f"ERROR: expected exactly 3 qa_types per (unsure_mode, pass_frac) family, "
                f"got {len(g)} at unsure_mode={unsure_mode}, pass_frac={pf}. "
                f"Family definition assumption violated -- do not trust correction below this point."
            )
        rej = holm_reject(list(g["mcnemar_p"]))
        df.loc[g.index, "holm_significant"] = rej

    df.to_csv(OUT_PATH, sep="\t", index=False)

    print("--- Holm-significant type(s) per (unsure_mode, pass_frac), 3-type family, alpha=0.05 ---")
    for unsure_mode, gm in df.groupby("unsure_mode"):
        print(f"\nunsure_mode={unsure_mode}")
        for pf, g in gm.groupby("pass_frac"):
            sig = sorted(g.loc[g["holm_significant"], "qa_type"].tolist())
            n_valid = g["mcnemar_p"].notna().sum()
            flag = "none" if not sig else str(sig)
            note = "" if n_valid == 3 else "  (family incomplete -- some n=0, skipped)"
            print(f"  pf={pf}: Holm-significant type(s) = {flag}{note}")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()