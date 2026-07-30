#!/usr/bin/env python3
"""
Diagnose which checklist criteria drive FAIL vs DISAGREE.

Fail rate and judge-disagreement rate are separated on purpose:
  - a criterion that fails (both judges say No) with LOW disagreement is a real,
    prompt-fixable generation defect.
  - a criterion with HIGH disagreement (judges split) is gate noise, not a
    generation defect, and tuning the generation prompt will not fix it.

Criterion_No is NOT stable across questions (criterion 6 is a different check on
different questions, checklists are 7 or 8 items), so criteria are bucketed by
their TEXT, not by number.

Run:
    python criterion_diagnostics.py path/to/judgments.tsv
"""

import sys
import re
import pandas as pd

# Ordered: first matching pattern wins, so put the specific quality axes before
# the generic "ask specifically" content bucket which would otherwise swallow them.
BUCKETS = [
    ("leakage",        r"giving away|give away|not already stat|does not itself mention|already stating|avoid mentioning the answer|not already list|leaking the answer|reveal(ing)? (its own |the )?answer|leading"),
    ("answerability",  r"answered? (entirely|fully)? ?(from|using) the supplied|documents alone|document alone|supplied document|without needing (any )?(outside|external)|without requiring (outside|external)"),
    ("single_reading", r"single (defensible|clear) (reading|interpretation|meaning)|single clear reading|one defensible reading|unambiguous|no ambiguity"),
    ("narrowness",     r"narrow enough|narrowly framed|narrow and|is the question narrow|narrow(s)? the topic|avoid being so broad|so that a (concise|short|response limited)"),
    ("retrieval",      r"require(s)? retrieval|demand retrieval|rather than being answerable from general|generic (knowledge|financial|risk)|general financial common sense|general knowledge about"),
    ("avoid_unrelated",r"avoid asking for unrelated|avoid broadening|avoid introducing unrelated|not mixing in|avoid mixing in|unrelated (home insurance|examples|figures|product)|other examples in the document"),
    ("coverage",       r"ask(s)? specifically|require(s)? the answer to mention|explicitly require|require the answer to|ask about|require(s)? (consideration|the exact|both)|focus(es)? on|clearly (anchor|concern|target|identify|require)|specifically (require|involve)|^does the question"),
]

def bucket(text):
    t = text.lower()
    for name, pat in BUCKETS:
        if re.search(pat, t):
            return name
    return "uncategorised"

def main(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in ("Root_ID","Question_No","QA_Type","Seed_Type","Judge","Criterion_No","Criterion","Verdict"):
        if col not in df.columns:
            sys.exit(f"missing column: {col}")
    df["Verdict"] = df["Verdict"].str.strip()
    df["Bucket"] = df["Criterion"].map(bucket)

    # coverage check: how much of the corpus fell through to uncategorised
    unc = (df["Bucket"] == "uncategorised").mean()
    print(f"uncategorised criteria: {unc:.1%} of all judge rows "
          f"({(df['Bucket']=='uncategorised').sum()} rows)  <- keep this low\n")

    # sanity: verdict distribution
    print("verdict counts:", df["Verdict"].value_counts().to_dict(), "\n")

    # --- FAIL RATE: fraction of judge rows scored No, per bucket x type x track ---
    df["is_no"] = (df["Verdict"] == "No").astype(int)
    fail = (df.groupby(["QA_Type","Seed_Type","Bucket"])
              .agg(judge_rows=("is_no","size"), no_rate=("is_no","mean"))
              .reset_index())

    # --- DISAGREEMENT RATE: per (question, criterion), do the two judges split? ---
    # a criterion instance = one (Root_ID, Question_No, Criterion) with 2 judge rows.
    piv = (df.pivot_table(index=["Root_ID","Question_No","QA_Type","Seed_Type","Bucket","Criterion"],
                          columns="Judge", values="Verdict", aggfunc="first"))
    piv = piv.reset_index()
    judge_cols = [c for c in piv.columns if c not in
                  ("Root_ID","Question_No","QA_Type","Seed_Type","Bucket","Criterion")]
    if len(judge_cols) < 2:
        print("WARNING: fewer than 2 judges found, disagreement undefined")
    # split = judges present and not all equal
    def is_split(row):
        vals = [row[j] for j in judge_cols if isinstance(row[j], str) and row[j] != ""]
        return 1 if len(vals) >= 2 and len(set(vals)) > 1 else 0
    piv["split"] = piv.apply(is_split, axis=1)
    dis = (piv.groupby(["QA_Type","Seed_Type","Bucket"])
              .agg(criterion_instances=("split","size"), disagree_rate=("split","mean"))
              .reset_index())

    out = fail.merge(dis, on=["QA_Type","Seed_Type","Bucket"], how="outer")
    out = out.sort_values(["QA_Type","Seed_Type","no_rate"], ascending=[True,True,False])
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False,
          formatters={"no_rate":"{:.3f}".format, "disagree_rate":"{:.3f}".format}))

    # --- headline: bucket-level, pooled across type and track ---
    print("\n=== pooled across type and track ===")
    pooled_fail = df.groupby("Bucket")["is_no"].agg(["size","mean"]).rename(
        columns={"size":"judge_rows","mean":"no_rate"})
    pooled_dis = piv.groupby("Bucket")["split"].agg(["size","mean"]).rename(
        columns={"size":"criterion_instances","mean":"disagree_rate"})
    pooled = pooled_fail.join(pooled_dis).sort_values("no_rate", ascending=False)
    print(pooled.to_string(formatters={"no_rate":"{:.3f}".format,
                                       "disagree_rate":"{:.3f}".format}))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python criterion_diagnostics.py judgments.tsv")
    main(sys.argv[1])