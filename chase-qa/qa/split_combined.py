#!/usr/bin/env python3
"""
Split combined_corpus.tsv into one TSV per QA type, for run_eval.sh.

run_eval.sh takes a single corpus and type 3 needs a different evaluator prompt,
so the combined corpus cannot be solved in one run. Splitting by QA_Type gives
three solver runs per (model, solver prompt) instead of three per batch.

    python split_combined.py -combined generation_outputs/combined_corpus.tsv

    ./run_eval.sh -i generation_outputs/combined_type1.tsv -r t1-gpt55-v8 \\
                  -m gpt-5.5 -t elm -p zero-shot-basic -e zero-shot-basic -s 8
    # type 3 needs -e type3-causal-check

Row order inside each file is the order dedup_batches.py wrote, which is run
order then type order. solver.py and evaluator.py join back positionally, so do
not reorder or rewrite these files once a run has started.
"""

import argparse
import os
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-combined", type=str,
                   default="generation_outputs/combined_corpus.tsv")
    p.add_argument("-out_dir", type=str, default="",
                   help="Default: the directory holding -combined")
    p.add_argument("-stem", type=str, default="combined",
                   help="Output files are <stem>_type<N>.tsv. Default: combined")
    args = p.parse_args()

    if not os.path.isfile(args.combined):
        sys.exit("no combined corpus at " + args.combined)
    out_dir = args.out_dir or os.path.dirname(args.combined) or "."

    kw = dict(sep="\t", dtype=str, keep_default_na=False)
    df = pd.read_csv(args.combined, **kw)
    for c in ("QA_Type", "QID", "Batch"):
        if c not in df.columns:
            sys.exit("no {} column in {}".format(c, args.combined))
    if df["QID"].str.strip().duplicated().any():
        sys.exit("duplicate QID in " + args.combined)

    for qa_type, sub in df.groupby(df["QA_Type"].str.strip(), sort=True):
        if not qa_type.startswith("type"):
            sys.exit("unexpected QA_Type value '{}'".format(qa_type))
        out = os.path.join(out_dir, "{}_{}.tsv".format(args.stem, qa_type))
        sub = sub.reset_index(drop=True)
        sub.to_csv(out, sep="\t", index=False)

        # Reg_Text and Documents carry literal newlines. Prove the write survives
        # a read before any solver run is started against it.
        rt = pd.read_csv(out, **kw)
        if len(rt) != len(sub) or list(rt.columns) != list(sub.columns):
            sys.exit("round-trip shape mismatch on " + out)
        if not rt.equals(sub.astype(str)):
            sys.exit("round-trip content mismatch on " + out + ". Do not proceed.")
        print("wrote {}  {} rows, {} batch(es), round-trip verified".format(
            out, len(sub), sub["Batch"].nunique()))


if __name__ == "__main__":
    main()