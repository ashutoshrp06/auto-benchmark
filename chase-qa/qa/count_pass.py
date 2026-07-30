#!/usr/bin/env python3
"""
Cumulative RocketEval PASS count across one or more batch runs.

Joins each batch's cleaned corpus to that batch's rocketeval/poll_scores.tsv on
(Root_ID, Question_No), applies the track filter and the Pass_Frac threshold,
then restricts to the QIDs that survived dedup_batches.py.

The detailed report goes to stderr. The final integer goes to stdout alone, so a
shell can capture it with $( ).

    python count_pass.py --runs v8-b1,v8-b2 --track reg --combined combined_corpus.tsv

Seed_Type note: seeded scenarios are 'reg'/'generic', stage-1 expanded scenarios
are 'dynamic_reg'/'dynamic_generic'. Both count toward their track. Any other
value is reported and counted in neither.
"""

import argparse
import os
import sys

import pandas as pd

TRACKS = {
    "reg": {"reg", "dynamic_reg"},
    "generic": {"generic", "dynamic_generic"},
}
KNOWN = TRACKS["reg"] | TRACKS["generic"]
CLEAN_STEM = "programmatic_data_modified_verified_cleaned.tsv"


def err(*a):
    print(*a, file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True,
                   help="comma-separated run names, e.g. v8-b1,v8-b2")
    p.add_argument("--types", default="1,2,3")
    p.add_argument("--prefix", default="elm")
    p.add_argument("--out-dir", default="generation_outputs")
    p.add_argument("--combined",
                   help="combined_corpus.tsv from dedup_batches.py. QIDs absent "
                        "from it are not counted. Omit to count every batch in "
                        "full, which double counts anything dedup would drop")
    p.add_argument("--pass-frac", type=float, default=0.75)
    p.add_argument("--track", default="reg",
                   choices=["reg", "generic", "all", "cleaned"],
                   help="'cleaned' counts rows with no PASS filter and needs no "
                        "poll_scores.tsv")
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    if not runs or not types:
        sys.exit("--runs and --types must be non-empty")
    kw = dict(sep="\t", dtype=str, keep_default_na=False)

    keep = None
    if args.combined:
        if not os.path.isfile(args.combined):
            sys.exit("no combined corpus at " + args.combined)
        comb = pd.read_csv(args.combined, **kw)
        for c in ("QID", "Batch", "QA_Type"):
            if c not in comb.columns:
                sys.exit("no {} column in {}. dedup_batches.py writes all three; "
                         "this file was not produced by it".format(c, args.combined))
        if comb["QID"].str.strip().duplicated().any():
            sys.exit("duplicate QID in " + args.combined)
        # Match on the full triple, not QID alone. dedup keeps one specific row
        # per QID and the copies were judged independently, so a bare QID test
        # would credit a batch whose own copy failed.
        keep = set(zip(comb["Batch"].str.strip(),
                       comb["QA_Type"].str.strip(),
                       comb["QID"].str.strip()))
        err("combined corpus: {} rows, all QIDs distinct".format(len(comb)))

    raw = 0
    passed = set()
    per = {}
    unknown = {}

    for run in runs:
        for t in types:
            d = os.path.join(args.out_dir,
                             "{}-docs-type{}-{}".format(args.prefix, t, run))
            clean_p = os.path.join(d, CLEAN_STEM)
            if not os.path.isfile(clean_p):
                sys.exit("no cleaned corpus at " + clean_p)
            clean = pd.read_csv(clean_p, **kw)
            for c in ("QID", "Root_ID", "Question_No", "Seed_Type"):
                if c not in clean.columns:
                    sys.exit("no {} column in {}".format(c, clean_p))

            for v in set(clean["Seed_Type"].str.strip()) - KNOWN:
                unknown.setdefault(v, set()).add(os.path.basename(d))

            if args.track in ("all", "cleaned"):
                mask = pd.Series(True, index=clean.index)
            else:
                mask = clean["Seed_Type"].str.strip().isin(TRACKS[args.track])

            if args.track != "cleaned":
                poll_p = os.path.join(d, "rocketeval", "poll_scores.tsv")
                if not os.path.isfile(poll_p):
                    sys.exit("no poll_scores.tsv at " + poll_p)
                poll = pd.read_csv(poll_p, **kw)
                for c in ("Root_ID", "Question_No", "Pass_Frac"):
                    if c not in poll.columns:
                        sys.exit("no {} column in {}".format(c, poll_p))
                key = ["Root_ID", "Question_No"]
                if poll.duplicated(key).any():
                    sys.exit("duplicate (Root_ID, Question_No) in {}. judge_panel.py "
                             "appends on resume. Delete {}/rocketeval and rejudge "
                             "this batch".format(poll_p, d))
                if clean.duplicated(key).any():
                    sys.exit("duplicate (Root_ID, Question_No) in " + clean_p)
                frac = pd.to_numeric(poll["Pass_Frac"], errors="coerce")
                if frac.isna().any():
                    sys.exit("non-numeric Pass_Frac in " + poll_p)
                m = clean[key].merge(
                    poll[key].assign(_p=(frac >= args.pass_frac - 1e-9)),
                    on=key, how="left", validate="one_to_one")
                if m["_p"].isna().any():
                    sys.exit("{} of {} rows in {} have no RocketEval verdict".format(
                        int(m["_p"].isna().sum()), len(clean), clean_p))
                mask = mask & pd.Series(m["_p"].to_numpy(dtype=bool),
                                        index=clean.index)

            if keep is not None:
                tag = ("type" + t)
                inkeep = [(run, tag, q) in keep for q in clean["QID"].str.strip()]
                mask = mask & pd.Series(inkeep, index=clean.index)

            n = int(mask.sum())
            per["{} type{}".format(run, t)] = (n, len(clean))
            raw += n
            passed |= set(clean.loc[mask, "QID"].str.strip())

    total = len(passed)

    for k in sorted(per):
        n, m = per[k]
        err("  {:<26} {:>5} counted of {:>5} cleaned".format(k, n, m))
    if raw != total:
        err("  {} passing rows collapse to {} distinct QID "
            "({} repeats across cells)".format(raw, total, raw - total))
    if unknown:
        err("WARNING unrecognised Seed_Type, counted in neither track:")
        for v, ds in sorted(unknown.items()):
            err("  '{}' in {}".format(v, ", ".join(sorted(ds))))
    err("track={}  pass_frac={}  TOTAL={}".format(
        args.track, "n/a" if args.track == "cleaned" else args.pass_frac, total))
    print(total)


if __name__ == "__main__":
    main()