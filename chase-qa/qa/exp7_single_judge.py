#!/usr/bin/env python
# exp7_single_judge.py -- single-judge counterfactual over stored panel verdicts.
# usage: python exp7_single_judge.py [--seed-type VALUE] run1/judgments.tsv [run2/judgments.tsv ...]
import sys
from pathlib import Path
import pandas as pd

REQUIRED = ["Root_ID", "Question_No", "QA_Type", "Seed_Type",
            "Judge", "Criterion_No", "Criterion", "Verdict"]
CRIT_KEY = ["Batch", "Root_ID", "Question_No", "QA_Type", "Criterion_No"]
Q_KEY    = ["Batch", "Root_ID", "Question_No", "QA_Type"]
PASS_FRAC, NOMAJ = 0.75, 0.34
SHIPPED = {"type1": 306, "type2": 363, "type3": 332}   # reg-track PASS, Table 3.2

args = sys.argv[1:]
seed_type = None
if args and args[0] == "--seed-type":
    if len(args) < 3:
        sys.exit("FATAL: --seed-type needs a value and at least one path")
    seed_type, args = set(args[1].split(",")), args[2:]
if not args:
    sys.exit("FATAL: pass at least one judgments.tsv path")

frames, labels = [], []
for p in args:
    path = Path(p).resolve()
    if not path.is_file():
        sys.exit(f"FATAL: {p} is not a file")
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"FATAL: {p} missing columns {missing}")
    label = path.parent.parent.name if path.parent.name == "rocketeval" else path.parent.name
    if label in labels:
        sys.exit(f"FATAL: duplicate batch label '{label}' from {p}; "
                 "Root_ID collides across batches and would silently merge")
    labels.append(label)
    df["Batch"] = label
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"batches: {labels}")
print(f"Seed_Type values present: {sorted(raw['Seed_Type'].dropna().unique())}")

if seed_type is not None:
    present = set(raw["Seed_Type"].dropna().unique())
    unknown = seed_type - present
    if unknown:
        sys.exit(f"FATAL: --seed-type values {sorted(unknown)} not in {sorted(present)}")
    raw = raw[raw["Seed_Type"].isin(seed_type)].copy()
    print(f"filtered to Seed_Type in {sorted(seed_type)}: {len(raw)} rows")

bad = set(raw["Verdict"].unique()) - {"Yes", "No", "Unsure"}
if bad:
    sys.exit(f"FATAL: unexpected Verdict values {sorted(bad)}")

# judge_panel.py writes with mode='a', so dedup on the full key before anything else.
# keep='last' takes the later write of a resumed run.
before = len(raw)
d = raw.drop_duplicates(subset=CRIT_KEY + ["Judge"], keep="last").copy()
print(f"rows {before} -> {len(d)} after dedup")

judges = sorted(d["Judge"].unique())
if len(judges) != 2:
    sys.exit(f"FATAL: expected exactly 2 judges, found {judges}")

counts = d.groupby(CRIT_KEY, sort=False).size()
if (counts > 2).any():
    sys.exit(f"FATAL: {int((counts > 2).sum())} criteria carry >2 judge rows after dedup")
short = int((counts < 2).sum())
if short:
    sys.exit(f"FATAL: {short} criteria graded by only 1 judge; the single-judge "
             "branches would run on different criterion denominators")

def label_panel(votes):
    kept = [v for v in votes if v != "Unsure"]
    if not kept:
        return "Undecided"
    if all(v == "Yes" for v in kept):
        return "Yes"
    if all(v == "No" for v in kept):
        return "No"
    return "DISAGREE"

def verdicts(crit_labels):
    g = crit_labels.groupby(Q_KEY, sort=False)["Label"]
    out = pd.DataFrame({
        "N_Criteria": g.size(),
        "N_Yes":      g.agg(lambda s: int((s == "Yes").sum())),
        "N_NoMaj":    g.agg(lambda s: int(s.isin(["DISAGREE", "Undecided"]).sum())),
    }).reset_index()
    if (out["N_Criteria"] == 0).any():
        sys.exit("FATAL: question with zero criteria")
    out["Pass_Frac"] = out["N_Yes"] / out["N_Criteria"]
    out["No_Majority_Frac"] = out["N_NoMaj"] / out["N_Criteria"]
    out["Verdict"] = "FAIL"
    out.loc[out["Pass_Frac"] >= PASS_FRAC, "Verdict"] = "PASS"
    out.loc[out["No_Majority_Frac"] > NOMAJ, "Verdict"] = "DISAGREE"
    return out

configs = {}
panel_lab = (d.groupby(CRIT_KEY, sort=False)["Verdict"]
               .agg(lambda s: label_panel(list(s)))
               .rename("Label").reset_index())
configs["panel"] = verdicts(panel_lab)

for j in judges:
    sub = d[d["Judge"] == j].copy()
    sub["Label"] = sub["Verdict"].where(sub["Verdict"] != "Unsure", "Undecided")
    configs[j] = verdicts(sub[CRIT_KEY + ["Label"]])

for name, v in configs.items():
    print(f"\n=== {name} (n={len(v)}) ===")
    print(pd.crosstab(v["QA_Type"], v["Verdict"]))

pan = configs["panel"]
print("\n=== reconciliation, panel branch against shipped PASS counts ===")
for t, want in SHIPPED.items():
    got = int(((pan["QA_Type"] == t) & (pan["Verdict"] == "PASS")).sum())
    print(f"  {t}: recomputed {got}, shipped {want}, {'match' if got == want else 'MISMATCH'}")

base = pan.set_index(Q_KEY)["Verdict"]
for j in judges:
    alt = configs[j].set_index(Q_KEY)["Verdict"]
    joined = pd.concat([base.rename("panel"), alt.rename("single")], axis=1)
    if joined.isna().any().any():
        sys.exit(f"FATAL: question set mismatch between panel and {j}")
    print(f"\n=== panel vs {j} ===")
    print(pd.crosstab(joined["panel"], joined["single"]))
    print(f"verdict agreement: {(joined['panel'] == joined['single']).mean():.4f}")
    print("panel DISAGREE reassigned to:",
          joined[joined["panel"] == "DISAGREE"]["single"]
          .value_counts(normalize=True).round(4).to_dict())
