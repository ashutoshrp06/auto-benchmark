#!/usr/bin/env python
# t47_question_level.py -- question-level self-preference, author judge vs independent judge.
# usage: python t47_question_level.py [--seed-type a,b] <judgments.tsv> [<judgments.tsv> ...]
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, binomtest

REQUIRED = ["Root_ID","Question_No","QA_Type","Seed_Type","Judge","Criterion_No","Criterion","Verdict"]
CRIT_KEY = ["Batch","Root_ID","Question_No","QA_Type","Criterion_No"]
Q_KEY    = ["Batch","Root_ID","Question_No","QA_Type"]
AUTHOR   = "gpt54mini"

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
        sys.exit(f"FATAL: duplicate batch label '{label}' from {p}; Root_ID collides across batches")
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
before = len(raw)
d = raw.drop_duplicates(subset=CRIT_KEY + ["Judge"], keep="last").copy()
print(f"rows {before} -> {len(d)} after dedup")

judges = sorted(d["Judge"].unique())
if len(judges) != 2:
    sys.exit(f"FATAL: expected exactly 2 judges, found {judges}")
if AUTHOR not in judges:
    sys.exit(f"FATAL: author judge '{AUTHOR}' not among {judges}")
OTHER = [j for j in judges if j != AUTHOR][0]

counts = d.groupby(CRIT_KEY, sort=False).size()
if (counts != 2).any():
    sys.exit(f"FATAL: {int((counts != 2).sum())} criteria not graded by exactly 2 judges")

d["Accept"] = (d["Verdict"] == "Yes").astype(int)
u = d.groupby("Judge")["Verdict"].apply(lambda s: (s == "Unsure").mean())
print("\nUnsure share by judge: " + ", ".join(f"{k} {v:.4%}" for k, v in u.items()))
print("Accept counts Yes only, so Unsure falls with No, matching the shipped table.")

# --- criterion level, to check against the shipped table ---
print("\n=== CRITERION LEVEL (reproduces the shipped table if the population matches) ===")
crit = d.groupby(["QA_Type", "Judge"])["Accept"].agg(["mean", "size"])
for t in sorted(d["QA_Type"].unique()):
    a, o = crit.loc[(t, AUTHOR)], crit.loc[(t, OTHER)]
    print(f"  {t}: {AUTHOR} {a['mean']:.4%} (n={int(a['size'])}), "
          f"{OTHER} {o['mean']:.4%} (n={int(o['size'])}), "
          f"diff {100*(a['mean']-o['mean']):+.2f}pp")
pa = d[d["Judge"] == AUTHOR]["Accept"]
po = d[d["Judge"] == OTHER]["Accept"]
print(f"  pooled: {AUTHOR} {pa.mean():.4%}, {OTHER} {po.mean():.4%}, "
      f"diff {100*(pa.mean()-po.mean()):+.2f}pp, criteria per judge n={len(pa)}, "
      f"total graded n={len(d)}")

# --- question level ---
q = (d.groupby(Q_KEY + ["Judge"], sort=False)
       .agg(Accept_Frac=("Accept", "mean"), N_Criteria=("Accept", "size"))
       .reset_index())

nlen = q.pivot(index=Q_KEY, columns="Judge", values="N_Criteria")
if nlen.isna().any().any():
    sys.exit("FATAL: a question is missing one judge entirely")
if (nlen[AUTHOR] != nlen[OTHER]).any():
    sys.exit(f"FATAL: {int((nlen[AUTHOR] != nlen[OTHER]).sum())} questions have "
             "different checklist lengths per judge; the pairing is not like for like")

w = q.pivot(index=Q_KEY, columns="Judge", values="Accept_Frac").reset_index()
if w[[AUTHOR, OTHER]].isna().any().any():
    sys.exit("FATAL: a question is missing one judge's accept fraction")
w["Diff"] = w[AUTHOR] - w[OTHER]

print("\n=== QUESTION LEVEL (each question one observation, both judges paired) ===")
rows = []
for t in sorted(w["QA_Type"].unique()) + ["POOLED"]:
    s = w if t == "POOLED" else w[w["QA_Type"] == t]
    dif = s["Diff"].to_numpy()
    n, nz = len(dif), int((dif != 0).sum())
    neg = int((dif < 0).sum())
    tt = ttest_rel(s[AUTHOR], s[OTHER])
    if nz == 0:
        wp = float("nan")
        sign_p = float("nan")
    else:
        _, wp = wilcoxon(dif, zero_method="wilcox", alternative="two-sided")
        sign_p = binomtest(neg, nz, 0.5).pvalue
    rows.append(dict(Type=t, n=n, nonzero=nz, author_stricter=neg,
                     author=s[AUTHOR].mean(), other=s[OTHER].mean(),
                     diff_pp=100*dif.mean(), median_pp=100*pd.Series(dif).median(),
                     wilcoxon_p=wp, sign_p=sign_p, paired_t_p=tt.pvalue))

pd.set_option("display.float_format", lambda v: f"{v:.6g}")
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nauthor judge = {AUTHOR}, independent judge = {OTHER}; "
      "diff is author minus independent, negative means the author is stricter")
print("n is questions, nonzero is questions where the two judges differ at all, "
      "author_stricter counts questions where the author accepts a smaller fraction")
