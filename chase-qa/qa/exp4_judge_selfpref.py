"""
RQ4b: judge self-preference test.

Does gpt-5.4-mini (checklist author) accept its own checklists' criteria more
often than gemini-3.5-flash-lite (independent judge) on the same criteria?

Input: generation_outputs/elm-docs-type{1,2,3}-v9-b*-r*/rocketeval/judgments.tsv
Each row = one (Root_ID, Question_No, Criterion_No, Judge) verdict.

Root_ID is only unique within a batch (confirmed: v9_frame.tsv join key is
(Batch, Root_ID, Question_No), not QID/Root_ID alone). judgments.tsv carries
no Batch column, so it is derived here from the folder name. Skipping this
would silently collide Root_IDs across batches and corrupt every pairing.

judge_panel.py appends in resume mode -> duplicate rows possible. Deduped on
(Batch, Root_ID, Question_No, Criterion_No, Judge), keeping first, with a
warning if duplicates disagree (real reproducibility problem, not resume noise).

Track: reg = Seed_Type in {reg, dynamic_reg}, else generic. Reg-only is used
for the primary result (see chat rationale: generic checklists have no
external anchor, so a leniency gap there is not attributable to the judge).
Unsure is collapsed into "No" for the binary McNemar test (consistent with
pass_frac / RQ3 methodology elsewhere in this project) and reported
separately as a raw diagnostic rate per judge.

Usage: python judge_selfpref.py [--root generation_outputs] [--all-tracks]
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    mcnemar = None

AUTHOR_JUDGE = "gpt54mini"
OTHER_JUDGE = "gemini35flashlite"
REG_SEED_TYPES = {"reg", "dynamic_reg"}

FOLDER_RE = re.compile(r"elm-docs-type(\d)-v9-(b\d+)-r\d+")


def find_files(root):
    pattern = os.path.join(root, "elm-docs-type*-v9-b*-r*", "rocketeval", "judgments.tsv")
    return sorted(glob.glob(pattern))


def load_all(root):
    files = find_files(root)
    if not files:
        sys.exit(
            "No judgments.tsv found under pattern '{}'. "
            "Check --root or confirm files exist.".format(
                os.path.join(root, "elm-docs-type*-v9-b*-r*", "rocketeval", "judgments.tsv")
            )
        )

    frames = []
    per_file_counts = []
    for fp in files:
        m = FOLDER_RE.search(fp)
        if not m:
            print("WARNING: could not parse type/batch from path, skipping: {}".format(fp), file=sys.stderr)
            continue
        qa_type, batch = m.group(1), m.group(2)
        df = pd.read_csv(fp, sep="\t", dtype=str, keep_default_na=False)
        required = {"Root_ID", "Question_No", "QA_Type", "Seed_Type", "Judge", "Criterion_No", "Criterion", "Verdict"}
        missing = required - set(df.columns)
        if missing:
            print("WARNING: {} missing columns {}, skipping.".format(fp, missing), file=sys.stderr)
            continue
        df["Batch"] = batch
        df["File_Type"] = "type" + qa_type
        df["_SourceFile"] = fp
        per_file_counts.append((fp, len(df)))
        frames.append(df)

    if not frames:
        sys.exit("All files failed to parse. Aborting.")

    all_df = pd.concat(frames, ignore_index=True)

    type_totals = all_df.groupby("File_Type").size().to_dict()
    print("Loaded {} files, {} raw rows. Per-type totals: {}".format(
        len(per_file_counts), len(all_df), type_totals))

    mismatch = all_df[all_df["File_Type"] != ("type" + all_df["QA_Type"].astype(str).str.extract(r"(\d)")[0])]
    if len(mismatch) > 0:
        print(
            "WARNING: {} rows where folder-derived File_Type disagrees with the "
            "in-data QA_Type column. Something is mislabeled upstream -- inspect "
            "before trusting any per-type split.".format(len(mismatch)),
            file=sys.stderr,
        )

    return all_df, per_file_counts


def dedupe(df, keep="first"):
    key_cols = ["File_Type", "Batch", "Root_ID", "Question_No", "Criterion_No", "Judge"]
    n_before = len(df)

    df["_Bin"] = df["Verdict"].apply(to_binary)

    grouped_raw = df.groupby(key_cols)["Verdict"].nunique()
    conflicting_raw = grouped_raw[grouped_raw > 1]

    grouped_bin = df.groupby(key_cols)["_Bin"].nunique()
    conflicting_bin = grouped_bin[grouped_bin > 1]

    print(
        "Duplicate-key diagnostic: {} keys with raw-verdict disagreement "
        "(includes harmless Unsure<->No swaps); {} keys with BINARY (accept/reject) "
        "disagreement -- these are the ones that can actually move the reported "
        "accept rate / McNemar result.".format(len(conflicting_raw), len(conflicting_bin))
    )
    if len(conflicting_bin) > 0:
        bin_conflict_keys = set(conflicting_bin.index)
        by_judge = {}
        for key in bin_conflict_keys:
            judge = key[-1]
            by_judge.setdefault(judge, 0)
            by_judge[judge] += 1
        print("  Binary-disagreement count by Judge: {}".format(by_judge))

        # first-vs-last direction check: for each conflicting key, compare the
        # binary outcome of the first occurrence vs the last occurrence.
        flips_first_yes_last_no = 0
        flips_first_no_last_yes = 0
        conflict_df = df[df.set_index(key_cols).index.isin(bin_conflict_keys)]
        for key, g in conflict_df.groupby(key_cols):
            first_bin = g["_Bin"].iloc[0]
            last_bin = g["_Bin"].iloc[-1]
            if first_bin == 1 and last_bin == 0:
                flips_first_yes_last_no += 1
            elif first_bin == 0 and last_bin == 1:
                flips_first_no_last_yes += 1
        print(
            "  Direction if switching keep='first'->keep='last': "
            "{} keys go accept->reject, {} go reject->accept "
            "(if these are roughly balanced, tie-break choice is not driving the result; "
            "if lopsided, it is, and the reported diff is not yet trustworthy).".format(
                flips_first_yes_last_no, flips_first_no_last_yes
            )
        )

    df = df.drop_duplicates(subset=key_cols, keep=keep)
    df = df.drop(columns=["_Bin"])
    n_after = len(df)
    if n_before != n_after:
        print("Deduped (keep='{}'): {} -> {} rows ({} duplicates removed).".format(
            keep, n_before, n_after, n_before - n_after))
    return df


def dump_conflicts(df, out_path, n_groups=15):
    """Write full raw rows for the first n_groups binary-conflicting keys, for manual inspection."""
    key_cols = ["File_Type", "Batch", "Root_ID", "Question_No", "Criterion_No", "Judge"]
    df = df.copy()
    df["_Bin"] = df["Verdict"].apply(to_binary)
    grouped_bin = df.groupby(key_cols)["_Bin"].nunique()
    conflicting = grouped_bin[grouped_bin > 1]
    if len(conflicting) == 0:
        print("No binary conflicts to dump.")
        return
    keys = list(conflicting.index)[:n_groups]
    cols = ["File_Type", "Batch", "Root_ID", "Question_No", "Criterion_No", "Judge",
            "QA_Type", "Seed_Type", "Verdict", "Criterion", "_SourceFile"]
    rows = []
    for key in keys:
        mask = pd.Series(True, index=df.index)
        for col, val in zip(key_cols, key):
            mask &= (df[col] == val)
        rows.append(df.loc[mask, cols])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_path, sep="\t", index=False)
    print("Dumped {} raw rows across {} conflicting keys to {}".format(len(out), len(keys), out_path))


def to_binary(verdict):
    return 1 if str(verdict).strip().lower() == "yes" else 0


def run_mcnemar(b, c):
    """b = author-Yes/other-No, c = author-No/other-Yes discordant counts."""
    if mcnemar is None:
        return None
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=(b + c < 25))
    return result.pvalue


def analyse_subset(df, label):
    key_cols = ["File_Type", "Batch", "Root_ID", "Question_No", "Criterion_No"]
    pivot = df.pivot_table(
        index=key_cols, columns="Judge", values="Verdict", aggfunc="first"
    )
    if AUTHOR_JUDGE not in pivot.columns or OTHER_JUDGE not in pivot.columns:
        print("[{}] Missing one of the two judges in this subset, skipping. Columns: {}".format(
            label, list(pivot.columns)))
        return None

    paired = pivot[[AUTHOR_JUDGE, OTHER_JUDGE]].dropna()
    n = len(paired)
    if n == 0:
        print("[{}] No paired criteria after alignment, skipping.".format(label))
        return None

    author_bin = paired[AUTHOR_JUDGE].apply(to_binary)
    other_bin = paired[OTHER_JUDGE].apply(to_binary)

    author_rate = author_bin.mean()
    other_rate = other_bin.mean()
    diff_pp = (author_rate - other_rate) * 100

    b = int(((author_bin == 1) & (other_bin == 0)).sum())  # author Yes, other No
    c = int(((author_bin == 0) & (other_bin == 1)).sum())  # author No, other Yes
    pval = run_mcnemar(b, c)

    author_unsure_rate = (paired[AUTHOR_JUDGE].str.lower() == "unsure").mean()
    other_unsure_rate = (paired[OTHER_JUDGE].str.lower() == "unsure").mean()

    print("\n=== {} (n={}) ===".format(label, n))
    print("  {} accept rate: {:.4f}".format(AUTHOR_JUDGE, author_rate))
    print("  {} accept rate: {:.4f}".format(OTHER_JUDGE, other_rate))
    print("  Diff (author - other): {:+.2f}pp".format(diff_pp))
    print("  Discordant: b(author Yes/other No)={} c(author No/other Yes)={}".format(b, c))
    if pval is not None:
        sig = "yes" if pval < 0.05 else "no"
        print("  McNemar p={:.4f} (sig @ .05: {})".format(pval, sig))
    else:
        print("  McNemar p=N/A (statsmodels not installed)")
    print("  Unsure rate -- {}: {:.4f}, {}: {:.4f} (diagnostic only, not in McNemar)".format(
        AUTHOR_JUDGE, author_unsure_rate, OTHER_JUDGE, other_unsure_rate))

    return {
        "label": label, "n": n, "author_rate": author_rate, "other_rate": other_rate,
        "diff_pp": diff_pp, "b": b, "c": c, "pval": pval,
    }


def inspect_solver_frame(path):
    """Print unique values of key columns so filter strings can be confirmed
    before running the real computation -- avoids guessing wrong and wasting a run."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"QID", "Batch", "QA_Type", "Seed_Type", "Verdict", "Pass_Frac", "N_Criteria", "N_Pass", "Model", "Cond", "Result", "Track"}
    missing = required - set(df.columns)
    if missing:
        print("WARNING: {} missing expected columns: {}".format(path, missing), file=sys.stderr)
    print("Rows: {}".format(len(df)))
    for col in ["Model", "Cond", "Track", "QA_Type", "Verdict", "Result"]:
        if col in df.columns:
            print("  {} unique values: {}".format(col, sorted(df[col].unique().tolist())[:20]))


def analyse_solver(path, model_val, cond_val, out_path=None):
    """Recompute reg+PASS accuracy by type for one model/condition from v9_frame.tsv,
    independent of any pre-computed table. Result column assumed to be 0/1 or a
    string coercible to float (e.g. '1', '0', '1.0')."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"QA_Type", "Model", "Cond", "Result", "Track", "Verdict"}
    missing = required - set(df.columns)
    if missing:
        sys.exit("v9_frame.tsv missing required columns: {}. Run --inspect-solver first.".format(missing))

    sub = df[(df["Model"] == model_val) & (df["Cond"] == cond_val) & (df["Track"] == "reg") & (df["Verdict"] == "PASS")]
    if len(sub) == 0:
        sys.exit(
            "Zero rows after filtering Model=='{}' Cond=='{}' Track=='reg' Verdict=='PASS'. "
            "Filter values are likely wrong -- run --inspect-solver and check the printed "
            "unique values before retrying.".format(model_val, cond_val)
        )

    try:
        sub = sub.copy()
        sub["_Result"] = sub["Result"].astype(float)
    except ValueError as e:
        sys.exit("Result column not numeric-coercible: {}. Inspect actual values before proceeding.".format(e))

    print("\n{} / {} / reg+PASS -- recomputed from {}".format(model_val, cond_val, path))
    for t in sorted(sub["QA_Type"].unique()):
        tsub = sub[sub["QA_Type"] == t]
        acc = tsub["_Result"].mean()
        print("  {}: n={}, accuracy={:.4f}".format(t, len(tsub), acc))
    overall = sub["_Result"].mean()
    print("  pooled: n={}, accuracy={:.4f}".format(len(sub), overall))

    if out_path:
        sub[["QID", "Batch", "QA_Type", "Model", "Cond", "Track", "Verdict", "_Result"]].to_csv(out_path, sep="\t", index=False)
        print("Wrote filtered rows to {}".format(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="generation_outputs")
    ap.add_argument("--all-tracks", action="store_true", help="Also report all-track (reg+generic) numbers alongside reg-only")
    ap.add_argument("--keep", choices=["first", "last"], default="first", help="Which duplicate occurrence to keep on resume-artefact collisions")
    ap.add_argument("--dump-conflicts", type=str, default=None, help="Path to write raw rows for the first 15 binary-conflicting keys, for manual inspection. Exits after dumping.")
    ap.add_argument("--solver-frame", type=str, default="analysis/v9_frame.tsv", help="Path to the merged solver analysis frame")
    ap.add_argument("--inspect-solver", action="store_true", help="Print unique column values from the solver frame and exit -- run this before --solver-check")
    ap.add_argument("--solver-check", action="store_true", help="Recompute reg+PASS solver accuracy by type, independent of any pre-computed table")
    ap.add_argument("--solver-model", type=str, default=None, help="Exact Model value to filter on, e.g. as printed by --inspect-solver")
    ap.add_argument("--solver-cond", type=str, default=None, help="Exact Cond value to filter on, e.g. as printed by --inspect-solver")
    ap.add_argument("--solver-out", type=str, default=None, help="Optional path to dump the filtered solver rows for inspection")
    args = ap.parse_args()

    if args.inspect_solver:
        inspect_solver_frame(args.solver_frame)
        return None

    if args.solver_check:
        if not args.solver_model or not args.solver_cond:
            sys.exit("--solver-check requires --solver-model and --solver-cond. Run --inspect-solver first to get exact values.")
        analyse_solver(args.solver_frame, args.solver_model, args.solver_cond, args.solver_out)
        return None

    all_df, per_file_counts = load_all(args.root)

    if args.dump_conflicts:
        dump_conflicts(all_df, args.dump_conflicts)
        return None

    all_df = dedupe(all_df, keep=args.keep)


    all_df["Track"] = all_df["Seed_Type"].apply(lambda s: "reg" if s in REG_SEED_TYPES else "generic")
    panel_df = all_df[all_df["Judge"].isin([AUTHOR_JUDGE, OTHER_JUDGE])]

    reg_df = panel_df[panel_df["Track"] == "reg"]

    print("\n" + "#" * 60)
    print("PRIMARY RESULT -- reg-track only")
    print("#" * 60)

    results = []
    for t in ["type1", "type2", "type3"]:
        sub = reg_df[reg_df["File_Type"] == t]
        r = analyse_subset(sub, "{} reg".format(t))
        if r:
            results.append(r)
    r_pooled = analyse_subset(reg_df, "pooled reg")
    if r_pooled:
        results.append(r_pooled)

    if args.all_tracks:
        print("\n" + "#" * 60)
        print("SECONDARY (reference only) -- all tracks, reg+generic")
        print("#" * 60)
        for t in ["type1", "type2", "type3"]:
            sub = panel_df[panel_df["File_Type"] == t]
            analyse_subset(sub, "{} all-tracks".format(t))
        analyse_subset(panel_df, "pooled all-tracks")

    return results


if __name__ == "__main__":
    main()