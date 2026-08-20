"""
exp2b_significance_sweep.py

Extends exp2's pass_frac x unsure_mode sweep with actual paired significance
testing (McNemar + cluster-bootstrap 95% CI), gemini31pro vs gpt55, matching
the exact method used in exp6_v9b_significance.py (that script's mcnemar_exact
and cluster_bootstrap_diff are reused verbatim below) so this sweep's numbers
are comparable to the v8/v9/v9b headline table at the matching pass_frac=0.75
point, not just directionally similar under a different test.

exp2's accuracy-only sweep answered "does the ranking change." This answers
"is the gap ever not statistically distinguishable from chance, at any
threshold" -- a stricter question. Gate-loading, aggregation, and corpus/
solver-loading logic is copied unmodified from the verified exp2_threshold_
sensitivity.py (same paths, same Batch-collision guard, same duplicate-ID
guard) -- do not edit that logic here without mirroring back.

v9 corpus only (matching exp2's scope). Not v9b -- that's a separate corpus
with its own directory structure (generation_outputs/v9b/...) and would need
CORPUS_DIR/GATE_PATTERN/RESULT_PATTERN repointed if run on it.

Cluster bootstrap is the expensive part: N_BOOT=10000 x 11 pass_fracs x 3
unsure_modes x 3 types = 99 cells, benchmarked at ~3.3s/cell worst case
(n~800, ~800 clusters) -> ~5.5 min ceiling, faster in practice since n shrinks
at high pass_frac. Set RUN_BOOTSTRAP=False below for a McNemar-only fast pass
first if you want to sanity-check before committing to the full run.

Run from repo root:
    python3 exp2b_significance_sweep.py

Inputs: identical to exp2_threshold_sensitivity.py.

Outputs:
    analysis/exp2b_significance_grid.tsv  -- one row per (unsure_mode, pass_frac, type):
        n, acc_gem, acc_gpt, diff, b, c, mcnemar_p, cluster_diff_mean, cluster95_lo,
        cluster95_hi, leader_matches_pf75
    stdout -- summary table + explicit flag on any cell where the leader flips
              relative to pass_frac=0.75 within the same (unsure_mode, type)
"""

import os
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TYPES = [1, 2, 3]
BATCHES = ["b1", "b2", "b3", "b4"]
UNSURE_MODES = ["exclude", "zero", "half"]
PASS_FRAC_GRID = np.round(np.arange(0.50, 1.001, 0.05), 2)
DISAGREE_FRAC = 0.34
REG_SEED_TYPES = {"reg", "dynamic_reg"}
RUN_BOOTSTRAP = True  # set False for a fast McNemar-only pass

N_BOOT = 10000
np.random.seed(42)  # matches exp6_v9b_significance.py exactly

GEN_DIR = "generation_outputs"
CORPUS_DIR = "generation_outputs/v9"
OUT_DIR = "outputs"
ANALYSIS_DIR = "analysis"

GATE_PATTERN = GEN_DIR + "/elm-docs-type{t}-v9-{b}/rocketeval/judgments.tsv"
CORPUS_PATTERN = CORPUS_DIR + "/combined_type{t}.tsv"
RESULT_PATTERN = OUT_DIR + "/type{t}-{m}-v9-noirrelevant-eval/result.tsv"


# ---------------------------------------------------------------------------
# Significance functions -- copied verbatim from exp6_v9b_significance.py.
# Do not modify without mirroring back; the whole point of this script is
# comparability to that table's numbers.
# ---------------------------------------------------------------------------
def mcnemar_exact(b, c):
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return float('nan')
    k = min(b, c)
    def binom_cdf_le(k, n):
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(2 * binom_cdf_le(k, n), 1.0)


def cluster_bootstrap_diff(a, b, clus, n_boot=N_BOOT):
    """a, b: bool arrays (correct/incorrect per row). clus: cluster id per row."""
    clusters = np.unique(clus)
    idx_by_cluster = {c: np.where(clus == c)[0] for c in clusters}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = np.random.choice(clusters, size=len(clusters), replace=True)
        idxs = np.concatenate([idx_by_cluster[c] for c in sampled])
        diffs[i] = a[idxs].mean() - b[idxs].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi


# ---------------------------------------------------------------------------
# Gate + consensus logic -- copied verbatim from exp2_threshold_sensitivity.py.
# ---------------------------------------------------------------------------
def verdict_to_score(v, unsure_mode):
    if v == "Yes":
        return 1.0
    if v == "No":
        return 0.0
    if unsure_mode == "zero":
        return 0.0
    if unsure_mode == "half":
        return 0.5
    return np.nan


def criterion_consensus(verdicts, unsure_mode):
    scores = [verdict_to_score(v, unsure_mode) for v in verdicts]
    valid = [s for s in scores if not np.isnan(s)]
    if len(valid) == 0:
        return "Undecided", np.nan, True
    yes = sum(1 for s in valid if s >= 0.75)
    no = sum(1 for s in valid if s <= 0.25)
    if yes > no:
        label = "Yes"
    elif no > yes:
        label = "No"
    else:
        label = "DISAGREE"
    unanimous = (yes == len(valid)) or (no == len(valid))
    return label, float(np.mean(valid)), (not unanimous)


def load_gate_judgments(qa_type):
    frames = []
    per_batch_unique = []
    for b in BATCHES:
        path = GATE_PATTERN.format(t=qa_type, b=b)
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: missing gate file {path}")
        df_b = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df_b["Batch"] = f"v9-{b}"
        per_batch_unique.append(df_b[["Root_ID", "Question_No"]].drop_duplicates().shape[0])
        frames.append(df_b)
    j = pd.concat(frames, ignore_index=True)

    group_cols = ["Batch", "Root_ID", "Question_No", "QA_Type", "Seed_Type"]
    miss = j[group_cols].isna().any(axis=1) | (j[group_cols] == "").any(axis=1)
    if miss.any():
        raise SystemExit(f"ERROR: {int(miss.sum())} gate row(s) missing group cols for type{qa_type}")

    dup_key = ["Batch", "Root_ID", "Question_No", "Criterion_No", "Judge"]
    n_dup = int(j.duplicated(subset=dup_key).sum())
    if n_dup:
        raise SystemExit(f"ERROR: {n_dup} duplicate (batch, question, criterion, judge) row(s) in type{qa_type}")

    n_q_concat = j[["Batch", "Root_ID", "Question_No"]].drop_duplicates().shape[0]
    n_q_expected = sum(per_batch_unique)
    if n_q_concat != n_q_expected:
        raise SystemExit(
            f"ERROR: type{qa_type} unexpected count after Batch-tagging: got {n_q_concat}, "
            f"expected {n_q_expected}. Investigate before proceeding."
        )
    print(f"type{qa_type} gate: {len(j)} criterion-rows, {n_q_concat} unique (batch, question) pairs")
    return j


def aggregate_gate(j, unsure_mode):
    rows = []
    group_cols = ["Batch", "Root_ID", "Question_No", "QA_Type", "Seed_Type"]
    for key, g in j.groupby(group_cols):
        batch, root_id, q_no, qa_type, seed_type = key
        labels = []
        no_majority = 0
        for cno, cg in g.groupby("Criterion_No"):
            label, _, _ = criterion_consensus(list(cg["Verdict"]), unsure_mode)
            labels.append(label)
            if label in ("DISAGREE", "Undecided"):
                no_majority += 1
        n_crit = len(labels)
        n_pass = sum(1 for l in labels if l == "Yes")
        rows.append({
            "Batch": batch, "Root_ID": root_id, "Question_No": q_no, "QA_Type": qa_type, "Seed_Type": seed_type,
            "N_Criteria": n_crit, "N_Pass": n_pass,
            "Pass_Frac": n_pass / n_crit if n_crit else 0.0,
            "No_Majority_Frac": no_majority / n_crit if n_crit else 0.0,
        })
    return pd.DataFrame(rows)


def apply_threshold(agg_df, pass_frac_thresh, disagree_frac=DISAGREE_FRAC):
    def verdict(row):
        if row["No_Majority_Frac"] > disagree_frac:
            return "DISAGREE"
        if row["Pass_Frac"] >= pass_frac_thresh:
            return "PASS"
        return "FAIL"
    out = agg_df.copy()
    out["Verdict"] = out.apply(verdict, axis=1)
    out["Track"] = np.where(out["Seed_Type"].isin(REG_SEED_TYPES), "reg", "generic")
    return out


def load_corpus(qa_type):
    path = CORPUS_PATTERN.format(t=qa_type)
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing corpus file {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.reset_index(drop=True)
    df["ID"] = df.index + 1
    return df[["ID", "Batch", "Root_ID", "Question_No", "Seed_Type", "Question"]]


def load_solver_results(qa_type, model, corpus_df):
    path = RESULT_PATTERN.format(t=qa_type, m=model)
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing result file {path}")
    res = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "Result" not in res.columns or "ID" not in res.columns:
        raise SystemExit(f"ERROR: {path} missing ID/Result columns")
    res["ID"] = res["ID"].astype(int)
    res["Result"] = res["Result"].astype(int)

    dup_ids = int(res["ID"].duplicated().sum())
    if dup_ids:
        raise SystemExit(f"ERROR: {path} has {dup_ids} duplicate ID value(s)")
    missing_ids = set(corpus_df["ID"]) - set(res["ID"])
    if missing_ids:
        raise SystemExit(f"ERROR: {path} missing {len(missing_ids)} ID(s) present in corpus")

    merged = corpus_df.merge(res[["ID", "Question", "Result"]], on="ID", suffixes=("", "_res"))
    if len(merged) != len(corpus_df):
        raise SystemExit(f"ERROR: {path} row count mismatch vs corpus for type{qa_type}")
    mismatches = int((merged["Question"] != merged["Question_res"]).sum())
    if mismatches:
        raise SystemExit(f"ERROR: {mismatches} positional Question mismatch(es) for {path}")
    return merged[["Batch", "Root_ID", "Question_No", "Result"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    gate_raw = {t: load_gate_judgments(t) for t in TYPES}
    corpus = {t: load_corpus(t) for t in TYPES}

    # Paired gem/gpt correctness per type, computed once (doesn't depend on
    # pass_frac/unsure_mode), then filtered to the reg+PASS subset at each grid point.
    paired = {}
    for t in TYPES:
        sres_gem = load_solver_results(t, "gemini31pro", corpus[t]).rename(columns={"Result": "Correct_gem"})
        sres_gem["Correct_gem"] = sres_gem["Correct_gem"].astype(bool)
        sres_gpt = load_solver_results(t, "gpt55", corpus[t]).rename(columns={"Result": "Correct_gpt"})
        sres_gpt["Correct_gpt"] = sres_gpt["Correct_gpt"].astype(bool)
        pr = sres_gem.merge(sres_gpt, on=["Batch", "Root_ID", "Question_No"], how="inner", validate="one_to_one")
        if len(pr) != len(sres_gem):
            raise SystemExit(f"ERROR: type{t} gem/gpt pairing dropped rows -- key mismatch between the two result files")
        pr["Cluster_Key"] = pr["Batch"] + "_" + pr["Root_ID"]
        paired[t] = pr
        print(f"type{t}: {len(pr)} paired gemini31pro/gpt55 rows")

    results = []
    for unsure_mode in UNSURE_MODES:
        agg = {t: aggregate_gate(gate_raw[t], unsure_mode) for t in TYPES}
        for pf in PASS_FRAC_GRID:
            for t in TYPES:
                thr = apply_threshold(agg[t], pf)
                reg_pass = thr[(thr["Verdict"] == "PASS") & (thr["Track"] == "reg")]
                sub = reg_pass[["Batch", "Root_ID", "Question_No"]].merge(
                    paired[t], on=["Batch", "Root_ID", "Question_No"], how="inner"
                )

                n = len(sub)
                if n == 0:
                    results.append({
                        "unsure_mode": unsure_mode, "pass_frac": pf, "qa_type": t, "n": 0,
                        "acc_gem": float("nan"), "acc_gpt": float("nan"), "diff": float("nan"),
                        "b": 0, "c": 0, "mcnemar_p": float("nan"),
                        "cluster_diff_mean": float("nan"), "cluster95_lo": float("nan"), "cluster95_hi": float("nan"),
                    })
                    continue

                a = sub["Correct_gem"].values
                g = sub["Correct_gpt"].values
                acc_gem = float(a.mean())
                acc_gpt = float(g.mean())
                b = int((a & ~g).sum())
                c = int((~a & g).sum())
                p = mcnemar_exact(b, c)

                if RUN_BOOTSTRAP:
                    cdm, clo, chi = cluster_bootstrap_diff(a, g, sub["Cluster_Key"].values)
                else:
                    cdm, clo, chi = float("nan"), float("nan"), float("nan")

                results.append({
                    "unsure_mode": unsure_mode, "pass_frac": pf, "qa_type": t, "n": n,
                    "acc_gem": acc_gem, "acc_gpt": acc_gpt, "diff": acc_gem - acc_gpt,
                    "b": b, "c": c, "mcnemar_p": p,
                    "cluster_diff_mean": cdm, "cluster95_lo": clo, "cluster95_hi": chi,
                })
            print(f"  unsure_mode={unsure_mode} pf={pf} done")

    res_df = pd.DataFrame(results)

    # leader_matches_pf75: same sign of diff as this (unsure_mode, type)'s own pf=0.75 cell
    res_df["leader_matches_pf75"] = None
    for unsure_mode in UNSURE_MODES:
        for t in TYPES:
            ref = res_df[(res_df.unsure_mode == unsure_mode) & (res_df.qa_type == t) & (res_df.pass_frac == 0.75)]
            if len(ref) == 0 or pd.isna(ref["diff"].iloc[0]):
                continue
            ref_sign = np.sign(ref["diff"].iloc[0])
            mask = (res_df.unsure_mode == unsure_mode) & (res_df.qa_type == t)
            res_df.loc[mask, "leader_matches_pf75"] = res_df.loc[mask, "diff"].apply(
                lambda d: (np.sign(d) == ref_sign) if not pd.isna(d) else None
            )

    res_df.to_csv(os.path.join(ANALYSIS_DIR, "exp2b_significance_grid.tsv"), sep="\t", index=False)

    print("\n--- Full significance grid (unsure_mode=exclude) ---")
    show = res_df[res_df.unsure_mode == "exclude"].sort_values(["qa_type", "pass_frac"])
    print(show[["qa_type", "pass_frac", "n", "acc_gem", "acc_gpt", "diff", "b", "c", "mcnemar_p", "leader_matches_pf75"]]
          .to_string(index=False))

    print("\n--- Cells where the leader flips relative to this type's own pass_frac=0.75 ---")
    flips = res_df[res_df["leader_matches_pf75"] == False]
    if len(flips) == 0:
        print("  none -- leader is consistent with the pf=0.75 result at every threshold tested, all unsure_modes")
    else:
        print(flips[["unsure_mode", "qa_type", "pass_frac", "n", "diff", "mcnemar_p"]].to_string(index=False))

    print(f"\nWrote {ANALYSIS_DIR}/exp2b_significance_grid.tsv")


if __name__ == "__main__":
    main()