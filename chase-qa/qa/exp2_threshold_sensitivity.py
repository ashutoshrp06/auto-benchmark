"""
Exp 2 -- Is the benchmark an artefact of my thresholds?

Sweeps the gate (pass_frac x unsure_mode) and re-derives, at every setting:
  - surviving reg-track corpus size, per type and overall
  - type composition of the surviving corpus
  - reg+PASS accuracy and 4-model ranking, per type

Also empirically checks the carried-forward claim that no DISAGREE row
survives above pass_frac 0.66 at disagree_frac=0.34 (structural, from
Pass_Frac + No_Majority_Frac <= 1) -- do not assume it, verify it here.

Rubric sensitivity (type3-causal-check vs zero-shot-basic) is NOT re-run
here -- already measured (+14.8pp / +15.1pp, ordering preserved). This
script covers the gate-threshold axis only.

Run from repo root:
    python3 exp2_threshold_sensitivity.py

Inputs (must exist, no fallback / no silent skip):
    generation_outputs/elm-docs-type{T}-v9-b{1..4}/rocketeval/judgments.tsv
    generation_outputs/v9/combined_type{T}.tsv
    outputs/type{T}-{MODEL}-v9-noirrelevant-eval/result.tsv
    outputs/type{T}-{MODEL}-v9-noirrelevant-zsb-eval/result.tsv   (type3 only, if present)

Outputs:
    analysis/exp2_accuracy_grid.tsv     -- one row per (unsure_mode, pass_frac, type, model)
    analysis/exp2_corpus_grid.tsv       -- one row per (unsure_mode, pass_frac, type), incl. pass_rate
    analysis/exp2_criteria_by_type.tsv  -- N_Criteria distribution per type + allowed-failures at pf=0.75
    analysis/exp2_disagree_check.txt    -- measured DISAGREE ceiling vs the carried-forward 0.66 bound
    stdout                              -- ranking-stability + composition + criteria summary
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TYPES = [1, 2, 3]
BATCHES = ["b1", "b2", "b3", "b4"]
MODELS = ["gpt55", "gemini31pro", "gemini31flashlite", "gpt54mini"]
UNSURE_MODES = ["exclude", "zero", "half"]
PASS_FRAC_GRID = np.round(np.arange(0.50, 1.001, 0.05), 2)
DISAGREE_FRAC = 0.34  # held fixed; this experiment sweeps pass_frac, not disagree_frac
REG_SEED_TYPES = {"reg", "dynamic_reg"}

GEN_DIR = "generation_outputs"
CORPUS_DIR = "generation_outputs/v9"
OUT_DIR = "outputs"
ANALYSIS_DIR = "analysis"

GATE_PATTERN = GEN_DIR + "/elm-docs-type{t}-v9-{b}/rocketeval/judgments.tsv"
CORPUS_PATTERN = CORPUS_DIR + "/combined_type{t}.tsv"
RESULT_PATTERN = OUT_DIR + "/type{t}-{m}-v9-noirrelevant-eval/result.tsv"


# ---------------------------------------------------------------------------
# Consensus logic -- copied verbatim from poll_aggregate.py so this script is
# self-contained. Any change there must be mirrored here or results diverge
# silently.
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
    return np.nan  # exclude


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


# ---------------------------------------------------------------------------
# Load gate judgments for one type, all 4 batches
# ---------------------------------------------------------------------------
def load_gate_judgments(qa_type):
    frames = []
    per_batch_unique = []
    for b in BATCHES:
        path = GATE_PATTERN.format(t=qa_type, b=b)
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: missing gate file {path}")
        df_b = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df_b["Batch"] = f"v9-{b}"  # confirmed format matches corpus Batch column exactly
        per_batch_unique.append(df_b[["Root_ID", "Question_No"]].drop_duplicates().shape[0])
        frames.append(df_b)
    j = pd.concat(frames, ignore_index=True)

    group_cols = ["Batch", "Root_ID", "Question_No", "QA_Type", "Seed_Type"]
    miss = j[group_cols].isna().any(axis=1) | (j[group_cols] == "").any(axis=1)
    if miss.any():
        raise SystemExit(f"ERROR: {int(miss.sum())} gate row(s) missing group cols for type{qa_type}")

    # Root_ID is scoped per batch, NOT globally unique (confirmed empirically:
    # concatenating without Batch produced 5822 dup rows, 2500 of them spanning
    # >1 batch -- a real collision, not judge_panel resume noise). Batch is now
    # part of every key below, resolving this.
    dup_key = ["Batch", "Root_ID", "Question_No", "Criterion_No", "Judge"]
    n_dup = int(j.duplicated(subset=dup_key).sum())
    if n_dup:
        raise SystemExit(
            f"ERROR: {n_dup} duplicate (batch, question, criterion, judge) row(s) in type{qa_type} "
            f"gate judgments. This is now a genuine within-batch resume duplicate (Batch is already "
            f"disambiguated) -- judge_panel.py appends on resume, dedupe before using."
        )

    n_q_concat = j[["Batch", "Root_ID", "Question_No"]].drop_duplicates().shape[0]
    n_q_expected = sum(per_batch_unique)
    if n_q_concat != n_q_expected:
        raise SystemExit(
            f"ERROR: type{qa_type} unexpected count after Batch-tagging: concatenated unique "
            f"(Batch, Root_ID, Question_No) = {n_q_concat}, expected {n_q_expected}. Something "
            f"other than the known cross-batch collision is going on -- investigate before proceeding."
        )

    print(f"type{qa_type} gate: {len(j)} criterion-rows, {n_q_concat} unique (batch, question) pairs "
          f"across {len(BATCHES)} batches, verified against per-batch counts {per_batch_unique}")
    return j


# ---------------------------------------------------------------------------
# Aggregate to question level for a given unsure_mode.
# Pass_Frac / No_Majority_Frac are threshold-independent; the pass_frac
# THRESHOLD is applied later in apply_threshold().
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Corpus + solver results, joined positionally, validated by exact Question match
# ---------------------------------------------------------------------------
def load_corpus(qa_type):
    path = CORPUS_PATTERN.format(t=qa_type)
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: missing corpus file {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.reset_index(drop=True)
    df["ID"] = df.index + 1  # 1-indexed, matches result.tsv ID
    return df[["ID", "Batch", "Root_ID", "Question_No", "Seed_Type", "Question"]]


def load_solver_results(qa_type, model, corpus_df, variant=""):
    path = RESULT_PATTERN.format(t=qa_type, m=model)
    if variant:
        path = path.replace("-eval/", f"-{variant}-eval/")
    if not os.path.exists(path):
        return None  # caller decides whether missing is fatal
    res = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "Result" not in res.columns or "ID" not in res.columns:
        raise SystemExit(f"ERROR: {path} missing ID/Result columns")
    res["ID"] = res["ID"].astype(int)
    res["Result"] = res["Result"].astype(int)

    dup_ids = int(res["ID"].duplicated().sum())
    if dup_ids:
        raise SystemExit(
            f"ERROR: {path} has {dup_ids} duplicate ID value(s). Documented failure mode: a "
            f"cancelled shard run resumed under a different -s value appends instead of "
            f"replacing (e.g. 222 rows -> 273). rm -rf the run dir and re-merge before using."
        )
    missing_ids = set(corpus_df["ID"]) - set(res["ID"])
    if missing_ids:
        raise SystemExit(f"ERROR: {path} missing {len(missing_ids)} ID(s) present in corpus, e.g. {sorted(missing_ids)[:5]}")

    merged = corpus_df.merge(res[["ID", "Question", "Result"]], on="ID", suffixes=("", "_res"))
    if len(merged) != len(corpus_df):
        raise SystemExit(f"ERROR: {path} row count {len(res)} does not match corpus {len(corpus_df)} for type{qa_type}")
    mismatches = int((merged["Question"] != merged["Question_res"]).sum())
    if mismatches:
        raise SystemExit(
            f"ERROR: {mismatches} positional Question mismatch(es) between corpus and {path}. "
            f"Positional join is invalid -- do not trust this file's ID ordering."
        )
    return merged[["Batch", "Root_ID", "Question_No", "Result"]]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    gate_raw = {t: load_gate_judgments(t) for t in TYPES}
    corpus = {t: load_corpus(t) for t in TYPES}

    # cache solver results per (type, model) -- loaded once, filtered many times
    solver_cache = {}
    for t in TYPES:
        for m in MODELS:
            r = load_solver_results(t, m, corpus[t])
            if r is None:
                print(f"WARNING: no result.tsv for type{t}/{m} -- excluded from ranking at all thresholds")
            solver_cache[(t, m)] = r

    accuracy_rows = []
    corpus_rows = []
    disagree_maxfrac_by_mode = {}
    agg_by_type_exclude = None  # captured below, needed after the loop for N_Criteria table

    for unsure_mode in UNSURE_MODES:
        agg = {t: aggregate_gate(gate_raw[t], unsure_mode) for t in TYPES}
        if unsure_mode == "exclude":
            agg_by_type_exclude = agg

        # disagree-claim check: fixed disagree_frac, verdict independent of pass_frac_thresh
        # for the DISAGREE branch itself, so check once per unsure_mode using any threshold
        probe = {t: apply_threshold(agg[t], 0.75) for t in TYPES}
        all_disagree_pf = pd.concat([probe[t].loc[probe[t]["Verdict"] == "DISAGREE", "Pass_Frac"] for t in TYPES])
        disagree_maxfrac_by_mode[unsure_mode] = float(all_disagree_pf.max()) if len(all_disagree_pf) else float("nan")

        for pf in PASS_FRAC_GRID:
            comp_counts = {}
            for t in TYPES:
                thr = apply_threshold(agg[t], pf)
                reg_pass = thr[(thr["Verdict"] == "PASS") & (thr["Track"] == "reg")]
                comp_counts[t] = len(reg_pass)
                n_reg_total = int((thr["Track"] == "reg").sum())
                corpus_rows.append({
                    "unsure_mode": unsure_mode, "pass_frac": pf, "qa_type": t,
                    "n_reg_pass": len(reg_pass),
                    "n_reg_total": n_reg_total,
                    "pass_rate": (len(reg_pass) / n_reg_total) if n_reg_total else float("nan"),
                })

                for m in MODELS:
                    sres = solver_cache[(t, m)]
                    if sres is None:
                        continue
                    j = reg_pass.merge(sres, on=["Batch", "Root_ID", "Question_No"], how="inner")
                    if len(j) == 0:
                        acc = float("nan")
                    else:
                        acc = float(j["Result"].mean())
                    accuracy_rows.append({
                        "unsure_mode": unsure_mode, "pass_frac": pf, "qa_type": t,
                        "model": m, "n": len(j), "accuracy": acc,
                    })

    acc_df = pd.DataFrame(accuracy_rows)
    corpus_df = pd.DataFrame(corpus_rows)
    acc_df.to_csv(os.path.join(ANALYSIS_DIR, "exp2_accuracy_grid.tsv"), sep="\t", index=False)
    corpus_df.to_csv(os.path.join(ANALYSIS_DIR, "exp2_corpus_grid.tsv"), sep="\t", index=False)

    # ------------------------------------------------------------------
    # Disagree claim: the carried-forward number was "cannot exceed 0.66"
    # (structural upper bound, from Pass_Frac + No_Majority_Frac <= 1 at
    # disagree_frac=0.34). That bound is real but loose -- report the
    # MEASURED max instead, which is what should go in the dissertation.
    # ------------------------------------------------------------------
    with open(os.path.join(ANALYSIS_DIR, "exp2_disagree_check.txt"), "w") as f:
        f.write("Structural upper bound (disagree_frac=0.34): DISAGREE rows cannot exceed Pass_Frac=0.66.\n")
        f.write("This is a ceiling, not a measured value -- do not cite 0.66 itself in the write-up.\n")
        f.write("Measured max Pass_Frac among actual DISAGREE rows, per unsure_mode (USE THIS NUMBER):\n")
        for mode, mx in disagree_maxfrac_by_mode.items():
            within_bound = "within 0.66 bound" if (np.isnan(mx) or mx <= 0.66) else "EXCEEDS 0.66 BOUND -- investigate"
            f.write(f"  unsure_mode={mode}: measured max = {mx:.4f}  ({within_bound})\n")
    print("\n--- DISAGREE ceiling: measured vs structural 0.66 bound ---")
    for mode, mx in disagree_maxfrac_by_mode.items():
        within_bound = "within bound" if (np.isnan(mx) or mx <= 0.66) else "EXCEEDS BOUND"
        print(f"  unsure_mode={mode}: measured max Pass_Frac among DISAGREE rows = {mx:.4f}  ({within_bound})")

    # ------------------------------------------------------------------
    # Ranking stability: for each (unsure_mode, type), does the accuracy-based
    # model ranking on reg+PASS change across the pass_frac grid?
    # ------------------------------------------------------------------
    print("\n--- Ranking stability across pass_frac grid (0.50-1.00, step 0.05) ---")
    for unsure_mode in UNSURE_MODES:
        for t in TYPES:
            sub = acc_df[(acc_df["unsure_mode"] == unsure_mode) & (acc_df["qa_type"] == t)]
            rankings = []
            for pf in PASS_FRAC_GRID:
                row = sub[sub["pass_frac"] == pf].dropna(subset=["accuracy"])
                if len(row) < 2:
                    continue
                order = tuple(row.sort_values("accuracy", ascending=False)["model"])
                rankings.append((pf, order))
            uniq_orders = set(o for _, o in rankings)
            stable = len(uniq_orders) == 1
            print(f"  unsure_mode={unsure_mode} type{t}: {'STABLE' if stable else 'CHANGES'} "
                  f"({len(uniq_orders)} distinct ranking(s) across {len(rankings)} settings)")
            if not stable:
                for pf, order in rankings:
                    print(f"      pf={pf}: {order}")

    # operating-threshold sanity number requested explicitly: pass_frac=0.75
    print("\n--- Operating threshold (pass_frac=0.75) reg+PASS accuracy, for the write-up table ---")
    op = acc_df[(acc_df["pass_frac"] == 0.75) & (acc_df["unsure_mode"] == "exclude")]
    print(op[["qa_type", "model", "n", "accuracy"]].sort_values(["qa_type", "model"]).to_string(index=False))

    # ------------------------------------------------------------------
    # Corpus composition + pass rate by type, across the grid (unsure_mode=exclude,
    # the default/reported mode). Printed directly so no follow-up command is needed.
    # ------------------------------------------------------------------
    print("\n--- reg n_reg_pass by type, across pass_frac grid (unsure_mode=exclude) ---")
    comp = corpus_df[corpus_df["unsure_mode"] == "exclude"]
    piv_n = comp.pivot(index="pass_frac", columns="qa_type", values="n_reg_pass")
    piv_n["total"] = piv_n.sum(axis=1)
    print(piv_n)
    print("\n--- pass_rate (n_reg_pass / n_reg_total) by type, across pass_frac grid (unsure_mode=exclude) ---")
    piv_rate = comp.pivot(index="pass_frac", columns="qa_type", values="pass_rate")
    print(piv_rate.round(4))

    # ------------------------------------------------------------------
    # N_Criteria distribution by type + exact allowed-failure count at pf=0.75.
    # Restricted to reg track, matching every other table in this experiment --
    # aggregate_gate() doesn't compute Track (that only happens in
    # apply_threshold()), so it's derived here directly from Seed_Type. Without
    # this filter the table would silently mix in generic-track checklists,
    # which may have a different N_Criteria distribution (different generation
    # template) and would misstate the reg-track standard being reported on.
    #
    # pass_frac>=0.75 means n_pass/n_crit >= 0.75, i.e. n_pass >= ceil(0.75*n_crit),
    # i.e. allowed_fail = n_crit - ceil(0.75*n_crit). Computed from real N_Criteria
    # values (unsure_mode does not affect N_Criteria, only which criteria score as
    # Yes/No/DISAGREE, so this is taken from the exclude-mode aggregation).
    # ------------------------------------------------------------------
    import math
    crit_rows = []
    for t in TYPES:
        reg_only = agg_by_type_exclude[t][agg_by_type_exclude[t]["Seed_Type"].isin(REG_SEED_TYPES)]
        vc = reg_only["N_Criteria"].value_counts().sort_index()
        for n_crit, count in vc.items():
            n_crit = int(n_crit)
            allowed_fail = n_crit - math.ceil(0.75 * n_crit)
            crit_rows.append({
                "qa_type": t, "N_Criteria": n_crit, "count": int(count),
                "allowed_fail_at_pf75": allowed_fail,
            })
    crit_df = pd.DataFrame(crit_rows)
    crit_df.to_csv(os.path.join(ANALYSIS_DIR, "exp2_criteria_by_type.tsv"), sep="\t", index=False)
    print("\n--- N_Criteria distribution by type (reg track only), allowed failures at pass_frac=0.75 ---")
    print(crit_df.to_string(index=False))

    print(f"\nWrote {ANALYSIS_DIR}/exp2_accuracy_grid.tsv, exp2_corpus_grid.tsv, "
          f"exp2_criteria_by_type.tsv, exp2_disagree_check.txt")


if __name__ == "__main__":
    main()