#!/usr/bin/env python3
"""
Dynamic split allocator for CHASE-Finance.

Reads per-type solver+evaluator output, computes per-type accuracy, applies the
saturation rule, and writes the weights for the next generation round.

Deliberately standalone: run_pipeline.sh does not know this exists. Small test
runs simply never invoke it.

RULE
    Any type with accuracy >= --threshold sheds --shift percentage points of
    generation weight. The total shed is given to the single lowest-accuracy
    type. No type may fall below --floor.

    If every type is at or above threshold there is no unsaturated recipient.
    Weights are left unchanged and the run is reported as fully saturated.

USAGE (real)
    python allocator.py \
        --results type1=outputs/type1-gpt55-v7-noirrelevant-eval+outputs/type1-gemini31pro-v7-noirrelevant-eval,type2=...,type3=... \
        --clean type1=generation_outputs/elm-docs-type1-v7/programmatic_data_modified_verified_cleaned.tsv,type2=...,type3=... \
        --poll type1=generation_outputs/elm-docs-type1-v7/rocketeval/poll_scores.tsv,type2=...,type3=... \
        --track reg --pass-frac 0.75 \
        --weights 33.3,33.3,33.4 \
        --out weights-round-2.json

    Several evaluator dirs per type, joined by '+', are different solver models.
    The per-type accuracy used by the rule is the MAXIMUM across them: a type
    stops being diagnostic once any frontier model tops out. This makes the
    mechanism monotone in panel size, so the panel is part of the definition.

    Accuracy is computed on the regulatory track only by default, restricted to
    RocketEval PASS. Generic ground truth is LLM-invented with no external
    grounding, so generic accuracy measures plausibility agreement and must not
    drive the split.

USAGE (simulation)
    Re-applies the rule for N rounds holding accuracy fixed. This shows how the
    mechanism moves weights, NOT how accuracy will actually evolve; accuracy is
    assumed constant, which it will not be. Use it to characterise the rule and
    to check that the round budget is enough to produce a visible trajectory.

    python allocator.py --simulate type1=0.567,type2=0.485,type3=0.512 \
        --weights 33.3,33.3,33.4 --rounds 6
"""

import argparse
import json
import os
import sys

import pandas as pd

TYPES = ["type1", "type2", "type3"]


def parse_kv(s, cast=str):
    out = {}
    for part in s.split(","):
        if "=" not in part:
            sys.exit("expected key=value pairs, got: " + part)
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in TYPES:
            sys.exit("unknown type key '{}', expected one of {}".format(k, TYPES))
        out[k] = cast(v.strip())
    missing = [t for t in TYPES if t not in out]
    if missing:
        sys.exit("missing types: " + ", ".join(missing))
    return out

def parse_kv_list(s):
    """Like parse_kv, but each value is a '+'-separated list of paths."""
    out = {}
    for part in s.split(","):
        if "=" not in part:
            sys.exit("expected key=value pairs, got: " + part)
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in TYPES:
            sys.exit("unknown type key '{}', expected one of {}".format(k, TYPES))
        paths = [x.strip() for x in v.split("+") if x.strip()]
        if not paths:
            sys.exit("no evaluator dirs given for " + k)
        out[k] = paths
    missing = [t for t in TYPES if t not in out]
    if missing:
        sys.exit("missing types: " + ", ".join(missing))
    return out


def build_mask(clean_path, poll_path, pass_frac, track, disagree_frac):
    """Row mask over the cleaned corpus: track filter AND RocketEval PASS.

    PASS is recomputed from Pass_Frac >= pass_frac. The Verdict column is NOT
    read: on v7 it marks Pass_Frac 0.875 as FAIL, so its threshold is unknown.
    Recomputing keeps the criterion explicit and makes a pass_frac sweep free.

    Returns (clean_df, mask). mask is a boolean Series on clean_df's index.
    """
    kw = dict(sep="\t", dtype=str, keep_default_na=False)
    if not os.path.isfile(clean_path):
        sys.exit("no cleaned corpus at " + clean_path)
    clean = pd.read_csv(clean_path, **kw)
    for col in ("Root_ID", "Question_No", "Seed_Type", "Question"):
        if col not in clean.columns:
            sys.exit("no {} column in {}".format(col, clean_path))

    if track == "all":
        mask = pd.Series(True, index=clean.index)
    else:
        want = {track, "dynamic_" + track}
        st = clean["Seed_Type"].str.strip()
        seen = sorted(set(st))
        if not (want & set(seen)):
            sys.exit("track '{}' absent from {} (found {})".format(
                track, clean_path, seen))
        mask = st.isin(want)

    if poll_path is None:
        return clean, mask, None

    if not os.path.isfile(poll_path):
        sys.exit("no poll_scores.tsv at " + poll_path)
    poll = pd.read_csv(poll_path, **kw)
    for col in ("Root_ID", "Question_No", "Pass_Frac"):
        if col not in poll.columns:
            sys.exit("no {} column in {}".format(col, poll_path))

    key = ["Root_ID", "Question_No"]
    if poll.duplicated(key).any():
        sys.exit("duplicate (Root_ID, Question_No) in {}; judge_panel.py "
                 "appends on resume, dedupe before use".format(poll_path))
    if clean.duplicated(key).any():
        sys.exit("duplicate (Root_ID, Question_No) in " + clean_path)

    frac = pd.to_numeric(poll["Pass_Frac"], errors="coerce")
    if frac.isna().any():
        sys.exit("non-numeric Pass_Frac in " + poll_path)
    poll = poll.assign(_pass=(frac >= pass_frac - 1e-9))

    merged = clean[key].merge(poll[key + ["_pass"]], on=key, how="left",
                              validate="one_to_one")
    if merged["_pass"].isna().any():
        sys.exit("{} of {} corpus rows have no RocketEval verdict in {}".format(
            int(merged["_pass"].isna().sum()), len(clean), poll_path))
    if "Seed_Type" in poll.columns:
        chk = clean[key + ["Seed_Type"]].merge(
            poll[key + ["Seed_Type"]], on=key, how="left", suffixes=("_c", "_p"))
        n_bad = int((chk["Seed_Type_c"].str.strip()
                     != chk["Seed_Type_p"].str.strip()).sum())
        if n_bad:
            sys.exit("{} rows disagree on Seed_Type between {} and {}".format(
                n_bad, clean_path, poll_path))
    passed = pd.Series(merged["_pass"].to_numpy(dtype=bool), index=clean.index)
    final = mask & passed

    dis = None
    if "No_Majority_Frac" in poll.columns:
        nm = pd.to_numeric(poll["No_Majority_Frac"], errors="coerce")
        if nm.isna().any():
            sys.exit("non-numeric No_Majority_Frac in " + poll_path)
        dmerge = clean[key].merge(poll[key].assign(_dis=(nm > disagree_frac)),
                                  on=key, how="left", validate="one_to_one")
        dser = pd.Series(dmerge["_dis"].to_numpy(dtype=bool), index=clean.index)
        dis = int((final & dser).sum())
    return clean, final, dis


def read_accuracy(run_dir, clean, mask):
    """Accuracy over the masked subset of an evaluator run directory.

    result.tsv has no Root_ID, only a 1-indexed ID, so the join to the corpus is
    positional. That is verified here by exact Question-string comparison on
    every row; a single mismatch aborts rather than silently scoring the wrong
    subset. Read with dtype=str and keep_default_na=False because Answer and 
    Explanation contain literal newlines.
    """
    path = os.path.join(run_dir, "result.tsv")
    if not os.path.isfile(path):
        sys.exit("no result.tsv in " + run_dir)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in ("Result", "Question"):
        if col not in df.columns:
            sys.exit("no {} column in {}".format(col, path))
    if len(df) != len(clean):
        sys.exit("row count mismatch: {} has {}, corpus has {}".format(
            path, len(df), len(clean)))
    bad = int((df["Question"].str.strip().to_numpy()
               != clean["Question"].str.strip().to_numpy()).sum())
    if bad:
        sys.exit("{} rows of {} do not align positionally with the corpus; "
                 "the positional join is invalid for this cell".format(bad, path))

    vals = df["Result"].str.strip()
    unexpected = sorted(set(vals) - {"0", "1"})
    if unexpected:
        sys.exit("unexpected Result values in {}: {}".format(path, unexpected))
    sel = vals.to_numpy()[mask.to_numpy()]
    n = len(sel)
    if n == 0:
        sys.exit("no rows left in {} after the track and PASS filters".format(path))
    correct = int((sel == "1").sum())
    return correct / n, correct, n


def apply_rule(weights, acc, threshold, shift, floor):
    """One round of the rule. Returns (new_weights, log_lines)."""
    log = []
    saturated = [t for t in TYPES if acc[t] >= threshold]
    unsaturated = [t for t in TYPES if acc[t] < threshold]

    if not saturated:
        log.append("no type at or above threshold {:.3f}; weights unchanged".format(threshold))
        return dict(weights), log

    if not unsaturated:
        log.append("ALL types at or above threshold {:.3f}: fully saturated, "
                   "no recipient, weights unchanged".format(threshold))
        return dict(weights), log

    recipient = min(unsaturated, key=lambda t: acc[t])
    log.append("saturated (acc >= {:.3f}): {}".format(threshold, ", ".join(saturated)))
    log.append("recipient (lowest accuracy among unsaturated): {} at {:.3f}".format(
        recipient, acc[recipient]))

    new = dict(weights)
    total_shed = 0.0
    for t in saturated:
        headroom = max(0.0, new[t] - floor)
        actual = min(shift, headroom)
        if actual < shift:
            log.append("  {} shed capped at {:.2f} by the {:.1f} floor "
                       "(wanted {:.2f})".format(t, actual, floor, shift))
        new[t] -= actual
        total_shed += actual
        log.append("  {}: {:.2f} -> {:.2f}".format(t, weights[t], new[t]))

    new[recipient] += total_shed
    log.append("  {}: {:.2f} -> {:.2f}  (+{:.2f})".format(
        recipient, weights[recipient], new[recipient], total_shed))

    total = sum(new.values())
    if abs(total - 100.0) > 1e-6:
        sys.exit("internal error: weights sum to {:.6f}, not 100".format(total))
    return new, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", help="type1=<dir>[+<dir>...],type2=...,type3=... "
                                     "evaluator run dirs. Several dirs per type are "
                                     "different solver models; the per-type MAXIMUM "
                                     "accuracy drives the rule")
    p.add_argument("--clean", help="type1=<path>,... cleaned corpus TSVs, required "
                                   "with --results")
    p.add_argument("--poll", help="type1=<path>,... rocketeval/poll_scores.tsv. "
                                  "Omit to skip the PASS filter")
    p.add_argument("--pass-frac", type=float, default=0.75,
                   help="RocketEval PASS threshold on Pass_Frac. Default 0.75")
    p.add_argument("--disagree-frac", type=float, default=0.34,
                   help="poll_aggregate.py's DISAGREE threshold on No_Majority_Frac. "
                        "Reported only. The allocator's PASS criterion is coverage "
                        "alone, so DISAGREE rows are not excluded")
    p.add_argument("--track", default="reg", choices=["reg", "generic", "all"],
                   help="which seed track drives the split. Default reg")
    p.add_argument("--simulate", help="type1=<acc>,type2=<acc>,type3=<acc> to run the rule "
                                      "on fixed accuracies instead of real output")
    p.add_argument("--rounds", type=int, default=1, help="simulation rounds. Default 1")
    p.add_argument("--weights", required=True, help="current weights as percentages, e.g. 33.3,33.3,33.4")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--shift", type=float, default=5.0, help="percentage points shed per saturated type")
    p.add_argument("--floor", type=float, default=5.0, help="minimum weight percentage per type")
    p.add_argument("--out", help="write the resulting weights to this JSON path")
    args = p.parse_args()

    if bool(args.results) == bool(args.simulate):
        sys.exit("give exactly one of --results or --simulate")

    parts = [x.strip() for x in args.weights.split(",")]
    if len(parts) != 3:
        sys.exit("--weights needs three comma-separated values")
    weights = {t: float(v) for t, v in zip(TYPES, parts)}
    if abs(sum(weights.values()) - 100.0) > 1e-6:
        sys.exit("weights must sum to 100, got {:.4f}".format(sum(weights.values())))
    for t in TYPES:
        if weights[t] < args.floor - 1e-9:
            sys.exit("{} weight {:.2f} is already below the floor {:.2f}".format(
                t, weights[t], args.floor))

    if args.results:
        dirs = parse_kv_list(args.results)
        if not args.clean:
            sys.exit("--clean is required with --results: the track and PASS "
                     "filters need Seed_Type from the cleaned corpus")
        cleans = parse_kv(args.clean)
        polls = parse_kv(args.poll) if args.poll else None
        if polls is None:
            print("WARNING: no --poll given. Accuracy is over the full corpus "
                  "with no RocketEval PASS filter.\n")
        if args.track != "reg":
            print("WARNING: track='{}'. Generic ground truth is LLM-invented "
                  "with no external grounding.\n".format(args.track))

        acc, detail = {}, {}
        for t in TYPES:
            clean, mask, n_dis = build_mask(cleans[t],
                                            polls[t] if polls else None,
                                            args.pass_frac, args.track,
                                            args.disagree_frac)
            per_model = {}
            for d in dirs[t]:
                a, c, n = read_accuracy(d, clean, mask)
                per_model[d] = {"correct": c, "n": n, "accuracy": a}
            driver = max(per_model, key=lambda d: per_model[d]["accuracy"])
            acc[t] = per_model[driver]["accuracy"]
            detail[t] = {
                "per_model": {d: {"correct": v["correct"], "n": v["n"],
                                  "accuracy": round(v["accuracy"], 4)}
                              for d, v in per_model.items()},
                "driver": driver,
                "accuracy": round(acc[t], 4),
                "n_after_filters": per_model[driver]["n"],
                "corpus_n": len(clean),
                "n_disagree_in_mask": n_dis,
                "disagree_frac": args.disagree_frac if polls else None,
            }

        print("per-type accuracy (track={}, pass_frac={}, aggregation=max)".format(
            args.track, args.pass_frac if polls else "n/a"))
        for t in TYPES:
            d = detail[t]
            print("  {}: {:.3f}   [{} of {} rows after filters]".format(
                t, d["accuracy"], d["n_after_filters"], d["corpus_n"]))
            if d["n_disagree_in_mask"] is not None:
                print("      {} of these {} would be DISAGREE at disagree_frac={} "
                      "(counted as PASS here; the allocator uses coverage only)".format(
                          d["n_disagree_in_mask"], d["n_after_filters"],
                          args.disagree_frac))
            for k, v in d["per_model"].items():
                flag = " <- driver" if k == d["driver"] else ""
                print("      {}/{} = {:.3f}  {}{}".format(
                    v["correct"], v["n"], v["accuracy"], k, flag))
        print()
        new, log = apply_rule(weights, acc, args.threshold, args.shift, args.floor)
        for line in log:
            print(line)
        print("\nnext weights: " + ", ".join(
            "{}={:.2f}".format(t, new[t]) for t in TYPES))

        payload = {
            "mode": "measured",
            "threshold": args.threshold,
            "shift": args.shift,
            "floor": args.floor,
            "track": args.track,
            "pass_frac": args.pass_frac if polls else None,
            "aggregation": "max across solver models",
            "shift_provisional": True,
            "accuracy": detail,
            "weights_in": {t: round(weights[t], 4) for t in TYPES},
            "weights_out": {t: round(new[t], 4) for t in TYPES},
            "log": log,
        }
    else:
        acc = parse_kv(args.simulate, cast=float)
        for t in TYPES:
            if not 0.0 <= acc[t] <= 1.0:
                sys.exit("simulated accuracy for {} must be in [0,1]".format(t))
        print("SIMULATION. Accuracy held constant across rounds; this characterises")
        print("the rule, it does not predict how accuracy will actually evolve.")
        print("fixed accuracy: " + ", ".join("{}={:.3f}".format(t, acc[t]) for t in TYPES))
        print()
        traj = [{t: round(weights[t], 4) for t in TYPES}]
        cur = dict(weights)
        print("round 0: " + ", ".join("{}={:6.2f}".format(t, cur[t]) for t in TYPES))
        for r in range(1, args.rounds + 1):
            cur, log = apply_rule(cur, acc, args.threshold, args.shift, args.floor)
            traj.append({t: round(cur[t], 4) for t in TYPES})
            print("round {}: ".format(r) + ", ".join(
                "{}={:6.2f}".format(t, cur[t]) for t in TYPES))
            if any("unchanged" in l for l in log):
                print("  " + log[0])
        new = cur
        payload = {
            "mode": "simulated",
            "threshold": args.threshold,
            "shift": args.shift,
            "floor": args.floor,
            "assumption": "accuracy held constant across rounds",
            "accuracy": {t: acc[t] for t in TYPES},
            "rounds": args.rounds,
            "trajectory": traj,
            "weights_out": {t: round(new[t], 4) for t in TYPES},
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print("\nwrote " + args.out)


if __name__ == "__main__":
    main()