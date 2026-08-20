"""Analyse the completed human evaluation and produce the Section 5.7 outputs.

The script joins the rating file to the key on item identifier and reports 2
quantities that never pool. One reports how the rater's correctness judgement
lands against each panel verdict, and it covers CHASE-Finance only, since the
baseline never passed through the gate and carries no verdict. The other
reports answer correctness across both corpora.

A CHASE-Finance item carries 3 judgements and a baseline item carries 1, so
the report runs 3 denominators: correctness over 60 items, clause governance
over 30, and tracing over 30. A baseline item records n/a against the 2
clause judgements, and the script halts where n/a appears on an item that
carries a clause or a verdict appears on an item that does not.

The script applies no significance test. At this sample size the free-text
reasons carry more than the rates, so the script groups every reason by frame,
verdict, and which judgement failed.
"""

import argparse
import collections
import csv
import os
import sys

YES = "yes"
NO = "no"
NOT_APPLICABLE = "n/a"
VERDICT_ORDER = ["PASS", "FAIL", "DISAGREE"]


def _raise_field_limit():
    """Raise the csv field limit to the largest value the platform accepts."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 2


def read_tsv(path):
    """Read a tab separated file and reject a ragged row rather than pad it."""
    if not os.path.exists(path):
        raise SystemExit(f"FAIL no file at {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise SystemExit(
                    f"FAIL ragged row: {path} line {line_no} holds {len(row)} "
                    f"fields against {len(header)} in the header"
                )
            rows.append({column: value for column, value in zip(header, row)})
    return rows


def normalize(value, field, item_id, allow_na=False):
    """Normalize a rating and halt on anything the instrument disallows."""
    cleaned = value.strip().lower()
    if cleaned in ("y", "yes", "1", "true"):
        return YES
    if cleaned in ("n", "no", "0", "false"):
        return NO
    if allow_na and cleaned in ("n/a", "na", "not applicable"):
        return NOT_APPLICABLE
    permitted = "yes, no, or n/a" if allow_na else "yes or no"
    raise SystemExit(
        f"FAIL item {item_id} carries {field} value {value!r}, which reads as "
        f"none of {permitted}"
    )


def join(ratings, key):
    """Join ratings to the key and halt on any mismatch or missing judgement."""
    keyed = {row["Item_ID"]: row for row in key}
    if len(keyed) != len(key):
        raise SystemExit("FAIL the key file repeats an item identifier")

    rated = {}
    for row in ratings:
        item_id = row["Item_ID"]
        if item_id in rated:
            raise SystemExit(f"FAIL the rating file repeats item {item_id}")
        rated[item_id] = row

    missing_key = sorted(set(rated) - set(keyed))
    missing_rating = sorted(set(keyed) - set(rated))
    if missing_key:
        raise SystemExit(f"FAIL rated items absent from the key: {missing_key}")
    if missing_rating:
        raise SystemExit(f"FAIL key items absent from the ratings: {missing_rating}")

    joined = []
    for item_id in sorted(keyed):
        rating, meta = rated[item_id], keyed[item_id]
        applicable = meta["Grounding_Applicable"] == "yes"
        correct = normalize(rating["Answer_Correct"], "Answer_Correct", item_id)
        governs = normalize(rating["Clause_Governs"], "Clause_Governs",
                            item_id, allow_na=True)
        traces = normalize(rating["Traces_To_Source"], "Traces_To_Source",
                           item_id, allow_na=True)
        for field, value in (("Clause_Governs", governs),
                             ("Traces_To_Source", traces)):
            if applicable and value == NOT_APPLICABLE:
                raise SystemExit(
                    f"FAIL item {item_id} carries a clause but records n/a "
                    f"against {field}"
                )
            if not applicable and value != NOT_APPLICABLE:
                raise SystemExit(
                    f"FAIL item {item_id} carries no clause but records "
                    f"{value} against {field}"
                )
        reason = rating.get("Reason", "").strip()
        if NO in (correct, governs, traces) and not reason:
            raise SystemExit(
                f"FAIL item {item_id} records a No without a reason"
            )
        joined.append({
            "Item_ID": item_id,
            "Frame": meta["Frame"],
            "Grounding_Applicable": meta["Grounding_Applicable"],
            "Verdict": meta["Verdict"],
            "Seed_Type": meta["Seed_Type"],
            "Root_ID": meta["Root_ID"],
            "Question_No": meta["Question_No"],
            "Batch": meta["Batch"],
            "Answer_Correct": correct,
            "Clause_Governs": governs,
            "Traces_To_Source": traces,
            "Reason": reason,
        })
    return joined


def correctness_by_corpus(joined):
    """Count correct answers within each corpus."""
    table = collections.OrderedDict()
    for frame in ("chase_finance", "baseline"):
        rows = [row for row in joined if row["Frame"] == frame]
        table[frame] = {
            "n": len(rows),
            "correct": sum(1 for row in rows if row["Answer_Correct"] == YES),
        }
    return table


def correctness_by_verdict(joined):
    """Count correct answers within each panel verdict, CHASE-Finance only."""
    table = collections.OrderedDict()
    for verdict in VERDICT_ORDER:
        rows = [row for row in joined
                if row["Frame"] == "chase_finance" and row["Verdict"] == verdict]
        table[verdict] = {
            "n": len(rows),
            "correct": sum(1 for row in rows if row["Answer_Correct"] == YES),
        }
    return table


def grounding_tables(joined):
    """Count the 2 clause judgements over the items that carry a clause."""
    reported = [row for row in joined if row["Grounding_Applicable"] == "yes"]

    def summarize(rows):
        return {
            "n": len(rows),
            "governs": sum(1 for row in rows if row["Clause_Governs"] == YES),
            "traces": sum(1 for row in rows if row["Traces_To_Source"] == YES),
        }

    by_verdict = collections.OrderedDict()
    for verdict in VERDICT_ORDER:
        by_verdict[verdict] = summarize(
            [row for row in reported if row["Verdict"] == verdict])
    return summarize(reported), by_verdict


def failure_rows(joined):
    """Collect every failed judgement alongside its reason."""
    rows = []
    for row in joined:
        for field, label in (("Answer_Correct", "correctness"),
                             ("Clause_Governs", "clause governance"),
                             ("Traces_To_Source", "grounding")):
            if row[field] == NO:
                rows.append({
                    "Item_ID": row["Item_ID"],
                    "Frame": row["Frame"],
                    "Verdict": row["Verdict"] or "n/a",
                    "Judgement": label,
                    "Reported": "yes",
                    "Reason": row["Reason"],
                })
    return rows


def latex_table(by_corpus, by_verdict, grounding, grounding_by_verdict):
    """Render the Section 5.7 counts as a booktabs table."""
    lines = [
        r"\begin{table}[h]", r"\centering", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Population & $n$ & Answer correct & Clause governs & Traces to clause \\",
        r"\midrule",
    ]
    for verdict in VERDICT_ORDER:
        correctness = by_verdict[verdict]
        ground = grounding_by_verdict[verdict]
        lines.append(
            f"CHASE-Finance {verdict} & {correctness['n']} & "
            f"{correctness['correct']} & {ground['governs']} & "
            f"{ground['traces']} \\\\"
        )
    finance = by_corpus["chase_finance"]
    baseline = by_corpus["baseline"]
    lines.extend([
        r"\midrule",
        f"CHASE-Finance, all & {finance['n']} & {finance['correct']} & "
        f"{grounding['governs']} & {grounding['traces']} \\\\",
        f"CHASE-QA (unmodified) & {baseline['n']} & {baseline['correct']} & "
        r"n/a & n/a \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Self-rated human evaluation counts. Correctness covers all "
        r"60 items. The 2 clause judgements cover the 30 CHASE-Finance "
        r"regulatory-track items alone, since the baseline preserves no seed "
        r"lineage and carries no source clause to read an answer against, so "
        r"the columns run different denominators.}",
        r"\label{tab:human-eval}", r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", default="human_eval")
    parser.add_argument("--outdir", default="human_eval")
    args = parser.parse_args()

    _raise_field_limit()
    ratings = read_tsv(os.path.join(args.indir, "ratings.tsv"))
    key = read_tsv(os.path.join(args.indir, "key.tsv"))
    joined = join(ratings, key)

    by_corpus = correctness_by_corpus(joined)
    by_verdict = correctness_by_verdict(joined)
    grounding, grounding_by_verdict = grounding_tables(joined)
    failures = failure_rows(joined)

    print("=" * 62)
    print("Answer correctness across the 2 corpora, never pooled")
    for frame, cell in by_corpus.items():
        print(f"  {frame:<16} {cell['correct']} of {cell['n']}")

    print("\nCHASE-Finance correctness by panel verdict")
    for verdict, cell in by_verdict.items():
        print(f"  {verdict:<9} {cell['correct']} of {cell['n']}")

    print("\nClause governance, CHASE-Finance regulatory items")
    print(f"  overall   {grounding['governs']} of {grounding['n']}")
    for verdict, cell in grounding_by_verdict.items():
        print(f"  {verdict:<9} {cell['governs']} of {cell['n']}")

    print("\nAnswer traces to the clause, CHASE-Finance regulatory items")
    print(f"  overall   {grounding['traces']} of {grounding['n']}")
    for verdict, cell in grounding_by_verdict.items():
        print(f"  {verdict:<9} {cell['traces']} of {cell['n']}")

    print(f"\nFailed judgements: {len(failures)}")
    grouped = collections.defaultdict(list)
    for row in failures:
        grouped[(row["Frame"], row["Judgement"], row["Verdict"])].append(row)
    for group in sorted(grouped):
        print(f"\n  {group[0]} / {group[1]} / verdict {group[2]}  "
              f"({len(grouped[group])} items)")
        for row in grouped[group]:
            print(f"    {row['Item_ID']}: {row['Reason']}")

    os.makedirs(args.outdir, exist_ok=True)
    joined_path = os.path.join(args.outdir, "joined.tsv")
    with open(joined_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", lineterminator="\n",
                                fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)

    failures_path = os.path.join(args.outdir, "failure_reasons.tsv")
    with open(failures_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", lineterminator="\n",
                                fieldnames=["Item_ID", "Frame", "Verdict",
                                            "Judgement", "Reported", "Reason"])
        writer.writeheader()
        writer.writerows(sorted(failures,
                                key=lambda row: (row["Frame"], row["Judgement"],
                                                 row["Verdict"], row["Item_ID"])))

    table_path = os.path.join(args.outdir, "table_5_7.tex")
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(latex_table(by_corpus, by_verdict, grounding,
                                 grounding_by_verdict) + "\n")

    print(f"\nwrote {joined_path}")
    print(f"wrote {failures_path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()