"""Build the blinded human evaluation set for Section 5.7.

The script samples 30 Type 2 questions from CHASE-Finance and 30 from the
unmodified CHASE-QA baseline, interleaves both frames into one shuffled
rating file, and writes corpus identity and panel verdict to a separate key.

A CHASE-Finance item shows the source clause its seed carries and takes 3
judgements. A baseline item carries no clause and takes 1. The rating file
therefore prints the clause on one frame alone, so the blind covers the panel
verdict and not corpus identity.

Verdict recomputation reads per-criterion judge votes and applies the shipped
gate setting: an Unsure vote drops out, a criterion carries the label both
surviving votes agree on, a question returns DISAGREE where the combined
DISAGREE and Undecided fraction exceeds 0.34, and otherwise returns PASS at a
pass fraction of at least 0.75. The script never reads the stored verdict
column, which the threshold sweep overwrote.

Every failure halts the run. The script raises rather than skipping a row,
interpolating a stratum, or overwriting a key file.
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

REG_SEED_TYPES = {"reg", "dynamic_reg"}
PASS_FRAC_THRESHOLD = 0.75
DISAGREE_THRESHOLD = 0.34
TYPE2_MIN_DOCS = 3


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
    """Read a tab separated file and return its header and rows.

    The reader rejects a ragged row rather than padding it, because a short row
    would silently shift every field to its left.
    """
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
            rows.append(row)
    return header, rows


def batch_from_path(path):
    """Derive the batch label from a judgment file's run directory."""
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if "-b" not in run_dir:
        raise SystemExit(f"FAIL run directory carries no batch suffix: {run_dir}")
    return "v9-b" + run_dir.rsplit("-b", 1)[-1]


def load_votes(pattern):
    """Collect per-criterion judge votes from every parent judgment file.

    The loader keys a vote on batch, root, question, criterion, and judge, and
    halts on a repeated key, because judge_panel.py appends on resume and a
    duplicated vote would weight one criterion twice.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"FAIL no judgment file matches {pattern}")

    votes = collections.defaultdict(dict)
    seed_type = {}
    qa_type = {}
    seen = set()

    for path in paths:
        batch = batch_from_path(path)
        header, rows = read_tsv(path)
        index = {column: position for position, column in enumerate(header)}
        required = ("Root_ID", "Question_No", "QA_Type", "Seed_Type",
                    "Judge", "Criterion_No", "Verdict")
        missing = [column for column in required if column not in index]
        if missing:
            raise SystemExit(f"FAIL {path} lacks columns {missing}")

        for row in rows:
            question_key = (batch, row[index["Root_ID"]], row[index["Question_No"]])
            vote_key = question_key + (row[index["Criterion_No"]], row[index["Judge"]])
            if vote_key in seen:
                raise SystemExit(f"FAIL duplicate judge vote for {vote_key} in {path}")
            seen.add(vote_key)
            votes[question_key + (row[index["Criterion_No"]],)][row[index["Judge"]]] = (
                row[index["Verdict"]].strip()
            )
            seed_type[question_key] = row[index["Seed_Type"]]
            qa_type[question_key] = row[index["QA_Type"]]

    return dict(votes), seed_type, qa_type, paths


def label_criterion(judge_votes):
    """Label one criterion after an Unsure vote drops out."""
    kept = [vote for vote in judge_votes.values() if vote.lower() != "unsure"]
    if not kept:
        return "UNDECIDED"
    if all(vote == kept[0] for vote in kept):
        return kept[0].capitalize()
    return "DISAGREE"


def recompute_verdicts(votes, seed_type, qa_type):
    """Recompute a panel verdict for every judged question."""
    per_question = collections.defaultdict(list)
    for key, judge_votes in votes.items():
        per_question[key[:3]].append(label_criterion(judge_votes))

    recomputed = {}
    for key, labels in per_question.items():
        total = len(labels)
        pass_frac = sum(1 for label in labels if label == "Yes") / total
        no_majority = sum(
            1 for label in labels if label in ("DISAGREE", "UNDECIDED")
        ) / total
        if no_majority > DISAGREE_THRESHOLD:
            verdict = "DISAGREE"
        elif pass_frac >= PASS_FRAC_THRESHOLD:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        recomputed[key] = {
            "Verdict": verdict,
            "Pass_Frac": round(pass_frac, 6),
            "No_Majority_Frac": round(no_majority, 6),
            "N_Criteria": total,
            "Seed_Type": seed_type[key],
            "QA_Type": qa_type[key],
        }
    return recomputed


def load_corpus(path):
    """Index the CHASE-Finance corpus on batch, root, and question number.

    Root_ID repeats across batches, so a bare Root_ID join would merge rows
    from different batches without reporting it.
    """
    header, rows = read_tsv(path)
    index = {column: position for position, column in enumerate(header)}
    required = ("Batch", "Root_ID", "Question_No", "QA_Type", "Seed_Type",
                "Question", "Answer", "Rel_Docs_List", "QID", "Reg_Text")
    missing = [column for column in required if column not in index]
    if missing:
        raise SystemExit(f"FAIL {path} lacks columns {missing}")

    corpus = {}
    for row in rows:
        key = (row[index["Batch"]], row[index["Root_ID"]], row[index["Question_No"]])
        if key in corpus:
            raise SystemExit(f"FAIL duplicate corpus key {key} in {path}")
        corpus[key] = {column: row[position] for column, position in index.items()}
    return corpus


def parse_documents(raw, context):
    """Parse a stored document list and return its entries in stored order.

    Part of the baseline stores the list double encoded, so the parser decodes
    repeatedly while a string comes back, up to a bounded depth. The bound
    stops a pathological value from looping and reports it instead.
    """
    decoded = raw
    for _ in range(4):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise SystemExit(f"FAIL {context} holds unparseable Rel_Docs_List: {error}")
        if not isinstance(decoded, str):
            break
    else:
        raise SystemExit(f"FAIL {context} nests Rel_Docs_List beyond 4 levels")
    if not isinstance(decoded, list) or not decoded:
        raise SystemExit(f"FAIL {context} holds an empty or malformed Rel_Docs_List")
    if not all(isinstance(entry, str) for entry in decoded):
        raise SystemExit(f"FAIL {context} holds a non-string document entry")
    return decoded


def baseline_document_count(raw, context):
    """Count the documents a baseline row scatters its answer points across."""
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"FAIL {context} holds unparseable Doc_Ans_Points: {error}")
    if not isinstance(mapping, dict):
        raise SystemExit(f"FAIL {context} holds a non-object Doc_Ans_Points")
    return len(mapping)


def sample_chase_finance(corpus, recomputed, strata, rng):
    """Draw the CHASE-Finance sample, stratified by recomputed verdict."""
    pools = collections.defaultdict(list)
    for key, record in recomputed.items():
        if record["QA_Type"] != "type2":
            raise SystemExit(f"FAIL judged row {key} carries QA_Type {record['QA_Type']}")
        if record["Seed_Type"] not in REG_SEED_TYPES:
            continue
        if key not in corpus:
            raise SystemExit(f"FAIL judged row {key} holds no matching corpus row")
        if corpus[key]["QA_Type"] != "type2":
            raise SystemExit(
                f"FAIL corpus row {key} carries QA_Type {corpus[key]['QA_Type']}"
            )
        if corpus[key]["Seed_Type"] not in REG_SEED_TYPES:
            raise SystemExit(
                f"FAIL Seed_Type disagrees between judgments and corpus at {key}"
            )
        pools[record["Verdict"]].append(key)

    available = {verdict: len(keys) for verdict, keys in pools.items()}
    for verdict, wanted in strata.items():
        if available.get(verdict, 0) < wanted:
            raise SystemExit(
                f"FAIL stratum {verdict} holds {available.get(verdict, 0)} rows "
                f"against {wanted} requested"
            )

    drawn = []
    for verdict in sorted(strata):
        drawn.extend(rng.sample(sorted(pools[verdict]), strata[verdict]))
    return drawn, available


def allocate_largest_remainder(population, wanted):
    """Allocate a sample across strata in proportion to their population sizes.

    The allocator floors every share, then hands each remaining place to the
    stratum carrying the largest fractional part. A tie resolves on the stratum
    key in ascending order, so the allocation stays deterministic.
    """
    total = sum(population.values())
    if total < wanted:
        raise SystemExit(
            f"FAIL population holds {total} rows against {wanted} requested"
        )
    exact = {stratum: wanted * count / total
             for stratum, count in population.items()}
    allocation = {stratum: int(share) for stratum, share in exact.items()}
    remaining = wanted - sum(allocation.values())
    order = sorted(exact, key=lambda stratum: (-(exact[stratum] - allocation[stratum]),
                                               stratum))
    for stratum in order[:remaining]:
        allocation[stratum] += 1
    if sum(allocation.values()) != wanted:
        raise SystemExit(
            f"FAIL allocation sums to {sum(allocation.values())} against {wanted}"
        )
    return allocation


def sample_baseline(path, wanted, rng):
    """Draw the baseline sample stratified on document count.

    The baseline carries no type column, and its rows do not uniformly scatter
    across the 3 documents the Type 2 shape needs, so the sampler draws from
    the whole corpus rather than restricting to the rows that do. Stratifying
    on document count fixes the shape composition in proportion to the
    population before any row is read, which keeps a single draw from
    over-representing one shape by luck.
    """
    header, rows = read_tsv(path)
    index = {column: position for position, column in enumerate(header)}
    required = ("Root_ID", "Question_No", "Question", "Answer",
                "Rel_Docs_List", "Doc_Ans_Points")
    missing = [column for column in required if column not in index]
    if missing:
        raise SystemExit(f"FAIL {path} lacks columns {missing}")

    records = {}
    for row in rows:
        key = (row[index["Root_ID"]], row[index["Question_No"]])
        if key in records:
            raise SystemExit(f"FAIL duplicate baseline key {key} in {path}")
        records[key] = {column: row[position] for column, position in index.items()}

    shapes = {}
    pools = collections.defaultdict(list)
    for key, record in records.items():
        points = baseline_document_count(record["Doc_Ans_Points"], f"baseline {key}")
        documents = len(parse_documents(record["Rel_Docs_List"], f"baseline {key}"))
        if points != documents:
            raise SystemExit(
                f"FAIL baseline row {key} records {points} answer points against "
                f"{documents} documents"
            )
        shapes[key] = {"points": points, "documents": documents}
        pools[documents].append(key)

    population = collections.Counter(
        {stratum: len(keys) for stratum, keys in pools.items()})
    allocation = allocate_largest_remainder(population, wanted)
    for stratum, count in allocation.items():
        if population[stratum] < count:
            raise SystemExit(
                f"FAIL baseline stratum {stratum} holds {population[stratum]} rows "
                f"against {count} allocated"
            )

    drawn = []
    for stratum in sorted(allocation):
        drawn.extend(rng.sample(sorted(pools[stratum]), allocation[stratum]))
    return drawn, records, shapes, population, allocation


def render_item(item):
    """Render one rating item as markdown.

    The renderer places the source clause after the documents, so judgement A
    reaches the rater before the clause can colour a reading of what the
    documents support. Judgement B asks whether that clause governs the
    question, and judgement C asks whether the answer traces to it. Asking B
    first puts the clause in front of the rater before the answer's own
    framing of it does.
    """
    lines = [f"## {item['Item_ID']}", "", "### Question", "", item["Question"], "",
             "### Constructed answer", "", item["Answer"], "", "### Documents", ""]
    for position, document in enumerate(item["Documents"], start=1):
        lines.extend([f"#### Document {position}", "", document, ""])
    lines.extend(["### Rating", "",
                  "A. Is the answer correct given the documents above?  Yes / No"])
    if item["Grounding_Applicable"] == "yes":
        lines.extend([
            "", "### Source clause", "", item["Reg_Text"], "",
            "B. Does the clause above govern the question asked?  Yes / No",
            "C. Does the answer trace to the clause above?  Yes / No",
        ])
    else:
        lines.extend(["", "This item carries no source clause. Record n/a "
                      "against B and C, which the rating file already holds."])
    lines.extend(["", "Reason, required on every No:", "", "---", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments-glob",
                        default="generation_outputs/elm-docs-type2-v9-b[1-4]/rocketeval/judgments.tsv")
    parser.add_argument("--corpus", default="generation_outputs/v9/combined_type2.tsv")
    parser.add_argument("--baseline",
                        default="../../chase-qa-base/qa/baseline_corpus.tsv")
    parser.add_argument("--outdir", default="human_eval")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-pass", type=int, default=12)
    parser.add_argument("--n-fail", type=int, default=8)
    parser.add_argument("--n-disagree", type=int, default=10)
    parser.add_argument("--n-baseline", type=int, default=30)
    parser.add_argument("--flag-grounding-inline", action="store_true",
                        help="print grounding applicability on the item, which "
                             "reveals corpus identity")
    parser.add_argument("--expect-pass", type=int, default=None)
    parser.add_argument("--expect-fail", type=int, default=None)
    parser.add_argument("--expect-disagree", type=int, default=None)
    args = parser.parse_args()

    _raise_field_limit()

    key_path = os.path.join(args.outdir, "key.tsv")
    if os.path.exists(key_path):
        raise SystemExit(f"FAIL key file already exists at {key_path}")

    votes, seed_type, qa_type, judgment_paths = load_votes(args.judgments_glob)
    recomputed = recompute_verdicts(votes, seed_type, qa_type)
    corpus = load_corpus(args.corpus)

    judged = set(recomputed)
    stored = set(corpus)
    if judged - stored:
        raise SystemExit(f"FAIL {len(judged - stored)} judged rows hold no corpus row")
    if stored - judged:
        raise SystemExit(f"FAIL {len(stored - judged)} corpus rows carry no judgment")

    reg_counts = collections.Counter(
        record["Verdict"] for record in recomputed.values()
        if record["Seed_Type"] in REG_SEED_TYPES
    )
    expected = {"PASS": args.expect_pass, "FAIL": args.expect_fail,
                "DISAGREE": args.expect_disagree}
    for verdict, target in expected.items():
        if target is not None and reg_counts[verdict] != target:
            raise SystemExit(
                f"FAIL recomputed {verdict} count {reg_counts[verdict]} "
                f"against expected {target}"
            )

    strata = {"PASS": args.n_pass, "FAIL": args.n_fail, "DISAGREE": args.n_disagree}
    rng = random.Random(args.seed)
    cf_keys, available = sample_chase_finance(corpus, recomputed, strata, rng)
    bl_keys, bl_records, bl_shapes, bl_population, bl_allocation = sample_baseline(
        args.baseline, args.n_baseline, rng)

    items = []
    for key in cf_keys:
        row = corpus[key]
        items.append({
            "Frame": "chase_finance",
            "Batch": key[0], "Root_ID": key[1], "Question_No": key[2],
            "QID": row["QID"],
            "Verdict": recomputed[key]["Verdict"],
            "Pass_Frac": recomputed[key]["Pass_Frac"],
            "No_Majority_Frac": recomputed[key]["No_Majority_Frac"],
            "N_Criteria": recomputed[key]["N_Criteria"],
            "Seed_Type": row["Seed_Type"],
            "N_Docs": len(parse_documents(row["Rel_Docs_List"], f"corpus {key}")),
            "N_Ans_Points": "",
            "Grounding_Applicable": "yes",
            "Reg_Text": row["Reg_Text"],
            "Question": row["Question"],
            "Answer": row["Answer"],
            "Documents": parse_documents(row["Rel_Docs_List"], f"corpus {key}"),
        })
    for key in bl_keys:
        row = bl_records[key]
        items.append({
            "Frame": "baseline",
            "Batch": "", "Root_ID": key[0], "Question_No": key[1],
            "QID": "",
            "Verdict": "", "Pass_Frac": "", "No_Majority_Frac": "", "N_Criteria": "",
            "Seed_Type": "",
            "N_Docs": bl_shapes[key]["documents"],
            "N_Ans_Points": bl_shapes[key]["points"],
            "Grounding_Applicable": "no",
            "Reg_Text": "",
            "Question": row["Question"],
            "Answer": row["Answer"],
            "Documents": parse_documents(row["Rel_Docs_List"], f"baseline {key}"),
        })

    for item in items:
        if item["Grounding_Applicable"] == "yes" and not item["Reg_Text"].strip():
            raise SystemExit(
                f"FAIL regulatory row {item['Root_ID']}/{item['Question_No']} in "
                f"batch {item['Batch']} carries no source clause"
            )

    rng.shuffle(items)
    for position, item in enumerate(items, start=1):
        item["Item_ID"] = f"I{position:03d}"

    os.makedirs(args.outdir, exist_ok=True)

    markdown = ["# Human evaluation rating set", "",
                f"Items: {len(items)}. Corpus identity and panel verdict sit in "
                "the key file and stay out of this document.", "",
                "Judgement A asks whether the answer is correct given the "
                "documents below it. Judgement B asks whether the answer traces "
                "to real FCA or HMRC text. Record a reason on every No.", ""]
    if args.flag_grounding_inline:
        markdown.append("Judgement B applies only where the item marks it "
                        "applicable.")
        markdown.append("")
    for item in items:
        if args.flag_grounding_inline:
            item = dict(item)
            item["Question"] = (
                f"[Judgement B applicable: {item['Grounding_Applicable']}]\n\n"
                + item["Question"]
            )
        markdown.append(render_item(item))

    with open(os.path.join(args.outdir, "items.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))

    with open(os.path.join(args.outdir, "items.jsonl"), "w", encoding="utf-8") as handle:
        for item in items:
            blind = {field: item[field] for field in
                     ("Item_ID", "Question", "Answer", "Documents", "Reg_Text")}
            handle.write(json.dumps(blind, ensure_ascii=False) + "\n")

    ratings_path = os.path.join(args.outdir, "ratings.tsv")
    with open(ratings_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Item_ID", "Answer_Correct", "Clause_Governs",
                         "Traces_To_Source", "Reason"])
        for item in items:
            applicable = item["Grounding_Applicable"] == "yes"
            filler = "" if applicable else "n/a"
            writer.writerow([item["Item_ID"], "", filler, filler, ""])

    with open(key_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["Item_ID", "Frame", "Grounding_Applicable", "Batch",
                         "Root_ID", "Question_No", "QID", "Verdict", "Pass_Frac",
                         "No_Majority_Frac", "N_Criteria", "Seed_Type",
                         "N_Docs", "N_Ans_Points"])
        for item in items:
            writer.writerow([item["Item_ID"], item["Frame"],
                             item["Grounding_Applicable"], item["Batch"],
                             item["Root_ID"], item["Question_No"], item["QID"],
                             item["Verdict"], item["Pass_Frac"],
                             item["No_Majority_Frac"], item["N_Criteria"],
                             item["Seed_Type"], item["N_Docs"],
                             item["N_Ans_Points"]])

    print("judgment files read:", len(judgment_paths))
    print("reg-track Type 2 questions available:", dict(reg_counts))
    print("strata drawn:", strata, "baseline drawn:", args.n_baseline)
    print("pool sizes:", available)
    print("baseline documents per row, whole corpus:",
          dict(sorted(bl_population.items())))
    print("baseline allocation, proportional:", dict(sorted(bl_allocation.items())))
    print("baseline rows below the Type 2 shape, whole corpus:",
          sum(count for points, count in bl_population.items()
              if points < TYPE2_MIN_DOCS), "of", sum(bl_population.values()))
    drawn_shape = collections.Counter(
        shape["documents"] for key, shape in bl_shapes.items() if key in set(bl_keys))
    print("baseline documents per drawn row:", dict(sorted(drawn_shape.items())))
    print("drawn baseline rows below the Type 2 shape:",
          sum(count for docs, count in drawn_shape.items() if docs < TYPE2_MIN_DOCS),
          "of", args.n_baseline)
    print("items written:", len(items), "seed:", args.seed)
    print("outputs:", args.outdir)


if __name__ == "__main__":
    main()