"""Exercise the sampler end to end on synthetic data.

The harness builds a corpus, a judgment set, and a baseline that match the real
schemas, then checks that the happy path produces the requested strata and that
each guard halts the run.
"""

import collections
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "synthetic")
SCRIPT = os.path.join(HERE, "build_human_eval.py")

CORPUS_COLS = ["Root_ID", "Question_No", "Persona", "Environment", "Question",
               "Answer", "Ans_Points", "Doc_Ans_Points", "Rel_Docs_List",
               "Adv_Question", "Adv_Answer", "Adv_Ans_Pts", "Adv_Doc_Ans_Pts",
               "Adv_Docs_List", "Reg_Text", "Numerics", "Seed_Type", "QID",
               "Documents", "Batch", "QA_Type"]
BASELINE_COLS = ["Root_ID", "Question_No", "Persona", "Environment", "Question",
                 "Answer", "Ans_Points", "Doc_Ans_Points", "Rel_Docs_List",
                 "Adv_Question", "Adv_Answer", "Adv_Ans_Pts", "Adv_Doc_Ans_Pts",
                 "Adv_Docs_List", "Documents"]
JUDGE_COLS = ["Root_ID", "Question_No", "QA_Type", "Seed_Type", "Judge",
              "Criterion_No", "Criterion", "Verdict"]


def write_tsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def docs(count, tag):
    return json.dumps([f"Title: {tag} document {n}\nText:\nAdviser: line one.\n"
                       f"Client: line two." for n in range(1, count + 1)])


def make_case(base, want_pass, want_fail, want_disagree, generic=6,
              corpus_qa_type="type2", baseline_docs=3, duplicate_vote=False,
              baseline_mixed=False, baseline_double_encode=False,
              baseline_broken_docs=False):
    """Build one synthetic case and return its paths.

    The builder targets 9 criteria per question and sets vote patterns so that
    the recomputation returns the requested verdict for each block: a PASS block
    agrees Yes on 8 of 9, a FAIL block agrees Yes on 4 of 9, and a DISAGREE
    block splits the judges on 4 of 9, which puts its no-majority fraction at
    0.444 and above the 0.34 threshold.
    """
    shutil.rmtree(base, ignore_errors=True)
    plan = ([("PASS", "reg")] * want_pass + [("FAIL", "dynamic_reg")] * want_fail
            + [("DISAGREE", "reg")] * want_disagree
            + [("PASS", "generic")] * generic)

    corpus_rows, judge_rows = [], []
    for position, (verdict, seed) in enumerate(plan, start=1):
        root, qno, batch = str(position), "1", "v9-b1"
        corpus_rows.append([
            root, qno, "persona", "environment", f"Question {position}?",
            f"- point one for {position}\n- point two for {position}",
            json.dumps(["point one", "point two"]),
            json.dumps({"1": ["point one"], "2": ["point two"], "3": ["point one"]}),
            docs(3, f"genuine {position}"),
            "adv question", "adv answer", "[]", "{}", "[]",
            "COBS 1.1 source text", "", seed, f"q{position:016d}",
            "Document 1:\nTitle: x\nText:\ny", batch, corpus_qa_type,
        ])
        for criterion in range(1, 10):
            if verdict == "PASS":
                votes = ("Yes", "Yes") if criterion <= 8 else ("No", "No")
            elif verdict == "FAIL":
                votes = ("Yes", "Yes") if criterion <= 4 else ("No", "No")
            else:
                votes = ("Yes", "No") if criterion <= 4 else ("Yes", "Yes")
            for judge, vote in zip(("gpt54mini", "gemini35flashlite"), votes):
                judge_rows.append([root, qno, "type2", seed, judge,
                                   str(criterion), "criterion text", vote])
    if duplicate_vote:
        judge_rows.append(list(judge_rows[0]))

    baseline_rows = []
    for position in range(1, 41):
        count = ((position % 4) + 1) if baseline_mixed else baseline_docs
        payload = docs(max(count, 1), f"baseline {position}")
        if baseline_broken_docs and position == 1:
            payload = json.dumps({"not": "a list"})
        elif baseline_double_encode and position % 3 == 0:
            payload = json.dumps(payload)
        baseline_rows.append([
            str(position), "1", "persona", "environment",
            f"Baseline question {position}?", f"- baseline point {position}",
            json.dumps(["baseline point"]),
            json.dumps({str(n): ["baseline point"] for n in range(1, count + 1)}),
            payload,
            "adv question", "adv answer", "[]", "{}", "[]",
            "Document 1:\nTitle: x\nText:\ny",
        ])

    corpus = os.path.join(base, "generation_outputs/v9/combined_type2.tsv")
    judgments = os.path.join(
        base, "generation_outputs/elm-docs-type2-v9-b1/rocketeval/judgments.tsv")
    baseline = os.path.join(base, "baseline_corpus.tsv")
    write_tsv(corpus, CORPUS_COLS, corpus_rows)
    write_tsv(judgments, JUDGE_COLS, judge_rows)
    write_tsv(baseline, BASELINE_COLS, baseline_rows)
    return corpus, judgments, baseline


def run(base, corpus, baseline, outdir, extra=()):
    command = [sys.executable, SCRIPT,
               "--judgments-glob",
               os.path.join(base, "generation_outputs/elm-docs-type2-v9-b[1-4]/"
                                  "rocketeval/judgments.tsv"),
               "--corpus", corpus, "--baseline", baseline,
               "--outdir", outdir, "--seed", "20260816"]
    command.extend(extra)
    return subprocess.run(command, capture_output=True, text=True)


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + ("  " + detail if detail else ""))
    if not condition:
        raise SystemExit(1)


def main():
    shutil.rmtree(ROOT, ignore_errors=True)

    # Case 1, happy path.
    base = os.path.join(ROOT, "happy")
    corpus, _, baseline = make_case(base, 20, 15, 18)
    out = os.path.join(base, "human_eval")
    result = run(base, corpus, baseline, out,
                 ["--expect-pass", "20", "--expect-fail", "15",
                  "--expect-disagree", "18"])
    check("happy path exits clean", result.returncode == 0,
          result.stdout + result.stderr)

    with open(os.path.join(out, "key.tsv"), newline="", encoding="utf-8") as handle:
        key = list(csv.DictReader(handle, delimiter="\t"))
    counts = {}
    for row in key:
        counts[(row["Frame"], row["Verdict"])] = counts.get(
            (row["Frame"], row["Verdict"]), 0) + 1
    check("key holds 60 items", len(key) == 60, str(len(key)))
    check("PASS stratum is 12", counts.get(("chase_finance", "PASS")) == 12)
    check("FAIL stratum is 8", counts.get(("chase_finance", "FAIL")) == 8)
    check("DISAGREE stratum is 10", counts.get(("chase_finance", "DISAGREE")) == 10)
    check("baseline frame is 30", counts.get(("baseline", "")) == 30)
    check("item ids run in order",
          [row["Item_ID"] for row in key] == [f"I{n:03d}" for n in range(1, 61)])
    frames = [row["Frame"] for row in key]
    check("shuffle interleaves the frames",
          frames != sorted(frames) and frames[:30] != ["chase_finance"] * 30)
    check("no generic row reaches the sample",
          all(row["Seed_Type"] in ("reg", "dynamic_reg", "") for row in key))
    check("grounding applicability tracks the frame",
          all((row["Grounding_Applicable"] == "yes") == (row["Frame"] == "chase_finance")
              for row in key))

    with open(os.path.join(out, "items.jsonl"), encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle]
    check("blinded items carry 5 fields only",
          all(set(item) == {"Item_ID", "Question", "Answer", "Documents",
                            "Reg_Text"}
              for item in items))
    check("the clause reaches the blinded set on chase-finance items only",
          sum(1 for item in items if item["Reg_Text"]) == 30)
    check("blinded items leak no verdict",
          not any(word in json.dumps(items)
                  for word in ("chase_finance", "DISAGREE", "Pass_Frac")))
    check("every item shows its documents",
          all(len(item["Documents"]) >= 1 for item in items))

    markdown_probe = open(os.path.join(out, "items.md"), encoding="utf-8").read()
    with open(os.path.join(out, "ratings.tsv"), newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle, delimiter="\t"))
    check("ratings template holds 60 rows with correctness blank",
          len(ratings) == 60 and all(row["Answer_Correct"] == "" for row in ratings))
    key_by_id = {row["Item_ID"]: row for row in key}
    check("baseline rows prefill the clause judgements with n/a",
          all((row["Clause_Governs"] == "n/a") ==
              (key_by_id[row["Item_ID"]]["Frame"] == "baseline")
              for row in ratings))
    check("clause judgements stay blank on chase-finance rows",
          all((row["Traces_To_Source"] == "") ==
              (key_by_id[row["Item_ID"]]["Frame"] == "chase_finance")
              for row in ratings))
    check("the source clause renders on chase-finance items only",
          markdown_probe.count("### Source clause") == 30,
          str(markdown_probe.count("### Source clause")))
    check("baseline items instruct the rater to record n/a",
          markdown_probe.count("carries no source clause") == 30)
    check("judgement B precedes judgement C in every item",
          all(block.index("B. Does the clause") < block.index("C. Does the answer")
              for block in markdown_probe.split("## I")[1:]
              if "B. Does the clause" in block))
    check("the clause sits after the documents",
          all(block.index("#### Document 1") < block.index("### Source clause")
              for block in markdown_probe.split("## I")[1:]
              if "### Source clause" in block))

    markdown = open(os.path.join(out, "items.md"), encoding="utf-8").read()
    check("markdown renders every item", markdown.count("## I") == 60)
    check("markdown carries no inline applicability flag by default",
          "Judgement B applicable" not in markdown)

    # Case 2, determinism.
    out2 = os.path.join(base, "human_eval_repeat")
    result = run(base, corpus, baseline, out2)
    with open(os.path.join(out2, "key.tsv"), newline="", encoding="utf-8") as handle:
        key2 = list(csv.DictReader(handle, delimiter="\t"))
    check("same seed reproduces the sample",
          [row["QID"] for row in key] == [row["QID"] for row in key2])

    # Case 3, a different seed draws a different sample.
    out3 = os.path.join(base, "human_eval_seed2")
    result = run(base, corpus, baseline, out3, [])
    command_seed = subprocess.run(
        [sys.executable, SCRIPT, "--judgments-glob",
         os.path.join(base, "generation_outputs/elm-docs-type2-v9-b[1-4]/"
                            "rocketeval/judgments.tsv"),
         "--corpus", corpus, "--baseline", baseline,
         "--outdir", os.path.join(base, "human_eval_seed3"), "--seed", "1"],
        capture_output=True, text=True)
    with open(os.path.join(base, "human_eval_seed3/key.tsv"), newline="",
              encoding="utf-8") as handle:
        key3 = list(csv.DictReader(handle, delimiter="\t"))
    check("a different seed draws differently",
          [row["QID"] for row in key] != [row["QID"] for row in key3])

    # Case 4, the key file already exists.
    result = run(base, corpus, baseline, out)
    check("existing key file halts the run",
          result.returncode != 0 and "key file already exists" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 5, a stratum comes up short.
    base = os.path.join(ROOT, "short")
    corpus, _, baseline = make_case(base, 20, 15, 6)
    result = run(base, corpus, baseline, os.path.join(base, "human_eval"))
    check("short stratum halts the run",
          result.returncode != 0 and "stratum DISAGREE holds 6" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 6, the corpus carries the wrong question type.
    base = os.path.join(ROOT, "type")
    corpus, _, baseline = make_case(base, 20, 15, 18, corpus_qa_type="type1")
    result = run(base, corpus, baseline, os.path.join(base, "human_eval"))
    check("type mismatch halts the run",
          result.returncode != 0 and "QA_Type type1" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 7, a mixed-shape baseline draws without aborting and records shape.
    base = os.path.join(ROOT, "shape")
    corpus, _, baseline = make_case(base, 20, 15, 18, baseline_mixed=True,
                                    baseline_double_encode=True)
    out7 = os.path.join(base, "human_eval")
    result = run(base, corpus, baseline, out7)
    check("mixed-shape baseline draws without aborting",
          result.returncode == 0, result.stdout + result.stderr)
    with open(os.path.join(out7, "key.tsv"), newline="", encoding="utf-8") as handle:
        key7 = list(csv.DictReader(handle, delimiter="\t"))
    bl7 = [row for row in key7 if row["Frame"] == "baseline"]
    check("every baseline row records its shape",
          len(bl7) == 30 and all(row["N_Docs"].isdigit() and row["N_Ans_Points"].isdigit()
                                 for row in bl7))
    check("recorded documents match recorded answer points",
          all(row["N_Docs"] == row["N_Ans_Points"] for row in bl7))
    check("the draw reaches rows below the Type 2 shape",
          any(int(row["N_Docs"]) < 3 for row in bl7))
    drawn7 = collections.Counter(int(row["N_Docs"]) for row in bl7)
    # the synthetic baseline spreads 40 rows evenly across 4 shapes, so a
    # proportional allocation of 30 hands 7 or 8 places to each shape.
    check("the baseline draw follows the proportional allocation",
          sorted(drawn7.values()) == [7, 7, 8, 8], str(dict(sorted(drawn7.items()))))
    check("the run summary reports the allocation",
          "baseline allocation, proportional" in result.stdout, result.stdout)
    check("the run summary reports the shape distribution",
          "baseline rows below the Type 2 shape" in result.stdout, result.stdout)
    check("double encoded rows decode rather than abort",
          all(int(row["N_Docs"]) >= 1 for row in bl7))
    check("chase-finance rows record their document count",
          all(int(row["N_Docs"]) == 3 for row in key7 if row["Frame"] == "chase_finance"))

    # Case 7b, a malformed document list still halts.
    base = os.path.join(ROOT, "broken")
    corpus, _, baseline = make_case(base, 20, 15, 18, baseline_broken_docs=True)
    result = run(base, corpus, baseline, os.path.join(base, "human_eval"))
    check("malformed document list halts the run",
          result.returncode != 0 and "malformed Rel_Docs_List" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 8, a duplicated judge vote.
    base = os.path.join(ROOT, "dupe")
    corpus, _, baseline = make_case(base, 20, 15, 18, duplicate_vote=True)
    result = run(base, corpus, baseline, os.path.join(base, "human_eval"))
    check("duplicate judge vote halts the run",
          result.returncode != 0 and "duplicate judge vote" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 9, the expected verdict counts disagree with the recomputation.
    base = os.path.join(ROOT, "expect")
    corpus, _, baseline = make_case(base, 20, 15, 18)
    result = run(base, corpus, baseline, os.path.join(base, "human_eval"),
                 ["--expect-pass", "19"])
    check("expected count mismatch halts the run",
          result.returncode != 0 and "recomputed PASS count 20" in result.stdout + result.stderr,
          result.stdout + result.stderr)

    # Case 10, the inline applicability flag.
    base = os.path.join(ROOT, "inline")
    corpus, _, baseline = make_case(base, 20, 15, 18)
    out = os.path.join(base, "human_eval")
    result = run(base, corpus, baseline, out, ["--flag-grounding-inline"])
    markdown = open(os.path.join(out, "items.md"), encoding="utf-8").read()
    check("inline flag reaches the markdown",
          result.returncode == 0 and markdown.count("Judgement B applicable") == 60,
          str(markdown.count("Judgement B applicable")))

    print("\nAll synthetic checks passed.")


if __name__ == "__main__":
    main()