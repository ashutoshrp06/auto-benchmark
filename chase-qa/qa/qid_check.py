import hashlib, pandas as pd
from collections import Counter

def make_qid(qa_type, persona, environment, question):
    payload = "\x00".join([str(qa_type), str(persona).strip(),
                           str(environment).strip(), str(question).strip()])
    return "q" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

rows = []
for t in (1, 2, 3):
    p = "generation_outputs/elm-docs-type{}-v7/programmatic_data_modified_verified_cleaned.tsv".format(t)
    d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    d["QID"] = [make_qid("type{}".format(t), a, b, c)
                for a, b, c in zip(d["Persona"], d["Environment"], d["Question"])]
    d["QA_Type"] = "type{}".format(t)
    print("type{}: {} rows, {} unique QIDs".format(t, len(d), d["QID"].nunique()))
    rows.append(d[["QA_Type", "Root_ID", "Question_No", "QID", "Question"]])

allr = pd.concat(rows, ignore_index=True)
print("TOTAL: {} rows, {} unique QIDs".format(len(allr), allr["QID"].nunique()))
c = Counter(allr["QID"])
dupes = [q for q, n in c.items() if n > 1]
if dupes:
    print(allr[allr["QID"].isin(dupes)].sort_values("QID").to_string())