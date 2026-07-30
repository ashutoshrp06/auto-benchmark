import argparse
import random
import pandas as pd
import pdb
import json
import tiktoken
import hashlib
from collections import Counter

tik_enc = tiktoken.encoding_for_model("gpt-4o")

def build_parser():
	parser = argparse.ArgumentParser(description='Post Process')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/gpt-4o-mini-docs/', help='Output Directory')
	parser.add_argument('-og_data', type=str, default='programmatic_data_modified_verified', help='Data filename')
	parser.add_argument('-seed_data', type=str, default='programmatic_data_modified_verified', help='Data filename')
	parser.add_argument('-threshold', type=int, default=0, help='Number of problems to take docs from')
	parser.add_argument('-qa_type', type=str, default='', help='QA type tag mixed into QID, e.g. type1. Keeps QID unique across the three type corpora.')
	parser.add_argument('-seed', type=int, default=-1, help='Seed for random.shuffle and random.randint. -1 leaves the RNG unseeded, which reproduces every run made before this flag existed. Any other value makes cleanup deterministic and therefore safely rerunnable.')
	
	return parser

def jaccard_similarity(set1, set2):
	# Calculate the intersection of the two sets
	intersection = set1.intersection(set2)
	
	# Calculate the union of the two sets
	union = set1.union(set2)
	
	# Calculate the Jaccard similarity coefficient
	if not union:
		return 100.0  # If both sets are empty, we define the similarity as 100%
	similarity = (len(intersection) / len(union)) * 100
	
	return similarity


def get_sim_score(ls, id1, id2):
	persona1 = ls[id1]["Persona"]
	persona2 = ls[id2]["Persona"]
	env1 = ls[id1]["Environment"]
	env2 = ls[id2]["Environment"]

	sim1 = jaccard_similarity(set(persona1.lower().split()), set(persona2.lower().split()))
	sim2 = jaccard_similarity(set(env1.lower().split()), set(env2.lower().split()))

	avg_sim = (sim1 + sim2)/2
	
	return avg_sim

def process(args, og_data, seed_data, threshold=0):
	ls = og_data.to_dict(orient='records')
	seed_ls = seed_data.to_dict(orient='records')

	max_context_size = 0
	avg_context_size = 0

	idxs_to_remove = []
	removed_records = []

	for i in range(len(seed_ls)):
		seed_ls[i]["QID"] = make_qid(
			args.qa_type,
			seed_ls[i]["Persona"],
			seed_ls[i]["Environment"],
			seed_ls[i]["Question"],
		)
		qno = seed_ls[i]["Question_No"]
		ex = None
		idx_chosen = None
		for j in range(len(ls)):
			if ls[j]["Question_No"] == qno:
				ex = ls[j]
				idx_chosen = j
				break
		
		if ex is None:
				raise RuntimeError(
					"cleanup.py: no example found for Question_No {}. The next "
					"statement dereferences it, so this cannot continue.".format(qno))
		
		try:
			rel_doc_list = json.loads(json.loads(ex["Rel_Docs_List"]))
		except:
			rel_doc_list = json.loads(ex["Rel_Docs_List"])
		try:
			adv_doc_list = json.loads(json.loads(ex["Adv_Docs_List"]))
		except:
			adv_doc_list = json.loads(ex["Adv_Docs_List"])
		
		if len(rel_doc_list) < 1:
			try:
				ans_pts = json.loads(json.loads(seed_ls[i]["Ans_Points"]))
			except:
				try:
					ans_pts = json.loads(seed_ls[i]["Ans_Points"])
				except:
					ans_pts = None
			if not isinstance(ans_pts, list):
				ans_pts = []
			rec = dict(seed_ls[i])
			rec["_Removed_Reason"] = "empty_rel_docs_list"
			rec["_N_Ans_Points"] = len(ans_pts)
			rec["_Conclusion_Only"] = bool(
				len(ans_pts) == 1
				and str(ans_pts[0]).lstrip("- ").lower().startswith("conclusion:")
			)
			removed_records.append(rec)
			idxs_to_remove.append(i)
			continue

		idxs = []
		while(len(idxs) < threshold):
			cur_idx = random.randint(0, len(ls)-1)
			if cur_idx in idxs or cur_idx == idx_chosen:
				continue
			if get_sim_score(ls, idx_chosen, cur_idx) < 20:
				idxs.append(cur_idx)

		all_docs_list = []
		for doc in rel_doc_list:
			all_docs_list.append(doc)
		for adv in adv_doc_list:
			for doc in adv:
				all_docs_list.append(doc)

		for cur_id in idxs:
			try:
				adv_doc_list = json.loads(json.loads(ls[cur_id]["Adv_Docs_List"]))
			except:
				adv_doc_list = json.loads(ls[cur_id]["Adv_Docs_List"])
			for adv in adv_doc_list:
				for doc in adv:
					all_docs_list.append(doc)

		random.shuffle(all_docs_list)

		new_docs = ""
		cnt = 1
		for doc in all_docs_list:
			new_docs = new_docs + "Document " + str(cnt) + ":\n" + doc + "\n\n"
			cnt += 1

		seed_ls[i]["Documents"] = new_docs.strip()

		context = seed_ls[i]["Documents"] + seed_ls[i]["Question"]
		context_size = len(tik_enc.encode(context))

		if context_size > max_context_size:
			max_context_size = context_size

		avg_context_size += context_size
	
	print("Max Context Size: ", str(max_context_size))

	for idx in sorted(idxs_to_remove, reverse=True):
		del seed_ls[idx]

	print("Rows removed (no relevant documents): ", len(removed_records))
	if removed_records:
		rem_df = pd.DataFrame(removed_records)
		print("  of which Ans_Points was Conclusion-only: ", int(rem_df["_Conclusion_Only"].sum()))
		rem_df.to_csv(args.out_dir + "/" + args.seed_data + "_removed.tsv", sep = '\t', index = None)

	print("Avg Context Size: ", str(avg_context_size / len(seed_ls)) if seed_ls else "n/a (0 rows)")

	qid_counts = Counter(r["QID"] for r in seed_ls)
	print("QID: qa_type='{}', {} unique across {} rows".format(args.qa_type, len(qid_counts), len(seed_ls)))
	if not str(args.qa_type).strip():
		print("WARNING: -qa_type is empty. QIDs will not match a run that sets it, and can collide across the three type corpora.")
	dupe_qids = set(q for q, n in qid_counts.items() if n > 1)
	if dupe_qids:
		print("WARNING: {} QID value(s) occur more than once. Identical (Persona, Environment, Question) inside this corpus.".format(len(dupe_qids)))
		pd.DataFrame([r for r in seed_ls if r["QID"] in dupe_qids]).to_csv(
			args.out_dir + "/" + args.seed_data + "_qid_collisions.tsv", sep = '\t', index = None)

	new_df = pd.DataFrame(seed_ls)
	new_df.to_csv(args.out_dir + "/" + args.seed_data + "_cleaned.tsv", sep = '\t', index = None)

	print("Size of the cleaned data: ", len(new_df))

def make_qid(qa_type, persona, environment, question):
	# Content-derived and position-free, so it survives merge, re-shard and
	# row drops, and matches across separately generated batches.
	# \x00 cannot appear in a TSV field, so no field boundary is ambiguous.
	# The 'q' prefix keeps the value non-numeric, so no reader can coerce it
	# to int64 and strip leading zeros.
	payload = "\x00".join([
		str(qa_type),
		str(persona).strip(),
		str(environment).strip(),
		str(question).strip(),
	])
	return "q" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def main(args):
	og_data = pd.read_csv(args.out_dir + "/" + args.og_data + ".tsv", sep='\t')
	seed_data = pd.read_csv(args.out_dir + "/" + args.seed_data + ".tsv", sep='\t')
	process(args, og_data, seed_data, args.threshold)
	
if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	if args.seed >= 0:
		random.seed(args.seed)

	main(args)