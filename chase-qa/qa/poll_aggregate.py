import os
import argparse
import pandas as pd
import numpy as np


def build_parser():
	parser = argparse.ArgumentParser(description='Aggregate judge panel verdicts (PoLL)')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, required=True, help='Run folder containing rocketeval/judgments.tsv')
	parser.add_argument('-unsure', type=str, default='exclude', choices=['exclude', 'zero', 'half'], help='How Unsure verdicts score: excluded from denominator, counted as 0, or as 0.5')
	parser.add_argument('-pass_frac', type=float, default=0.75, help='Fraction of criteria that must pass consensus for the question to PASS. 0.75 per D10 sweep; N_Criteria is 8-9 so 0.9 is de facto unanimity.')
	parser.add_argument('-disagree_frac', type=float, default=0.34, help='If more than this fraction of criteria are DISAGREE, the question verdict is DISAGREE')
	return parser


def verdict_to_score(v, unsure_mode):
	"""Map a single verdict to a numeric score, or NaN if excluded."""
	if v == "Yes":
		return 1.0
	if v == "No":
		return 0.0
	# Unsure
	if unsure_mode == "zero":
		return 0.0
	if unsure_mode == "half":
		return 0.5
	return np.nan  # exclude


def criterion_consensus(verdicts, unsure_mode):
	"""Consensus for one criterion across judges.

	Returns (label, mean_score, disagree_flag).
	label in {Yes, No, DISAGREE, Undecided}.
	  - Yes/No: strict majority of scoring judges on that side
	  - DISAGREE: scoring judges split evenly, or tie
	  - Undecided: no judge produced a scoring verdict (all excluded)
	disagree_flag is True when judges did not reach the same side.
	"""
	scores = [verdict_to_score(v, unsure_mode) for v in verdicts]
	valid = [s for s in scores if not np.isnan(s)]
	if len(valid) == 0:
		return "Undecided", np.nan, True
	mean = float(np.mean(valid))
	yes = sum(1 for s in valid if s >= 0.75)
	no = sum(1 for s in valid if s <= 0.25)
	# 'half' contributes to neither side, sits in the middle
	if yes > no:
		label = "Yes"
	elif no > yes:
		label = "No"
	else:
		label = "DISAGREE"
	unanimous = (yes == len(valid)) or (no == len(valid))
	return label, mean, (not unanimous)


def main(args):
	rk_dir = os.path.join(args.out_dir, args.folder_name, "rocketeval")
	j = pd.read_csv(os.path.join(rk_dir, "judgments.tsv"), sep='\t')

	# --- input integrity -------------------------------------------------
	group_cols_chk = ["Root_ID", "Question_No", "QA_Type", "Seed_Type"]
	miss = j[group_cols_chk].isna().any(axis=1)
	if miss.any():
		raise SystemExit(
			"ERROR: {} row(s) have a missing value in {}. groupby drops these silently.".format(
				int(miss.sum()), group_cols_chk))

	# judge_panel.py writes with mode='a', so a resumed run appends. Uniform
	# duplication leaves the distribution constant, so print it unconditionally
	# and read it against the known panel size rather than testing for variance.
	dup_key = ["Root_ID", "Question_No", "Criterion_No", "Judge"]
	n_dup = int(j.duplicated(subset=dup_key).sum())
	if n_dup:
		raise SystemExit(
			"ERROR: {} row(s) repeat a (question, criterion, judge) key. judge_panel.py writes "
			"with mode='a', so a resumed run appends instead of replacing. Delete judgments.tsv "
			"and rerun, or de-duplicate before aggregating.".format(n_dup))

	sizes = j.groupby(["Root_ID", "Question_No", "Criterion_No"])["Judge"].nunique()
	print("INFO: judges per criterion: {}".format(dict(sorted(sizes.value_counts().items()))))
	if sizes.nunique() > 1:
		print("WARNING: judges per criterion is not constant. Partial judge failure.")

	known = {"Yes", "No", "Unsure"}
	bad = j.loc[~j["Verdict"].isin(known), "Verdict"]
	if len(bad):
		print("WARNING: {} verdict(s) outside {}. These score as excluded, so a failed "
		      "judge call is indistinguishable from a genuine Unsure. Values: {}".format(
		      len(bad), sorted(known), dict(bad.value_counts(dropna=False))))
	# ---------------------------------------------------------------------

	crit_rows = []
	q_rows = []
	group_cols = ["Root_ID", "Question_No", "QA_Type", "Seed_Type"]
	for key, g in j.groupby(group_cols):
		root_id, q_no, qa_type, seed_type = key
		labels = []
		non_unanimous = 0      # judges did not all agree, a disagreement-rate metric
		no_majority = 0        # no side won: even split, tie, or all-excluded
		crit_means = []
		for cno, cg in g.groupby("Criterion_No"):
			verdicts = list(cg["Verdict"])
			label, mean, dis = criterion_consensus(verdicts, args.unsure)
			labels.append(label)
			if dis:
				non_unanimous += 1
			if label in ("DISAGREE", "Undecided"):
				no_majority += 1
			if not np.isnan(mean):
				crit_means.append(mean)
			crit_rows.append({
				"Root_ID": root_id, "Question_No": q_no, "QA_Type": qa_type, "Seed_Type": seed_type,
				"Criterion_No": cno, "Consensus": label, "Mean_Score": round(mean, 4) if not np.isnan(mean) else "",
				"N_Judges": len(verdicts), "Non_Unanimous": dis,
			})

		n_crit = len(labels)
		n_pass = sum(1 for l in labels if l == "Yes")
		pass_frac = n_pass / n_crit if n_crit else 0.0
		disagreement_rate = non_unanimous / n_crit if n_crit else 0.0   # logged for RQ3
		no_majority_frac = no_majority / n_crit if n_crit else 0.0      # drives DISAGREE verdict
		q_mean = float(np.mean(crit_means)) if crit_means else np.nan

		if no_majority_frac > args.disagree_frac:
			verdict = "DISAGREE"
		elif pass_frac >= args.pass_frac:
			verdict = "PASS"
		else:
			verdict = "FAIL"

		q_rows.append({
			"Root_ID": root_id, "Question_No": q_no, "QA_Type": qa_type, "Seed_Type": seed_type,
			"N_Criteria": n_crit, "N_Pass": n_pass, "Pass_Frac": round(pass_frac, 4),
			"No_Majority_Frac": round(no_majority_frac, 4),
			"Disagreement_Rate": round(disagreement_rate, 4),
			"Mean_Score": round(q_mean, 4) if not np.isnan(q_mean) else "",
			"Verdict": verdict,
		})

	crit_df = pd.DataFrame(crit_rows)
	q_df = pd.DataFrame(q_rows)
	if len(q_df) == 0:
		raise SystemExit("ERROR: no questions aggregated. judgments.tsv is empty or malformed. Nothing written.")
	crit_df.to_csv(os.path.join(rk_dir, "poll_criterion_scores.tsv"), sep='\t', index=None)
	q_df.to_csv(os.path.join(rk_dir, "poll_scores.tsv"), sep='\t', index=None)

	# Summary, overall and split by Seed_Type and QA_Type.
	summary_lines = []
	summary_lines.append("Config: unsure={}  pass_frac={}  disagree_frac={}".format(args.unsure, args.pass_frac, args.disagree_frac))
	summary_lines.append("Questions: {}".format(len(q_df)))

	def block(df, title):
		if len(df) == 0:
			return
		vc = df["Verdict"].value_counts().to_dict()
		summary_lines.append("{}: PASS={} FAIL={} DISAGREE={}  mean_pass_frac={:.4f}".format(
			title, vc.get("PASS", 0), vc.get("FAIL", 0), vc.get("DISAGREE", 0), df["Pass_Frac"].mean()))

	block(q_df, "Overall")
	for st, g in q_df.groupby("Seed_Type"):
		block(g, "Seed_Type={}".format(st))
	for qt, g in q_df.groupby("QA_Type"):
		block(g, "QA_Type={}".format(qt))

	summary = "\n".join(summary_lines)
	with open(os.path.join(rk_dir, "poll_summary.txt"), "w") as f:
		f.write(summary + "\n")
	print(summary)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	main(args)