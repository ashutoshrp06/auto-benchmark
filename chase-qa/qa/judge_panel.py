import os
import re
import json
import argparse
import pandas as pd
import openai
import anthropic
import google.generativeai as genai
from models import LargeLanguageModel
from rocketeval_prompts import get_grading_prompt
from checklist_generator import unescape_answer, load_docs


# Panel definition. Each judge is (name, model_type, model_string).
# name is a stable label used in output and for resume keys.
# llama-3.3 ELM model string is a placeholder, replace when known.
DEFAULT_PANEL = [
	("gpt54mini", "elm", "gpt-5.4-mini"),
	("gemini35flashlite", "gemini", "gemini-3.5-flash-lite"),
	#("llama33", "elm", "meta-llama/Llama-3.3-70B-Instruct"),
]

_VERDICT_PATTERN = re.compile(r'^\s*(\d+)\s*[\.\)]\s*(yes|no|unsure)\b', re.IGNORECASE)
_FENCE_PATTERN = re.compile(r'^\s*```')


def build_parser():
	parser = argparse.ArgumentParser(description='Grade RocketEval checklists with a judge panel')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, required=True, help='Run folder containing data + rocketeval/checklists.jsonl')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Data filename without extension')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-max_tokens', type=int, default=2000, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='presence penalty')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='frequency penalty')
	return parser


def parse_verdicts(text, n_expected):
	"""Map criterion index -> Yes/No/Unsure. Missing or unparseable indices become Unsure.

	Returns a list of length n_expected aligned to criteria order (1-based in the prompt).
	"""
	found = {}
	for line in str(text).split("\n"):
		if _FENCE_PATTERN.match(line):
			continue
		m = _VERDICT_PATTERN.match(line)
		if not m:
			continue
		idx = int(m.group(1))
		verdict = m.group(2).capitalize()
		if 1 <= idx <= n_expected and idx not in found:
			found[idx] = verdict
	return [found.get(i, "Unsure") for i in range(1, n_expected + 1)]


def load_checklists(path):
	"""Return dict (Root_ID, Question_No) -> record."""
	checklists = {}
	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			checklists[(int(rec["Root_ID"]), int(rec["Question_No"]))] = rec
	return checklists


def load_done(path):
	"""Return set of (Root_ID, Question_No, Judge) already graded, for resume."""
	done = set()
	if not os.path.exists(path):
		return done
	df = pd.read_csv(path, sep='\t')
	for _, r in df.iterrows():
		done.add((int(r["Root_ID"]), int(r["Question_No"]), str(r["Judge"])))
	return done


def append_rows(path, rows):
	"""Append long-format rows, writing header only if file is new."""
	df = pd.DataFrame(rows)
	header = not os.path.exists(path)
	df.to_csv(path, sep='\t', index=None, mode='a', header=header)


def main(args):
	data_path = os.path.join(args.out_dir, args.folder_name, args.data + ".tsv")
	data = pd.read_csv(data_path, sep='\t')
	data_by_key = {(int(r["Root_ID"]), int(r["Question_No"])): r for _, r in data.iterrows()}

	rk_dir = os.path.join(args.out_dir, args.folder_name, "rocketeval")
	checklists = load_checklists(os.path.join(rk_dir, "checklists.jsonl"))
	out_path = os.path.join(rk_dir, "judgments.tsv")
	log_path = os.path.join(rk_dir, "judgment_logs.txt")

	done = load_done(out_path)
	print("Already graded (row x judge): {}".format(len(done)))

	# One model instance per judge. The Gemini branch fixes sys_prompt at __init__,
	# so the grading sys_prompt must be supplied here, not per call.
	_, grading_sys = get_grading_prompt(("", "", ""))
	judges = []
	for name, mtype, mstr in DEFAULT_PANEL:
		m = LargeLanguageModel(model_type=mtype, model=mstr, peft_model="none", sys_prompt=grading_sys, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)
		judges.append((name, m))

	keys = sorted(checklists.keys())
	total = len(keys) * len(judges)
	graded = 0
	for ki, key in enumerate(keys):
		rec = checklists[key]
		if key not in data_by_key:
			with open(log_path, "a") as f:
				f.write("WARNING: checklist key {} has no matching data row, skipped.\n".format(key))
			continue
		row = data_by_key[key]
		question = str(row["Question"])
		documents = load_docs(row["Rel_Docs_List"])
		criteria = rec["Checklist"]
		n = len(criteria)
		criteria_block = "\n".join("{}. {}".format(i + 1, c) for i, c in enumerate(criteria))

		for name, model in judges:
			if (key[0], key[1], name) in done:
				continue
			prompt, sys_prompt = get_grading_prompt((question, documents, criteria_block))
			og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
			verdicts = parse_verdicts(og_pred, n)
			n_unsure = sum(1 for v in verdicts if v == "Unsure")

			with open(log_path, "a") as f:
				f.write("Root_ID: {}  Question_No: {}  Judge: {}  Unsure: {}/{}\n".format(key[0], key[1], name, n_unsure, n))
				f.write(str(og_pred) + "\n")
				f.write("---------------------------------------------------------\n")

			rows = []
			for cidx, (crit, verdict) in enumerate(zip(criteria, verdicts)):
				rows.append({
					"Root_ID": key[0],
					"Question_No": key[1],
					"QA_Type": rec["QA_Type"],
					"Seed_Type": rec["Seed_Type"],
					"Judge": name,
					"Criterion_No": cidx + 1,
					"Criterion": crit,
					"Verdict": verdict,
				})
			append_rows(out_path, rows)
			graded += 1
			print("Completed {} / {} (row {}/{}, judge {})...".format(graded + len(done), total, ki + 1, len(keys), name), end='\r', flush=True)

	print("\nJudgments appended: {}".format(graded))


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
	main(args)