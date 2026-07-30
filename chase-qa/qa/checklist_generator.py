import os
import re
import json
import argparse
import pandas as pd
import openai
import anthropic
import google.generativeai as genai
from models import LargeLanguageModel
from rocketeval_prompts import get_checklist_prompt


def build_parser():
	parser = argparse.ArgumentParser(description='Generate RocketEval checklists')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, required=True, help='Run folder containing the data file')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Data filename without extension')
	parser.add_argument('-qa_type', type=str, required=True, choices=['type1', 'type2', 'type3'], help='Question type of this directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-model_type', type=str, default='elm', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic', 'elm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-5.4-mini', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=4000, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.3, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases likelihood to repeat lines')
	parser.add_argument('-min_criteria', type=int, default=4, help='Rows yielding fewer criteria than this are flagged')
	return parser


_CRITERION_PATTERN = re.compile(r'^\s*(\d+)\s*[\.\)]\s*(.+?)\s*$')
_FENCE_PATTERN = re.compile(r'^\s*```[a-zA-Z]*\s*$')


def unescape_answer(ans):
	"""Answer is stored with literal backslash-n by generator.py, restore real newlines."""
	return str(ans).replace('\\n', '\n').strip()


def load_docs(rel_docs_list):
	"""Rel_Docs_List is a JSON-encoded list of document strings."""
	docs = json.loads(rel_docs_list)
	if isinstance(docs, str):
		docs = [docs]
	return "\n\n".join(str(d).strip() for d in docs)


def parse_criteria(text):
	"""Extract numbered criteria. Returns criteria in emitted order, ignoring numbering gaps."""
	criteria = []
	for line in str(text).split("\n"):
		if _FENCE_PATTERN.match(line):
			continue
		m = _CRITERION_PATTERN.match(line)
		if not m:
			continue
		body = m.group(2).strip()
		body = re.sub(r'^\*\*(.*?)\*\*$', r'\1', body).strip()
		if body:
			criteria.append(body)
	return criteria


def load_done(path):
	"""Return set of (Root_ID, Question_No) already written, so runs are resumable."""
	done = set()
	if not os.path.exists(path):
		return done
	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			done.add((str(rec["Root_ID"]), str(rec["Question_No"])))
	return done


def main(args):
	data_path = os.path.join(args.out_dir, args.folder_name, args.data + ".tsv")
	data = pd.read_csv(data_path, sep='\t')

	args.checklist_dir = os.path.join(args.out_dir, args.folder_name, "rocketeval")
	if not os.path.exists(args.checklist_dir):
		os.makedirs(args.checklist_dir)
	out_path = os.path.join(args.checklist_dir, "checklists.jsonl")
	log_path = os.path.join(args.checklist_dir, "checklist_logs.txt")

	done = load_done(out_path)
	print("Already done: {}".format(len(done)))

	_, sys_prompt = get_checklist_prompt(args.qa_type, ("", ""))
	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	written = 0
	skipped = 0
	for i in range(len(data)):
		row = data.iloc[i]
		key = (str(row["Root_ID"]), str(row["Question_No"]))
		if key in done:
			continue

		answer = unescape_answer(row["Answer"])
		documents = load_docs(row["Rel_Docs_List"])
		prompt, sys_prompt = get_checklist_prompt(args.qa_type, (answer, documents))
		og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
		criteria = parse_criteria(og_pred)

		with open(log_path, "a") as f:
			f.write("Root_ID: {}  Question_No: {}\n".format(key[0], key[1]))
			f.write("Parsed criteria: {}\n\n".format(len(criteria)))
			f.write(str(og_pred) + "\n")
			f.write("---------------------------------------------------------\n")

		if len(criteria) < args.min_criteria:
			skipped += 1
			with open(log_path, "a") as f:
				f.write("SKIPPED: only {} criteria parsed (min {}).\n".format(len(criteria), args.min_criteria))
				f.write("---------------------------------------------------------\n")
			print("Completed {} / {}... (skipped, {} criteria)".format(i + 1, len(data), len(criteria)), end='\r', flush=True)
			continue

		record = {
			"Root_ID": int(row["Root_ID"]),
			"Question_No": int(row["Question_No"]),
			"QA_Type": args.qa_type,
			"Seed_Type": str(row["Seed_Type"]),
			"Checklist": criteria,
		}
		with open(out_path, "a") as f:
			f.write(json.dumps(record) + "\n")
		written += 1
		print("Completed {} / {}...".format(i + 1, len(data)), end='\r', flush=True)

	print("\nChecklists written: {}   Skipped: {}".format(written, skipped))


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
	main(args)