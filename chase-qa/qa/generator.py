import pandas as pd # type: ignore

import os
import argparse
import random
import json
import re
import tiktoken # type: ignore
import pdb

import openai # type: ignore
import anthropic # type: ignore
import google.generativeai as genai # type: ignore

from utils import sample_scenarios, process_naive
from models import LargeLanguageModel
from prompts import get_generator_prompt, get_verification_prompt

import datetime

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-adversarial_name', type=str, default='gpt-4o-mini-adv', help='Adversarial name')
	parser.add_argument('-questions_name', type=str, default='gpt-4o-mini-qa', help='Questions name')
	parser.add_argument('-scenarios_name', type=str, default='gpt-4o-mini-scenarios', help='Scenarios name')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-exp_type', type=str, default='programmatic_scenarios', help='Exp type')
	parser.add_argument('-prompt_type', type=str, default='programmatic_scenarios', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic', 'elm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-3.5-turbo', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=8000, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=1.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-num_iters', type=int, default=5, help='number of iterations to run')

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

def programmatic_scenario_generation(model, prompt_type, num_iters, max_tokens, temperature, stop, tik_encoding):
	with open("reg_clauses.json", "r") as f:
		reg_clauses = json.load(f)

	pred_ls = [
		[1, "Retail client seeking pension consolidation advice", "FCA COBS 19 pension transfer rules", 0],
		[2, "First-time investor consulting a financial adviser", "FCA Consumer Duty (PRIN 2A) and COBS 4 communication rules", 0],
		[3, "Client approaching retirement reviewing drawdown options", "FCA COBS 9.2 suitability requirements and HMRC PTM056510 money purchase annual allowance rules", 0],
		[4, "Client weighing long-term care funding options against their pension", "FCA COBS 16.6 communications on long-term care insurance and drawdown pensions", 0],
		[5, "Homebuyer seeking mortgage affordability advice", "FCA MCOB 11.6 responsible lending and affordability assessment rules", 0],
		[6, "High-net-worth individual seeking estate planning strategies", "HMRC inheritance tax nil-rate band rules (gov.uk) and FCA COBS 9.2 suitability requirements for advisers", 0],
		[7, "Individual considering equity release on their property", "FCA MCOB 8 and MCOB 9 equity release advising and disclosure rules", 0],
		[8, "Parent planning children's education savings", "FCA COBS 9 suitability and COBS 4 disclosure rules for Junior ISAs", 0],
		[9, "Small business owner setting up workplace pension schemes", "Pensions Act 2008 auto-enrolment duties and FCA workplace pension default fund charge cap rules", 0],
		[10, "Investor evaluating cryptoasset investment products", "FCA PS23/6 cryptoasset financial promotion rules", 0]
	]

	for row in pred_ls:
		row.append(reg_clauses.get(str(row[0]), ""))

	tot_ip_tokens = 0
	tot_op_tokens = 0

	pred_id = len(pred_ls)

	# Write seeded scenarios immediately so num_iters=0 still produces a valid scenarios.tsv
	pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Persona', 'Environment', 'Similarity', 'Reg_Text'])
	pred_df.to_csv(args.out_dir + "/scenarios.tsv", sep = '\t', index = None)

	for i in range(num_iters):
		if i < 20:
			with open("annotated_scenarios.txt", "r") as f:
				sampled_scenarios = f.read()
		else:
			sampled_scenarios = sample_scenarios(pred_ls)

		prompt, sys_prompt = get_generator_prompt(prompt_type, question=sampled_scenarios)

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		tot_ip_tokens += ip_tokens
		tot_op_tokens += op_tokens

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Scenario Set " + str(i) + ":\n\n" + og_pred + "\n\n")
			f.write("------------------------\n")

		lines = og_pred.strip().split("\n")
		for l in range(len(lines)):
			if lines[l][:12] == "USER_PERSONA":
				persona = lines[l].split(":")[1].strip()
				if lines[l+1][:13] == "COLLECTION_OF":
					env = lines[l+1].split(":")[1].strip()
				avg_sim = 0
				for j in range(len(pred_ls)):
					prev_per = pred_ls[j][1]
					prev_env = pred_ls[j][2]
					sim1 = jaccard_similarity(set(persona.lower().split()), set(prev_per.lower().split()))
					sim2 = jaccard_similarity(set(env.lower().split()), set(prev_env.lower().split()))
					cur_sim = (sim1 + sim2)/2
					if cur_sim > avg_sim:
						avg_sim = cur_sim
				if avg_sim < 60:
					pred_ls.append([pred_id+1, persona, env, avg_sim, ""])
					pred_id += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Persona', 'Environment', 'Similarity', 'Reg_Text'])
		pred_df.to_csv(args.out_dir + "/scenarios.tsv", sep = '\t', index = None)

		i += 1
		print("Completed {} / {}...".format(i, num_iters), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)

def strip_quotes(s):
	s = s.strip()
	quote_chars = '"\'“”‘’'
	while len(s) >= 2 and s[0] in quote_chars and s[-1] in quote_chars:
		s = s[1:-1].strip()
	return s

def reg_pregen_grounding_check(args, model, ques, reg_text, answer):
	prompt, sys_prompt = get_verification_prompt("reg_pregen_grounding", params=(ques, reg_text, answer))
	og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)

	grounded_res = True
	if "false" in og_pred.split("Flagged Point")[0].split("Grounded:")[1].lower():
		grounded_res = False

	with open(args.out_dir + "/reg_pregen_logs.txt", "a") as f:
		f.write("Question: " + str(ques) + "\n\n")
		f.write("Regulatory Source Text:\n" + str(reg_text) + "\n\n")
		f.write("Answer:\n" + str(answer) + "\n\n")
		f.write("Prediction:\n" + og_pred + "\n")
		f.write("---------------------------------------------------------\n")

	if grounded_res:
		return True, None, None

	flagged_point = og_pred.split("Flagged Point")[1].split("Corrected Point")[0].split(":", 1)[1].strip()
	corrected_point = og_pred.split("Corrected Point")[1].split(":", 1)[1].strip()

	flagged_point = strip_quotes(flagged_point)
	corrected_point = strip_quotes(corrected_point)

	return False, flagged_point, corrected_point


def reg_pregen_retry(args, model, ques, reg_text, answer):
	if str(reg_text).strip() == "" or str(reg_text).strip().lower() == "nan":
		return answer, False

	grounded, flagged_point, corrected_point = reg_pregen_grounding_check(args, model, ques, reg_text, answer)

	if grounded:
		return answer, False

	if flagged_point in answer:
		new_answer = answer.replace(flagged_point, corrected_point)
	else:
		with open(args.out_dir + "/reg_pregen_logs.txt", "a") as f:
			f.write("WARNING: flagged point not found verbatim in answer, leaving unmodified before recheck.\nFlagged Point: " + flagged_point + "\n")
			f.write("---------------------------------------------------------\n")
		with open(args.out_dir + "/reg_pregen_logs.txt", "a") as f:
			f.write("DISCARD: splice failed, no correction made, skipping recheck.\n")
			f.write("---------------------------------------------------------\n")
		return answer, True
	regrounded, _, _ = reg_pregen_grounding_check(args, model, ques, reg_text, new_answer)

	if regrounded:
		with open(args.out_dir + "/reg_pregen_logs.txt", "a") as f:
			f.write("CORRECTED IN PLACE\nFlagged Point: " + flagged_point + "\nCorrected Point: " + corrected_point + "\n")
			f.write("---------------------------------------------------------\n")
		return new_answer, False

	with open(args.out_dir + "/reg_pregen_logs.txt", "a") as f:
		f.write("DISCARD: still ungrounded after one correction attempt.\n")
		f.write("---------------------------------------------------------\n")
	return new_answer, True

def programmatic_qa_generation(scenarios_data, model, prompt_type, num_iters, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0

	for i in range(len(scenarios_data)):
		persona = scenarios_data.loc[i]["Persona"]
		env = scenarios_data.loc[i]["Environment"]
		reg_text = scenarios_data.loc[i]["Reg_Text"]
		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Scenario " + str(i+1) + ":\n\n")
			f.write("Persona: " + persona + "\n")
			f.write("Environment: " + env + "\n")
			f.write("\n")

		group = []

		for xy in range(num_iters):
		
			prompt, sys_prompt = get_generator_prompt(prompt_type, question=(persona, env, reg_text	))

			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

			ip_tokens = len(tik_encoding.encode(prompt))
			op_tokens = len(tik_encoding.encode(og_pred))

			tot_ip_tokens += ip_tokens
			tot_op_tokens += op_tokens

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write(og_pred + "\n\n")

			resp = og_pred.strip().split("\n")
			for j in range(len(resp)):
				if resp[j][:8] == "Question":
					question = resp[j].split(":")[1].strip()
					if question == "":
						question = resp[j+1]
				if resp[j][:6] == "Answer":
					for zj in range(len(resp)):
						if resp[zj][:16] == "Document 1 Title":
							lno = zj
							break
					answer = "\n".join(resp[j:lno]).split("Answer:")[1].strip()
				if resp[j][:10] == "Document 1":
					docs_info = "\n".join(resp[j:])
					break

			main_sim = 0
			for prev_q in group:
				sim = jaccard_similarity(set(prev_q.lower().split()), set(question.lower().split()))
				if main_sim < sim:
					main_sim = sim

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Similarity: " + str(main_sim) + "\n\n")
				f.write("------------------------------------------------------------------\n\n")

			if main_sim < 60:
				group.append(question)
				answer, discard_row = reg_pregen_retry(args, model, question, reg_text, answer)
				if discard_row:
					print("Completed {} / {}... (discarded)".format(i+1, len(scenarios_data)), end = '\r', flush = True)
					continue
				pred_ls.append([i+1, persona, env, question, answer, docs_info, main_sim, reg_text])
				cnt += 1

				pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Persona', 'Environment', 'Question', 'Answer', 'Documents_Info', 'Similarity', 'Reg_Text'])
				pred_df['Answer'] = pred_df['Answer'].str.replace('\n', '\\n')
				pred_df['Documents_Info'] = pred_df['Documents_Info'].str.replace('\n', '\\n')
				pred_df.to_csv(args.out_dir + "/prog_qa.tsv", sep = '\t', index = None, quoting=1)

		print("Completed {} / {}...".format(i+1, len(scenarios_data)), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Data points generated: ", cnt)


def programmatic_adversarial_generation(questions_data, model, prompt_type, num_iters, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0

	for i in range(len(questions_data)):
		id1 = questions_data.loc[i]["ID"]
		persona = questions_data.loc[i]["Persona"]
		env = questions_data.loc[i]["Environment"]
		ques = questions_data.loc[i]["Question"]
		ans = questions_data.loc[i]["Answer"]
		main_sim = questions_data.loc[i]["Similarity"]
		ans_pts = questions_data.loc[i]["Ans_Points"]
		doc_ans_pts = questions_data.loc[i]["Doc_Ans_Points"]
		reg_text = questions_data.loc[i]["Reg_Text"]

		if str(ans) == "nan":
			continue

		docs_info = questions_data.loc[i]["Documents_Info"]
		
		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("ID: " + str(id1) + "\n")
			f.write("Persona: " + persona + "\n")
			f.write("Environment: " + env + "\n")
			f.write("Question: " + ques + "\n")
			f.write("Answer: " + str(ans) + "\n")
			f.write("Documents Information:\n" + docs_info + "\n")
			f.write("\n\n")

		num_loops = 0
		adv_ques_ls = []
		adv_ans_ls = []
		adv_docs_info_ls = []

		while(num_loops < num_iters):
			prompt, sys_prompt = get_generator_prompt(prompt_type, question=(persona, env, ques, ans, docs_info, adv_ques_ls))

			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

			ip_tokens = len(tik_encoding.encode(prompt))
			op_tokens = len(tik_encoding.encode(og_pred))

			tot_ip_tokens += ip_tokens
			tot_op_tokens += op_tokens

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Adversarial Prediction:\n")
				f.write(og_pred + "\n\n")
				f.write("------------------------------------------------------------------\n\n")

			resp = og_pred.strip().split("\n")
			for j in range(len(resp)):
				if resp[j][:8] == "Question":
					adv_question = resp[j].split(":")[1].strip()
					if adv_question == "":
						adv_question = resp[j+1]
				if resp[j][:6] == "Answer":
					for zj in range(len(resp)):
						if resp[zj][:16] == "Document 1 Title":
							lno = zj
							break
					adv_answer = "\n".join(resp[j:lno]).split("Answer:")[1].strip()
				if resp[j][:10] == "Document 1":
					adv_docs_info = "\n".join(resp[j:])
					break

			adv_ques_ls.append(adv_question)
			adv_ans_ls.append(adv_answer)
			adv_docs_info_ls.append(adv_docs_info)

			num_loops += 1

		pred_ls.append([id1, persona, env, ques, ans, docs_info, adv_ques_ls, adv_ans_ls, adv_docs_info_ls, main_sim, ans_pts, doc_ans_pts, reg_text])
		cnt += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Persona', 'Environment', 'Question', 'Answer', 'Documents_Info', 'Adv_Question', 'Adv_Answer', 'Adv_Documents_Info', 'Similarity', 'Ans_Points', 'Doc_Ans_Points', 'Reg_Text'])
		pred_df['Adv_Question'] = pred_df['Adv_Question'].apply(json.dumps)
		pred_df['Adv_Answer'] = pred_df['Adv_Answer'].apply(json.dumps)
		pred_df['Adv_Documents_Info'] = pred_df['Adv_Documents_Info'].apply(json.dumps)
		pred_df.to_csv(args.out_dir + "/prog_qa.tsv", sep = '\t', index = None, quoting=1)

		print("Completed {} / {}...".format(i+1, len(questions_data)), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Data points generated: ", cnt)

def strip_titles(d_info):
	new_docs_info = ""
	for docline in d_info.split("\n"):
		if "title" not in docline.lower():
			new_docs_info = new_docs_info + docline + "\n"
	return new_docs_info.strip()

def programmatic_doc_generation(questions_data, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0

	for i in range(len(questions_data)):
		persona = questions_data.loc[i]["Persona"]
		env = questions_data.loc[i]["Environment"]

		ques = json.loads(questions_data.loc[i]["Questions"])
		ans = json.loads(questions_data.loc[i]["Answers"])
		docs_info = json.loads(questions_data.loc[i]["Documents_Info"])
		ans_pts = questions_data.loc[i]["Ans_Points"]
		doc_ans_pts = json.loads(questions_data.loc[i]["Doc_Ans_Points"])
		reg_text = questions_data.loc[i]["Reg_Text"]

		if len(ques) != len(docs_info):
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Number of questions and docs info len are not same!\n\n")
				f.write("------------------------------------------------------------------\n\n")
			continue

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Persona: " + persona + "\n")
			f.write("Environment: " + env + "\n\n")

		documents_list = []

		for idx in range(len(ques)):
			cur_ques = ques[idx]
			cur_ans = ans[idx]
			cur_docs_info = strip_titles(docs_info[idx])
			
			adv_ques_ls = ques.copy()
			adv_ques_ls.remove(cur_ques)
			adv_ans_ls = ans.copy()
			adv_ans_ls.remove(cur_ans)

			adv_info = ""
			for jdx in range(len(adv_ques_ls)):
				adv_info = adv_info + "Adversarial Question: " + adv_ques_ls[jdx] + "\nAdversarial Answer:\n" + adv_ans_ls[jdx] + "\n\n"
			adv_info = adv_info.strip()

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Question: " + cur_ques + "\n")
				f.write("Answer: " + str(cur_ans) + "\n")
				f.write("Documents Information:\n" + cur_docs_info + "\n\n")
				f.write("------Adversarial Information------\n")
				f.write(adv_info + "\n\n")
		
			prompt, sys_prompt = get_generator_prompt(prompt_type, question=(persona, env, cur_ques, cur_ans, cur_docs_info, adv_info))

			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

			ip_tokens = len(tik_encoding.encode(prompt))
			op_tokens = len(tik_encoding.encode(og_pred))

			tot_ip_tokens += ip_tokens
			tot_op_tokens += op_tokens

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Input Tokens: " + str(ip_tokens) + "\n")
				f.write("Output Tokens: " + str(op_tokens) + "\n\n")
				f.write("Relevant Documents:\n\n" + og_pred + "\n\n")
				f.write("------------------------------------------------------------------\n\n")

			# Define a regex pattern to match the document header
			pattern = r"(Document \d+:)"

			# Split the documents by the pattern, keeping the pattern as a delimiter
			split_docs = re.split(pattern, og_pred)

			# Remove the first element if it's empty
			if len(split_docs)%2 == 1:
				split_docs = split_docs[1:]

			# Reconstruct the documents by combining the header with the following content
			try:
				doc_list = [split_docs[i] + "\n" + split_docs[i+1].split("Question:")[0].strip() + "\nText:\n"  + split_docs[i+1].split("Text:")[-1].strip() for i in range(0, len(split_docs), 2)]
			except:
				print("Error in splitting docs")
				continue

			if len(doc_list) != len(doc_ans_pts[idx].keys()):
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Number of relevant docs and doc ans pts mismatch!\n\n")
					f.write("------------------------------------------------------------------\n\n")

			documents_list.append(doc_list)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("=======================================================================================\n")
			f.write("=======================================================================================\n\n")

		pred_ls.append([i+1, persona, env, ques, ans, docs_info, ans_pts, doc_ans_pts, documents_list, reg_text])
		cnt += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Persona', 'Environment', 'Questions', 'Answers', 'Documents_Info', 'Ans_Points', 'Doc_Ans_Points', 'Docs_List', 'Reg_Text'])

		pred_df['Questions'] = pred_df['Questions'].apply(json.dumps)
		pred_df['Answers'] = pred_df['Answers'].apply(json.dumps)
		pred_df['Documents_Info'] = pred_df['Documents_Info'].apply(json.dumps)
		pred_df['Docs_List'] = pred_df['Docs_List'].apply(json.dumps)
		pred_df['Doc_Ans_Points'] = pred_df['Doc_Ans_Points'].apply(json.dumps)
		
		pred_df.to_csv(args.out_dir + "/programmatic_data.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(questions_data)), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Data points generated: ", cnt)


def naive_generation(seed_data, model, prompt_type, num_iters, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0

	for i in range(num_iters):
		seed = seed_data.sample(1).reset_index(drop=True)
		seed_ex = "Documents:\n\n" + seed.loc[0]["Documents"] + "\n\n" + "Question: " + seed.loc[0]["Question"] + "\n\nAnswer:\n" + seed.loc[0]["Answer"]

		prompt, sys_prompt = get_generator_prompt(prompt_type, question=(seed_ex))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Complete Output:\n\n" + og_pred + "\n\n")

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		tot_ip_tokens += ip_tokens
		tot_op_tokens += op_tokens

		try:
			docs, ques, ans = process_naive(og_pred)
		except:
			print("Error in processing naive output")
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Error in processing naive output\n\n")
				f.write("--------------------------------------------------------------------------------------------\n\n")
			continue

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Documents:\n\n" + docs + "\n\n")
			f.write("Question: " + ques + "\n\n")
			f.write("Answer:\n\n" + ans + "\n\n")
			f.write("--------------------------------------------------------------------------------------------\n\n")

		pred_ls.append([i+1, docs, ques, ans])
		cnt += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Documents', 'Question', 'Answer'])

		pred_df.to_csv(args.out_dir + "/naive_data.tsv", sep = '\t', index = None)

		print("Total data points generated: ", cnt)

		print("Completed {} / {}...".format(i+1, num_iters), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Data points generated: ", cnt)


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")
	
	_, sys_prompt = get_generator_prompt(args.prompt_type, question=("", "", "", "", "", "", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	if args.exp_type == "programmatic_scenarios":
		programmatic_scenario_generation(model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "programmatic_qa":
		scenarios_data = pd.read_csv(args.out_dir_name + args.scenarios_name + "/scenarios.tsv", sep='\t')
		programmatic_qa_generation(scenarios_data, model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "programmatic_adversarial":
		questions_data = pd.read_csv(args.out_dir_name + args.questions_name + "/prog_qa_modified.tsv", sep='\t')
		programmatic_adversarial_generation(questions_data, model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "programmatic_docs":
		adversarial_data = pd.read_csv(args.out_dir_name + args.adversarial_name + "/prog_qa_modified_verified.tsv", sep='\t')
		programmatic_doc_generation(adversarial_data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "naive_baseline":
		seed_data = pd.read_csv("data/chase_qa.tsv", sep='\t')
		naive_generation(seed_data, model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.exp_type + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)