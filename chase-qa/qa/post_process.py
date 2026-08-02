import argparse
import pandas as pd # type: ignore
import pdb
import json
import re

# line[9] compares a single character against str(doc_no), so it is wrong for
# every document numbered 10 or above. Compare the parsed number instead.
DOC_NUM_RE = re.compile(r'^Document\s+(\d+)')

def build_parser():
	parser = argparse.ArgumentParser(description='Post Process')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, default='gpt-4o-mini-qa', help='Folder name')
	parser.add_argument('-data', type=str, default='prog_qa', help='Data filename')
	parser.add_argument('-exp_type', type=str, default='programmatic_qa', help='Exp type')
	parser.add_argument('-verbose', type=bool, default=False, help='Verbose')
	
	return parser

def programmatic_qa_process(data):
	ls = data.to_dict(orient='records')
	for row in ls:
		if isinstance(row['Answer'], str):
			row['Answer'] = row['Answer'].replace('\\n', '\n')
		if isinstance(row['Documents_Info'], str):
			row['Documents_Info'] = row['Documents_Info'].replace('\\n', '\n')
	exceptions_ls = []
	new_ls = []

	for i in range(len(ls)):
		try:
			answer = ls[i]["Answer"]

			ans_points_og = answer.split("\n")
			ans_points = []
			for ans_pt in ans_points_og:
				ans_pt = ans_pt.strip()
				if ans_pt == "":
					continue
				if ans_pt[0] == "-":
					ans_points.append(ans_pt[1:].strip())
				else:
					ans_points.append(ans_pt)

			ans_points_copy = ans_points.copy()

			docs_info = ls[i]["Documents_Info"]

			doc_ans_points = {1: []}
			doc_no = 1
			cur_doc_label = "1"
			for line in docs_info.split("\n"):
				if len(line) > 2:
					if line[:8] == "Document":
						m = DOC_NUM_RE.match(line)
						if m and m.group(1) != cur_doc_label:
							cur_doc_label = m.group(1)
							doc_no += 1
							doc_ans_points[doc_no] = []
						if "title:" not in line.lower():
							parts = line.strip().split(":", 1)
							if len(parts) > 1 and len(parts[1]) > 2:
								pt_candidate = parts[1].strip()
								if pt_candidate[0] == "-":
									doc_ans_points[doc_no].append(pt_candidate[1:].strip())
									matches = [x for x in ans_points_copy if x.strip() == pt_candidate[1:].strip()]
									if matches: ans_points_copy.remove(matches[0])
								else:
									doc_ans_points[doc_no].append(pt_candidate)
									matches = [x for x in ans_points_copy if x.strip() == pt_candidate.strip()]
									if matches: ans_points_copy.remove(matches[0])
					else:
						stripped_line = line.strip()
						if stripped_line == "":
							continue
						if stripped_line[0] == "-":
							pt = stripped_line[1:].strip()
						else:
							pt = stripped_line
						doc_ans_points[doc_no].append(pt)
						matches = [x for x in ans_points_copy if x.strip() == pt]
						if matches:
							ans_points_copy.remove(matches[0])

			if len(ans_points_copy) > 0.5:
				raise Exception("Some points did not match!")

			ls[i]["Ans_Points"] = ans_points
			ls[i]["Doc_Ans_Points"] = doc_ans_points

			new_ls.append(ls[i])
		except Exception as e:
			if args.verbose:
				print("At index: ", str(i))
				print("Question: ", ls[i]["Question"])
				print("Answer:\n", ls[i]["Answer"])
				print("Documents Info:\n", ls[i]["Documents_Info"])
				print("Exception: ", str(e))
				print("Note: list remove errors occur when there is a mismatch between the answer points and the document answer points. The expectation is that all points in the answer should be assigned (written again) in the document answer points.")
				print()
			exceptions_ls.append(ls[i])
			continue

	if len(new_ls) > 0:
		new_df = pd.DataFrame(new_ls)
		try:
			new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
			new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
		except:
			pass
		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None, quoting=1)

	if len(exceptions_ls) > 0:
		exc_df = pd.DataFrame(exceptions_ls)
		exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep = '\t', index = None)

def programmatic_qa_process_type3(data):
    ls = data.to_dict(orient='records')
    exceptions_ls = []
    new_ls = []

    for i in range(len(ls)):
        try:
            if isinstance(ls[i]["Answer"], str):
                ls[i]["Answer"] = ls[i]["Answer"].replace('\\n', '\n')
            if isinstance(ls[i]["Documents_Info"], str):
                ls[i]["Documents_Info"] = ls[i]["Documents_Info"].replace('\\n', '\n')
            answer = ls[i]["Answer"]

            ans_points_og = answer.split("\n")
            ans_points = []
            for ans_pt in ans_points_og:
                if ans_pt.strip().startswith("-"):
                    ans_points.append(ans_pt[1:].strip())
                elif ans_pt.strip():
                    ans_points.append(ans_pt.strip())

            docs_info = ls[i]["Documents_Info"]

            doc_evidence = {1: []}
            doc_no = 1
            cur_doc_label = "1"
            for line in docs_info.split("\n"):
                if len(line) > 2:
                    if line[:8] == "Document":
                        m = DOC_NUM_RE.match(line)
                        if m and m.group(1) != cur_doc_label:
                            cur_doc_label = m.group(1)
                            doc_no += 1
                            doc_evidence[doc_no] = []
                    else:
                        if line.strip().startswith("-"):
                            doc_evidence[doc_no].append(line.strip()[1:].strip())

            ls[i]["Ans_Points"] = ans_points
            ls[i]["Doc_Ans_Points"] = doc_evidence
            new_ls.append(ls[i])

        except Exception as e:
            if args.verbose:
                print("At index: ", str(i))
                print("Exception: ", str(e))
            exceptions_ls.append(ls[i])
            continue

    if len(new_ls) > 0:
        new_df = pd.DataFrame(new_ls)
        try:
            new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
            new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
        except:
            pass
        new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep='\t', index=None, quoting=1)

    if len(exceptions_ls) > 0:
        exc_df = pd.DataFrame(exceptions_ls)
        exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep='\t', index=None)

def programmatic_adversarial_process(data):
	ls = data.to_dict(orient='records')
	exceptions_ls = []
	partial_drops_ls = []
	new_ls = []

	for i in range(len(ls)):
		try:
			og_answer = json.loads(ls[i]["Adv_Answer"])
			og_question = json.loads(ls[i]["Adv_Question"])
			og_docs_info = json.loads(ls[i]["Adv_Documents_Info"])

			ls_ans_pts = []
			ls_doc_ans_pts = []
			kept_indices = []

			for j in range(len(og_answer)):
				try:
					answer = og_answer[j]
					if not isinstance(answer, str) or answer.strip() == "":
						raise Exception("Empty adversarial answer")

					ans_points_og = answer.split("\n")
					ans_points = []
					for ans_pt in ans_points_og:
						ans_pt = ans_pt.strip()
						if ans_pt == "":
							continue
						if ans_pt[0] == "-":
							ans_points.append(ans_pt[1:].strip())
						else:
							ans_points.append(ans_pt.strip())

					if len(ans_points) == 0:
						raise Exception("No answer points parsed")

					ans_points_copy = ans_points.copy()

					docs_info = og_docs_info[j]
					if not isinstance(docs_info, str) or docs_info.strip() == "":
						raise Exception("Empty adversarial documents info")

					doc_ans_points = {1: []}
					doc_no = 1
					cur_doc_label = "1"
					for line in docs_info.split("\n"):
						if len(line) > 2:
							if line[:8] == "Document":
								m = DOC_NUM_RE.match(line)
								if m and m.group(1) != cur_doc_label:
									cur_doc_label = m.group(1)
									doc_no += 1
									doc_ans_points[doc_no] = []
								if "title:" not in line.lower():
									parts = line.strip().split(":", 1)
									if len(parts) > 1 and len(parts[1]) > 2:
										pt_candidate = parts[1].strip()
										if pt_candidate[0] == "-":
											doc_ans_points[doc_no].append(pt_candidate[1:].strip())
											matches = [x for x in ans_points_copy if x.strip() == pt_candidate[1:].strip()]
											if matches: ans_points_copy.remove(matches[0])
										else:
											doc_ans_points[doc_no].append(pt_candidate)
											matches = [x for x in ans_points_copy if x.strip() == pt_candidate.strip()]
											if matches: ans_points_copy.remove(matches[0])
							else:
								stripped_line = line.strip()
								if stripped_line == "":
									continue
								if stripped_line[0] == "-":
									pt = stripped_line[1:].strip()
									doc_ans_points[doc_no].append(pt)
									matches = [x for x in ans_points_copy if x.strip() == pt.strip()]
									if matches:
										ans_points_copy.remove(matches[0])
								else:
									pt = stripped_line
									doc_ans_points[doc_no].append(pt)
									matches = [x for x in ans_points_copy if x.strip() == pt.strip()]
									if matches:
										ans_points_copy.remove(matches[0])

					if len(ans_points_copy) > 0.5:
						raise Exception("Some points did not match!")

					ls_ans_pts.append(ans_points)
					ls_doc_ans_pts.append(doc_ans_points)
					kept_indices.append(j)

				except Exception as inner_e:
					if args.verbose:
						print("Row", i, "adversarial question", j, "dropped:", str(inner_e))
					partial_drops_ls.append({
						"Row_Index": i,
						"Adv_Question_Index": j,
						"Adv_Question": og_question[j] if j < len(og_question) else None,
						"Reason": str(inner_e),
					})
					continue

			if len(kept_indices) == 0:
				raise Exception("All adversarial questions failed to match, dropping row")

			ls[i]["Adv_Question"] = json.dumps([og_question[j] for j in kept_indices])
			ls[i]["Adv_Answer"] = json.dumps([og_answer[j] for j in kept_indices])
			ls[i]["Adv_Documents_Info"] = json.dumps([og_docs_info[j] for j in kept_indices])
			ls[i]["Adv_Ans_Points"] = ls_ans_pts
			ls[i]["Adv_Doc_Ans_Points"] = ls_doc_ans_pts

			new_ls.append(ls[i])
		except Exception as e:
			if args.verbose:
				print("At index: ", str(i))
				print("Adv Question: ", ls[i]["Adv_Question"])
				print("Adv Answer:\n", ls[i]["Adv_Answer"])
				print("Adv Documents Info:\n", ls[i]["Adv_Documents_Info"])
				print("Exception: ", str(e))
				print()
			exceptions_ls.append(ls[i])
			continue

	if len(new_ls) > 0:
		final_ls = []

		for i in range(len(new_ls)):
			id1 = new_ls[i]["ID"]
			persona = new_ls[i]["Persona"]
			env = new_ls[i]["Environment"]
			sim = new_ls[i]["Similarity"]
			reg_text = new_ls[i]["Reg_Text"]
			adv_questions = json.loads(new_ls[i]["Adv_Question"])
			adv_answers = json.loads(new_ls[i]["Adv_Answer"])
			adv_docs_info = json.loads(new_ls[i]["Adv_Documents_Info"])
			adv_ans_pts = new_ls[i]["Adv_Ans_Points"]
			adv_doc_ans_pts = new_ls[i]["Adv_Doc_Ans_Points"]

			questions_ls = [new_ls[i]["Question"]]
			answers_ls = [new_ls[i]["Answer"]]
			docs_info_ls = [new_ls[i]["Documents_Info"]]
			ans_pts_ls = [json.loads(new_ls[i]["Ans_Points"])]
			doc_ans_pts_ls = [json.loads(new_ls[i]["Doc_Ans_Points"])]
			numerics = new_ls[i]["Numerics"]
			seed_type = new_ls[i]["Seed_Type"]

			for j in range(len(adv_questions)):
				questions_ls.append(adv_questions[j])
				answers_ls.append(adv_answers[j])
				docs_info_ls.append(adv_docs_info[j])
				ans_pts_ls.append(adv_ans_pts[j])
				doc_ans_pts_ls.append(adv_doc_ans_pts[j])

			final_ls.append([id1, persona, env, sim, reg_text, numerics, seed_type, questions_ls, answers_ls, docs_info_ls, ans_pts_ls, doc_ans_pts_ls])

		new_df = pd.DataFrame(final_ls, columns = ['ID', 'Persona', 'Environment', 'Similarity', 'Reg_Text', 'Numerics', 'Seed_Type', 'Questions', 'Answers', 'Documents_Info', 'Ans_Points', 'Doc_Ans_Points'])
		new_df['Questions'] = new_df['Questions'].apply(json.dumps)
		new_df['Answers'] = new_df['Answers'].apply(json.dumps)
		new_df['Documents_Info'] = new_df['Documents_Info'].apply(json.dumps)
		new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
		new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None, quoting=1)
		print("Length of Final Data: ", str(len(new_df)))

	if len(partial_drops_ls) > 0:
		pd.DataFrame(partial_drops_ls).to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_partial_drops.tsv", sep = '\t', index = None)
		print("Individually dropped adversarial questions (row survived): ", len(partial_drops_ls))

	if len(exceptions_ls) > 0:
		exc_df = pd.DataFrame(exceptions_ls)
		exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep = '\t', index = None)

def strip_conclusion_from_docs_info(d_info):
	new_lines = []
	for line in d_info.split("\n"):
		stripped = line.strip()
		if stripped.lower().startswith("conclusion:") or stripped.lower().startswith("- conclusion:"):
			continue
		idx = line.lower().find("assigned:")
		if idx != -1:
			trailing = line[idx + len("assigned:"):].strip()
			if trailing.lower().startswith("conclusion:"):
				line = line[:idx + len("assigned:")]
		new_lines.append(line)
	return "\n".join(new_lines)

def programmatic_adversarial_process_type3(data):
	ls = data.to_dict(orient='records')
	exceptions_ls = []
	partial_drops_ls = []
	new_ls = []

	for i in range(len(ls)):
		try:
			ls_ans_pts = []
			ls_doc_ans_pts = []
			kept_indices = []
			og_answer = json.loads(ls[i]["Adv_Answer"])
			og_question = json.loads(ls[i]["Adv_Question"])
			og_docs_info = json.loads(ls[i]["Adv_Documents_Info"])

			for j in range(len(og_answer)):
				try:
					answer = og_answer[j]
					if not isinstance(answer, str) or answer.strip() == "":
						raise Exception("Empty adversarial answer")

					ans_points_og = answer.split("\n")
					ans_points = []
					for ans_pt in ans_points_og:
						ans_pt = ans_pt.strip()
						if ans_pt == "":
							continue
						if ans_pt[0] == "-":
							ans_points.append(ans_pt[1:].strip())
						else:
							ans_points.append(ans_pt)

					if len(ans_points) == 0:
						raise Exception("No answer points parsed")

					ans_points_copy = [x for x in ans_points.copy() if not x.strip().lower().startswith("conclusion:")]

					if args.verbose:
						print(f"  [debug] index={i}, j={j}, ans_points={ans_points}")

					docs_info = og_docs_info[j]
					if not isinstance(docs_info, str) or docs_info.strip() == "":
						raise Exception("Empty adversarial documents info")

					doc_ans_points = {1: []}
					doc_no = 1
					cur_doc_label = "1"
					for line in docs_info.split("\n"):
						if len(line) > 2:
							if line[:8] == "Document":
								m = DOC_NUM_RE.match(line)
								if m and m.group(1) != cur_doc_label:
									cur_doc_label = m.group(1)
									doc_no += 1
									doc_ans_points[doc_no] = []
								if "title:" not in line.lower():
									parts = line.strip().split(":", 1)
									if len(parts) > 1 and len(parts[1]) > 2:
										pt_candidate = parts[1].strip()
										if pt_candidate[0] == "-":
											doc_ans_points[doc_no].append(pt_candidate[1:].strip())
											matches = [x for x in ans_points_copy if x.strip() == pt_candidate[1:].strip()]
											if matches: ans_points_copy.remove(matches[0])
										else:
											doc_ans_points[doc_no].append(pt_candidate)
											matches = [x for x in ans_points_copy if x.strip() == pt_candidate.strip()]
											if matches: ans_points_copy.remove(matches[0])
							else:
								stripped_line = line.strip()
								if stripped_line == "":
									continue
								if stripped_line[0] == "-":
									pt = stripped_line[1:].strip()
								else:
									pt = stripped_line
								doc_ans_points[doc_no].append(pt)
								matches = [x for x in ans_points_copy if x.strip() == pt]
								if matches: ans_points_copy.remove(matches[0])

					for dn in doc_ans_points:
						doc_ans_points[dn] = [pt for pt in doc_ans_points[dn] if not pt.strip().lower().startswith("conclusion:")]

					if len(ans_points_copy) > 0.5:
						if args.verbose:
							print(f"  [debug] index={i}, j={j} FAILED -- unmatched points: {ans_points_copy}")
						raise Exception("Some points did not match!")

					ls_ans_pts.append(ans_points)
					ls_doc_ans_pts.append(doc_ans_points)
					kept_indices.append(j)

				except Exception as inner_e:
					if args.verbose:
						print("Row", i, "adversarial question", j, "dropped:", str(inner_e))
					partial_drops_ls.append({
						"Row_Index": i,
						"Adv_Question_Index": j,
						"Adv_Question": og_question[j] if j < len(og_question) else None,
						"Reason": str(inner_e),
					})
					continue

			if len(kept_indices) == 0:
				raise Exception("All adversarial questions failed to match, dropping row")

			ls[i]["Adv_Question"] = json.dumps([og_question[j] for j in kept_indices])
			ls[i]["Adv_Answer"] = json.dumps([og_answer[j] for j in kept_indices])
			ls[i]["Adv_Documents_Info"] = json.dumps([og_docs_info[j] for j in kept_indices])
			ls[i]["Adv_Ans_Points"] = ls_ans_pts
			ls[i]["Adv_Doc_Ans_Points"] = ls_doc_ans_pts

			new_ls.append(ls[i])
		except Exception as e:
			if args.verbose:
				print("At index: ", str(i))
				print("Adv Question: ", ls[i].get("Adv_Question"))
				print("Adv Answer:\n", ls[i].get("Adv_Answer"))
				print("Adv Documents Info:\n", ls[i].get("Adv_Documents_Info"))
				print("Exception: ", str(e))
				print()
			exceptions_ls.append(ls[i])
			continue

	if len(new_ls) > 0:
		final_ls = []

		for i in range(len(new_ls)):
			id1 = new_ls[i]["ID"]
			persona = new_ls[i]["Persona"]
			env = new_ls[i]["Environment"]
			sim = new_ls[i]["Similarity"]
			reg_text = new_ls[i]["Reg_Text"]
			adv_questions = json.loads(new_ls[i]["Adv_Question"])
			adv_answers = json.loads(new_ls[i]["Adv_Answer"])
			adv_docs_info = json.loads(new_ls[i]["Adv_Documents_Info"])
			adv_ans_pts = new_ls[i]["Adv_Ans_Points"]
			adv_doc_ans_pts = new_ls[i]["Adv_Doc_Ans_Points"]

			questions_ls = [new_ls[i]["Question"]]
			answers_ls = [new_ls[i]["Answer"]]
			docs_info_ls = [strip_conclusion_from_docs_info(new_ls[i]["Documents_Info"])]
			ans_pts_ls = [json.loads(new_ls[i]["Ans_Points"])]
			doc_ans_pts_ls = [json.loads(new_ls[i]["Doc_Ans_Points"])]
			numerics = new_ls[i]["Numerics"]
			seed_type = new_ls[i]["Seed_Type"]

			for j in range(len(adv_questions)):
				questions_ls.append(adv_questions[j])
				answers_ls.append(adv_answers[j])
				docs_info_ls.append(strip_conclusion_from_docs_info(adv_docs_info[j]))
				ans_pts_ls.append(adv_ans_pts[j])
				doc_ans_pts_ls.append(adv_doc_ans_pts[j])

			final_ls.append([id1, persona, env, sim, reg_text, numerics, seed_type, questions_ls, answers_ls, docs_info_ls, ans_pts_ls, doc_ans_pts_ls])

		new_df = pd.DataFrame(final_ls, columns = ['ID', 'Persona', 'Environment', 'Similarity', 'Reg_Text', 'Numerics', 'Seed_Type', 'Questions', 'Answers', 'Documents_Info', 'Ans_Points', 'Doc_Ans_Points'])
		new_df['Questions'] = new_df['Questions'].apply(json.dumps)
		new_df['Answers'] = new_df['Answers'].apply(json.dumps)
		new_df['Documents_Info'] = new_df['Documents_Info'].apply(json.dumps)
		new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
		new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None, quoting=1)
		print("Length of Final Data: ", str(len(new_df)))

	if len(partial_drops_ls) > 0:
		pd.DataFrame(partial_drops_ls).to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_partial_drops.tsv", sep = '\t', index = None)
		print("Individually dropped adversarial questions (row survived): ", len(partial_drops_ls))

	if len(exceptions_ls) > 0:
		exc_df = pd.DataFrame(exceptions_ls)
		exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep = '\t', index = None)

def programmatic_docs_process(data):
	ls = data.to_dict(orient='records')
	new_ls = []
	skipped_rows = []
	
	tot_cnt = 1

	for i in range(len(ls)):
		id1 = ls[i]["ID"]
		persona = ls[i]["Persona"]
		env = ls[i]["Environment"]
		questions = json.loads(ls[i]["Questions"])
		answers = json.loads(ls[i]["Answers"])
		docs_info = json.loads(ls[i]["Documents_Info"])
		ans_pts = json.loads(ls[i]["Ans_Points"])
		doc_ans_pts = json.loads(ls[i]["Doc_Ans_Points"])
		docs_list = json.loads(ls[i]["Docs_List"])
		reg_text = ls[i]["Reg_Text"]
		numerics = ls[i]["Numerics"]
		seed_type = ls[i]["Seed_Type"]

		# Stage 7 does `continue` on a doc-split failure, so Docs_List can be
		# shorter than Questions. Indexing past the end kills the whole shard.
		# json.loads can return a str when a field is double-encoded, and len()
		# of a str is its character count, so type is checked before length.
		fields = {"Questions": questions, "Answers": answers, "Docs_List": docs_list,
		          "Ans_Points": ans_pts, "Doc_Ans_Points": doc_ans_pts}
		bad_type = [k for k, v in fields.items() if not isinstance(v, list)]
		if bad_type:
			print("SKIP row index {} (ID {}): not a list after decode: {}".format(i, id1, bad_type))
			skipped_rows.append({"Row_Index": i, "ID": id1, "Reason": "non_list:" + ",".join(bad_type)})
			continue
		n_q = len(questions)
		short = {k: len(v) for k, v in fields.items() if len(v) < n_q}
		if short:
			print("SKIP row index {} (ID {}): shorter than Questions ({}): {}".format(i, id1, n_q, short))
			rec = {"Row_Index": i, "ID": id1, "Reason": "short_field", "N_Questions": n_q}
			rec.update(short)
			skipped_rows.append(rec)
			continue

		modified_docs_list = []
		for doc_ls in docs_list:
			new_doc_ls = []
			for doc in doc_ls:
				if "Title:" in doc:
					mod_doc = "Title: " + doc.split("Title:")[1].strip()
				else:
					# mod_doc has function scope. Without this branch a document
					# lacking "Title:" re-emits the previous document and loses
					# its own, silently, or raises NameError on the first one.
					mod_doc = doc
				mod_doc = mod_doc.split("In conclusion,")[0].strip()
				mod_doc = mod_doc.split("In summary,")[0].strip()
				mod_doc = mod_doc.split("To summarize")[0].strip()
				new_doc_ls.append(mod_doc)
			modified_docs_list.append(new_doc_ls)

		for j in range(len(questions)):
			cur_ques = questions[j]
			cur_ans = answers[j]
			cur_ans_pts = ans_pts[j]
			cur_doc_ans_pts = doc_ans_pts[j]
			cur_docs = modified_docs_list[j]

			# list.remove() deletes the first *equal* element, not this one. Two
			# identical siblings, which nothing prevents, would leave Adv_*
			# describing the wrong sibling. Index-based is exact.
			others = [k for k in range(len(questions)) if k != j]
			adv_ques = [questions[k] for k in others]
			adv_ans = [answers[k] for k in others]
			adv_ans_pts = [ans_pts[k] for k in others]
			adv_doc_ans_pts = [doc_ans_pts[k] for k in others]
			adv_docs_list = [modified_docs_list[k] for k in others]

			new_ls.append([id1, tot_cnt, persona, env, cur_ques, cur_ans, cur_ans_pts, cur_doc_ans_pts, cur_docs, adv_ques, adv_ans, adv_ans_pts, adv_doc_ans_pts, adv_docs_list, reg_text, numerics, seed_type])
			
			tot_cnt += 1
	
	if skipped_rows:
		print("Rows skipped on length mismatch: ", len(skipped_rows))
		pd.DataFrame(skipped_rows).to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_skipped.tsv", sep = '\t', index = None)

	new_df = pd.DataFrame(new_ls, columns = ['Root_ID', 'Question_No', 'Persona', 'Environment', 'Question', 'Answer', 'Ans_Points', 'Doc_Ans_Points', 'Rel_Docs_List', 'Adv_Question', 'Adv_Answer', 'Adv_Ans_Pts', 'Adv_Doc_Ans_Pts', 'Adv_Docs_List', 'Reg_Text', 'Numerics', 'Seed_Type'])
	new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
	new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
	new_df['Rel_Docs_List'] = new_df['Rel_Docs_List'].apply(json.dumps)
	new_df['Adv_Question'] = new_df['Adv_Question'].apply(json.dumps)
	new_df['Adv_Answer'] = new_df['Adv_Answer'].apply(json.dumps)
	new_df['Adv_Ans_Pts'] = new_df['Adv_Ans_Pts'].apply(json.dumps)
	new_df['Adv_Doc_Ans_Pts'] = new_df['Adv_Doc_Ans_Pts'].apply(json.dumps)
	new_df['Adv_Docs_List'] = new_df['Adv_Docs_List'].apply(json.dumps)
	new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None, quoting=1)
	print("Length of Final Data: ", str(len(new_df)))

	
def main(args):
	if args.exp_type == "programmatic_qa":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_qa_process(data)
	elif args.exp_type == "programmatic_adversarial":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_adversarial_process(data)
	elif args.exp_type == "programmatic_docs":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_docs_process(data)
	elif args.exp_type == "programmatic_qa_type3":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_qa_process_type3(data)
	elif args.exp_type == "programmatic_adversarial_type3":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_adversarial_process_type3(data)
	
if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	main(args)