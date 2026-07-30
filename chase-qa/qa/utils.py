import random
import json

def sample_scenarios(ls, num_samples=5, reg_index=None):
	"""Exemplars for the stage 1 expansion prompt.

	Emits REGULATORY_AREA_ID because the prompt's format block requires it and
	the parser discards any scenario that lacks it. A two-line exemplar is what
	teaches the model to omit the field, so a row whose area id cannot be
	resolved is skipped rather than shown without it.

	dynamic_reg rows do not store their area id, so it is recovered from
	Reg_Text via reg_index. That is exact: Reg_Text was assigned verbatim from
	reg_clauses[area_id].
	"""
	if reg_index is None:
		with open("reg_clauses.json", "r") as f:
			_rc = json.load(f)
		reg_index = {}
		for _k, _v in _rc.items():
			_v = str(_v).strip()
			if _v and _v not in reg_index:
				reg_index[_v] = _k
	pool = list(ls)
	random.shuffle(pool)
	out = []
	for row in pool:
		seed_type = row[6] if len(row) > 6 else ""
		if seed_type in ("generic", "dynamic_generic"):
			area = "GENERIC"
		else:
			clause = str(row[4]).strip() if len(row) > 4 else ""
			area = reg_index.get(clause) if clause else None
			if area is None:
				continue
		out.append("USER_PERSONA: " + str(row[1])
		           + "\nCOLLECTION_OF_DOCS: " + str(row[2])
		           + "\nREGULATORY_AREA_ID: " + area)
		if len(out) == num_samples:
			break
	if len(out) < num_samples:
		print("WARNING: only {} of {} exemplars had a resolvable "
		      "REGULATORY_AREA_ID".format(len(out), num_samples))
	return "\n\n".join(out)

def process_naive(output):
	output = output.replace("**", "")
	output = output.replace("##", "")

	lines = output.split("\n")

	for line_no in range(len(lines)):
		if lines[line_no].strip().startswith("Documents:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Question:"):
					end_line = temp_no
					break
			docs = "\n".join(lines[line_no:end_line]).strip().split("Documents:")[1].strip()
			if docs[0] == "<" and docs[-1] == ">":
				docs = docs[1:-1]
		elif lines[line_no].strip().startswith("Question:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Answer:"):
					end_line = temp_no
					break
			ques = "\n".join(lines[line_no:end_line]).strip().split("Question:")[1].strip()
			if ques[0] == "<" and ques[-1] == ">":
				ques = ques[1:-1]
		elif lines[line_no].strip().startswith("Answer:"):
			ans = "\n".join(lines[line_no:len(lines)]).strip().split("Answer:")[1].strip()
			if ans[0] == "<" and ans[-1] == ">":
				ans = ans[1:-1]

	return docs, ques, ans