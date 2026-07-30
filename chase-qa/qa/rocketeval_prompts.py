"""Prompt templates for the RocketEval + PoLL question-quality validation pass.

Convention matches prompts.py: each getter returns (prompt, sys_prompt).

Direction of evidence:
	generation  ->  criteria are written from the Answer + relevant documents
	grading     ->  judges score the Question against those criteria

The Answer is withheld from the judges. They see only the Question, the documents, and the criteria, so they assess the question itself rather than being anchored by a finished answer sitting next to it.

Answer correctness is NOT established here. It is assumed, having been checked upstream by the presence/extra/pred verification and the pre-generation grounding checks. Criteria inherit any error in the Answer, so a question aligned to a wrong answer scores clean. This pass validates question quality and question-answer alignment only.

Prose in the templates is deliberately left unwrapped: any linebreak here is sent verbatim to the model.
"""

CHECKLIST_SYS = "You are a benchmark quality assessor for financial advisory question-answering data. You write precise, binary, self-contained evaluation criteria. You never speculate beyond the material supplied to you."

GRADING_SYS = "You are an expert evaluator of financial advisory benchmark questions. You answer each criterion strictly with Yes, No, or Unsure. You never explain, justify, or add commentary."

_CHECKLIST = """# Task

You are given an answer and the documents that answer was built from. You are NOT given the question. A question exists that this answer is supposed to answer.

Write binary criteria that will be used to judge whether that question is a good benchmark question. The criteria will be applied to the question alone, without the answer visible, so each criterion must carry enough detail to be checked on its own.

# Documents

<|begin_of_documents|>

{documents}

<|end_of_documents|>

# Answer

<|begin_of_answer|>

{answer}

<|end_of_answer|>

# What to cover

Write {count} criteria in total, covering these concerns:

- Alignment. Does the question actually ask for the specific content this answer provides? Name the specific figures, rules, entities, or effects that the question must be asking about for this answer to be the correct response to it.
- Scope. Is the question specific enough that this answer is required, rather than a much shorter or more general reply being sufficient? Is it narrow enough that the answer does not contain material the question never asked for?
- Answerability. Can the question be answered from the supplied documents alone, without outside knowledge?
- Ambiguity. Does the question admit a single defensible reading?
- Leakage. Does the question avoid stating the content of its own answer?
- Reasoning demand. {type_line}

# Requirements

Each criterion must be answerable Yes or No alone, and phrased so that Yes means the question is good. Embed the specific figure, rule reference, term, or entity being checked directly in the criterion text. Write "Does the question ask specifically about the money purchase annual allowance trigger on first flexible access?" rather than "Is the question relevant?".

Do not write vague, duplicated, or overlapping criteria. Do not introduce facts that appear in neither the answer nor the documents.

# Output Format

Output only the criteria, one per line, numbered, nothing else. No preamble, no headings, no trailing commentary.

```
1. {{criterion}}
2. {{criterion}}
...
```"""

_TYPE_LINE = {
	"type1": "Does the question require retrieving a specific stated fact or figure, rather than being answerable from general financial knowledge without consulting any document?",
	"type2": "Does the question require combining information that originates in more than one document, rather than being fully answerable from any single document on its own?",
	"type3": "Does the question require inferring a causal link, sequence, or trend that is not stated verbatim in any document, rather than being answerable by retrieving stated facts? Name the specific causal relationship the question must be asking about.",
}

_COUNT = {"type1": "6 to 8", "type2": "7 to 9", "type3": "7 to 10"}


def get_checklist_prompt(qa_type, params):
	"""qa_type in {type1, type2, type3}. params = (answer, documents)."""
	qa_type = qa_type.lower()
	if qa_type not in _TYPE_LINE:
		raise ValueError("Unknown qa_type: {}".format(qa_type))
	answer, documents = params
	prompt = _CHECKLIST.format(
		documents=documents,
		answer=answer,
		count=_COUNT[qa_type],
		type_line=_TYPE_LINE[qa_type],
	)
	return prompt, CHECKLIST_SYS


_GRADING = """# Task

Judge a benchmark question against a list of criteria.

# Documents

<|begin_of_documents|>

{documents}

<|end_of_documents|>

# Question

<|begin_of_question|>

{question}

<|end_of_question|>

# Criteria

<|begin_of_criteria|>

{criteria}

<|end_of_criteria|>

# Instructions

Answer every criterion, judging the Question above. Judge each criterion independently of the others.

Answer Yes only if the criterion is clearly satisfied by the question as written. Answer No if it is clearly not satisfied. Answer Unsure only if the question and documents genuinely do not settle it.

Judge the question exactly as written. Do not credit a question for what it could have asked, and do not penalise it for anything a criterion does not raise.

# Output Format

Output one line per criterion, numbered to match the criteria above, nothing else. No preamble, no explanation, no trailing commentary.

```
1. Yes
2. No
3. Unsure
...
```"""


def get_grading_prompt(params):
	"""params = (question, documents, criteria_block)."""
	question, documents, criteria_block = params
	prompt = _GRADING.format(
		documents=documents,
		question=question,
		criteria=criteria_block,
	)
	return prompt, GRADING_SYS