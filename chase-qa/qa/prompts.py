GENERIC_NUMERIC_INSTRUCTION = """

This scenario requires you to invent realistic, plausible figures appropriate to the environment described (e.g. contribution amounts, interest rates, fund values, mortgage balances, ages, terms) and perform a genuine multi-step calculation using them. The calculation must involve at least 3 dependent steps, where later steps use the numeric result of earlier steps; do not use a single-step or plug-and-chug calculation. Before the Question and Answer, output a Calculation section showing your full working. Each step must be on its own line in the exact format below, using plain decimal numbers only (no currency symbols, percent signs, or thousands separators), where later steps must literally re-state the numeric value produced by an earlier step wherever that earlier result is used:

Calculation:
Step 1: <short description> = <arithmetic expression using literal numbers only> = <resulting value>
Step 2: <short description> = <arithmetic expression using literal numbers only, re-stating any prior step's value literally> = <resulting value>
Step 3: <short description> = <arithmetic expression using literal numbers only, re-stating any prior step's value literally> = <resulting value>
(add further steps if needed)

The Question and Answer you generate afterwards must be grounded in exactly these calculated figures. The Answer's bullet points must state the specific numeric results from the Calculation section (rounded naturally as a human adviser would state them), not the raw calculation steps themselves. The calculation must be non-trivial: it must require genuine multi-step reasoning to answer, not something solvable by reading a single number off a table."""

def get_verification_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "presence":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and an answer point. You must check two things:
1. Presence: Is the point mentioned in the document?
2. Relevance: Is the point discussed in a manner such that it can be used to partially answer the question?

Document:
{params[2]}

Question: {params[0]}

Answer Point:
{params[1]}

Give output in the following format:
Presence: True/False
Explanation for Presence:

Relevance: True/False
Explanation for Relevance:
"""

	elif prompt_type == "predict":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given a document followed by a question. You need to generate the answer for that question based on only the information provided in the document. Provide the answer in 1-3 bullet points, so that it is easier to objectively evaluate. Note that the document might not provide all the necessary information for answering the question, but you must just form the partial answer based on whatever information is given. Generate only the answer and nothing else. There is no need to mention what information is not provided in the document, just write the answer as best as you can. If the document does not provide any information relevant to the question, and there is no way you can attempt to generate even a partial answer, then you must only generate "No relevant information found in this document." and nothing else.

Document:
{params[1]}

Question: {params[0]}

Answer:"""

	elif prompt_type == "compare":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. You need to evaluate whether the prediction contains any major additional points that are not mentioned in the ground-truth answer. You can discard any points in the prediction that mention lack of information in the document about specific aspects. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of any new broad ideas or main points in the prediction that are not at all discussed in the ground-truth answer. The order of points mentioned is irrelevant. You should ignore the points in the prediction that just elaborate or provide more specifics or details over a major point in the ground-truth answer. You should also ignore if the ground-truth answer provides additional points that are not mentioned in the prediction.

If the prediction is not introducing major new points compared to the ground-truth answer, output "False" to "Presence of Extra Points" without giving any explanation. Otherwise, if and only if the prediction discusses major additional points that are not in the ground-truth answer, output "True" and mention only the extra major points part exactly as they are written in the prediction.

Question: {params[0]}

Ground-truth Answer: {params[1]}

Prediction: {params[2]}

Give output in the following format:
Presence of Extra Points: True/False
Extra Points Mentioned (if any):
"""

	elif prompt_type == "extra":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and some answer points. You must check whether there are any additional major points in the document that provide relevant information for answering the question that are currently missing from the answer. Follow these instructions:
1. Do not look for exact phrases or explicit mentions since the answer can have points that are a paraphrase of the same broad information. 
2. It is ok if the document provides more specifics or details about the points already in the answer or if it discusses them in more depth by introducing related information so you can ignore that. 
3. Check if the document introduces a new "major" idea or point that is crucial for answering the question and is not at all mentioned in the answer and is not an extension of the existing points in the answer.
4. Your job is not to check if the question can be sufficiently answered. You should ignore if the document or answer points are missing any points that are needed in the answer to the question.

If the document is not introducing major new points pertaining to the answer, output "False" to "Presence of Extra Points" without giving any explanation. Otherwise, if and only if the document discusses major additional points that are necessary to answer the question, output "True" and mention only the extra major points discussed.

Document:
{params[2]}

Question: {params[0]}

Answer Points:
{params[1]}

Give output in the following format:
Presence of Extra Points: True/False
Extra Points Mentioned (if any):
"""

	elif prompt_type == "extra_adv":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question. You must check whether there is any information in the document that is relevant for answering the question. If not, output "False" to "Presence of Relevant Information" without giving any explanation. Otherwise, if and only if the document discusses information that is necessary to answer the question, output "True" and mention only that information, summarized as bullet points.

{params[1]}

Question: {params[0]}

Give output in the following format:
Presence of Relevant Information: True/False
Relevant Points (if any):
"""
		
	elif prompt_type == "presence_type3":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and an evidence component. You must check two things:
1. Presence: Is the evidence component mentioned in the document?
2. Absence of Conclusion: Does the document avoid explicitly stating the causal conclusion to the question, and instead only provides supporting evidence?

Document:
{params[2]}

Question: {params[0]}

Evidence Component:
{params[1]}

Give output in the following format:
Presence: True/False
Explanation for Presence:

Absence of Conclusion: True/False
Explanation for Absence:
"""

	elif prompt_type == "extra_type3":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and some evidence components. You must check two things:
1. Whether there are additional major evidence components in the document relevant to the question that are missing from the list.
2. Whether the document explicitly states the causal conclusion to the question rather than just providing evidence.

If neither is true, output "False" to "Presence of Extra Points". Otherwise output "True" and describe what was found.

Document:
{params[2]}

Question: {params[0]}

Evidence Components:
{params[1]}

Give output in the following format:
Presence of Extra Points: True/False
Extra Points Mentioned (if any):
"""

	elif prompt_type == "remove":
		sys_prompt = "You are an expert at correcting data."
		prompt = f"""You are given a document followed by a question. Then you are provided with some information to be removed. Your job is to re-write the document exactly as it is, only removing those lines that refer to the information to be removed.

{params[3]}

Question: {params[0]}

Information to be removed:
{params[2]}"""

		if params[1] is not None:
			prompt = prompt + f"""

Note that the following points are crucial for answering the question and hence make sure that they must still be conveyed in the document:
{params[1]}"""
		
		prompt = prompt + f"""

Give output in the following format:
Title: <Title>
Text:
<Text>
"""

	elif prompt_type == "add":
		sys_prompt = "You are an expert at correcting data."
		prompt = f"""You are given a document followed by a question. Then you are provided with some information relevant for answering the question that must be added to the document. Your job is to re-write the document exactly as it is and placing this new relevant information somewhere inside. Do not elaborate or provide extra details or specifics surrounding this new information. Just directly mention it in a suitable way in context of the question.

{params[2]}

Question: {params[0]}

Information to be added:
{params[1]}

Give output in the following format:
Title: <Title>
Text:
<Text>
"""

	elif prompt_type == "adv_cross_check":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a question and an answer. You must check whether the answer is even partially relevant for answering the question. If the answer is not relevant at all, output "False" to "Relevance". Otherwise, if and only if the answer discusses information that is at least partially necessary to answer the question, output "True".

Question: {params[0]}

Answer:
{params[1]}

Give output in the following format:
Relevance: True/False
"""
	elif prompt_type == "reg_pregen_grounding":
		sys_prompt = "You are an expert at verifying regulatory accuracy."
		prompt = f"""You are given a question, an answer written as bullet points, and the real current regulatory source text that the answer's facts must be consistent with. You must check whether every regulatory fact, figure, rule, or rate stated in the answer matches the real regulatory source text. The answer may contain facts not covered by the source text at all; ignore those, you are only checking for contradiction with what the source text does state, not for completeness.

Only flag statements that assert a specific regulatory fact, figure, rule, or rate. Do not flag narrative, behavioural, or causal claims about outcomes, trends, client behaviour, or adviser behaviour, since these cannot be verified as true or false against regulatory source text, only genuine regulatory facts can.

If every regulatory fact in the answer that is covered by the source text is consistent with it, output "True" to "Grounded" without giving any explanation. Otherwise, if and only if the answer states a regulatory fact, figure, rule, or rate that contradicts the source text, output "False", quote the exact bullet point from the answer that is wrong exactly as written, and provide a corrected version of that same bullet point consistent with the source text, keeping the same approximate length and style.

Question: {params[0]}

Regulatory Source Text:
{params[1]}

Answer:
{params[2]}

Give output in the following format:
Grounded: True/False
Flagged Point (if False): <exact bullet point as written in the answer>
Corrected Point (if False): <corrected bullet point>
"""

	return prompt, sys_prompt



def get_evaluator_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "zero-shot-basic":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, an adversarial wrong answer, the ground-truth answer, and a prediction. You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. All the main points in the ground-truth answer must be mentioned in the prediction. The order of points mentioned is irrelevant. It is allowed for the prediction to elaborate or provide more specifics or details over the major points in the ground-truth answer. However, the prediction should not contain additional major points that are contradictory or irrelevant for answering the question. Importantly, the prediction must not discuss any of the points mentioned in the "adversarial wrong answer". The first word in your response must be either True or False. If False, explain why you think the prediction is wrong in detail.

Question: {question[0]}

Adversarial Wrong Answers: {question[3]}

Ground-truth Answer: {question[1]}

Prediction: {question[2]}

Result:
"""
	
	elif prompt_type == "zero-shot-step-by-step":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. 

Question: {question[0]}

Ground-truth Answer:
{question[1]}

Prediction:
{question[2]}

You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. All the main points in the ground-truth answer must be mentioned in the prediction. The order of points mentioned is irrelevant. It is allowed for the prediction to elaborate or provide more specifics or details over the major points in the ground-truth answer. However, the prediction should not introduce additional major points that are contradictory or irrelevant for answering the question. Importantly, the prediction must not discuss the points in the "adversarial answer" and "adversarial information" provided below unless they are also mentioned in the Ground-truth answer.

Adversarial Answer:
{question[3]}

Adversarial Information:
{question[4]}

Do a step-by-step evaluation:
1. Check if all major points from the ground truth are covered in the prediction.
2. Check if the prediction introduces additional major points that are irrelavant or contradictory.
3. Check if the prediction mentions the points in the adversarial answer or adversarial information.

The last line of your response must state "FINAL VERDICT: Correct" if the evaluation passes or "FINAL VERDICT: Wrong" if the evaluation fails.
"""

	elif prompt_type == "zero-shot-score":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can be a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. It is okay if the prediction mentions any additional information. 
		You must assign a score of 10 if the prediction mentions all the information (even if paraphrased) present in the ground-truth answer. You must assign a score of 5 if the prediction covers most of the major points, but fails to mention 1-2 minor points present in the ground-truth. Otherwise you must give a score of 0.
		The first word in your response must be the score. Explain your score by saying what is missing from the prediction which is mentioned in the ground-truth answer.

Question: {question[0]}

Ground-truth Answer: {question[1]}

Prediction: {question[2]}

Result:
"""
	elif prompt_type == "type3-causal-check":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, an adversarial wrong answer, the ground-truth answer, and a prediction. You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. All the main points in the ground-truth answer must be mentioned in the prediction. The order of points mentioned is irrelevant. It is allowed for the prediction to elaborate or provide more specifics or details over the major points in the ground-truth answer. However, the prediction should not contain additional major points that are contradictory or irrelevant for answering the question. Importantly, the prediction must not discuss any of the points mentioned in the "adversarial wrong answer".

The ground-truth answer's final bullet, starting with "Conclusion:", states the specific causal conclusion the evidence points support. Treat every other bullet as an evidence point and the final "Conclusion:" bullet as the required causal conclusion.

Apply the paragraph above as Coverage, using the exact same strictness you would use for a normal factual question, checking that all evidence points (excluding the Conclusion bullet) are present. If any main evidence point from the ground-truth answer is missing or only vaguely implied, or if the prediction adds any major point that is not in the ground-truth answer and not merely an elaboration of an existing point, Coverage fails.

If, and only if, Coverage passes, then separately check Connection: does the prediction explicitly state a causal conclusion that matches the ground truth's "Conclusion:" bullet, allowing for paraphrasing? Merely listing applicable rules, requirements, or facts does NOT satisfy Connection, even if the prediction uses words like "because", "therefore", or "as a result" while doing so. A prediction that lists what a regulation requires, or restates the evidence points as a flat list, has NOT satisfied Connection unless it also states the specific conclusion given in the ground truth's "Conclusion:" bullet, in a way that directly answers the causal question asked.

Output the single word "True" or "False" as the first word of your response, based on: False if Coverage fails, False if Coverage passes but Connection fails, True only if both pass. After that first word, always explain your reasoning in detail, covering both your Coverage assessment and, if reached, your Connection assessment, regardless of whether the verdict is True or False.

Question: {question[0]}

Adversarial Wrong Answers: {question[3]}

Ground-truth Answer (evidence points): {question[1]}

Prediction: {question[2]}

Result:
"""

	return prompt, sys_prompt




def get_solver_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "zero-shot-basic":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given some documents followed by a question. You need to generate the answer for that question. Provide the answer in bullet points, so that it is easier to objectively evaluate. Answering the question correctly requires information from multiple documents. If you find no relevant information in the documents for answering the question, you must only generate "No relevant information found in the documents." and nothing else.

Documents:
{question[0]}

Question: {question[1]}

Answer:
"""
	elif prompt_type == "zero-shot-no-irrelevant":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given some documents followed by a question. You need to generate the answer for that question. Provide the answer in bullet points, so that it is easier to objectively evaluate. Answering the question correctly requires information from multiple documents. You must only generate the points necessary for answering the question, without mentioning anything irrelevant to the question. If you find no relevant information in the documents for answering the question, you must only generate "No relevant information found in the documents." and nothing else.

Documents:
{question[0]}

Question: {question[1]}

Answer:
"""

	return prompt, sys_prompt





def get_generator_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "programmatic_docs":
		sys_prompt = "You are an expert data generator specialising in personal financial advice. Following the instruction, you must generate long and accurate documents grounded in FCA regulations and adviser-client contexts."
		prompt = f"""You need to generate the documents for an example of a retrieval based Question Answering Task. 
The task consists of n documents provided in English text that consist of information about different topics and a question. To answer the question correctly compulsorily requires using some of the information in some subset of the documents provided.

Given below is a situation faced by {question[0]} when searching {question[1]}. The question-answer pair is:
Question: {question[2]}
Answer: {question[3]}

Given below are the assigned answer points for each document.
{question[4]}

Your job is to create long documents according to this information. For each document, first create 10-12 unique other points that are in no way related to the topic of the question and answer (different points for each document). These points should discuss very different things about a similar but different topic. The unrelated points must be specific and detailed, with the same level of concrete factual content (figures, named rules, named products) as the assigned answer points, not generic statements. They should sit in a closely adjacent area of personal financial advice so they are difficult to distinguish from the genuine answer points on style or specificity alone, while remaining clearly about a different underlying question. Then use these points along with the assigned answer points to create a long document (at least 1200-1500 words long). The assigned answer points must be discussed taking into account the question. You must only discuss about these points and nothing else. Change the order of the points so that the answer points are embedded inside the document. When embedding each assigned answer point into the dialogue, you may restructure the sentence to fit natural conversation, but every specific fact, figure, rate, date, and named rule within that point must be preserved exactly as given, without paraphrasing, rounding, generalising, or omitting any of it. Assign an appropriate title to the document. Do not summarize or conclude the document in the end.

Format the document text as a transcript of a meeting between a financial adviser and their client, not as prose. The text must begin with a single header line in the format "Meeting: <topic>" summarising the meeting's subject, followed by a blank line, then the dialogue. Each line of dialogue must start with "Adviser:" or "Client:". Either speaker may have multiple consecutive lines before the other speaks. Do not use any other speaker labels or formatting. The assigned answer points and the unrelated points must be naturally embedded within the dialogue, spoken by either the adviser or the client as appropriate, and each unrelated point must arise naturally from the flow of conversation rather than appearing as an unconnected aside.

Additionally, ensure that the documents you create do not have any information related to the following adversarial question-answer pairs. You should create documents that discuss topics that are completely different from the following information.
{question[-1]}

Give output in the following format:
Document 1:
Title: <Title>
Question: {question[2]}
Answer points assigned [Only these points must be covered with respect to the question]: <Points>
Other unrelated points created: <Points>
Text:
<Document Text>

Document 2:
Title: <Title>
Question: {question[2]}
Answer points assigned [Only these points must be covered with respect to the question]: <Points>
Other unrelated points created: <Points>
Text:
<Document Text>

and so on...
"""

	elif prompt_type == "programmatic_adversarial":
		sys_prompt = "You are an expert generator of data. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make hard data to test an advanced question answering system. You are given a question that a {question[0]} might want answered, along with the corresponding answer, and information of documents from {question[1]} that are important for answering that question. 

Original Question: {question[2]}

Original Answer:
{question[3]}

Original Documents:
{question[4]}

You must generate an adversarial question, adversarial answer, and corresponding adversarial documents that ask for something different but on similar topics or type so that it is difficult to answer the original question. The adversarial question must come from a genuinely different regulatory area than the original question, not a different angle on the same area. For example, if the original question is about pension transfers, the adversarial question must not be about any other aspect of pensions (drawdown, contributions, MPAA, pension sharing) -- it must come from an entirely separate domain such as mortgages, ISAs, equity release, inheritance tax, or cryptoassets. Pick the regulatory area for the adversarial question first, ensure it shares no underlying subject matter with the original question's domain, and only then construct the question and answer. Never include the literal characters backslash-n in your response. Use only genuine line breaks between points. Examples of how adversarial questions should look like are provided below:

Original Question: What must an adviser verify before recommending a pension transfer?
Adversarial Question: What are the disclosure requirements for platform fees when an adviser recommends a Stocks and Shares ISA?

Original Question: How does triggering the MPAA affect a client's contribution capacity?
Adversarial Question: What loan-to-value limits apply when a client takes out a lifetime mortgage under an equity release plan?

Original Question: What disclosure obligations apply when an adviser recommends a specific investment platform?
Adversarial Question: What registration and customer due diligence requirements apply to a cryptoasset business under FCA rules?

Also provide an answer to the adversarial question, which is similar in style to the original answer, but differs significantly in information or specifics. The answer points for the adversarial question should be written in context of that adversarial question, so that they cannot be confused with the original question. Note that none of the points appearing in the original answer should be present in the answer to the adversarial question. Every point in the adversarial answer must be a substantive answer point -- never a document title, document name, or document reference stated as its own point (for example, never write a point like "Document 1: Halifax Drawdown Factsheet"). Document titles belong only in the "Document N Title:" lines further below, never inside the Answer itself.

The answer to the adversarial question you craft must be scattered across different documents (at least 3) separate from the original answer documents. Assign each point of the adversarial answer to a specific document in which that point will be discussed. You may assign multiple points to the same adversarial document, but each point must only be assigned to a single adversarial document. You must state the title and adversarial answer points assigned for each of the adversarial documents. Each adversarial answer point assigned to a document must be repeated verbatim, exactly as written in the adversarial answer above -- do not paraphrase, merge multiple points into one line, or split a single point across multiple lines. These adversarial documents should not have any overlapping information with the original answer documents."""
		if len(question[5]) > 0:
			prev_adv_questions = "\n\n".join(question[5])
			prompt = prompt + f"""

The following are adversarial questions I have already generated. Make a very different adversarial question.
{prev_adv_questions}"""
		prompt = prompt + f"""

Answer in the following format:

Question: <Question>
Answer: <Answer>

Document 1 Title: <Title>
Document 1 Answer points assigned: <Points>

Document 2 Title: <Title>
Document 2 Answer points assigned: <Points>

and so on...
"""

	elif prompt_type == "programmatic_adversarial_type3":
		sys_prompt = "You are an expert generator of data. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make hard data to test an advanced question answering system. You are given a question that a {question[0]} might want answered, along with the corresponding answer, and information of documents from {question[1]} that are important for answering that question. 

Original Question: {question[2]}

Original Answer:
{question[3]}

Original Documents:
{question[4]}

You must generate an adversarial question, adversarial answer, and corresponding adversarial documents that ask for something different but on similar topics or type so that it is difficult to answer the original question. The adversarial question must come from a genuinely different regulatory area than the original question, not a different angle on the same area. For example, if the original question is about pension transfers, the adversarial question must not be about any other aspect of pensions (drawdown, contributions, MPAA, pension sharing) -- it must come from an entirely separate domain such as mortgages, ISAs, equity release, inheritance tax, or cryptoassets. Pick the regulatory area for the adversarial question first, ensure it shares no underlying subject matter with the original question's domain, and only then construct the question and answer. Never include the literal characters backslash-n in your response. Use only genuine line breaks between points. Examples of how adversarial questions should look like are provided below:

Original Question: What must an adviser verify before recommending a pension transfer?
Adversarial Question: What are the disclosure requirements for platform fees when an adviser recommends a Stocks and Shares ISA?

Original Question: How does triggering the MPAA affect a client's contribution capacity?
Adversarial Question: What loan-to-value limits apply when a client takes out a lifetime mortgage under an equity release plan?

Original Question: What disclosure obligations apply when an adviser recommends a specific investment platform?
Adversarial Question: What registration and customer due diligence requirements apply to a cryptoasset business under FCA rules?

The adversarial question must ask about a causal relationship, trend, or sequence of events, matching the style of the original question. The adversarial answer must consist of 3-6 evidence bullet points followed by exactly one final bullet starting with "- Conclusion: " that states the causal conclusion the evidence supports. Each evidence bullet must state a plain fact, cause, effect, or connecting mechanism only -- do NOT use causal connective language such as "which means", "as a result", "so that", "which leads to", "this results in", or similar phrasing. The Conclusion bullet must state the causal conclusion explicitly and must be the final bullet in the adversarial answer. Every bullet in the Adv_Answer must be a substantive evidence point or the final Conclusion bullet -- never a document title, document name, or document reference stated as its own bullet (for example, never write a bullet like "- Document 1: Halifax Drawdown Factsheet"). Document titles belong only in the "Document N Title:" lines further below, never inside the Answer section.

Also provide an answer to the adversarial question, which is similar in style to the original answer, but differs significantly in information or specifics. The answer points for the adversarial question should be written in context of that adversarial question, so that they cannot be confused with the original question. Note that none of the points appearing in the original answer should be present in the answer to the adversarial question.

The evidence bullets of the adversarial answer must be scattered across different documents (at least 3), separate from the original answer documents. Assign each evidence bullet to a specific document in which that point will be discussed. You may assign multiple evidence bullets to the same document, but each evidence bullet must only be assigned to a single document. You must state the title and evidence bullets assigned for each of the adversarial documents. Each evidence bullet assigned to a document must be repeated verbatim, exactly as written in the adversarial answer above -- do not paraphrase, merge multiple points into one line, or split a single point across multiple lines. The final "- Conclusion: " bullet must NEVER be assigned to any document and must NEVER appear in any document's text or answer points assigned. These adversarial documents should not have any overlapping information with the original answer documents."""
		if len(question[5]) > 0:
			prev_adv_questions = "\n\n".join(question[5])
			prompt = prompt + f"""

The following are adversarial questions I have already generated. Make a very different adversarial question.
{prev_adv_questions}"""
		prompt = prompt + f"""

Answer in the following format:

Question: <Question>
Answer:
- <evidence point 1>
- <evidence point 2>
...
- Conclusion: <the causal conclusion the evidence points support>

Document 1 Title: <Title>
Document 1 Answer points assigned: <evidence points, never the Conclusion bullet>

Document 2 Title: <Title>
Document 2 Answer points assigned: <evidence points, never the Conclusion bullet>

and so on...
"""
	
	elif prompt_type == "programmatic_qa_type1":
		sys_prompt = "You are an expert generator of data specialising in personal financial advice. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system focused on personal financial advice.
Give me an example question and corresponding answer that a {question[0]} may ask that compulsorily requires searching a {question[1]}. The question must require specific factual or numerical information that cannot be answered from general knowledge. The answer must be very specific and written in bullet points. Depending on the question, the answer can have anything between 3-6 bullet points without any sub-points.

The answer to the question must be contained within a single document. Assign all answer points to that one document. You must state the title and answer points assigned for that document. Each answer point must be on its own line starting with "- ".

Answer in the following format:

Question: <Question>
Answer:
- <point 1>
- <point 2>
- <point 3>

Document 1 Title: <Title>
Document 1 Answer points assigned:
- <point 1>
- <point 2>
- <point 3>"""
		if len(question) > 3 and question[3] == "generic":
			prompt = prompt + GENERIC_NUMERIC_INSTRUCTION

	elif prompt_type == "programmatic_qa_type2":
		sys_prompt = "You are an expert generator of data specialising in personal financial advice. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system focused on personal financial advice.
Give me an example question and corresponding answer that a {question[0]} may ask that compulsorily requires searching a {question[1]}. Make questions that cannot be answered directly with general knowledge but necessarily require specific information spread across multiple documents. The answer must be very specific and written in bullet points. Depending on the question, the answer can havre anything between 3-6 bullet points without any sub-points.

The answer to the question you create must be scattered across different documents (at least 3). Assign each point of the answer to a specific document in which that point will be discussed. You may assign multiple points to the same document, but each point must only be assigned to a single document. You must state the title and answer points assigned for each of the documents. Each answer point must be on its own line starting with "- ".

Answer in the following format:

Question: <Question>
Answer:
- <point 1>
- <point 2>
...

Document 1 Title: <Title>
Document 1 Answer points assigned: <Points>

Document 2 Title: <Title>
Document 2 Answer points assigned: <Points>

and so on...

Do not use numbers, letters, or any other shorthand (e.g. "1, 2" or "A, B") to refer to answer points under each document. You must repeat the full text of each assigned answer point under its document, exactly as written in the Answer section above."""
		if len(question) > 3 and question[3] == "generic":
			prompt = prompt + GENERIC_NUMERIC_INSTRUCTION
		
	elif prompt_type == "programmatic_qa_type3":
		sys_prompt = "You are an expert generator of data specialising in personal financial advice. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system focused on personal financial advice.
Give me an example question and corresponding answer that a {question[0]} may ask that compulsorily requires searching a {question[1]}. The question must ask about a causal relationship, trend, or sequence of events in a personal financial advisory context -- for example, how one regulatory change caused a shift in client behaviour, or what sequence of events leads to a specific financial outcome. The question must not be answerable from general knowledge but must require specific causal evidence spread across multiple documents. The answer must be very specific and written in bullet points. Depending on the question, the answer can have anything between 3-6 evidence bullet points without any sub-points, plus exactly one final Conclusion bullet as described below. Each evidence bullet point must describe either a cause, an effect, or a connecting mechanism, stated as a plain factual observation only -- do NOT use causal connective language such as "which means", "as a result", "so that", "which leads to", "this results in", or similar phrasing that pre-states how the point relates to the overall conclusion. State only the isolated fact or observation itself.

The evidence points in the answer must be scattered across different documents (at least 3). Assign each evidence point to a specific document in which that point will be discussed. You may assign multiple evidence points to the same document, but each evidence point must only be assigned to a single document. You must state the title and one evidence component per document. The evidence component must be on its own line starting with "- ", worded as a plain fact with no causal connective language. The causal conclusion must never be stated, implied, or hinted at in any document -- only the isolated supporting evidence, with no connective phrasing suggesting how it relates to the outcome.

After the evidence bullets in the Answer section, add exactly one final bullet starting with "- Conclusion: " that explicitly states the causal conclusion the evidence supports. This Conclusion bullet must NOT be assigned to any document and must NOT appear in any document's text.

Answer in the following format:

Question: <Question>
Answer:
- <point 1>
- <point 2>
...
- Conclusion: <the causal conclusion the evidence points support>

Document 1 Title: <Title>
Document 1 Evidence component:
- <evidence that supports the causal conclusion without stating it explicitly>

Document 2 Title: <Title>
Document 2 Evidence component:
- <evidence that supports the causal conclusion without stating it explicitly>

and so on..."""
		if len(question) > 3 and question[3] == "generic":
			prompt = prompt + GENERIC_NUMERIC_INSTRUCTION

	elif prompt_type == "programmatic_scenarios":
		sys_prompt = "You are an expert generator of data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system focused on UK personal financial advice.
Give me 5 examples of real-life scenarios where a USER_PERSONA may seek information in a COLLECTION_OF_DOCS. All scenarios must involve a client seeking personal financial advice in the UK, governed by FCA regulations. Do not consider educational, historical, or non-financial scenarios.

Some examples are:
{question}

Answer in the following format:
USER_PERSONA:
COLLECTION_OF_DOCS:
"""
	
	elif prompt_type == "naive_baseline":
		sys_prompt = "You are an expert generator of data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system.
Given below is an example of a hard problem consisting of a set of documents, a question, and the corresponding answer. 

{question}

You need to generate a different problem for this same task. You must provide a different set of documents, a different question, and a different answer. The question must be such that it cannot be answered directly with general knowledge but necessarily requires some uncommon information that is present in the documents. The answer must be very specific and written in 3-4 bullet points, so that it is easier to objectively evaluate.

Answer in the following format:
Documents:
<Documents>

Question: <Question>

Answer:
<Answer>
"""

	return prompt, sys_prompt