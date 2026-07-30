"""Solver accuracy split by RocketEval PoLL verdict.

Zero API cost. Reads results that already exist on disk.

Verdicts are recomputed from poll_scores.tsv rather than read from its Verdict
column, so -pass_frac can be swept without re-running poll_aggregate.py. The
rule mirrors poll_aggregate.py exactly: DISAGREE if No_Majority_Frac exceeds
-disagree_frac, otherwise PASS if Pass_Frac >= -pass_frac, else FAIL.

The join to result.tsv is positional. This is safe because shard_tsv.py verifies
order preservation on every split, and merge_eval.py concatenates contiguous
shards in order, so corpus row i corresponds to result.tsv row i. The script
asserts equal lengths before relying on it.
"""

import argparse
import os
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Solver accuracy split by RocketEval verdict')

	parser.add_argument('-gen_dir', type=str, default='generation_outputs/', help='Generation output root')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Solver/evaluator output root')
	parser.add_argument('-run', type=str, default='v7', help='Run tag used in folder names')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Corpus filename stem')
	parser.add_argument('-models', type=str, default='gpt55,gemini31pro', help='Comma separated model tags as used in run names')
	parser.add_argument('-prompts', type=str, default='basic,noirrelevant', help='Comma separated solver prompt tags as used in run names')
	parser.add_argument('-eval_suffix', type=str, default='-eval', help='Evaluator run directory suffix')
	parser.add_argument('-pass_frac', type=float, default=0.75, help='Fraction of criteria that must pass')
	parser.add_argument('-disagree_frac', type=float, default=0.34, help='No-majority fraction above which a question is DISAGREE')
	parser.add_argument('-types', type=str, default='1,2,3', help='Comma separated QA types')

	return parser


def verdict_of(pass_frac, no_majority_frac, args):
	if no_majority_frac > args.disagree_frac:
		return 'DISAGREE'
	return 'PASS' if pass_frac >= args.pass_frac else 'FAIL'


def main():
	args = build_parser().parse_args()

	types = [t.strip() for t in args.types.split(',') if t.strip()]
	models = [m.strip() for m in args.models.split(',') if m.strip()]
	prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]

	print('pass_frac={}  disagree_frac={}  eval_suffix={}'.format(
		args.pass_frac, args.disagree_frac, args.eval_suffix))

	rows = []

	for t in types:
		folder = 'elm-docs-type{}-{}'.format(t, args.run)
		corpus_path = os.path.join(args.gen_dir, folder, args.data + '.tsv')
		poll_path = os.path.join(args.gen_dir, folder, 'rocketeval', 'poll_scores.tsv')

		if not os.path.exists(corpus_path):
			sys.exit('ERROR: missing ' + corpus_path)
		if not os.path.exists(poll_path):
			sys.exit('ERROR: missing ' + poll_path + '. Run RocketEval first.')

		corpus = pd.read_csv(corpus_path, sep='\t')
		poll = pd.read_csv(poll_path, sep='\t')

		for c in ('Root_ID', 'Question_No'):
			if c not in corpus.columns:
				sys.exit('ERROR: corpus lacks ' + c)
			if c not in poll.columns:
				sys.exit('ERROR: poll_scores lacks ' + c)
		for c in ('Pass_Frac', 'No_Majority_Frac', 'N_Criteria'):
			if c not in poll.columns:
				sys.exit('ERROR: poll_scores lacks ' + c)

		keys = corpus[['Root_ID', 'Question_No', 'Seed_Type']].copy()
		keys = keys.merge(
			poll[['Root_ID', 'Question_No', 'Pass_Frac', 'No_Majority_Frac', 'N_Criteria']],
			on=['Root_ID', 'Question_No'], how='left')

		if len(keys) != len(corpus):
			sys.exit('ERROR: poll_scores does not join one-to-one with the corpus')
		n_null = int(keys['Pass_Frac'].isna().sum())
		if n_null:
			print('WARNING: type{} has {} questions with no PoLL score; treated as DISAGREE'.format(t, n_null))

		keys['Verdict'] = [
			'DISAGREE' if pd.isna(pf) else verdict_of(pf, nm, args)
			for pf, nm in zip(keys['Pass_Frac'], keys['No_Majority_Frac'])
		]

		counts = keys['Verdict'].value_counts().to_dict()
		print('')
		print('type{}  n={}  criteria/q mean {:.1f}  {}'.format(
			t, len(keys), keys['N_Criteria'].mean(), counts))

		for m in models:
			for p in prompts:
				run = 'type{}-{}-{}-{}'.format(t, m, args.run, p)
				res_path = os.path.join(args.out_dir, run + args.eval_suffix, 'result.tsv')
				if not os.path.exists(res_path):
					print('  {:<11} {:<13} MISSING {}'.format(m, p, res_path))
					continue

				r = pd.read_csv(res_path, sep='\t')
				if len(r) != len(corpus):
					print('  {:<11} {:<13} SKIPPED: {} rows vs corpus {}'.format(
						m, p, len(r), len(corpus)))
					continue

				# Positional join, asserted safe by the length check above.
				r = r.assign(Verdict=keys['Verdict'].values,
				             Seed_Type=keys['Seed_Type'].values)

				full = r['Result'].mean()
				out = '  {:<11} {:<13} full {:>3}/{:<3}={:.3f}'.format(
					m, p, int(r['Result'].sum()), len(r), full)

				for v in ('PASS', 'FAIL', 'DISAGREE'):
					s = r[r['Verdict'] == v]
					if len(s):
						out += ' | {} {:>3}/{:<3}={:.3f}'.format(
							v[:4], int(s['Result'].sum()), len(s), s['Result'].mean())
					else:
						out += ' | {} n/a'.format(v[:4])
				print(out)

				sub = r[r['Verdict'] == 'PASS']
				for st, g in sub.groupby('Seed_Type'):
					rows.append({'Type': 'type' + t, 'Model': m, 'Prompt': p,
					             'Seed_Type': st, 'PASS_n': len(g),
					             'PASS_acc': round(g['Result'].mean(), 3)})

	if rows:
		print('')
		print('PASS-only accuracy by seed type')
		df = pd.DataFrame(rows)
		piv = df.pivot_table(index=['Type', 'Seed_Type'], columns=['Model', 'Prompt'],
		                     values='PASS_acc')
		n = df.pivot_table(index=['Type', 'Seed_Type'], columns=['Model', 'Prompt'],
		                   values='PASS_n')
		print(piv.to_string())
		print('')
		print('n per cell')
		print(n.to_string())


if __name__ == '__main__':
	main()