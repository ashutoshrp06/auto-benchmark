import argparse
import os
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Merge sharded solver or evaluator outputs')

	parser.add_argument('-out_dir', type=str, default='outputs/', help='Solver/evaluator output root')
	parser.add_argument('-base_run', type=str, required=True, help='Base run name without shard suffix')
	parser.add_argument('-shards', type=int, required=True, help='Number of shards that were launched')
	parser.add_argument('-mode', type=str, required=True, choices=['solver', 'evaluator'], help='Which output to merge')
	parser.add_argument('-strict', action='store_true', help='Fail instead of warn on a shard that ran but produced no output')

	return parser


def main():
	args = build_parser().parse_args()

	stem = 'predictions' if args.mode == 'solver' else 'result'

	frames = []
	missing = []
	for s in range(1, args.shards + 1):
		d = os.path.join(args.out_dir, '{}-s{:02d}'.format(args.base_run, s))
		p = os.path.join(d, stem + '.tsv')
		if not os.path.exists(p):
			# A shard whose input slice was empty never launched and has no run
			# directory. A shard that launched and failed does have one.
			missing.append((p, os.path.isdir(d)))
			continue
		part = pd.read_csv(p, sep='\t')
		if len(part) == 0:
			continue
		frames.append(part)

	if missing:
		msg = 'missing shard outputs:\n  ' + '\n  '.join(p for p, _ in missing)
		real = [p for p, ran in missing if ran]
		if real and args.strict:
			sys.exit('ERROR: ' + msg)
		print('WARNING: ' + msg)

	if not frames:
		sys.exit('ERROR: no shard outputs found for ' + args.base_run)

	cols = list(frames[0].columns)
	for fr in frames[1:]:
		if list(fr.columns) != cols:
			sys.exit('ERROR: column mismatch between shards. Refusing to merge.')

	# Shards were contiguous slices in ascending order, so plain concat in shard
	# order restores the unsharded row order exactly.
	df = pd.concat(frames, ignore_index=True)

	if args.mode == 'evaluator':
		for c in ('ID', 'Result'):
			if c not in df.columns:
				sys.exit('ERROR: evaluator output lacks column ' + c)
		# ID is assigned positionally (i+1) and is therefore per-shard. Renumber.
		df['ID'] = range(1, len(df) + 1)

	dst_dir = os.path.join(args.out_dir, args.base_run)
	os.makedirs(dst_dir, exist_ok=True)
	dst = os.path.join(dst_dir, stem + '.tsv')
	df.to_csv(dst, sep='\t', index=False)

	print('Merged {} shards -> {}'.format(len(frames), dst))
	print('  rows: {}'.format(len(df)))

	if args.mode == 'evaluator':
		# Accuracy in any individual shard's eval_logs.txt covers that shard only.
		# This is the only correct total for the run.
		score = float(df['Result'].sum())
		acc = score / len(df)
		line = 'Accuracy: {} ({}/{})'.format(acc, score, len(df))
		print('  ' + line)
		with open(os.path.join(dst_dir, 'eval_logs.txt'), 'a') as f:
			f.write('\n=== MERGED TOTAL ===\n' + line + '\n')


if __name__ == '__main__':
	main()