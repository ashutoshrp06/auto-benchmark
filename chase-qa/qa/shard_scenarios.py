import argparse
import os
import shutil
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Split scenarios.tsv into N shard directories (round-robin)')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-scenarios_name', type=str, required=True, help='Unsharded scenarios folder name')
	parser.add_argument('-shards', type=int, required=True, help='Number of shards')
	parser.add_argument('-manifest', type=str, default='', help='Path to write shard suffix manifest. Default: <out_dir>/<scenarios_name>/shards.manifest')
	parser.add_argument('-data', type=str, default='scenarios', help='Scenarios filename without extension')

	return parser


def main():
	args = build_parser().parse_args()

	if args.shards < 1:
		sys.exit('ERROR: -shards must be >= 1')

	src_dir = os.path.join(args.out_dir, args.scenarios_name)
	src_file = os.path.join(src_dir, args.data + '.tsv')

	if not os.path.isdir(src_dir):
		sys.exit('ERROR: missing scenarios dir ' + src_dir)
	if not os.path.exists(src_file):
		sys.exit('ERROR: missing ' + src_file)

	df = pd.read_csv(src_file, sep='\t')
	n_rows = len(df)
	if n_rows == 0:
		sys.exit('ERROR: ' + src_file + ' has 0 rows')

	if args.shards > n_rows:
		print('WARNING: -shards {} exceeds row count {}. Empty shards will be skipped.'.format(args.shards, n_rows))

	manifest_path = args.manifest or os.path.join(src_dir, 'shards.manifest')

	# Replicate every sibling file into each shard dir so any auxiliary input
	# stage 2 might read is present. The scenarios file itself is written per
	# shard below. The manifest is excluded only if it lives in src_dir.
	mf_in_src = os.path.dirname(os.path.abspath(manifest_path)) == os.path.abspath(src_dir)
	carry = [
		f for f in os.listdir(src_dir)
		if os.path.isfile(os.path.join(src_dir, f))
		and f != args.data + '.tsv'
		and not (mf_in_src and f == os.path.basename(manifest_path))
	]

	suffixes = []
	for s in range(args.shards):
		# Round-robin so reg (IDs 1-10) and generic (IDs 11-15) seeds spread evenly.
		part = df.iloc[s::args.shards]
		if len(part) == 0:
			print('  s{:02d}: empty, skipped'.format(s + 1))
			continue

		sfx = '-s{:02d}'.format(s + 1)
		dst_dir = os.path.join(args.out_dir, args.scenarios_name + sfx)
		os.makedirs(dst_dir, exist_ok=True)

		for f in carry:
			shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

		part.to_csv(os.path.join(dst_dir, args.data + '.tsv'), sep='\t', index=False)
		suffixes.append(sfx)
		print('  {}: {} rows -> {}'.format(sfx, len(part), dst_dir))

	if not suffixes:
		sys.exit('ERROR: no non-empty shards produced')

	# Read back every shard and prove the union equals the source exactly.
	# Reg_Text carries embedded newlines, so quoting must survive the rewrite.
	check = []
	for sfx in suffixes:
		p = os.path.join(args.out_dir, args.scenarios_name + sfx, args.data + '.tsv')
		check.append(pd.read_csv(p, sep='\t'))
	rt = pd.concat(check, ignore_index=True)

	if len(rt) != n_rows:
		sys.exit('ERROR: round-trip row count {} != source {}'.format(len(rt), n_rows))
	if list(rt.columns) != list(df.columns):
		sys.exit('ERROR: round-trip columns differ from source')

	sort_col = list(df.columns)[0]
	a = df.sort_values(sort_col, kind='mergesort').reset_index(drop=True).fillna('').astype(str)
	b = rt.sort_values(sort_col, kind='mergesort').reset_index(drop=True).fillna('').astype(str)
	if not a.equals(b):
		sys.exit('ERROR: round-trip content mismatch. Quoting is not preserved; do not proceed.')
	print('  round-trip verified: {} rows, {} columns'.format(len(rt), len(rt.columns)))

	with open(manifest_path, 'w') as f:
		for sfx in suffixes:
			f.write(sfx + '\n')

	print('Wrote manifest: {} ({} non-empty shards, {} rows total)'.format(manifest_path, len(suffixes), n_rows))


if __name__ == '__main__':
	main()