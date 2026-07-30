import argparse
import os
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Split a TSV into N contiguous shard files')

	parser.add_argument('-input_tsv', type=str, required=True, help='TSV to split')
	parser.add_argument('-out_dir', type=str, required=True, help='Directory to write shard files into')
	parser.add_argument('-base', type=str, required=True, help='Base name for shard files, e.g. type1-v7')
	parser.add_argument('-shards', type=int, required=True, help='Number of shards')
	parser.add_argument('-manifest', type=str, default='', help='Manifest path. Default: <out_dir>/<base>.manifest')

	return parser


def main():
	args = build_parser().parse_args()

	if args.shards < 1:
		sys.exit('ERROR: -shards must be >= 1')
	if not os.path.exists(args.input_tsv):
		sys.exit('ERROR: missing ' + args.input_tsv)

	df = pd.read_csv(args.input_tsv, sep='\t')
	n = len(df)
	if n == 0:
		sys.exit('ERROR: ' + args.input_tsv + ' has 0 rows')

	os.makedirs(args.out_dir, exist_ok=True)
	manifest_path = args.manifest or os.path.join(args.out_dir, args.base + '.manifest')

	# Remove shard files from a previous run with a larger -shards, otherwise a
	# stale s07 would be picked up by a later 4-shard run.
	for f in os.listdir(args.out_dir):
		if f.startswith(args.base + '-s') and f.endswith('.tsv'):
			os.remove(os.path.join(args.out_dir, f))

	# Contiguous split. Order within and across shards matches the input exactly,
	# so concatenating shard outputs in shard order reproduces unsharded order.
	k, r = divmod(n, args.shards)
	paths = []
	start = 0
	for s in range(args.shards):
		size = k + (1 if s < r else 0)
		if size == 0:
			print('  s{:02d}: empty, skipped'.format(s + 1))
			continue
		part = df.iloc[start:start + size]
		start += size
		p = os.path.join(args.out_dir, '{}-s{:02d}.tsv'.format(args.base, s + 1))
		part.to_csv(p, sep='\t', index=False)
		paths.append(p)
		print('  s{:02d}: rows {}..{} ({}) -> {}'.format(s + 1, start - size, start - 1, size, p))

	if start != n:
		sys.exit('ERROR: split covered {} of {} rows'.format(start, n))
	if not paths:
		sys.exit('ERROR: no non-empty shards produced')

	# Prove the split is lossless AND order preserving before anything runs on it.
	rt = pd.concat([pd.read_csv(p, sep='\t') for p in paths], ignore_index=True)
	if len(rt) != n:
		sys.exit('ERROR: round-trip row count {} != source {}'.format(len(rt), n))
	if list(rt.columns) != list(df.columns):
		sys.exit('ERROR: round-trip columns differ from source')
	a = df.reset_index(drop=True).fillna('').astype(str)
	b = rt.reset_index(drop=True).fillna('').astype(str)
	if not a.equals(b):
		sys.exit('ERROR: round-trip content mismatch. Do not proceed.')
	print('  round-trip verified: {} rows, {} columns, order preserved'.format(n, len(df.columns)))

	with open(manifest_path, 'w') as f:
		for p in paths:
			f.write(p + '\n')
	print('Wrote manifest: {} ({} shards)'.format(manifest_path, len(paths)))


if __name__ == '__main__':
	main()