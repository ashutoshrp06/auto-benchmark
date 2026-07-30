import argparse
import os
import shutil
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Split a run folder into N shard folders for RocketEval')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, required=True, help='Canonical run folder, e.g. elm-docs-type1-v7')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Data filename without extension')
	parser.add_argument('-shards', type=int, required=True, help='Number of shards')
	parser.add_argument('-suffix', type=str, default='-r', help='Shard folder suffix prefix. Default: -r')

	return parser


def main():
	args = build_parser().parse_args()

	if args.shards < 1:
		sys.exit('ERROR: -shards must be >= 1')

	src_dir = os.path.join(args.out_dir, args.folder_name)
	src_file = os.path.join(src_dir, args.data + '.tsv')
	if not os.path.exists(src_file):
		sys.exit('ERROR: missing ' + src_file)

	df = pd.read_csv(src_file, sep='\t')
	n = len(df)
	if n == 0:
		sys.exit('ERROR: ' + src_file + ' has 0 rows')

	for col in ('Root_ID', 'Question_No'):
		if col not in df.columns:
			sys.exit('ERROR: data lacks column ' + col)
	if df.duplicated(subset=['Root_ID', 'Question_No']).any():
		sys.exit('ERROR: (Root_ID, Question_No) is not unique. Merge would be ambiguous.')

	# Gated on presence so pre-QID corpora (v7) are unaffected.
	if 'QID' in df.columns:
		if df['QID'].isna().any() or (df['QID'].astype(str).str.strip() == '').any():
			sys.exit('ERROR: QID column has empty values. Rerun cleanup.py.')
		n_dupe = int(df.duplicated(subset=['QID']).sum())
		if n_dupe:
			sys.exit('ERROR: QID is not unique ({} duplicate row(s)). Investigate, do not suppress.'.format(n_dupe))

	# Drop shard folders from any previous run so a larger prior -shards cannot
	# leave stale directories that the merge would pick up.
	parent = args.out_dir
	pref = args.folder_name + args.suffix
	for d in os.listdir(parent):
		if d.startswith(pref) and os.path.isdir(os.path.join(parent, d)):
			shutil.rmtree(os.path.join(parent, d))

	k, r = divmod(n, args.shards)
	made = []
	start = 0
	for s in range(args.shards):
		size = k + (1 if s < r else 0)
		if size == 0:
			print('  {}{:02d}: empty, skipped'.format(args.suffix, s + 1))
			continue
		part = df.iloc[start:start + size]
		start += size
		d = os.path.join(parent, '{}{}{:02d}'.format(args.folder_name, args.suffix, s + 1))
		os.makedirs(d, exist_ok=True)
		part.to_csv(os.path.join(d, args.data + '.tsv'), sep='\t', index=False)
		made.append(d)
		print('  {}{:02d}: {} rows -> {}'.format(args.suffix, s + 1, size, d))

	if start != n:
		sys.exit('ERROR: split covered {} of {} rows'.format(start, n))
	if not made:
		sys.exit('ERROR: no non-empty shards produced')

	rt = pd.concat([pd.read_csv(os.path.join(d, args.data + '.tsv'), sep='\t') for d in made],
	               ignore_index=True)
	if len(rt) != n or list(rt.columns) != list(df.columns):
		sys.exit('ERROR: round-trip shape mismatch')
	a = df.reset_index(drop=True).fillna('').astype(str)
	b = rt.reset_index(drop=True).fillna('').astype(str)
	if not a.equals(b):
		sys.exit('ERROR: round-trip content mismatch. Do not proceed.')
	print('  round-trip verified: {} rows, {} columns, order preserved'.format(n, len(df.columns)))


if __name__ == '__main__':
	main()