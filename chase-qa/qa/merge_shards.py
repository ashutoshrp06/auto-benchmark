import argparse
import os
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Merge sharded stage-9 outputs and renumber positional IDs')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-base_name', type=str, required=True, help='Canonical folder name without shard suffix, e.g. elm-docs-type1-v7')
	parser.add_argument('-scenarios_name', type=str, required=True, help='Unsharded scenarios folder name, used to recover original ordering')
	parser.add_argument('-manifest', type=str, required=True, help='Shard suffix manifest written by shard_scenarios.py')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified', help='Data filename without extension')
	parser.add_argument('-scenarios_data', type=str, default='scenarios', help='Scenarios filename without extension')
	parser.add_argument('-qno_mode', type=str, default='global', choices=['global', 'perroot', 'keep'], help='How to renumber Question_No after merge')
	parser.add_argument('-strict', action='store_true', help='Fail instead of warn on unmapped rows or missing shard files')

	return parser


def _key(persona, environment):
	# Join key only. Stored values are never modified by this normalisation.
	return (str(persona).strip(), str(environment).strip())


def main():
	args = build_parser().parse_args()

	with open(args.manifest) as f:
		suffixes = [ln.strip() for ln in f if ln.strip()]
	if not suffixes:
		sys.exit('ERROR: empty manifest ' + args.manifest)

	sc_file = os.path.join(args.out_dir, args.scenarios_name, args.scenarios_data + '.tsv')
	if not os.path.exists(sc_file):
		sys.exit('ERROR: missing ' + sc_file)
	sc = pd.read_csv(sc_file, sep='\t')

	for col in ['ID', 'Persona', 'Environment']:
		if col not in sc.columns:
			sys.exit('ERROR: scenarios file lacks column ' + col)

	keys = [_key(p, e) for p, e in zip(sc['Persona'], sc['Environment'])]
	if len(set(keys)) != len(keys):
		msg = 'scenarios (Persona, Environment) is not unique; original ordering cannot be recovered'
		if args.strict:
			sys.exit('ERROR: ' + msg)
		print('WARNING: ' + msg + '. Falling back to shard-then-row ordering.')
		orig_map = {}
	else:
		orig_map = dict(zip(keys, sc['ID'].astype(int)))

	SENTINEL = 10 ** 9
	frames = []
	missing = []

	for si, sfx in enumerate(suffixes):
		path = os.path.join(args.out_dir, args.base_name + sfx, args.data + '.tsv')
		if not os.path.exists(path):
			missing.append(path)
			continue
		part = pd.read_csv(path, sep='\t')
		if len(part) == 0:
			continue
		part['_shard'] = si
		frames.append(part)

	if missing:
		msg = 'missing shard outputs:\n  ' + '\n  '.join(missing)
		if args.strict:
			sys.exit('ERROR: ' + msg)
		print('WARNING: ' + msg)

	if not frames:
		sys.exit('ERROR: no shard outputs found for ' + args.base_name)

	# Real columns only. Nothing in the pipeline schema starts with an underscore.
	cols = [c for c in frames[0].columns if not c.startswith('_')]
	for fr in frames[1:]:
		if [c for c in fr.columns if not c.startswith('_')] != cols:
			sys.exit('ERROR: column mismatch between shards. Refusing to merge.')

	df = pd.concat(frames, ignore_index=True)

	for col in ['Root_ID', 'Question_No', 'Persona', 'Environment']:
		if col not in df.columns:
			sys.exit('ERROR: merged data lacks column ' + col)

	if orig_map:
		df['_orig'] = [
			orig_map.get(_key(p, e), SENTINEL)
			for p, e in zip(df['Persona'], df['Environment'])
		]
		n_unmapped = int((df['_orig'] == SENTINEL).sum())
		if n_unmapped:
			msg = '{} rows could not be mapped back to a scenario; they sort last'.format(n_unmapped)
			if args.strict:
				sys.exit('ERROR: ' + msg)
			print('WARNING: ' + msg)
	else:
		df['_orig'] = SENTINEL

	df['_old_root'] = df['Root_ID']
	df['_old_qno'] = df['Question_No']

	# One group per original pipeline row. Sort by the smallest original
	# scenario ID in the group, then shard, then shard-local Root_ID.
	grp = df.groupby(['_shard', '_old_root'], sort=False)['_orig'].min().reset_index()
	grp = grp.sort_values(['_orig', '_shard', '_old_root'], kind='mergesort').reset_index(drop=True)
	grp['_new_root'] = grp.index + 1

	df = df.merge(grp[['_shard', '_old_root', '_new_root']], on=['_shard', '_old_root'], how='left')
	if df['_new_root'].isna().any():
		sys.exit('ERROR: Root_ID remap produced nulls. Aborting.')

	df = df.sort_values(['_new_root', '_old_qno'], kind='mergesort').reset_index(drop=True)
	df['Root_ID'] = df['_new_root'].astype(int)

	if args.qno_mode == 'global':
		df['Question_No'] = range(1, len(df) + 1)
	elif args.qno_mode == 'perroot':
		df['Question_No'] = df.groupby('Root_ID').cumcount() + 1
	# 'keep' leaves Question_No untouched

	df = df[cols]

	dst_dir = os.path.join(args.out_dir, args.base_name)
	os.makedirs(dst_dir, exist_ok=True)
	dst = os.path.join(dst_dir, args.data + '.tsv')
	df.to_csv(dst, sep='\t', index=False)

	print('Merged {} shards -> {}'.format(len(frames), dst))
	print('  rows: {}  Root_IDs: {}  qno_mode: {}'.format(len(df), df['Root_ID'].nunique(), args.qno_mode))


if __name__ == '__main__':
	main()