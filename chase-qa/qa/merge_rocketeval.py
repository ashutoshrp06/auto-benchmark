import argparse
import glob
import os
import sys

import pandas as pd


def build_parser():
	parser = argparse.ArgumentParser(description='Merge sharded RocketEval judgments and checklists')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, required=True, help='Canonical run folder')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Data filename without extension')
	parser.add_argument('-suffix', type=str, default='-r', help='Shard folder suffix prefix')

	return parser


def main():
	args = build_parser().parse_args()

	parent = args.out_dir
	dirs = sorted(glob.glob(os.path.join(parent, args.folder_name + args.suffix + '[0-9][0-9]')))
	if not dirs:
		sys.exit('ERROR: no shard folders matching ' + args.folder_name + args.suffix + 'NN')

	jf, cl, missing = [], [], []
	for d in dirs:
		jp = os.path.join(d, 'rocketeval', 'judgments.tsv')
		cp = os.path.join(d, 'rocketeval', 'checklists.jsonl')
		if not os.path.exists(jp):
			missing.append(jp)
			continue
		part = pd.read_csv(jp, sep='\t')
		if len(part):
			jf.append(part)
		if os.path.exists(cp):
			with open(cp) as f:
				cl.extend(l for l in f if l.strip())

	if missing:
		sys.exit('ERROR: missing shard judgments:\n  ' + '\n  '.join(missing))
	if not jf:
		sys.exit('ERROR: all shard judgments are empty')

	cols = list(jf[0].columns)
	for x in jf[1:]:
		if list(x.columns) != cols:
			sys.exit('ERROR: column mismatch between shard judgments')

	j = pd.concat(jf, ignore_index=True)

	# judgments.tsv is written with mode='a'. A resumed shard can therefore carry
	# a repeated header row, and a rerun can duplicate judgements.
	before = len(j)
	j = j[j['Root_ID'].astype(str) != 'Root_ID']
	j = j.drop_duplicates(subset=['Root_ID', 'Question_No', 'Judge', 'Criterion_No'], keep='last')
	if len(j) != before:
		print('  dropped {} repeated header or duplicate judgement rows'.format(before - len(j)))

	corpus = pd.read_csv(os.path.join(parent, args.folder_name, args.data + '.tsv'), sep='\t')
	want = set(zip(corpus['Root_ID'].astype(str), corpus['Question_No'].astype(str)))
	got = set(zip(j['Root_ID'].astype(str), j['Question_No'].astype(str)))
	if want - got:
		print('WARNING: {} of {} corpus questions have no judgements'.format(len(want - got), len(want)))
	if got - want:
		sys.exit('ERROR: judgements reference {} questions absent from the corpus'.format(len(got - want)))

	dst_dir = os.path.join(parent, args.folder_name, 'rocketeval')
	os.makedirs(dst_dir, exist_ok=True)
	j.to_csv(os.path.join(dst_dir, 'judgments.tsv'), sep='\t', index=None)
	if cl:
		with open(os.path.join(dst_dir, 'checklists.jsonl'), 'w') as f:
			f.writelines(cl)

	print('Merged {} shards -> {}'.format(len(jf), os.path.join(dst_dir, 'judgments.tsv')))
	print('  judgement rows: {}  questions: {}  judges: {}'.format(
		len(j), len(got), j['Judge'].nunique()))


if __name__ == '__main__':
	main()