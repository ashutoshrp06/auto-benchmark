import argparse
import os
import sys

import pandas as pd

# Single source of truth for the hash. Two copies of a hash function is exactly
# the divergence that silently splits a corpus in half.
# Note: importing cleanup runs its module-level tiktoken init once.
from cleanup import make_qid


def build_parser():
	parser = argparse.ArgumentParser(description='Combine per-batch cleaned corpora, tag Batch, drop duplicate questions')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-prefix', type=str, default='elm', help='Directory prefix used by run_pipeline.sh')
	parser.add_argument('-run_names', type=str, required=True, help='Comma-separated run names in batch order, e.g. v8-b01,v8-b02')
	parser.add_argument('-types', type=str, default='1,2,3', help='Comma-separated QA types')
	parser.add_argument('-data', type=str, default='programmatic_data_modified_verified_cleaned', help='Data filename without extension')
	parser.add_argument('-jaccard', type=float, default=60.0, help='Near-duplicate threshold on question tokens, percent')
	parser.add_argument('-drop_scope', type=str, default='none', choices=['none', 'cross_batch', 'all'], help='Which near-duplicates to drop. Exact QID duplicates are always dropped.')
	parser.add_argument('-cross_type', action='store_true', help='Also drop near-duplicates spanning two QA types. Off by default: a type1 and type2 question can share most tokens and still be different items.')
	parser.add_argument('-out_file', type=str, default='', help='Combined corpus path. Default: <out_dir>/combined_corpus.tsv')
	parser.add_argument('-report_only', action='store_true', help='Print the duplicate report and write nothing')
	parser.add_argument('-strict', action='store_true', help='Fail instead of warn on missing folders')

	return parser


def jaccard_similarity(set1, set2):
	# Identical to generator.py and cleanup.py so thresholds are comparable
	# with the stage-1 and stage-2 gates.
	union = set1.union(set2)
	if not union:
		return 100.0
	return (len(set1.intersection(set2)) / len(union)) * 100


def main():
	args = build_parser().parse_args()

	runs = [r.strip() for r in args.run_names.split(',') if r.strip()]
	types = [t.strip() for t in args.types.split(',') if t.strip()]
	if not runs:
		sys.exit('ERROR: -run_names is empty')
	if not types:
		sys.exit('ERROR: -types is empty')
	if len(set(runs)) != len(runs):
		sys.exit('ERROR: -run_names contains a repeat')

	frames = []
	missing = []
	base_cols = None
	n_synth_qid = 0

	for run in runs:
		for t in types:
			folder = '{}-docs-type{}-{}'.format(args.prefix, t, run)
			path = os.path.join(args.out_dir, folder, args.data + '.tsv')
			if not os.path.exists(path):
				missing.append(path)
				continue
			# Reg_Text carries literal newlines. dtype=str and keep_default_na=False
			# are mandatory, never count these files by line.
			df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
			if len(df) == 0:
				print('WARNING: {} has 0 rows'.format(path))
				continue
			for col in ('Question', 'Persona', 'Environment'):
				if col not in df.columns:
					sys.exit('ERROR: {} has no {} column'.format(path, col))

			if 'QID' not in df.columns:
				df['QID'] = [make_qid('type' + t, p, e, q) for p, e, q
				             in zip(df['Persona'], df['Environment'], df['Question'])]
				n_synth_qid += len(df)
				print('NOTE: {} predates QID, computed in memory (file not modified)'.format(path))
			elif (df['QID'].str.strip() == '').any():
				sys.exit('ERROR: {} has empty QID values. Rerun cleanup.py.'.format(path))

			cols = [c for c in df.columns if c != 'QID']
			if base_cols is None:
				base_cols = cols
			elif cols != base_cols:
				sys.exit('ERROR: column mismatch at {}. Refusing to combine.'.format(path))

			df['Batch'] = run
			df['QA_Type'] = 'type' + t
			frames.append(df)

	if missing:
		msg = 'missing cleaned corpora:\n  ' + '\n  '.join(missing)
		if args.strict:
			sys.exit('ERROR: ' + msg)
		print('WARNING: ' + msg)
	if not frames:
		sys.exit('ERROR: no corpora found')

	df = pd.concat(frames, ignore_index=True)
	n_in = len(df)
	print('Loaded {} rows from {} folder(s)'.format(n_in, len(frames)))
	if n_synth_qid:
		print('QID computed in memory for {} row(s)'.format(n_synth_qid))
	print(df.groupby(['Batch', 'QA_Type']).size().to_string())

	# ---------------------------------------------------------- exact QID
	dup_mask = df.duplicated(subset=['QID'], keep='first')
	n_exact = int(dup_mask.sum())
	exact_rows = df[dup_mask].copy()
	if n_exact:
		first_batch = df.drop_duplicates(subset=['QID'], keep='first').set_index('QID')['Batch']
		exact_rows['_Drop_Reason'] = 'exact_qid'
		exact_rows['_Matched_Batch'] = exact_rows['QID'].map(first_batch)
		n_cross = int((exact_rows['_Matched_Batch'] != exact_rows['Batch']).sum())
		print('Exact QID duplicates: {} ({} cross-batch, {} within-batch)'.format(n_exact, n_cross, n_exact - n_cross))
	else:
		print('Exact QID duplicates: 0')

	kept_df = df[~dup_mask].reset_index(drop=True)

	# ----------------------------------------------------- near duplicates
	toks = [set(str(q).lower().split()) for q in kept_df['Question']]
	batches = list(kept_df['Batch'])
	qtypes = list(kept_df['QA_Type'])
	n = len(kept_df)

	buckets = [50.0, 60.0, 70.0, 80.0, 90.0]
	# Record from the lower of the histogram floor and the drop threshold, or a
	# threshold below 50 would silently record nothing and still report drops.
	record_at = min(buckets[0], args.jaccard)

	hist_all = {b: 0 for b in buckets}
	hist_cross_batch = {b: 0 for b in buckets}
	hist_cross_type = {b: 0 for b in buckets}
	pairs = {}

	for i in range(n):
		ti = toks[i]
		for j in range(i + 1, n):
			s = jaccard_similarity(ti, toks[j])
			if s < record_at:
				continue
			for b in buckets:
				if s >= b:
					hist_all[b] += 1
					if batches[i] != batches[j]:
						hist_cross_batch[b] += 1
					if qtypes[i] != qtypes[j]:
						hist_cross_type[b] += 1
			if s >= args.jaccard:
				pairs.setdefault(j, []).append((i, s))

	print('Near-duplicate question pairs, cumulative at each threshold:')
	for b in buckets:
		print('  >= {:>4.0f}%: {:>7} total, {:>7} cross-batch, {:>7} cross-type'.format(
			b, hist_all[b], hist_cross_batch[b], hist_cross_type[b]))

	near_rows = []
	if args.drop_scope == 'none':
		print('Near-duplicates dropped: 0 (-drop_scope none)')
	else:
		drop_idx = set()
		kept_pos = []
		for j in range(n):
			hit = None
			for (i, s) in pairs.get(j, []):
				if i in drop_idx:
					continue
				if not args.cross_type and qtypes[i] != qtypes[j]:
					continue
				if args.drop_scope == 'all' or batches[i] != batches[j]:
					hit = (i, s)
					break
			if hit is None:
				kept_pos.append(j)
			else:
				drop_idx.add(j)
				rec = dict(kept_df.iloc[j])
				rec['_Drop_Reason'] = 'near_dup_jaccard'
				rec['_Matched_QID'] = kept_df.iloc[hit[0]]['QID']
				rec['_Matched_Batch'] = batches[hit[0]]
				rec['_Similarity'] = round(hit[1], 2)
				near_rows.append(rec)
		print('Near-duplicates dropped at >= {:.0f}% (scope {}, cross_type {}): {}'.format(
			args.jaccard, args.drop_scope, args.cross_type, len(drop_idx)))
		kept_df = kept_df.iloc[sorted(kept_pos)].reset_index(drop=True)

	print('FINAL: {} rows in, {} rows out'.format(n_in, len(kept_df)))
	print(kept_df.groupby(['Batch', 'QA_Type']).size().to_string())

	if args.report_only:
		print('-report_only set, nothing written.')
		return

	out_file = args.out_file or os.path.join(args.out_dir, 'combined_corpus.tsv')
	kept_df.to_csv(out_file, sep='\t', index=False)

	# Prove the write round-trips before anything downstream reads it.
	rt = pd.read_csv(out_file, sep='\t', dtype=str, keep_default_na=False)
	lhs = kept_df.reset_index(drop=True).astype(str)
	if len(rt) != len(lhs) or list(rt.columns) != list(lhs.columns):
		sys.exit('ERROR: round-trip shape mismatch on ' + out_file)
	if not rt.reset_index(drop=True).equals(lhs):
		sys.exit('ERROR: round-trip content mismatch on ' + out_file + '. Do not proceed.')
	print('Wrote {} ({} rows, round-trip verified)'.format(out_file, len(rt)))

	dropped = []
	if len(exact_rows):
		dropped.append(exact_rows)
	if near_rows:
		dropped.append(pd.DataFrame(near_rows))
	if dropped:
		dpath = os.path.splitext(out_file)[0] + '_dropped.tsv'
		pd.concat(dropped, ignore_index=True).to_csv(dpath, sep='\t', index=False)
		print('Wrote {}'.format(dpath))


if __name__ == '__main__':
	main()