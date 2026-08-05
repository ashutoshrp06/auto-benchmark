"""
exp6_v9b_significance.py
McNemar exact + cluster bootstrap (clustered by Root_ID) for the
gemini31pro vs gpt55 crossover, replicated on v9b.

result.tsv only has a flat 1..N positional 'ID', not Root_ID. Root_ID/
Question_No are recovered by joining back to the source corpus file on
sorted-ID position, verified zero-mismatch on Question text for all
three types before this script was written.
"""
import pandas as pd
import numpy as np
import math

np.random.seed(42)
N_BOOT = 10000

def mcnemar_exact(b, c):
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return float('nan')
    k = min(b, c)
    def binom_cdf_le(k, n):
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(2 * binom_cdf_le(k, n), 1.0)

def cluster_bootstrap_diff(df, correct_a, correct_b, cluster_col, n_boot=N_BOOT):
    clusters = df[cluster_col].unique()
    diffs = np.empty(n_boot)
    a = df[correct_a].values
    b = df[correct_b].values
    clus = df[cluster_col].values
    idx_by_cluster = {c: np.where(clus == c)[0] for c in clusters}
    for i in range(n_boot):
        sampled = np.random.choice(clusters, size=len(clusters), replace=True)
        idxs = np.concatenate([idx_by_cluster[c] for c in sampled])
        diffs[i] = a[idxs].mean() - b[idxs].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi

def load_result_with_keys(result_path, corpus_path):
    res = pd.read_csv(result_path, sep='\t', dtype=str, keep_default_na=False)
    corpus = pd.read_csv(corpus_path, sep='\t', dtype=str, keep_default_na=False)

    if len(res) != len(corpus):
        raise ValueError(f"{result_path}: row count mismatch vs {corpus_path} "
                          f"({len(res)} vs {len(corpus)})")

    r = pd.to_numeric(res['Result'], errors='coerce')
    n_null = r.isna().sum()
    if n_null:
        raise ValueError(f"{result_path}: {n_null} unparseable Result value(s)")
    res['Correct'] = r.astype(bool)

    res['ID_int'] = res['ID'].astype(int)
    res = res.sort_values('ID_int').reset_index(drop=True)

    mismatches = (res['Question'].str.strip() != corpus['Question'].str.strip()).sum()
    if mismatches:
        raise ValueError(f"{result_path}: {mismatches} Question mismatches at "
                          f"matched position, positional join unsafe")

    res['Root_ID'] = corpus['Root_ID'].values
    res['Question_No'] = corpus['Question_No'].values
    return res

for t in [1, 2, 3]:
    corpus_path = f'generation_outputs/v9b/combined_type{t}.tsv'
    pg = load_result_with_keys(f'outputs/type{t}-gemini31pro-v9b-noirrelevant-eval/result.tsv', corpus_path)
    pp = load_result_with_keys(f'outputs/type{t}-gpt55-v9b-noirrelevant-eval/result.tsv', corpus_path)

    m = pg.merge(pp, on=['Root_ID', 'Question_No'], suffixes=('_gem', '_gpt'), validate='one_to_one')
    if len(m) != len(pg) or len(m) != len(pp):
        print(f"type{t}: WARNING merge row count mismatch, "
              f"gem={len(pg)} gpt={len(pp)} merged={len(m)}")
        continue

    b = int((m['Correct_gem'] & ~m['Correct_gpt']).sum())
    c = int((~m['Correct_gem'] & m['Correct_gpt']).sum())
    p = mcnemar_exact(b, c)

    diff_mean, lo, hi = cluster_bootstrap_diff(m, 'Correct_gem', 'Correct_gpt', 'Root_ID')

    acc_gem = m['Correct_gem'].mean()
    acc_gpt = m['Correct_gpt'].mean()

    print(f"type{t}: n={len(m)}  gem={acc_gem:.4f}  gpt={acc_gpt:.4f}  "
          f"diff={acc_gem-acc_gpt:+.4f}  b={b} c={c}  "
          f"McNemar p={p:.4f}  cluster95=[{lo:+.4f}, {hi:+.4f}]")