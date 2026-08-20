"""
Experiment 1: Does the quality gate remove the hard questions?

Question (Barry): does RocketEval+PoLL filtering select for tractable
questions (PASS easier than FAIL -> corpus is systematically easier than
raw output) or does it do real quality work (FAIL easier than PASS,
gate removes malformed-but-easy items)?

Sections:
  1. Core join: panel verdict (PASS/NOCON/FAIL) x solver accuracy, per
     type, per model, all-corpus and reg-track.
  2. Control 2: FAIL cell size check (n>=50 to trust a cell).
  3. Control 1: type3 held-rubric check, same breakdown under both
     causal-check and zero-shot-basic evaluators.
  4. Discriminative-signal companion: does filtering to reg+PASS widen
     the Gemini-vs-GPT gap vs unfiltered, computed fresh from data
     (not recalled), with an exact McNemar p-value on reg+PASS.
  5. Structural correlates: N_Criteria, question length, doc count,
     Seed_Type composition, by verdict, reg track.
  6. Criterion-level: position-in-checklist fail rate, heuristic
     keyword-tagged category fail rate vs baseline (exploratory, needs
     manual validation before citing in the write-up).

No API calls. Everything reads existing files on disk. ~5 min runtime.
Run from the qa/ repo root.
"""

import pandas as pd, numpy as np, glob, re, sys, json, math

PASS_T = 0.75
NOCON_T = 0.34
MODELS = {'gpt55': 'type{t}-gpt55-v9-noirrelevant-eval',
          'gemini': 'type{t}-gemini31pro-v9-noirrelevant-eval'}
TYPE3_ZSB = {'gpt55': 'type3-gpt55-v9-noirrelevant-zsb-eval',
             'gemini': 'type3-gemini31pro-v9-noirrelevant-zsb-eval'}
FAIL_MIN_N = 50


def parse_list(x):
    if not isinstance(x, str) or not x.strip():
        return np.nan
    try:
        p = json.loads(x)
    except Exception:
        return np.nan
    return len(p) if isinstance(p, list) else np.nan


def mcnemar_exact(b, c):
    """Exact two-sided binomial McNemar test on discordant pairs b, c."""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return float('nan')
    k = min(b, c)
    def binom_cdf_le(k, n):
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = 2 * binom_cdf_le(k, n)
    return min(p, 1.0)


def load_batches(t):
    batches = sorted(d for d in glob.glob(f'generation_outputs/elm-docs-type{t}-v9-b*')
                      if re.fullmatch(rf'generation_outputs/elm-docs-type{t}-v9-b\d+', d))
    assert len(batches) == 4, f'type{t}: expected 4 unsharded batch dirs, found {len(batches)}'
    return batches


def build_verdict_frame(t, batches, verbose=False):
    """Join cleaned corpus + poll_scores + poll_criterion_scores(+judgment text) per type."""
    poll_parts, crit_parts = [], []
    for d in batches:
        clean = pd.read_csv(f'{d}/programmatic_data_modified_verified_cleaned.tsv', sep='\t',
                             dtype=str, keep_default_na=False)
        poll = pd.read_csv(f'{d}/rocketeval/poll_scores.tsv', sep='\t', dtype=str, keep_default_na=False)
        crit = pd.read_csv(f'{d}/rocketeval/poll_criterion_scores.tsv', sep='\t', dtype=str, keep_default_na=False)
        judg = pd.read_csv(f'{d}/rocketeval/judgments.tsv', sep='\t', dtype=str, keep_default_na=False)

        keep = [c for c in ('Root_ID', 'Question_No', 'Pass_Frac', 'No_Majority_Frac', 'N_Criteria') if c in poll.columns]
        m = clean.merge(poll[keep], on=['Root_ID', 'Question_No'], how='left', validate='one_to_one')
        poll_parts.append(m)

        crit_text = judg.drop_duplicates(['Root_ID', 'Question_No', 'Criterion_No'])[
            ['Root_ID', 'Question_No', 'Criterion_No', 'Criterion']]
        c2 = crit.merge(crit_text, on=['Root_ID', 'Question_No', 'Criterion_No'], how='left', validate='one_to_one')
        crit_parts.append(c2)

    v = pd.concat(poll_parts, ignore_index=True)
    crit_all = pd.concat(crit_parts, ignore_index=True)

    if verbose:
        print('cleaned/poll columns:', list(v.columns), file=sys.stderr)
        print('criterion columns:', list(crit_all.columns), file=sys.stderr)

    pf = pd.to_numeric(v.get('Pass_Frac'), errors='coerce')
    nm = pd.to_numeric(v['No_Majority_Frac'], errors='coerce') if 'No_Majority_Frac' in v else pd.Series(0.0, index=v.index)
    cat = pd.Series('FAIL', index=v.index)
    cat[pf >= PASS_T] = 'PASS'
    cat[nm >= NOCON_T] = 'NOCON'
    cat[pf.isna()] = 'MISSING'
    v['Verdict_cat'] = cat
    v['is_reg'] = v.Seed_Type.isin(['reg', 'dynamic_reg'])
    return v, crit_all


def load_result(run_dir, comb):
    r = pd.read_csv(f'outputs/{run_dir}/result.tsv', sep='\t', dtype=str, keep_default_na=False).reset_index(drop=True)
    assert len(r) == len(comb), f'length mismatch {run_dir}: {len(r)} vs {len(comb)}'
    qc = [c for c in ('Question', 'question') if c in r.columns and c in comb.columns]
    if qc:
        assert (r[qc[0]] == comb[qc[0]]).all(), f'ALIGNMENT FAIL {run_dir}'
    return (r.Result == '1').values


# ============================================================
# 1 + 2. CORE JOIN + FAIL CELL SIZE CONTROL
# ============================================================
print('=' * 70)
print('SECTION 1+2: core join, PASS/NOCON/FAIL x model x type, + FAIL cell size control')
print('=' * 70)

verdict_frames = {}
comb_frames = {}
ok_frames = {}  # ok_frames[t][model] = bool array aligned to comb order

for t in (1, 2, 3):
    batches = load_batches(t)
    v, crit_all = build_verdict_frame(t, batches, verbose=(t == 1))
    verdict_frames[t] = (v, crit_all)

    comb = pd.read_csv(f'generation_outputs/v9/combined_type{t}.tsv', sep='\t', dtype=str, keep_default_na=False).reset_index(drop=True)
    j = comb.merge(v[['QID', 'Verdict_cat', 'is_reg']], on='QID', how='left')
    assert len(j) == len(comb)
    comb_frames[t] = j

    reg = j.is_reg.values
    cat = j.Verdict_cat.values
    print(f'\n--- type{t} n={len(j)} reg={reg.sum()} verdicts={pd.Series(cat).value_counts().to_dict()} ---')

    ok_frames[t] = {}
    for name, tmpl in MODELS.items():
        ok = load_result(tmpl.format(t=t), comb)
        ok_frames[t][name] = ok
        for tk, msk in (('all', np.ones(len(j), dtype=bool)), ('reg', reg)):
            row = []
            for c in ('PASS', 'NOCON', 'FAIL'):
                sel = (cat == c) & msk
                n = sel.sum()
                flag = '' if (c != 'FAIL' or n >= FAIL_MIN_N) else '  <-- UNDERPOWERED (n<50)'
                row.append(f'{c} {ok[sel].mean()*100:5.1f}% (n={n:4d}){flag}' if n else f'{c}    n/a')
            print(f'  {name:7s} {tk:4s} ' + '  '.join(row))

# ============================================================
# 2b. GRADIENT SIGNIFICANCE: PASS vs NOCON and NOCON vs FAIL
#     Completes the 3-way gradient claim with real p-values.
#     Reg track only. Holm-corrected across all 12 tests
#     (3 types x 2 models x 2 comparisons).
# ============================================================
print()
print('=' * 70)
print('SECTION 2b: gradient significance, PASS-vs-NOCON and NOCON-vs-FAIL, reg track')
print('=' * 70)

def two_prop_ztest(n1, x1, n2, x2):
    """Two-proportion z-test. x1, x2 are counts correct, n1, n2 are sample sizes."""
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1, p2, float('nan'), float('nan')
    z = (p1 - p2) / se
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p1, p2, z, p_val

gradient_tests = []
for t in (1, 2, 3):
    j = comb_frames[t]
    cat = j.Verdict_cat.values
    reg = j.is_reg.values
    for name in ('gpt55', 'gemini'):
        ok = ok_frames[t][name]
        cells = {}
        for c in ('PASS', 'NOCON', 'FAIL'):
            sel = (cat == c) & reg
            n = int(sel.sum())
            x = int(ok[sel].sum())
            cells[c] = (n, x)
        n_p, x_p = cells['PASS']
        n_n, x_n = cells['NOCON']
        n_f, x_f = cells['FAIL']

        p1, p2, z, p_val = two_prop_ztest(n_p, x_p, n_n, x_n)
        gradient_tests.append((f'type{t}_{name}_PASSvNOCON', p_val, p1, p2, n_p, n_n))

        p1b, p2b, zb, p_valb = two_prop_ztest(n_n, x_n, n_f, x_f)
        gradient_tests.append((f'type{t}_{name}_NOCONvFAIL', p_valb, p1b, p2b, n_n, n_f))

print(f'{"comparison":30s} {"n1":>5s} {"p1":>7s}  {"n2":>5s} {"p2":>7s}  {"diff":>8s}  {"raw p":>8s}')
for label, p_val, p1, p2, n1, n2 in gradient_tests:
    diff = 100 * (p1 - p2)
    print(f'{label:30s} {n1:5d} {p1*100:6.1f}%  {n2:5d} {p2*100:6.1f}%  {diff:+7.1f}pp  {p_val:8.4f}')

print()
print('-- Holm correction across these 12 tests --')
sorted_tests = sorted(gradient_tests, key=lambda x: x[1])
n_tests = len(sorted_tests)
for i, (label, p_val, p1, p2, n1, n2) in enumerate(sorted_tests):
    thresh = 0.05 / (n_tests - i)
    verdict = 'REJECT (significant)' if p_val < thresh else 'fail to reject'
    print(f'{label:30s} p={p_val:.4f}  needs <{thresh:.4f}  {verdict}')


# ============================================================
# 3. CONTROL 1: type3 held-rubric check (causal-check vs zero-shot-basic)
# ============================================================
print()
print('=' * 70)
print('SECTION 3: control 1, type3 PASS/FAIL breakdown under BOTH evaluators')
print('=' * 70)

t = 3
j = comb_frames[3]
comb3 = pd.read_csv('generation_outputs/v9/combined_type3.tsv', sep='\t', dtype=str, keep_default_na=False).reset_index(drop=True)
cat = j.Verdict_cat.values
reg = j.is_reg.values

for name, tmpl in MODELS.items():
    ok_primary = ok_frames[3][name]
    ok_zsb = load_result(TYPE3_ZSB[name], comb3)
    for tk, msk in (('all', np.ones(len(j), dtype=bool)), ('reg', reg)):
        row_p, row_z = [], []
        for c in ('PASS', 'FAIL'):
            sel = (cat == c) & msk
            n = sel.sum()
            row_p.append(f'{c} {ok_primary[sel].mean()*100:5.1f}% (n={n})')
            row_z.append(f'{c} {ok_zsb[sel].mean()*100:5.1f}% (n={n})')
        print(f'  {name:7s} {tk:4s} causal-check: ' + '  '.join(row_p))
        print(f'  {name:7s} {tk:4s} zero-shot-basic: ' + '  '.join(row_z))
        gap_p = 100 * (ok_primary[(cat == "PASS") & msk].mean() - ok_primary[(cat == "FAIL") & msk].mean())
        gap_z = 100 * (ok_zsb[(cat == "PASS") & msk].mean() - ok_zsb[(cat == "FAIL") & msk].mean())
        print(f'  {name:7s} {tk:4s} PASS-FAIL gap: causal-check={gap_p:+.1f}pp  zero-shot-basic={gap_z:+.1f}pp')


# ============================================================
# 4. DISCRIMINATIVE SIGNAL: does filtering widen the gemini-gpt gap?
#    Computed fresh from data, not recalled from memory.
# ============================================================
print()
print('=' * 70)
print('SECTION 4: discriminative signal, unfiltered vs reg+PASS gap (gemini - gpt55)')
print('=' * 70)

for t in (1, 2, 3):
    j = comb_frames[t]
    cat = j.Verdict_cat.values
    reg = j.is_reg.values
    ok_g = ok_frames[t]['gemini']
    ok_p = ok_frames[t]['gpt55']

    acc_g_unfilt, acc_p_unfilt = ok_g.mean(), ok_p.mean()
    gap_unfilt = 100 * (acc_g_unfilt - acc_p_unfilt)

    sel = (cat == 'PASS') & reg
    acc_g_filt, acc_p_filt = ok_g[sel].mean(), ok_p[sel].mean()
    gap_filt = 100 * (acc_g_filt - acc_p_filt)

    b = int((ok_g[sel] & ~ok_p[sel]).sum())   # gemini right, gpt wrong
    c = int((~ok_g[sel] & ok_p[sel]).sum())   # gpt right, gemini wrong
    p_val = mcnemar_exact(b, c)

    print(f'\ntype{t}: unfiltered gap={gap_unfilt:+.2f}pp (n={len(j)})   '
          f'reg+PASS gap={gap_filt:+.2f}pp (n={sel.sum()})   widened_by={gap_filt-gap_unfilt:+.2f}pp')
    print(f'  reg+PASS McNemar: discordant b={b} c={c}  p={p_val:.4f}')


# ============================================================
# 5. STRUCTURAL CORRELATES
# ============================================================
print()
print('=' * 70)
print('SECTION 5: structural correlates by verdict, reg track')
print('=' * 70)

for t in (1, 2, 3):
    v, crit_all = verdict_frames[t]
    reg = v.is_reg
    print(f'\n===== TYPE {t} =====')
    ncrit = pd.to_numeric(v.get('N_Criteria'), errors='coerce')
    qlen = v['Question'].str.split().str.len() if 'Question' in v.columns else None
    doc_count = v['Rel_Docs_List'].apply(parse_list) if 'Rel_Docs_List' in v.columns else None

    for name, series in (('N_Criteria', ncrit), ('Question word count', qlen), ('Doc count (Rel_Docs_List)', doc_count)):
        if series is None:
            print(f'  {name}: column not found, skipped')
            continue
        for c in ('PASS', 'NOCON', 'FAIL'):
            sel = (v['Verdict_cat'] == c).values & reg.values
            if sel.sum():
                print(f'  {name:28s} {c:5s} mean={series[sel].mean():.2f}  n={sel.sum()}')

    print('  Seed_Type composition by verdict (full corpus, not reg-filtered):')
    ct = pd.crosstab(v['Verdict_cat'], v['Seed_Type'], normalize='index') * 100
    print(ct.round(1).to_string())


# ============================================================
# 6. CRITERION-LEVEL: position + heuristic category tagging
#    (exploratory -- validate manually before citing category %s)
# ============================================================
print()
print('=' * 70)
print('SECTION 6: criterion-level position and heuristic category fail rates (reg track)')
print('=' * 70)

for t in (1, 2, 3):
    v, crit_all = verdict_frames[t]
    cn = pd.to_numeric(crit_all['Criterion_No'], errors='coerce')
    consensus_fail = (crit_all['Consensus'] == 'No')
    reg_crit = crit_all.Seed_Type.isin(['reg', 'dynamic_reg'])
    max_pos = int(cn[reg_crit].max())

    print(f'\n===== TYPE {t} =====')
    print(f'-- criterion position fail rate, position 1..{max_pos} --')
    for pos in range(1, max_pos + 1):
        sel = (cn == pos).values & reg_crit.values
        if sel.sum():
            print(f'  pos {pos}: fail_rate={consensus_fail[sel].mean()*100:5.1f}%  n={sel.sum()}')

    baseline_sel = reg_crit.values
    baseline = consensus_fail[baseline_sel].mean() * 100
    print(f'-- baseline overall consensus fail rate = {baseline:.1f}% (n={baseline_sel.sum()}) --')

    text = crit_all['Criterion'].fillna('').str.lower()
    tags = {
        'numeric_precision': text.str.contains(r'exact|precise|specific (?:figure|number|amount|percentage|rate)|£|%|numeric'),
        'citation_specificity': text.str.contains(r'cite|citation|section|clause|rule number|cobs|handbook|reference the'),
        'causal_reasoning': text.str.contains(r'because|causal|reason|why|explain the (?:link|connection|relationship)|therefore'),
        'coverage_completeness': text.str.contains(r'all (?:relevant|required|necessary)|cover|complete|comprehensive|every'),
        'conclusion_gate': text.str.contains(r'conclusion|final (?:answer|point|bullet)'),
    }
    print('-- heuristic category fail rate vs baseline (NOT mutually exclusive, validate manually before citing) --')
    for name, mask in tags.items():
        sel = mask.values & reg_crit.values
        if sel.sum() >= 20:
            fr = consensus_fail[sel].mean() * 100
            print(f'  {name:22s} fail_rate={fr:5.1f}%  gap={fr-baseline:+5.1f}pp  n={sel.sum()}')
        else:
            print(f'  {name:22s} n={sel.sum()} too small, skipped')
    other = ~np.logical_or.reduce(list(tags.values()))
    sel = other & reg_crit.values
    print(f'  {"untagged/other":22s} fail_rate={consensus_fail[sel].mean()*100:5.1f}%  n={sel.sum()}')

print()
print('=' * 70)
print('DONE. Sections 1-4 are the core experiment + controls. Sections 5-6 are')
print('supporting structural evidence for the write-up, section 6 category tags')
print('need manual validation before citing as anything beyond a lead.')
print('=' * 70)
# Section 5: PASS-vs-FAIL two-proportion z-test, reg track, noirrelevant cond, per type/model
# Independent samples (disjoint question sets), NOT McNemar (no pairing).
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("analysis/v9_frame.tsv", sep="\t")
mask = (df["Track"] == "reg") & (df["Cond"] == "noirrelevant") & (df["Verdict"].isin(["PASS", "FAIL"]))
sub = df[mask]

print("SECTION 5: PASS-vs-FAIL two-proportion z-test, reg track, noirrelevant cond")
results = []
for qa_type in ["type1", "type2", "type3"]:
    for model in ["gpt55", "gemini31pro"]:
        cell = sub[(sub["QA_Type"] == qa_type) & (sub["Model"] == model)]
        p = cell[cell["Verdict"] == "PASS"]["Result"]
        f = cell[cell["Verdict"] == "FAIL"]["Result"]
        count = [p.sum(), f.sum()]
        nobs = [len(p), len(f)]
        stat, pval = proportions_ztest(count, nobs)
        results.append((qa_type, model, len(p), p.mean(), len(f), f.mean(), pval))
        print(f"{qa_type}_{model}_PASSvFAIL  n_pass={len(p)} pass_acc={p.mean():.4f}  "
              f"n_fail={len(f)} fail_acc={f.mean():.4f}  p={pval:.5f}")

pvals = [r[6] for r in results]
reject, corrected, _, _ = multipletests(pvals, method="holm")
print("\n-- Holm correction across these 6 tests --")
for r, rej, corr in sorted(zip(results, reject, corrected), key=lambda x: x[0][6]):
    qa_type, model, n_p, acc_p, n_f, acc_f, pval = r
    tag = "REJECT (significant)" if rej else "fail to reject"
    print(f"{qa_type}_{model}_PASSvFAIL  raw_p={pval:.5f}  holm_corrected_p={corr:.5f}  {tag}")
