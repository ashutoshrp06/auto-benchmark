import pandas as pd, numpy as np, glob, re, sys, json

PASS_T=0.75; NOCON_T=0.34

def parse_list(x):
    if not isinstance(x, str) or not x.strip():
        return np.nan
    try:
        p = json.loads(x)
    except Exception:
        return np.nan
    return len(p) if isinstance(p, list) else np.nan

for t in (1,2,3):
    batches = sorted(d for d in glob.glob(f'generation_outputs/elm-docs-type{t}-v9-b*')
                      if re.fullmatch(rf'generation_outputs/elm-docs-type{t}-v9-b\d+', d))
    assert len(batches)==4, f'type{t}: expected 4 batch dirs, got {len(batches)}'

    poll_parts=[]; crit_parts=[]
    for d in batches:
        clean=pd.read_csv(f'{d}/programmatic_data_modified_verified_cleaned.tsv',sep='\t',dtype=str,keep_default_na=False)
        poll=pd.read_csv(f'{d}/rocketeval/poll_scores.tsv',sep='\t',dtype=str,keep_default_na=False)
        crit=pd.read_csv(f'{d}/rocketeval/poll_criterion_scores.tsv',sep='\t',dtype=str,keep_default_na=False)
        judg=pd.read_csv(f'{d}/rocketeval/judgments.tsv',sep='\t',dtype=str,keep_default_na=False)
        keep=[c for c in ('Root_ID','Question_No','Pass_Frac','No_Majority_Frac','N_Criteria') if c in poll.columns]
        m=clean.merge(poll[keep],on=['Root_ID','Question_No'],how='left',validate='one_to_one')
        poll_parts.append(m)
        crit_text = judg.drop_duplicates(['Root_ID','Question_No','Criterion_No'])[['Root_ID','Question_No','Criterion_No','Criterion']]
        c2 = crit.merge(crit_text, on=['Root_ID','Question_No','Criterion_No'], how='left', validate='one_to_one')
        crit_parts.append(c2)

    v = pd.concat(poll_parts, ignore_index=True)
    crit_all = pd.concat(crit_parts, ignore_index=True)

    if t==1:
        print('cleaned/poll columns:', list(v.columns), file=sys.stderr)
        print('criterion columns:', list(crit_all.columns), file=sys.stderr)

    pf=pd.to_numeric(v.get('Pass_Frac'),errors='coerce')
    nm=pd.to_numeric(v['No_Majority_Frac'],errors='coerce') if 'No_Majority_Frac' in v else pd.Series(0.0,index=v.index)
    cat=pd.Series('FAIL',index=v.index)
    cat[pf>=PASS_T]='PASS'; cat[nm>=NOCON_T]='NOCON'; cat[pf.isna()]='MISSING'
    v['Verdict_cat']=cat
    reg = v.Seed_Type.isin(['reg','dynamic_reg'])

    print(f'\n===== TYPE {t} =====')

    # ---------- structural correlates ----------
    ncrit = pd.to_numeric(v.get('N_Criteria'), errors='coerce')
    qlen = v['Question'].str.split().str.len() if 'Question' in v.columns else None
    doc_count = v['Rel_Docs_List'].apply(parse_list) if 'Rel_Docs_List' in v.columns else None

    print('-- structural correlates, reg track --')
    for name, series in (('N_Criteria', ncrit), ('Question word count', qlen), ('Doc count (Rel_Docs_List)', doc_count)):
        if series is None:
            print(f'  {name}: column not found, skipped')
            continue
        for c in ('PASS','NOCON','FAIL'):
            sel = (v['Verdict_cat']==c).values & reg.values
            if sel.sum():
                print(f'  {name:28s} {c:5s} mean={series[sel].mean():.2f}  n={sel.sum()}')

    print('  Seed_Type composition by verdict:')
    ct = pd.crosstab(v['Verdict_cat'], v['Seed_Type'], normalize='index')*100
    print(ct.round(1).to_string())

    # ---------- criterion position ----------
    cn = pd.to_numeric(crit_all['Criterion_No'], errors='coerce')
    consensus_fail = (crit_all['Consensus']=='No')
    reg_crit = crit_all.Seed_Type.isin(['reg','dynamic_reg'])
    max_pos = int(cn[reg_crit].max())
    print(f'-- criterion position fail rate (reg track, Consensus==No), position 1..{max_pos} --')
    for pos in range(1, max_pos+1):
        sel = (cn==pos).values & reg_crit.values
        if sel.sum():
            print(f'  pos {pos}: fail_rate={consensus_fail[sel].mean()*100:5.1f}%  n={sel.sum()}')

    # ---------- heuristic keyword tagging (approximate, validate manually before citing) ----------
    text = crit_all['Criterion'].fillna('').str.lower()
    tags = {
        'numeric_precision': text.str.contains(r'exact|precise|specific (?:figure|number|amount|percentage|rate)|£|%|numeric'),
        'citation_specificity': text.str.contains(r'cite|citation|section|clause|rule number|cobs|handbook|reference the'),
        'causal_reasoning': text.str.contains(r'because|causal|reason|why|explain the (?:link|connection|relationship)|therefore'),
        'coverage_completeness': text.str.contains(r'all (?:relevant|required|necessary)|cover|complete|comprehensive|every'),
        'conclusion_gate': text.str.contains(r'conclusion|final (?:answer|point|bullet)'),
    }
    baseline_sel = reg_crit.values
    print(f'-- baseline: overall consensus fail rate, reg track = {consensus_fail[baseline_sel].mean()*100:.1f}% (n={baseline_sel.sum()}) --')
    print('-- heuristic criterion-category fail rate vs baseline (reg track) --')
    for name, mask in tags.items():
        sel = mask.values & reg_crit.values
        if sel.sum() >= 20:
            print(f'  {name:22s} fail_rate={consensus_fail[sel].mean()*100:5.1f}%  n={sel.sum()}')
        else:
            print(f'  {name:22s} n={sel.sum()} too small, skipped')
    other = ~np.logical_or.reduce(list(tags.values()))
    sel = other & reg_crit.values
    print(f'  {"untagged/other":22s} fail_rate={consensus_fail[sel].mean()*100:5.1f}%  n={sel.sum()}')