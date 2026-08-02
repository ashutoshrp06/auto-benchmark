import pandas as pd, glob, os, re
NEEDLE='valuation charges'
EMPTY=re.compile(r'^\s*-\s*$')
def broken(s):
    ls=s.replace('\\n','\n').split('\n'); e=len(ls)
    while e>0 and (ls[e-1].strip()=='' or EMPTY.match(ls[e-1])): e-=1
    return 'BROKEN' if any(EMPTY.match(x) for x in ls[:e]) else 'clean '
STAGES=[('1 qa','elm-qa-type1-v8-b1-s*/prog_qa.tsv'),
        ('2 qa-mod','elm-qa-type1-v8-b1-s*/prog_qa_modified.tsv'),
        ('  qa-exc','elm-qa-type1-v8-b1-s*/prog_qa_exceptions.tsv'),
        ('3 adv','elm-adv-type1-v8-b1-s*/prog_qa.tsv'),
        ('4 adv-mod','elm-adv-type1-v8-b1-s*/prog_qa_modified.tsv'),
        ('5 adv-ver','elm-adv-type1-v8-b1-s*/prog_qa_modified_verified.tsv'),
        ('  adv-drop','elm-adv-type1-v8-b1-s*/prog_qa_partial_drops.tsv'),
        ('6 docs','elm-docs-type1-v8-b1-s*/programmatic_data.tsv'),
        ('7 docs-mod','elm-docs-type1-v8-b1-s*/programmatic_data_modified.tsv'),
        ('8 docs-ver','elm-docs-type1-v8-b1-s*/programmatic_data_modified_verified.tsv')]
for lbl,pat in STAGES:
    fs=sorted(glob.glob('generation_outputs/'+pat)); n=0; bad=0
    for f in fs:
        try: d=pd.read_csv(f,sep='\t',dtype=str,keep_default_na=False)
        except Exception as e: bad+=1; print(f'{lbl:11s} PARSE FAIL {os.path.basename(f)}: {e}'); continue
        for col in [c for c in d.columns if 'Ans' in c]:
            for v in d.loc[d[col].str.contains(NEEDLE,regex=False),col]:
                n+=1
                if n<=4: print(f'{lbl:11s} {col:14s} [{broken(v)}] {v[:190]!r}')
    print(f'{lbl:11s} files={len(fs)} parse_fail={bad} matches={n}\n')
