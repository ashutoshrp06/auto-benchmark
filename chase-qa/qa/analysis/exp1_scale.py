import pandas as pd, re
EMPTY=re.compile(r'^\s*-\s*$'); BULLET=re.compile(r'^\s*-\s+\S')
B=('v8-b1','v8-b2','v8-b3')

def shape(a):
    ls=a.replace('\\n','\n').split('\n')
    e=len(ls)
    while e>0 and (ls[e-1].strip()=='' or EMPTY.match(ls[e-1])): e-=1
    core=ls[:e]
    return pd.Series({'interior':sum(1 for x in core if EMPTY.match(x)),
                      'trailing':sum(1 for x in ls[e:] if EMPTY.match(x)),
                      'bullets':sum(1 for x in core if BULLET.match(x))})

def pct(mask,ok):
    return f'{ok[mask].mean()*100:.1f}% (n={int(mask.sum())})' if mask.sum() else 'n/a'

for t in (1,2,3):
    cl=pd.concat([pd.read_csv(f'generation_outputs/elm-docs-type{t}-{b}/programmatic_data_modified_verified_cleaned.tsv',
                              sep='\t',dtype=str,keep_default_na=False) for b in B],ignore_index=True)
    assert 'Answer' in cl.columns and cl.QID.is_unique
    j=pd.read_csv(f'analysis/type{t}_v8.tsv',sep='\t',dtype=str,keep_default_na=False)
    assert j.QID.is_unique
    for c in ('ok_gpt55','ok_gemini'):
        assert set(j[c].unique())<={'True','False'}, f'{c} domain={set(j[c].unique())}'
    ans=cl[['QID','Answer']] if 'Answer' not in j.columns else cl[['QID']].assign(Answer=None)
    j=j.merge(cl[['QID','Answer']].rename(columns={'Answer':'_ans'}),on='QID',how='left',validate='one_to_one')
    assert j._ans.notna().all(), f'type{t}: QID join incomplete'
    j=pd.concat([j,j._ans.apply(shape)],axis=1)
    j['_lit']=j._ans.str.contains(r'\\n',regex=True)
    j['_broken']=j.interior>0
    s=j[j.Seed_Type.isin(['reg','dynamic_reg'])]
    print(f'--- type{t} reg n={len(s)}')
    print(f"  literal \\n rows:        {int(s._lit.sum())} ({s._lit.mean()*100:.1f}%)")
    print(f"  interior empty bullets: {int(s._broken.sum())} ({s._broken.mean()*100:.1f}%)   <- content loss")
    print(f"  trailing empty only:    {int(((s.trailing>0)&~s._broken).sum())}   <- likely benign")
    print('  bullets by verdict:'); print(s.groupby('Verdict3').bullets.agg(['mean','median','count']).round(2).to_string())
    print('  verdict % by broken:');  print((pd.crosstab(s._broken,s.Verdict3,normalize='index')*100).round(1).to_string())
    for m in ('ok_gpt55','ok_gemini'):
        ok=s[m]=='True'
        print(f'  {m}: broken {pct(s._broken,ok)} | clean {pct(~s._broken,ok)}')
