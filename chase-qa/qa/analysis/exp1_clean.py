import pandas as pd, numpy as np, re
from scipy.stats import binomtest
EMPTY=re.compile(r'^\s*-\s*$'); rng=np.random.default_rng(0); B=('v8-b1','v8-b2','v8-b3')
def interior(a):
    ls=a.replace('\\n','\n').split('\n'); e=len(ls)
    while e>0 and (ls[e-1].strip()=='' or EMPTY.match(ls[e-1])): e-=1
    return sum(1 for x in ls[:e] if EMPTY.match(x))
for t in (1,2,3):
    cl=pd.concat([pd.read_csv(f'generation_outputs/elm-docs-type{t}-{b}/programmatic_data_modified_verified_cleaned.tsv',
                  sep='\t',dtype=str,keep_default_na=False) for b in B],ignore_index=True)
    assert cl.QID.is_unique
    j=pd.read_csv(f'analysis/type{t}_v8.tsv',sep='\t',dtype=str,keep_default_na=False)
    j=j.merge(cl[['QID','Answer']].rename(columns={'Answer':'_ans'}),on='QID',how='left',validate='one_to_one')
    assert j._ans.notna().all()
    j['_broken']=j._ans.apply(interior)>0
    base=j.Seed_Type.isin(['reg','dynamic_reg'])&(j.Verdict3=='PASS')
    for lbl,msk in (('PASS all',base.values),('PASS clean',(base&~j._broken).values)):
        a=(j.ok_gpt55=='True').values[msk]; g=(j.ok_gemini=='True').values[msk]; ro=j.Root_ID.values[msk]
        bb=int((a&~g).sum()); cc=int((~a&g).sum())
        p=binomtest(bb,bb+cc,0.5).pvalue if bb+cc else float('nan')
        d=g.astype(int)-a.astype(int)
        grp=[np.where(ro==x)[0] for x in pd.unique(ro)]; K=len(grp)
        bs=np.array([d[np.concatenate([grp[k] for k in rng.integers(0,K,K)])].mean() for _ in range(2000)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        print(f'type{t} {lbl:10s} n={int(msk.sum()):4d} | gpt {a.mean()*100:.1f}% gem {g.mean()*100:.1f}% | gem-gpt {d.mean()*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}] | p={p:.4g}')
