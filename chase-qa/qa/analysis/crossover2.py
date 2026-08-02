import pandas as pd, numpy as np, glob, re, os
from scipy.stats import binomtest
EMPTY=re.compile(r'^\s*-\s*$'); rng=np.random.default_rng(0)
def real(a): return sum(1 for x in a.replace('\\n','\n').split('\n') if x.strip().startswith('-') and not EMPTY.match(x))
for t in (1,2,3):
    recs=[]
    for d in sorted(glob.glob(f'generation_outputs/elm-docs-type{t}-v8-b*-s*')):
        fm,fv=f'{d}/programmatic_data_modified.tsv',f'{d}/programmatic_data_modified_verified.tsv'
        if not (os.path.exists(fm) and os.path.exists(fv)): continue
        m=pd.read_csv(fm,sep='\t',dtype=str,keep_default_na=False)
        v=pd.read_csv(fv,sep='\t',dtype=str,keep_default_na=False)
        need={'Root_ID','Question_No','Question','Answer'}
        assert need<=set(m.columns), f'{fm} cols={list(m.columns)}'
        k=['Root_ID','Question_No']
        if m.duplicated(k).any() or v.duplicated(k).any(): print(f'  DUP KEYS {d}'); continue
        j=m[k+['Question','Answer']].merge(v[k+['Answer']],on=k,suffixes=('_p','_v'),validate='one_to_one')
        j['_lost']=j.Answer_p.map(real)-j.Answer_v.map(real)
        recs.append(j[['Question','_lost']])
    r=pd.concat(recs,ignore_index=True)
    amb=r.groupby('Question')._lost.nunique()
    bad=set(amb[amb>1].index)
    lut=r[~r.Question.isin(bad)].drop_duplicates('Question').set_index('Question')._lost
    j=pd.read_csv(f'analysis/type{t}_v8.tsv',sep='\t',dtype=str,keep_default_na=False)
    qc=[c for c in ('Question','question') if c in j.columns][0]
    j['_lost']=j[qc].map(lut)
    base=(j.Seed_Type.isin(['reg','dynamic_reg'])&(j.Verdict3=='PASS')).values
    print(f'--- type{t}: stage rows={len(r)} ambiguous-text keys dropped={len(bad)} | matched {int(j._lost.notna().sum())}/{len(j)} | reg+PASS unmatched={int((base&j._lost.isna().values).sum())}')
    intact=(j._lost==0).values
    for lbl,msk in (('PASS all',base),('PASS matched',base&j._lost.notna().values),('PASS intact',base&intact)):
        if msk.sum()<30: print(f'  {lbl}: n={int(msk.sum())} too small'); continue
        a=(j.ok_gpt55=='True').values[msk]; g=(j.ok_gemini=='True').values[msk]; ro=j.Root_ID.values[msk]
        b=int((a&~g).sum()); c=int((~a&g).sum()); p=binomtest(b,b+c,0.5).pvalue if b+c else float('nan')
        d=g.astype(int)-a.astype(int)
        grp=[np.where(ro==x)[0] for x in pd.unique(ro)]; K=len(grp)
        bs=np.array([d[np.concatenate([grp[k] for k in rng.integers(0,K,K)])].mean() for _ in range(2000)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        print(f'  {lbl:12s} n={int(msk.sum()):4d} | gpt {a.mean()*100:.1f}% gem {g.mean()*100:.1f}% | gem-gpt {d.mean()*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}] | p={p:.4g}')
