import pandas as pd, glob, re, os
EMPTY=re.compile(r'^\s*-\s*$')
def slots(a):
    b=[x for x in a.replace('\\n','\n').split('\n') if x.strip().startswith('-')]
    return len(b), sum(1 for x in b if EMPTY.match(x))
for t in (1,2,3):
    rows=[]; miss=0; fail=0; dropped=0; total=0; dirs=0
    for d in sorted(glob.glob(f'generation_outputs/elm-docs-type{t}-v8-b*-s*')):
        fm,fv=f'{d}/programmatic_data_modified.tsv',f'{d}/programmatic_data_modified_verified.tsv'
        if not (os.path.exists(fm) and os.path.exists(fv)): miss+=1; continue
        try:
            m=pd.read_csv(fm,sep='\t',dtype=str,keep_default_na=False)
            v=pd.read_csv(fv,sep='\t',dtype=str,keep_default_na=False)
        except Exception as e: fail+=1; print(f'  PARSE FAIL {d}: {e}'); continue
        k=['Root_ID','Question_No']
        if not (set(k+['Answer'])<=set(m.columns)&set(v.columns)): fail+=1; print(f'  COLS {d}'); continue
        if m.duplicated(k).any() or v.duplicated(k).any(): fail+=1; print(f'  DUP KEYS {d}'); continue
        dirs+=1; total+=len(m); dropped+=len(m)-len(v)
        j=m[k+['Answer']].merge(v[k+['Answer']],on=k,suffixes=('_pre','_post'),validate='one_to_one')
        for _,r in j.iterrows():
            bp,ep=slots(r.Answer_pre); ba,ea=slots(r.Answer_post)
            rows.append((bp-ep,ba-ea,ep))
    print(f'--- type{t}  dirs={dirs} missing={miss} failed={fail}')
    if not rows: print('  no pairs\n'); continue
    df=pd.DataFrame(rows,columns=['pre','post','pre_empty'])
    df['lost']=df.pre-df.post
    print(f'  rows in: {total} | dropped whole by verification: {dropped} ({dropped/total*100:.1f}%) | surviving pairs: {len(df)}')
    print(f'  pre-existing empties before stage 8: {int((df.pre_empty>0).sum())}')
    print(f'  surviving rows losing >=1 point: {int((df.lost>0).sum())} ({(df.lost>0).mean()*100:.1f}%)')
    ok=df.pre>0
    print(f'  surviving rows losing >=half:    {int((df.lost[ok]>=df.pre[ok]/2).sum())} ({(df.lost[ok]>=df.pre[ok]/2).mean()*100:.1f}%)')
    print(f'  mean points: pre={df.pre.mean():.2f} post={df.post.mean():.2f}')
    print(df.lost.value_counts().sort_index().head(8).to_string()+'\n')
