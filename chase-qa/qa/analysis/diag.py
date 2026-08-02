import pandas as pd, re
EMPTY=re.compile(r'^\s*-\s*$')
def inter(a):
    ls=a.replace('\\n','\n').split('\n'); e=len(ls)
    while e>0 and (ls[e-1].strip()=='' or EMPTY.match(ls[e-1])): e-=1
    return sum(1 for x in ls[:e] if EMPTY.match(x))
for t in (1,2,3):
    cl=pd.concat([pd.read_csv(f'generation_outputs/elm-docs-type{t}-{b}/programmatic_data_modified_verified_cleaned.tsv',
                  sep='\t',dtype=str,keep_default_na=False) for b in ('v8-b1','v8-b2','v8-b3')],ignore_index=True)
    b=cl[cl.Answer.apply(inter)>0]
    print(f'--- type{t}: {len(b)} broken of {len(cl)}')
    for c in ('Ans_Points','Doc_Ans_Points','Adv_Ans_Pts'):
        if c in cl.columns: print(f'  {c}: also empty-bulleted in {int(b[c].apply(inter).gt(0).sum())}/{len(b)}')
    if len(b):
        r=b.iloc[0]
        print(f'  sample QID={r.QID} root={r.Root_ID} q={r.Question_No}')
        print(f'  Answer     : {r.Answer[:400]!r}')
        print(f'  Ans_Points : {r.Ans_Points[:400]!r}')
