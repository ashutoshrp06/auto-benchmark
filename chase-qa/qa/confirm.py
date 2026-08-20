"""Reproduce every figure the Section 5.7, 5.6.1 and 3.9 drafts rely on.

The script reads the ratings file, the key, the corpus, the baseline, and the
generation logs, and prints each quantity beside the file it came from. It
computes nothing from recollection and asserts nothing it cannot show.
"""

import collections
import csv
import glob
import json
import os
import re
import sys

lim = sys.maxsize
while True:
    try:
        csv.field_size_limit(lim)
        break
    except OverflowError:
        lim //= 2

HE = 'human_eval'
CORPUS = 'generation_outputs/v9/combined_type2.tsv'
BASELINE = '../../chase-qa-base/qa/baseline_corpus.tsv'


def rows(path):
    return list(csv.DictReader(open(path, newline='', encoding='utf-8'),
                               delimiter='\t'))


def head(section):
    print('\n' + '=' * 70)
    print(section)
    print('=' * 70)


low = lambda v: (v or '').strip().lower()
is_no = lambda v: low(v) in ('n', 'no')
is_yes = lambda v: low(v) in ('y', 'yes')

rat = {r['Item_ID']: r for r in rows(os.path.join(HE, 'ratings.tsv'))}
key = {r['Item_ID']: r for r in rows(os.path.join(HE, 'key.tsv'))}
if set(rat) != set(key):
    raise SystemExit('FAIL ratings and key disagree on item identifiers')
cf = [i for i in sorted(key) if key[i]['Frame'] == 'chase_finance']
bl = [i for i in sorted(key) if key[i]['Frame'] == 'baseline']

head('SOURCE: human_eval/ratings.tsv joined to human_eval/key.tsv')
print('items', len(key), 'chase_finance', len(cf), 'baseline', len(bl))

# ---------------------------------------------------------------- 1
head('1. FAILURE SHAPE SPLIT  (prefix of Reason on governance failures)')
shape = collections.Counter()
unlabelled = []
for i in cf:
    if not is_no(rat[i]['Clause_Governs']):
        continue
    m = re.match(r'\s*"?(domain|instrument|provision)\s*:', rat[i]['Reason'])
    if m:
        shape[m.group(1)] += 1
    else:
        unlabelled.append(i)
print('governance failures', sum(1 for i in cf if is_no(rat[i]['Clause_Governs'])))
print('shape counts', dict(sorted(shape.items())))
print('sum of shapes', sum(shape.values()))
print('unlabelled reasons', unlabelled)
for s in sorted(shape):
    ids = [i for i in cf if is_no(rat[i]['Clause_Governs'])
           and re.match(r'\s*"?' + s + r'\s*:', rat[i]['Reason'])]
    print(f'  {s:<11}', ' '.join(ids))

# ---------------------------------------------------------------- 2
head('2. GOVERNANCE BY PANEL VERDICT  (both readings)')
print(f"{'verdict':<10}{'n':>4}{'governs':>9}{'fails':>7}{'sum check':>11}")
for v in ('PASS', 'FAIL', 'DISAGREE'):
    sub = [i for i in cf if key[i]['Verdict'] == v]
    y = sum(1 for i in sub if is_yes(rat[i]['Clause_Governs']))
    n = sum(1 for i in sub if is_no(rat[i]['Clause_Governs']))
    print(f'{v:<10}{len(sub):>4}{y:>9}{n:>7}{"ok" if y + n == len(sub) else "MISMATCH":>11}')
tot_y = sum(1 for i in cf if is_yes(rat[i]['Clause_Governs']))
tot_n = sum(1 for i in cf if is_no(rat[i]['Clause_Governs']))
print(f"{'TOTAL':<10}{len(cf):>4}{tot_y:>9}{tot_n:>7}")

# ---------------------------------------------------------------- 5
head('5. ITEM-LEVEL IDENTITY OF B AND C FAILURES')
gov_fail = {i for i in cf if is_no(rat[i]['Clause_Governs'])}
tra_fail = {i for i in cf if is_no(rat[i]['Traces_To_Source'])}
print('governance failures', len(gov_fail))
print('tracing failures   ', len(tra_fail))
print('identical item sets:', gov_fail == tra_fail)
print('governance only:', sorted(gov_fail - tra_fail))
print('tracing only   :', sorted(tra_fail - gov_fail))

# ---------------------------------------------------------------- 3
head('3. CLAUSE HEADS ACROSS THE 30 CHASE-FINANCE ITEMS')
corp = {(r['Batch'], r['Root_ID'], r['Question_No']): r for r in rows(CORPUS)}
fam = collections.Counter()
detail = []
for i in cf:
    m = key[i]
    row = corp[(m['Batch'], m['Root_ID'], m['Question_No'])]
    h = row['Reg_Text'].strip().split('\n')[0]
    tag = re.match(r'\s*((?:COBS|MCOB|ICOBS|CONC|SYSC|PRIN)\s*[\d.]+|'
                   r'HMRC[^.,:]*|Pensions Act \d+)', h)
    tag = tag.group(1).strip() if tag else 'OTHER: ' + h[:40]
    fam[tag] += 1
    detail.append((i, tag, 'governs' if is_yes(rat[i]['Clause_Governs']) else 'fails',
                   row['Question'][:70]))
print('clause head counts across the sample:')
for k, v in sorted(fam.items(), key=lambda x: (-x[1], x[0])):
    print(f'  {v:>3}  {k}')
print('\nMCOB items only:')
for i, tag, verd, q in detail:
    if tag.startswith('MCOB'):
        print(f'  {i}  {tag:<12} governance={verd}  q={q}')

# ---------------------------------------------------------------- 6
head('6. FINAL ALLOCATION ON BOTH SIDES  (from key.tsv)')
print('chase_finance by verdict :',
      dict(sorted(collections.Counter(key[i]['Verdict'] for i in cf).items())))
print('baseline by document count:',
      dict(sorted(collections.Counter(int(key[i]['N_Docs']) for i in bl).items())))
print('baseline N_Docs equals N_Ans_Points on every row:',
      all(key[i]['N_Docs'] == key[i]['N_Ans_Points'] for i in bl))

# ---------------------------------------------------------------- 10
head('10. ALLOCATION AS A FUNCTION OF POPULATION, NOT OF THE DRAW')
allrows = rows(BASELINE)


def keycount(raw):
    return len(json.loads(raw))


pop = collections.Counter(keycount(r['Doc_Ans_Points']) for r in allrows)
total = sum(pop.values())
exact = {s: 30 * c / total for s, c in pop.items()}
alloc = {s: int(v) for s, v in exact.items()}
order = sorted(exact, key=lambda s: (-(exact[s] - alloc[s]), s))
for s in order[:30 - sum(alloc.values())]:
    alloc[s] += 1
print('baseline population by answer-point keys:', dict(sorted(pop.items())),
      'total', total)
print('exact shares  :', {k: round(v, 3) for k, v in sorted(exact.items())})
print('allocation    :', dict(sorted(alloc.items())), 'sum', sum(alloc.values()))
drawn = collections.Counter(int(key[i]['N_Docs']) for i in bl)
print('drawn         :', dict(sorted(drawn.items())))
print('allocation matches draw:', dict(alloc) == dict(drawn))
print('seed recorded in the run command: 20260816 (see run history)')

# ---------------------------------------------------------------- 7
head('7. JUDGEMENT WORDING AND ORDER, AS RENDERED')
md = open(os.path.join(HE, 'items.md'), encoding='utf-8').read()
for pat in (r'^A\..*$', r'^B\..*$', r'^C\..*$', r'^This item carries no.*$'):
    found = sorted(set(re.findall(pat, md, re.M)))
    for f in found:
        print(' ', f)
print('items rendered      ', md.count('\n## I'))
print('source clause blocks', md.count('### Source clause'))
blocks = md.split('\n## I')[1:]
order_ok = all(b.index('B. ') < b.index('C. ') for b in blocks if 'B. ' in b)
clause_after_docs = all(b.index('#### Document 1') < b.index('### Source clause')
                        for b in blocks if '### Source clause' in b)
print('B precedes C in every clause item :', order_ok)
print('clause printed after the documents:', clause_after_docs)

# ---------------------------------------------------------------- 8
head('8. WHAT THE ITEM FILE SHOWED  (documents per item against corpus fields)')
items = [json.loads(l) for l in open(os.path.join(HE, 'items.jsonl'),
                                     encoding='utf-8')]
by_id = {it['Item_ID']: it for it in items}
print('CHASE-FINANCE: documents shown against Rel_Docs_List and full Documents field')
for i in cf[:4]:
    m = key[i]
    row = corp[(m['Batch'], m['Root_ID'], m['Question_No'])]
    rel = json.loads(row['Rel_Docs_List'])
    full = row['Documents'].count('\nDocument ') + row['Documents'].startswith('Document ')
    print(f"  {i}  shown={len(by_id[i]['Documents'])}  Rel_Docs_List={len(rel)}  "
          f"Documents_field={full}")
base = {(r['Root_ID'], r['Question_No']): r for r in rows(BASELINE)}
print('BASELINE: documents shown against Rel_Docs_List and full Documents field')
for i in bl[:4]:
    m = key[i]
    row = base[(m['Root_ID'], m['Question_No'])]
    raw = json.loads(row['Rel_Docs_List'])
    while isinstance(raw, str):
        raw = json.loads(raw)
    full = row['Documents'].count('\nDocument ') + row['Documents'].startswith('Document ')
    print(f"  {i}  shown={len(by_id[i]['Documents'])}  Rel_Docs_List={len(raw)}  "
          f"Documents_field={full}")

def decode(raw):
    d = json.loads(raw)
    while isinstance(d, str):
        d = json.loads(d)
    return d

mism = {'chase_finance': [], 'baseline': []}
shown_lt_full = {'chase_finance': 0, 'baseline': 0}
for i in sorted(key):
    m = key[i]
    row = (corp[(m['Batch'], m['Root_ID'], m['Question_No'])]
           if m['Frame'] == 'chase_finance' else base[(m['Root_ID'], m['Question_No'])])
    rel = len(decode(row['Rel_Docs_List']))
    full = row['Documents'].count('\nDocument ') + row['Documents'].startswith('Document ')
    if len(by_id[i]['Documents']) != rel:
        mism[m['Frame']].append(i)
    if rel < full:
        shown_lt_full[m['Frame']] += 1
print('\nALL 60 ITEMS:')
print('  items where shown count differs from Rel_Docs_List:', mism)
print('  items where Rel_Docs_List is smaller than the full Documents field:',
      shown_lt_full, 'of', {'chase_finance': len(cf), 'baseline': len(bl)})
print('  reading: a smaller Rel_Docs_List means adversarial documents were '
      'dropped for that item')

# ---------------------------------------------------------------- 9
head('9. WHAT THE KEY RECORDED AND WHAT THE ITEM FILE WITHHELD')
print('key columns  :', list(next(iter(key.values())).keys()))
print('item fields  :', sorted(items[0].keys()))
structural = set(items[0].keys())
print('structural leak, key fields appearing as item fields:',
      sorted(structural & {'Frame', 'Verdict', 'Pass_Frac', 'No_Majority_Frac',
                           'Seed_Type', 'Grounding_Applicable', 'Batch', 'QID'}))
subs = {w: sum(1 for it in items if w in json.dumps(it))
        for w in ('PASS', 'FAIL', 'DISAGREE', 'chase_finance')}
print('substring hits in item text, free prose not labels:', subs)
print('items carrying clause text:', sum(1 for it in items if it['Reg_Text']))

# ---------------------------------------------------------------- 11
head('11. GROUNDING CHECK VERDICT DENOMINATOR')
sep = '-' * 57
gr = collections.Counter()
for path in glob.glob('generation_outputs/elm-*/reg_pregen_logs.txt'):
    run = os.path.basename(os.path.dirname(path))
    draw = 'v9b' if '-v9b-' in run else ('v9' if '-v9-' in run else 'other')
    typ = next((t for t in ('type1', 'type2', 'type3') if t in run), 'other')
    for block in open(path, encoding='utf-8', errors='replace').read().split(sep):
        m = re.findall(r'Grounded:\s*(\w+)', block)
        if m:
            gr[(draw, typ, m[0].lower())] += 1
print('grounding verdicts by draw, type and value:')
for k in sorted(gr):
    print('  ', k, gr[k])
print('total', sum(gr.values()))
print('NOTE: these count calls including rechecks after a correction, not rows.')

head('12. TOPIC CHECK RUNS')
tp = collections.Counter()
for path in glob.glob('generation_outputs/elm-*/topic_pregen_logs.txt'):
    run = os.path.basename(os.path.dirname(path))
    draw = 'v9b' if '-v9b-' in run else ('v9' if '-v9-' in run else 'other')
    for block in open(path, encoding='utf-8', errors='replace').read().split(sep):
        m = re.findall(r'^On_Topic:\s*(True|False)\s*$', block, re.M)
        if m:
            tp[(draw, m[-1])] += 1
print('topic verdicts by draw:', dict(sorted(tp.items())))
print('total', sum(tp.values()))
