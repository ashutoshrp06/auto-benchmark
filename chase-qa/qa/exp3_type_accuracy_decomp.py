import pandas as pd

pd.set_option('display.width', 160)
pd.set_option('display.max_rows', 200)

df = pd.read_csv('analysis/v9_frame.tsv', sep='\t', dtype=str, keep_default_na=False)
df['Result'] = df['Result'].astype(int)

def acc(sub):
    return sub.groupby(['QA_Type', 'Cond', 'Model'])['Result'].agg(['mean', 'count'])

unf_full = df
reg_full = df[df['Track'] == 'reg']
rp_full = df[(df['Track'] == 'reg') & (df['Verdict'] == 'PASS')]

print("=== UNFILTERED ===")
print(acc(unf_full).to_string())

print("\n=== REG-ONLY ===")
print(acc(reg_full).to_string())

print("\n=== REG+PASS ===")
print(acc(rp_full).to_string())

# Stage decomposition: where does the type2 direction split enter, reg filter or PASS filter
print("\n=== STAGE DECOMPOSITION (reg-filter step vs PASS-filter step) ===")
unf = df.groupby(['QA_Type', 'Cond', 'Model'])['Result'].mean().rename('unfiltered')
regonly = reg_full.groupby(['QA_Type', 'Cond', 'Model'])['Result'].mean().rename('reg_only')
rp = rp_full.groupby(['QA_Type', 'Cond', 'Model'])['Result'].mean().rename('reg_pass')
stage = pd.concat([unf, regonly, rp], axis=1)
stage['reg_step_pp'] = (stage['reg_only'] - stage['unfiltered']) * 100
stage['pass_step_pp'] = (stage['reg_pass'] - stage['reg_only']) * 100
print(stage.to_string())

# Rubric effect: type3 reg+PASS, causal-check (noirrelevant) vs zsb (noirrelevant-zsb), same predictions
print("\n=== RUBRIC EFFECT (type3, reg+PASS, causal-check vs zsb) ===")
t3 = rp_full[rp_full['QA_Type'] == 'type3']
t3_cc = t3[t3['Cond'] == 'noirrelevant'].groupby('Model')['Result'].mean()
t3_zsb = t3[t3['Cond'] == 'noirrelevant-zsb'].groupby('Model')['Result'].mean()
rubric = pd.concat([t3_cc.rename('causal_check'), t3_zsb.rename('zsb')], axis=1)
rubric['delta_pp'] = (rubric['zsb'] - rubric['causal_check']) * 100
print(rubric.to_string())

# Construction effect: does conditioning on reg+PASS lift one type more than others (full lift, unfiltered -> reg+PASS)
print("\n=== CONSTRUCTION EFFECT (lift: unfiltered -> reg+PASS, by type/cond/model) ===")
lift = pd.concat([unf, rp], axis=1)
lift['lift_pp'] = (lift['reg_pass'] - lift['unfiltered']) * 100
print(lift.to_string())

print("\n=== CONSTRUCTION EFFECT SUMMARY (mean lift_pp by type, noirr-family conds only) ===")
lift_noirr = lift.reset_index()
lift_noirr = lift_noirr[lift_noirr['Cond'].isin(['noirrelevant', 'noirrelevant-zsb'])]
print(lift_noirr.groupby('QA_Type')['lift_pp'].agg(['mean', 'std', 'count']).to_string())

# Difficulty residual: after rubric correction (type3 measured under zsb), reg+PASS, noirr-family, across all 4 models where available
print("\n=== DIFFICULTY RESIDUAL (rubric-corrected, reg+PASS, noirr family) ===")
base = rp_full
t1 = base[(base['QA_Type'] == 'type1') & (base['Cond'] == 'noirrelevant')].groupby('Model')['Result'].mean().rename('type1_noirr')
t2 = base[(base['QA_Type'] == 'type2') & (base['Cond'] == 'noirrelevant')].groupby('Model')['Result'].mean().rename('type2_noirr')
t3z = base[(base['QA_Type'] == 'type3') & (base['Cond'] == 'noirrelevant-zsb')].groupby('Model')['Result'].mean().rename('type3_zsb')
resid = pd.concat([t1, t2, t3z], axis=1)
print(resid.to_string())
print("\nrange_pp (max-min per model, rubric-corrected):")
print(((resid.max(axis=1) - resid.min(axis=1)) * 100).round(2).to_string())

# n check per row, since coverage varies by model/type/cond
print("\n=== N CHECK (reg+PASS, noirr family) ===")
n_check = base[base['Cond'].isin(['noirrelevant', 'noirrelevant-zsb'])].groupby(['QA_Type', 'Cond', 'Model'])['Result'].count()
print(n_check.to_string())