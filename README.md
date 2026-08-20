# auto-benchmark

CHASE-Finance adapts [CHASE-QA](https://github.com/McGill-NLP/CHASE) (McGill-NLP/Mila) into a pipeline that generates multi-hop question-answering benchmarks for UK personal financial advice, grounded in the FCA Handbook and HMRC guidance, entirely by LLM, with no human writing any of the questions.

Ground truth is fixed before any supporting document exists: the pipeline writes the answer first, builds a document around it, then checks that the document still says what it was supposed to say. A verifier checks each document against its answer, a two-judge panel scores every question against a per-question RocketEval checklist, and a grounding check tests each document for non-contradiction with its source clause. Three question types come out the other end: single-document factual retrieval, multi-document compositional reasoning, and causal reasoning.

This is an MSc dissertation project (University of Edinburgh), built on top of a vendored, unmodified copy of the upstream CHASE-QA repository.

## Layout

```
auto-benchmark/
├── chase-qa-base/    Vendored upstream McGill-NLP/CHASE repo (git subtree).
│                     qa/ carries two small patches and a baseline corpus,
│                     see "CHASE-QA baseline generation" below.
└── chase-qa/         Working copy. CHASE-Finance lives in qa/; code/ and
                      math/ are the untouched upstream CHASE-Code and
                      CHASE-Math domains.
    ├── qa/           CHASE-Finance pipeline, evaluation, and analysis
    ├── code/         (upstream CHASE-Code, untouched)
    ├── math/         (upstream CHASE-Math, untouched)
    └── requirements.txt
```

`chase-qa/README.md` and `chase-qa/qa/README.md` are the original upstream CHASE-QA documentation and describe the generic (non-Finance) pipeline. This file documents the CHASE-Finance extension that actually runs in `qa/`.

Fastest way in: skim [The final run](#the-final-run) to see the concrete commands that produced the shipped corpus, then use [Pipeline](#pipeline-chase-qaqa) as the reference for running any stage on its own.

## CHASE-QA baseline generation

`chase-qa-base/qa/generator.py` and `chase-qa-base/qa/models.py` carry two small patches on top of the vendored upstream, needed to actually run the vanilla CHASE-QA pipeline against the ELM proxy rather than a locally hosted vLLM model: the `vllm` import in `models.py` is wrapped in a `try`/`except` so its absence doesn't hard-fail the module, and `_get_chat_response` retries with the offending parameter stripped or renamed (`max_tokens` to `max_completion_tokens`) on an OpenAI `unsupported_parameter` error. `generator.py`'s `programmatic_scenario_generation` also had its five hardcoded example scenario rows emptied out, since scenario generation here reads from an external `annotated_scenarios.txt` instead.

That file, `chase-qa-base/qa/annotated_scenarios.txt`, holds the 15-persona seed pool used to produce the CHASE-QA ablation baseline. `runs/shard1`, `runs/shard2`, and `runs/shard3` each hold the persona subset used for that shard; `runs/smoketest` is an identical copy of `shard1`, used as a single-shard dry run before committing to all three. `chase-qa-base/qa/baseline_corpus.tsv` is the resulting corpus: 145 rows, split 53/40/52 across the three shards (`_s1`/`_s2`/`_s3` suffixes on `Root_ID`), confirmed by reading the file directly rather than trusting the row count in any log.

## Dependencies and install

```shell
git clone https://github.com/ashutoshrp06/auto-benchmark.git
cd auto-benchmark/chase-qa
python -m venv chasenv && source chasenv/bin/activate
pip install -r requirements.txt
```

Everything CHASE-Finance runs on lives in `qa/` and its `requirements.txt` is the one at `chase-qa/requirements.txt`, inherited unmodified from upstream CHASE-QA. It pulls in the full upstream dependency set, including GPU/vLLM packages (torch, vllm, the `nvidia-*` CUDA wheels) used in the original paper for local model hosting. CHASE-Finance never hosts a model locally: generation and judging run entirely through `-model_type elm` or `-model_type gemini`, so `pip install -r requirements.txt` on a CPU-only machine works but pulls weight it does not need. The packages actually imported on the CHASE-Finance path, checked against every top-level import in `qa/*.py`: `pandas`, `numpy`, `scipy`, `statsmodels`, `openai` (the ELM proxy is OpenAI-compatible, so the same client handles both), `anthropic`, `tiktoken`, `tenacity`, and both `google-generativeai` and `google-genai`.

That last point is a real gap, not a style choice. `requirements.txt` pins `google-generativeai==0.8.0`, the deprecated SDK, and every script that talks to Gemini still calls its `genai.configure(...)` for the API key side effect. But `models.py`, the module that actually makes the Gemini call, imports the new SDK unconditionally at the top of the file (`from google import genai`), and `google-genai` is not in `requirements.txt` at all. A fresh `pip install -r requirements.txt` will not give you a working install: anything that imports `models.py`, which is most of the pipeline, fails on `ModuleNotFoundError` until `google-genai` is installed separately:

```shell
pip install google-genai
```

`python-dotenv` is also pinned in `requirements.txt` but never imported anywhere under `qa/`; it is not on the CHASE-Finance path.

Model access. Generation and judging models are called via Edinburgh's ELM proxy (`-model_type elm`) or the Gemini API directly (`-model_type gemini`). Set whichever of these a given run needs:

```shell
export ELM_API_KEY=...
export ELM_BASE_URL=...
export GEMINI_API_KEY=...
```

`run_pipeline.sh`, `run_batches.sh`, and `run_eval.sh` all check for the relevant key(s) before doing any work and fail fast if one is missing.

## The final run

Every headline number in the dissertation traces back to one corpus: the CHASE-Finance final run, named `v9` on disk and in every command below. 2,296 rows (714 type1, 770 type2, 812 type3) across 4 batches, reg-track total 1,001 against a target of 900. "Reg-track" means questions seeded from an actual FCA/HMRC clause (`Seed_Type` `reg` or its stage-1-expanded form `dynamic_reg`), as opposed to the generic track (`generic`/`dynamic_generic`), which isn't tied to a specific regulatory clause; `--target-track reg` below is what makes the batch loop count only the regulatory questions toward the target. Here's exactly how it was built, reconstructed from `qa/v9.log`, which `run_batches.sh` writes with its own parameter echo at the top of the file:

```shell
cd chase-qa/qa
./run_batches.sh --stem v9 -m gpt-5.4-mini --scenario-iters 20 \
                  --target-total 900 --target-track reg --max-batches 6 \
                  --shards 16 2>&1 | tee v9.log
```

The loop stopped itself after batch 4 (cumulative reg count 1,001, over target), so batches 5 and 6 never ran despite `--max-batches 6`. RocketEval judging ran sharded at 16 as well, confirmed from the `-- RocketEval type1 across 16 shard(s)` lines in the log; `--re-shards` was left at its default of 0, which falls back to `--shards`. `pass_frac 0.75` and `disagree_frac 0.34` are the script's own defaults and were not overridden.

Output: `generation_outputs/v9/combined_corpus.tsv`, then split by `split_combined.py` into `generation_outputs/v9/combined_type{1,2,3}.tsv` (714 / 770 / 812 rows, each confirmed round-trip verified in the log).

`v9b.log` and `v9b_resume.log` record a second run, same target and track, run in the background with a wider flag set than the final run's header alone shows. The literal command:

```shell
cd chase-qa/qa
nohup ./run_batches.sh --stem v9b -m gpt-5.4-mini \
    --scenario-iters 20 --max-batches 6 \
    --target-total 900 --target-track reg \
    --shards 16 --max-parallel 0 --re-shards 16 \
    --min-scenarios 3 --cleanup-seed 42 \
    --combined generation_outputs/v9b/combined_corpus.tsv \
    > v9b.log 2>&1 &

sleep 20; tail -5 v9b.log; pgrep -f run_batches >/dev/null && echo ALIVE || echo DEAD
```

`--min-scenarios 3` is far below the script's own default of 16, so a batch that only expanded 3 scenarios would not abort. `--cleanup-seed 42` makes `cleanup.py` deterministic across reruns, unlike the final run, which left cleanup unseeded (`-1`, the default). Used for the significance replication in `exp6_v9b_significance.py`. `v9b_resume.log` shows the script's resume behaviour on a later invocation of the same command: batch 1 was already complete, so generation and judging were both skipped and only dedup and the cumulative count re-ran.

For the final run itself, only the parameters visible in the log's own header (Stem/Model/Target/Track/Batches/Types/Shards) are recoverable from `v9.log` directly, since `run_batches.sh` does not echo every flag it was called with. Whether the final run also ran under `nohup` in the background, and whether it used non-default `--min-scenarios`/`--cleanup-seed`/`--re-shards` the way v9b did, is not something the log content confirms either way.

Solving and scoring: one confirmed `run_eval.sh` invocation, type2 against gpt-5.4-mini with the no-irrelevant-context solver prompt, scored by the standard zero-shot-basic evaluator:

```shell
./run_eval.sh -i generation_outputs/v9/combined_type2.tsv \
    -r type2-gpt54mini-v9-noirrelevant -m gpt-5.4-mini -t elm \
    -p zero-shot-no-irrelevant -e zero-shot-basic -s 16
```

`zero-shot-no-irrelevant` is a real solver prompt (`prompts.py`), and `eval_outputs/` already carries a `-noirrelevant-solve`/`-noirrelevant-judge` naming convention from earlier runs, so this fits the project's established pattern rather than being a one-off. I don't have equivalent confirmed commands for the other three solver models (gpt-5.5, gemini-3.1-pro-preview, gemini-3.1-flash-lite) or for types 1 and 3 against the final run - those would follow the same `-i .../combined_type{N}.tsv -m <model> -p <prompt> -e <eval prompt>` shape, with `-e type3-causal-check` for type 3, but I'm not stating exact run names for combinations I haven't seen.

## Pipeline (`chase-qa/qa/`)

Ten stages, run end to end or individually. Everything from a seed clause to a verified, documented question goes through `run_pipeline.sh`.

### Single run: `run_pipeline.sh`

Runs the full generation pipeline for one batch, stages 1–10, with optional sharding across stages 2–9:

```shell
./run_pipeline.sh -n v7 -m gpt-5.4-mini
./run_pipeline.sh -n v7 -m gpt-5.4-mini --shards 12 --max-parallel 6
```

| Stage | What happens |
|---|---|
| 1 | Scenario generation (seeded, optionally self-expanded via `--scenario-iters`) |
| 2 | QA generation, per type (1/2/3), via `generator.py` |
| 3 | Post-process QA output into compatible format |
| 4 | Adversarial QA generation |
| 5 | Post-process adversarial output |
| 6 | Adversarial verification |
| 7 | Document generation |
| 8 | Post-process documents |
| 9 | Document verification |
| 9b | Merge shards, renumber positional IDs (sharded runs only) |
| 10 | `cleanup.py` - cross-row, unsharded, always runs after merge |

Stage 1 and stage 10 always run unsharded; stage 10 is cross-row so it cannot be sharded. The script prints the RocketEval commands to run next (`checklist_generator.py`, `judge_panel.py`, `poll_aggregate.py`) once it finishes.

Run a subset of stages with `--from`/`--to`, e.g. to redo document generation and verification only:

```shell
./run_pipeline.sh -n v7 -m gpt-5.4-mini --from 7 --to 9
```

Each stage is also a standalone script and can be called directly, which is what `run_pipeline.sh` does under the hood, for cases where the wrapper's shape does not fit (rerunning one stage on hand-edited input, debugging a single call). The exact invocation per stage, unsharded, single type:

```shell
# Stage 1 - scenarios
python generator.py -exp_type programmatic_scenarios -prompt_type programmatic_scenarios \
    -model_type elm -model gpt-5.4-mini -num_iters 20 -run_name elm-scenarios-v7

# Stage 2 - QA generation, per type (programmatic_qa_type1/2/3)
python generator.py -exp_type programmatic_qa -prompt_type programmatic_qa_type1 \
    -scenarios_name elm-scenarios-v7 -model_type elm -model gpt-5.4-mini \
    -num_iters 1 -run_name elm-qa-type1-v7

# Stage 3 - post-process QA
python post_process.py -exp_type programmatic_qa -folder_name elm-qa-type1-v7 -data prog_qa

# Stage 4 - adversarial QA
python generator.py -exp_type programmatic_adversarial -prompt_type programmatic_adversarial \
    -questions_name elm-qa-type1-v7 -model_type elm -model gpt-5.4-mini \
    -num_iters 3 -run_name elm-adv-type1-v7

# Stage 5 - post-process adversarial
python post_process.py -exp_type programmatic_adversarial -folder_name elm-adv-type1-v7 -data prog_qa

# Stage 6 - adversarial verify
python verification.py -exp_type programmatic_adversarial -folder_name elm-adv-type1-v7 \
    -data prog_qa_modified -model_type elm -model gpt-5.4-mini

# Stage 7 - document generation
python generator.py -exp_type programmatic_docs -prompt_type programmatic_docs \
    -adversarial_name elm-adv-type1-v7 -model_type elm -model gpt-5.4-mini \
    -run_name elm-docs-type1-v7

# Stage 8 - post-process documents
python post_process.py -exp_type programmatic_docs -folder_name elm-docs-type1-v7 -data programmatic_data

# Stage 9 - document verify
python verification.py -exp_type programmatic_docs -folder_name elm-docs-type1-v7 \
    -data programmatic_data_modified -model_type elm -model gpt-5.4-mini

# Stage 10 - cleanup (cross-row, run once per type after all types finish stage 9)
python cleanup.py -out_dir generation_outputs/elm-docs-type1-v7/ \
    -og_data programmatic_data_modified_verified -seed_data programmatic_data_modified_verified \
    -qa_type type1 -seed -1 -threshold 0
```

Type 3 swaps `programmatic_qa_type3` / `programmatic_adversarial_type3` / `programmatic_docs_type3` prompt types in stages 2, 4, and 9 - see the per-type branches in `run_pipeline.sh`.

### Corpus scale-up: `run_batches.sh`

Loops `run_pipeline.sh` across batches until a target corpus size is reached, running the RocketEval judge chain and cross-batch dedup after each batch:

```shell
./run_batches.sh --stem v8 -m gpt-5.4-mini --scenario-iters 40 \
                  --target-total 900 --target-track reg --max-batches 8
```

Each batch is self-contained (`<stem>-b1`, `<stem>-b2`, ...); nothing from an earlier batch is regenerated or re-judged. The script is resumable: batches whose cleaned corpora already exist are skipped, as is RocketEval judging for any type whose `poll_scores.tsv` already exists. `dedup_batches.py` merges completed batches into a combined corpus and drops near-duplicates (Jaccard threshold, default 60); `split_combined.py` then splits the result by question type for `run_eval.sh`.

`run_pipeline.sh` itself is never modified by this loop - it is treated as the reproducibility record for whichever run produced the shipped corpus.

### Solving and evaluation: `run_eval.sh`

Sharded solver + evaluator for one (corpus, model, solver-prompt) combination:

```shell
./run_eval.sh -i generation_outputs/elm-docs-type1-v7/programmatic_data_modified_verified_cleaned.tsv \
              -r solver-type1-gpt55-v7-basic -m gpt-5.5 -t elm \
              -p zero-shot-basic -e zero-shot-basic -s 8
```

Type 3 corpora require `-e type3-causal-check`. Both `solver.py` and `evaluator.py` resume from their own output, so rerunning the same command after a failure continues rather than restarting. Only the `MERGED TOTAL` line in `outputs/<run>-eval/eval_logs.txt` is the correct accuracy; per-shard figures cover that shard alone.

### RocketEval judging

Three scripts, run per corpus folder after generation:

- `checklist_generator.py` - builds a per-question grading checklist (criteria count per type: type1 6–8, type2 7–9, type3 7–10).
- `judge_panel.py` - runs the two-judge PoLL panel (`gpt-5.4-mini`, `gemini-3.5-flash-lite`) against each checklist. Output is append-only; its output file must be deleted before any rerun or rows duplicate.
- `poll_aggregate.py` - aggregates panel verdicts into PASS / DISAGREE / FAIL per question, with configurable `-pass_frac` (default 0.75) and `-disagree_frac` (default 0.34) thresholds, and configurable handling of Unsure votes (`-unsure exclude|zero|half`).

## Experiments

Every claim in Chapter 5 has a script behind it. Run the corresponding one and you get the exact number that's in the dissertation, not an approximation of it.

All experiment scripts run from `chase-qa/qa/` and read committed or already-generated TSVs (`analysis/v9_frame.tsv`, `generation_outputs/v9/...`, `generation_outputs/elm-docs-type{1,2,3}-v9-b*/...`), so they need the CHASE-Finance final run's corpus and solve/eval outputs to already exist. None of them take a `run_pipeline.sh`-style flag set; most are plain scripts with paths hardcoded to the final run, run with no arguments:

```shell
cd chase-qa/qa

python exp1_quality_gate.py           # does the RocketEval+PoLL gate select for
                                       # tractable questions or do real quality work
python exp1_deep.py                   # deeper follow-up on exp1, asserts exactly
                                       # 4 v9 batch dirs exist per type
python exp2_threshold_sensitivity.py  # pass_frac x unsure_mode sweep. Also runs the
                                       # McNemar + cluster-bootstrap significance test
                                       # (gemini31pro vs gpt55) at every grid point and
                                       # writes analysis/exp2b_significance_grid.tsv.
                                       # RUN_BOOTSTRAP=True by default inside the file,
                                       # ~5.5 min ceiling; set it False in the source
                                       # for a McNemar-only fast pass first
python analysis/exp2b_holm_correction.py  # reads exp2b_significance_grid.tsv,
                                           # applies Holm-Bonferroni within each
                                           # (unsure_mode, pass_frac) 3-type family
python exp3_type_accuracy_decomp.py   # accuracy by QA_Type x Cond x Model,
                                       # unfiltered / reg-only / reg+PASS
python exp6_v9b_significance.py       # McNemar exact + cluster bootstrap for the
                                       # gemini31pro vs gpt55 crossover, on v9b
```

`exp2b_significance_grid.tsv` and `exp2b_holm_grid.tsv` are both already committed under `analysis/`, so `exp2b_holm_correction.py` works out of the box without rerunning the sweep. Rerun `exp2_threshold_sensitivity.py` only to regenerate the grid from scratch.

Three scripts take explicit arguments:

```shell
# exp4 - judge self-preference (RQ4b). Run --inspect-solver first to see
# valid --solver-model / --solver-cond values before --solver-check.
python exp4_judge_selfpref.py --inspect-solver
python exp4_judge_selfpref.py --solver-check \
    --solver-model gpt-5.5 --solver-cond zero-shot-basic

# exp7 - single-judge counterfactual over stored panel verdicts
python exp7_single_judge.py generation_outputs/elm-docs-type1-v9-b1/rocketeval/judgments.tsv \
    generation_outputs/elm-docs-type1-v9-b2/rocketeval/judgments.tsv [...]

# t47 - question-level self-preference, author judge vs independent judge
python t47_question_level.py --seed-type reg,dynamic_reg \
    generation_outputs/elm-docs-type1-v9-b1/rocketeval/judgments.tsv [...]
```

`analysis/` also holds standalone diagnostics used ad hoc rather than as part of a numbered experiment (`crossover2.py`, `diag.py`, `trace.py`, `verstrip.py`) - each reads its own hardcoded input path; check the top of the file before running.

## Human evaluation

```shell
cd chase-qa/qa
python build_human_eval.py      # builds the blinded 60-item rating set (§5.7):
                                 # 30 CHASE-Finance regulatory-track Type 2 +
                                 # 30 unmodified CHASE-QA baseline, interleaved
                                 # and shuffled, key kept separate from the
                                 # rating file
python analyse_human_eval.py    # joins ratings to the key, reports answer
                                 # correctness (n=60), clause governance (n=30),
                                 # answer tracing (n=30) - no pooling across
                                 # the 3 denominators
```

Join key throughout the analysis layer: `(Batch, Root_ID, Question_No)`. `QID` (content hash, from `cleanup.py`/`dedup_batches.py`) is the key for cross-batch identity, since `Root_ID`/`Question_No` restart from 1 in every batch and are not globally unique.

`test_synthetic.py` exercises `build_human_eval.py` end to end against fixtures in `synthetic/` (a fake corpus, judgment set, and baseline matching the real schemas), checking that the happy path samples the right strata and that every guard actually halts the run. Run it with `python test_synthetic.py` to confirm `build_human_eval.py` still works before spending real API calls on a fresh sample.

The pipeline description above makes it easy to assume the grounding check settles the matter: document written, checked against its clause, done. The human evaluation tested that directly, by hand, on the 30 CHASE-Finance items. Clause governance held on 10 of 30. Answer correctness held on all 60. Both automated gates, topic adherence and grounding, are sound on their own terms: they catch what they're built to catch. Neither one establishes that the clause actually bears on the question being asked, and that's a separate property from non-contradiction, which is the one this evaluation was built to test.

## Data

- `qa/reg_clauses.json` - regulatory seed clauses (FCA Handbook, HMRC).
- `qa/generic_seeds.json` - non-regulatory generic seeds.
- `qa/data/chase_qa.tsv` - original CHASE-QA released corpus (baseline).
- `qa/generation_outputs/` - per-run pipeline output, one directory per `(prefix, type, run_name)` combination.
- `qa/outputs/` - solver and evaluator output.

## License

`chase-qa/LICENSE` is the upstream CHASE-QA MIT license, carried over with the fork. There is no separate license file for the CHASE-Finance additions in `qa/`; this repository is a dissertation project, not a released package.

## Citation

CHASE-Finance builds on:

```
@misc{patel2025llmgeneratechallengingproblems,
      title={How to Get Your LLM to Generate Challenging Problems for Evaluation},
      author={Arkil Patel and Siva Reddy and Dzmitry Bahdanau},
      year={2025},
      eprint={2502.14678},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14678},
}
```