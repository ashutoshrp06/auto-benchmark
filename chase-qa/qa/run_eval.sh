#!/bin/bash
# Sharded solver + evaluator for one (corpus, model, solver-prompt) combination.
#
#   ./run_eval.sh -i generation_outputs/elm-docs-type1-v7/programmatic_data_modified_verified_cleaned.tsv \
#                 -r solver-type1-gpt55-v7-basic -m gpt-5.5 -t elm \
#                 -p zero-shot-basic -e zero-shot-basic -s 8
#
# Type 3 corpora must use -e type3-causal-check.
#
# Merged solver output:    outputs/<run>/predictions.tsv
# Merged evaluator output: outputs/<run>-eval/result.tsv
#
# Only the "=== MERGED TOTAL ===" line in outputs/<run>-eval/eval_logs.txt is the
# correct accuracy. Per-shard eval_logs.txt figures cover that shard alone.
#
# Both solver.py and evaluator.py resume from their own output, so rerunning the
# same command after a failure continues rather than restarting.

set -euo pipefail

INPUT=""
RUN=""
MODEL=""
MODEL_TYPE="elm"
SOLVER_PROMPT="zero-shot-basic"
EVAL_PROMPT="zero-shot-basic"
EVAL_MODEL="gpt-5.4-mini"
EVAL_MODEL_TYPE="elm"
EVAL_SUFFIX="-eval"
SHARDS=8
MAX_PARALLEL=0
SOLVER_MAX_TOKENS=4096
EVAL_MAX_TOKENS=2048
TEMPERATURE=1
STAGE="all"

usage() {
	cat <<EOF
Usage: $0 -i <input_tsv> -r <run_name> -m <model> [options]

Required:
  -i, --input           Cleaned corpus TSV
  -r, --run             Base run name (shards get -sNN appended)
  -m, --model           Solver model

Options:
  -t, --model-type      Solver model type. Default: elm
  -p, --prompt          Solver prompt type. Default: zero-shot-basic
  -e, --eval-prompt     Evaluator prompt type. Default: zero-shot-basic
                        Use type3-causal-check for type 3 corpora.
      --eval-model      Judge model. Default: gpt-5.4-mini
      --eval-model-type Judge model type. Default: elm
      --eval-suffix     Suffix for the evaluator run directory. Default: -eval
                        Use a distinct value per judge, e.g. -eval-gemini, so a
                        second judge over the same predictions does not resume
                        into the first judge's results.
  -s, --shards          Number of shards. Default: 8
      --max-parallel    Max concurrent shards. Default: 0 (all)
      --solver-tokens   Solver max_tokens. Default: 4096
      --eval-tokens     Evaluator max_tokens. Default: 2048
      --temperature     Default: 1
      --stage           all | solve | eval. Default: all
EOF
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-i|--input) INPUT="$2"; shift 2 ;;
		-r|--run) RUN="$2"; shift 2 ;;
		-m|--model) MODEL="$2"; shift 2 ;;
		-t|--model-type) MODEL_TYPE="$2"; shift 2 ;;
		-p|--prompt) SOLVER_PROMPT="$2"; shift 2 ;;
		-e|--eval-prompt) EVAL_PROMPT="$2"; shift 2 ;;
		--eval-model) EVAL_MODEL="$2"; shift 2 ;;
		--eval-model-type) EVAL_MODEL_TYPE="$2"; shift 2 ;;
		--eval-suffix) EVAL_SUFFIX="$2"; shift 2 ;;
		-s|--shards) SHARDS="$2"; shift 2 ;;
		--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
		--solver-tokens) SOLVER_MAX_TOKENS="$2"; shift 2 ;;
		--eval-tokens) EVAL_MAX_TOKENS="$2"; shift 2 ;;
		--temperature) TEMPERATURE="$2"; shift 2 ;;
		--stage) STAGE="$2"; shift 2 ;;
		-h|--help) usage ;;
		*) echo "Unknown option: $1"; usage ;;
	esac
done

[[ -z "$INPUT" ]] && { echo "ERROR: -i/--input is required"; usage; }
[[ -z "$RUN" ]] && { echo "ERROR: -r/--run is required"; usage; }
[[ -z "$MODEL" ]] && { echo "ERROR: -m/--model is required"; usage; }
[[ -f "$INPUT" ]] || { echo "ERROR: no such file: $INPUT"; exit 1; }

if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [[ "$SHARDS" -lt 1 ]]; then
	echo "ERROR: --shards must be a positive integer"; exit 1
fi
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
	echo "ERROR: --max-parallel must be a non-negative integer"; exit 1
fi
case "$STAGE" in
	all|solve|eval) ;;
	*) echo "ERROR: --stage must be all, solve or eval"; exit 1 ;;
esac

if [[ "$MODEL_TYPE" == "elm" || "$EVAL_MODEL_TYPE" == "elm" ]]; then
	[[ -z "${ELM_API_KEY:-}" ]] && { echo "ERROR: ELM_API_KEY is not set"; exit 1; }
	[[ -z "${ELM_BASE_URL:-}" ]] && { echo "ERROR: ELM_BASE_URL is not set"; exit 1; }
fi
if [[ "$MODEL_TYPE" == "gemini" || "$EVAL_MODEL_TYPE" == "gemini" ]]; then
	[[ -z "${GEMINI_API_KEY:-}" ]] && { echo "ERROR: GEMINI_API_KEY is not set"; exit 1; }
fi

SHARD_DIR="outputs/_shards-${RUN}"
LOG_DIR="outputs/_logs-${RUN}"
EVAL_RUN="${RUN}${EVAL_SUFFIX}"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

# Base name of the shard files the current phase reads. Reassigned before the
# eval phase. run_shards() resolves it at call time.
BASE=""

echo "Input:      $INPUT"
echo "Run:        $RUN"
echo "Solver:     $MODEL ($MODEL_TYPE) / $SOLVER_PROMPT"
echo "Judge:      $EVAL_MODEL ($EVAL_MODEL_TYPE) / $EVAL_PROMPT"
echo "Shards:     $SHARDS"
echo "Stage:      $STAGE"

# Warm the tiktoken BPE cache once so concurrent shards do not race on the same
# download. Mirrors the try/except in solver.py and evaluator.py.
python -c "
import sys, tiktoken
try:
    tiktoken.encoding_for_model(sys.argv[1])
except Exception:
    tiktoken.encoding_for_model('gpt-4')
" "$MODEL" >/dev/null 2>&1 || true

# $1 = log label, $2 = function to invoke with the shard suffix.
# Launches one background job per existing shard file and collects exit codes.
run_shards() {
	local label="$1"
	local fn="$2"
	local PIDS=() NAMES=() FAILED=()
	local s sfx i rc

	for s in $(seq 1 "$SHARDS"); do
		sfx=$(printf -- "-s%02d" "$s")
		[[ -f "${SHARD_DIR}/${BASE}${sfx}.tsv" ]] || continue
		if [[ "$MAX_PARALLEL" -gt 0 ]]; then
			while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do sleep 5; done
		fi
		"$fn" "$sfx" < /dev/null > "${LOG_DIR}/${label}${sfx}.log" 2>&1 &
		PIDS+=($!)
		NAMES+=("$sfx")
		echo "  launched ${label}${sfx} (pid $!)"
	done

	if [[ ${#PIDS[@]} -eq 0 ]]; then
		echo "ERROR: no shard files matched ${SHARD_DIR}/${BASE}-sNN.tsv"
		exit 1
	fi

	for i in "${!PIDS[@]}"; do
		rc=0
		wait "${PIDS[$i]}" || rc=$?
		if [[ "$rc" -eq 0 ]]; then
			echo "  ${NAMES[$i]}: OK"
		else
			echo "  ${NAMES[$i]}: FAILED ($rc)"
			FAILED+=("${NAMES[$i]}")
		fi
	done

	if [[ ${#FAILED[@]} -gt 0 ]]; then
		echo ""
		echo "FAILURES: ${FAILED[*]}"
		echo "Logs in ${LOG_DIR}/. Both scripts resume from their own output,"
		echo "so rerunning the same command continues rather than restarting."
		exit 1
	fi
}

# Asserts the merged file has exactly as many rows as the file it was built from.
check_rows() {
	python -c "
import pandas as pd, sys
a = len(pd.read_csv(sys.argv[1], sep='\t'))
b = len(pd.read_csv(sys.argv[2], sep='\t'))
print('  merged {} of {} rows'.format(b, a))
sys.exit(0 if a == b else 1)
" "$1" "$2"
}

solve_one() {
	local sfx="$1"
	python solver.py \
		-input_tsv "${SHARD_DIR}/${BASE}${sfx}.tsv" \
		-prompt_type "$SOLVER_PROMPT" \
		-model_type "$MODEL_TYPE" -model "$MODEL" \
		-max_tokens "$SOLVER_MAX_TOKENS" -temperature "$TEMPERATURE" \
		-run_name "${RUN}${sfx}"
}

eval_one() {
	local sfx="$1"
	python evaluator.py \
		-predictions_tsv "${SHARD_DIR}/${BASE}${sfx}.tsv" \
		-prompt_type "$EVAL_PROMPT" \
		-model_type "$EVAL_MODEL_TYPE" -model "$EVAL_MODEL" \
		-max_tokens "$EVAL_MAX_TOKENS" -temperature "$TEMPERATURE" \
		-run_name "${EVAL_RUN}${sfx}"
}

# ------------------------------------------------------------------ SOLVE
if [[ "$STAGE" == "all" || "$STAGE" == "solve" ]]; then
	BASE=$(basename "$INPUT" .tsv)

	echo ""
	echo "--- splitting corpus ---"
	python shard_tsv.py -input_tsv "$INPUT" -out_dir "$SHARD_DIR" -base "$BASE" -shards "$SHARDS"

	echo ""
	echo "--- solver ---"
	run_shards solver solve_one

	echo ""
	echo "--- merging predictions ---"
	python merge_eval.py -out_dir outputs/ -base_run "$RUN" -shards "$SHARDS" -mode solver -strict
	check_rows "$INPUT" "outputs/${RUN}/predictions.tsv"
fi

# ------------------------------------------------------------------- EVAL
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
	PRED="outputs/${RUN}/predictions.tsv"
	[[ -f "$PRED" ]] || { echo "ERROR: missing $PRED. Run --stage solve first."; exit 1; }

	BASE="predictions-${RUN}"

	echo ""
	echo "--- splitting predictions ---"
	python shard_tsv.py -input_tsv "$PRED" -out_dir "$SHARD_DIR" -base "$BASE" -shards "$SHARDS"

	echo ""
	echo "--- evaluator ---"
	run_shards eval eval_one

	echo ""
	echo "--- merging results ---"
	python merge_eval.py -out_dir outputs/ -base_run "$EVAL_RUN" -shards "$SHARDS" -mode evaluator -strict
	check_rows "$PRED" "outputs/${EVAL_RUN}/result.tsv"
fi

echo ""
echo "=================================================================="
echo "  Done: $RUN"
echo "=================================================================="
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
	echo "  Accuracy: outputs/${EVAL_RUN}/eval_logs.txt (MERGED TOTAL line)"
fi