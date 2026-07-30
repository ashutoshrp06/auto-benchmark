#!/bin/bash
# Sharded RocketEval: checklist generation + judge panel, then a single
# unsharded PoLL aggregation over the merged judgements.
#
#   ./run_rocketeval.sh -f elm-docs-type1-v7 -q type1 -m gpt-5.4-mini -s 8
#
# Shard folders <folder>-rNN are created, run, merged, then removed only if
# --clean is passed. poll_aggregate.py is cross-question and runs once.

set -euo pipefail

FOLDER=""
QA_TYPE=""
MODEL="gpt-5.4-mini"
MODEL_TYPE="elm"
OUT_DIR="generation_outputs/"
DATA="programmatic_data_modified_verified_cleaned"
SHARDS=8
MAX_PARALLEL=0
UNSURE="exclude"
PASS_FRAC="0.8"
DISAGREE_FRAC="0.34"
STAGE="all"
CLEAN=0

usage() {
	cat <<EOF
Usage: $0 -f <folder_name> -q <qa_type> [options]

Required:
  -f, --folder          Run folder, e.g. elm-docs-type1-v7
  -q, --qa-type         type1 | type2 | type3

Options:
  -m, --model           Checklist model. Default: gpt-5.4-mini
  -t, --model-type      Default: elm
  -s, --shards          Default: 8
      --max-parallel    Max concurrent shards. Default: 0 (all)
      --data            Data filename stem. Default: $DATA
      --unsure          exclude | zero | half. Default: exclude
      --pass-frac       Default: 0.8
      --disagree-frac   Default: 0.34
      --stage           all | run | aggregate. Default: all
      --clean           Remove shard folders after a successful merge
EOF
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-f|--folder) FOLDER="$2"; shift 2 ;;
		-q|--qa-type) QA_TYPE="$2"; shift 2 ;;
		-m|--model) MODEL="$2"; shift 2 ;;
		-t|--model-type) MODEL_TYPE="$2"; shift 2 ;;
		-s|--shards) SHARDS="$2"; shift 2 ;;
		--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
		--data) DATA="$2"; shift 2 ;;
		--unsure) UNSURE="$2"; shift 2 ;;
		--pass-frac) PASS_FRAC="$2"; shift 2 ;;
		--disagree-frac) DISAGREE_FRAC="$2"; shift 2 ;;
		--stage) STAGE="$2"; shift 2 ;;
		--clean) CLEAN=1; shift ;;
		-h|--help) usage ;;
		*) echo "Unknown option: $1"; usage ;;
	esac
done

[[ -z "$FOLDER" ]] && { echo "ERROR: -f/--folder is required"; usage; }
[[ -z "$QA_TYPE" ]] && { echo "ERROR: -q/--qa-type is required"; usage; }
if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [[ "$SHARDS" -lt 1 ]]; then
	echo "ERROR: --shards must be a positive integer"; exit 1
fi
case "$STAGE" in all|run|aggregate) ;; *) echo "ERROR: bad --stage"; exit 1 ;; esac
[[ -f "${OUT_DIR}${FOLDER}/${DATA}.tsv" ]] || { echo "ERROR: no such file: ${OUT_DIR}${FOLDER}/${DATA}.tsv"; exit 1; }

LOG_DIR="${OUT_DIR}_rklogs-${FOLDER}"
mkdir -p "$LOG_DIR"

echo "Folder:   $FOLDER"
echo "QA type:  $QA_TYPE"
echo "Model:    $MODEL ($MODEL_TYPE)"
echo "Shards:   $SHARDS"
echo "Stage:    $STAGE"

shard_one() {
	local sfx="$1"
	python checklist_generator.py -out_dir "$OUT_DIR" -folder_name "${FOLDER}${sfx}" \
		-data "$DATA" -qa_type "$QA_TYPE" -model_type "$MODEL_TYPE" -model "$MODEL" || exit 1
	python judge_panel.py -out_dir "$OUT_DIR" -folder_name "${FOLDER}${sfx}" -data "$DATA" || exit 2
}

if [[ "$STAGE" == "all" || "$STAGE" == "run" ]]; then
	echo ""
	echo "--- splitting folder ---"
	python shard_folder.py -out_dir "$OUT_DIR" -folder_name "$FOLDER" -data "$DATA" -shards "$SHARDS"

	echo ""
	echo "--- checklists + judge panel ---"
	PIDS=(); NAMES=(); FAILED=()
	for s in $(seq 1 "$SHARDS"); do
		sfx=$(printf -- "-r%02d" "$s")
		[[ -d "${OUT_DIR}${FOLDER}${sfx}" ]] || continue
		if [[ "$MAX_PARALLEL" -gt 0 ]]; then
			while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do sleep 5; done
		fi
		shard_one "$sfx" < /dev/null > "${LOG_DIR}/rk${sfx}.log" 2>&1 &
		PIDS+=($!); NAMES+=("$sfx")
		echo "  launched ${sfx} (pid $!)"
	done
	[[ ${#PIDS[@]} -eq 0 ]] && { echo "ERROR: no shard folders found"; exit 1; }
	for i in "${!PIDS[@]}"; do
		rc=0; wait "${PIDS[$i]}" || rc=$?
		if [[ "$rc" -eq 0 ]]; then echo "  ${NAMES[$i]}: OK"
		else echo "  ${NAMES[$i]}: FAILED (stage $rc)"; FAILED+=("${NAMES[$i]}"); fi
	done
	if [[ ${#FAILED[@]} -gt 0 ]]; then
		echo ""
		echo "FAILURES: ${FAILED[*]}"
		echo "Logs in ${LOG_DIR}/. Both scripts resume, so rerun the same command."
		exit 1
	fi

	echo ""
	echo "--- merging judgements ---"
	python merge_rocketeval.py -out_dir "$OUT_DIR" -folder_name "$FOLDER" -data "$DATA"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "aggregate" ]]; then
	echo ""
	echo "--- PoLL aggregate (unsharded, cross-question) ---"
	python poll_aggregate.py -out_dir "$OUT_DIR" -folder_name "$FOLDER" \
		-unsure "$UNSURE" -pass_frac "$PASS_FRAC" -disagree_frac "$DISAGREE_FRAC" < /dev/null
fi

if [[ "$CLEAN" -eq 1 && "$STAGE" != "aggregate" ]]; then
	echo ""
	echo "--- removing shard folders ---"
	rm -rf "${OUT_DIR}${FOLDER}"-r[0-9][0-9]
fi

echo ""
echo "Done: $FOLDER"
echo "  verdicts: ${OUT_DIR}${FOLDER}/rocketeval/poll_scores.tsv"
echo "  summary:  ${OUT_DIR}${FOLDER}/rocketeval/poll_summary.txt"