#!/bin/bash
# Batch loop for CHASE-Finance corpus scale-up.
#
#   ./run_batches.sh --stem v8 -m gpt-5.4-mini --scenario-iters 40 \
#                    --target-total 900 --target-track reg --max-batches 8
#
# Each batch is a self-contained run of run_pipeline.sh stages 1-10 under its own
# run name (<stem>-b1, <stem>-b2, ...), followed by that batch's RocketEval chain.
# Nothing from an earlier batch is regenerated or re-judged. Batches meet only in
# dedup_batches.py and in the cumulative count.
#
# The loop stops when the cumulative post-dedup count reaches --target-total, or
# at --max-batches, whichever comes first. Rerunning the same command resumes:
# any batch whose cleaned corpora already exist is not regenerated, and any type
# whose poll_scores.tsv already exists is not re-judged.
#
# run_pipeline.sh is not modified. It is the artefact that produced v7 and is the
# reproducibility record.

set -euo pipefail

STEM=""
MODEL=""
MODEL_TYPE="elm"
PREFIX="elm"
TARGET_TOTAL=900
TARGET_TRACK="reg"
MAX_BATCHES=8
SCENARIO_ITERS=0
MIN_SCENARIOS=16
QA_ITERS=1
ADV_ITERS=3
SHARDS=8
MAX_PARALLEL=0
TEMPERATURE=1
TYPES="1 2 3"
PASS_FRAC=0.75
DISAGREE_FRAC=0.34
JACCARD=60
RE_SHARDS=0
RE_SUFFIX="-r"
CLEANUP_SEED=-1
DROP_SCOPE="cross_batch"
COMBINED="generation_outputs/combined_corpus.tsv"
SEED_BATCH="first"

usage() {
	cat <<EOF
Usage: $0 --stem <name> -m <model> --scenario-iters <n> [options]

Required:
      --stem            Run name stem. Batches are <stem>-b1, <stem>-b2, ...
  -m, --model           Generation model
      --scenario-iters  Stage 1 self-expansion iterations per batch. Must be > 0
                        when --max-batches > 1, otherwise every batch would
                        generate from the same ten seeds and produce duplicates.

Options:
  -t, --model-type      Default: elm
  -p, --prefix          Directory prefix. Default: elm
      --target-total    Stop when the cumulative count reaches this. Default: 900
      --target-track    reg | generic | all | cleaned. Default: reg
                        reg counts Seed_Type reg and dynamic_reg at RocketEval PASS.
                        cleaned counts every cleaned row with no PASS filter.
      --max-batches     Hard stop. Default: 8
      --min-scenarios   Abort a batch if stage 1 added fewer EXPANDED scenarios
                        than this. Counts Seed_Type dynamic_reg and
                        dynamic_generic only, so seeded rows never inflate it.
                        Default: 16.
      --qa-iters        Passed to run_pipeline.sh. Default: 1
      --adv-iters       Default: 3
      --shards          Default: 8
      --max-parallel    Default: 0 (all)
      --temperature     Default: 1
      --types           Space-separated, quoted. Default: "1 2 3"
      --pass-frac       RocketEval PASS threshold. Default: 0.75
      --disagree-frac   poll_aggregate.py DISAGREE threshold. Default: 0.34
      --re-shards       Shards for the RocketEval chain per type. 0 means use
                        --shards. 1 disables sharding. checklist_generator.py
                        and judge_panel.py are sequential, so this is the main
                        lever on judging wall clock. Default: 0
      --re-suffix       shard_folder.py -suffix. Default: -r
      --cleanup-seed    run_pipeline.sh --cleanup-seed. -1 leaves cleanup
                        unseeded, as before. Set it once for the whole run so
                        every batch is cleaned the same way. Default: -1
      --jaccard         dedup_batches.py near-duplicate threshold. Default: 60
      --drop-scope      none | cross_batch | all. Default: cross_batch
      --seed-batch      first | none. Which batch emits the seeded scenarios.
                        first: batch 1 runs -emit_scope all, later batches run
                        new, so the seeds are never regenerated. none: every
                        batch runs new, giving a purely expanded corpus.
                        Default: first
      --combined        Combined corpus path, passed to dedup_batches.py -out_file.
                        Default: generation_outputs/combined_corpus.tsv
EOF
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--stem) STEM="$2"; shift 2 ;;
		-m|--model) MODEL="$2"; shift 2 ;;
		-t|--model-type) MODEL_TYPE="$2"; shift 2 ;;
		-p|--prefix) PREFIX="$2"; shift 2 ;;
		--target-total) TARGET_TOTAL="$2"; shift 2 ;;
		--target-track) TARGET_TRACK="$2"; shift 2 ;;
		--max-batches) MAX_BATCHES="$2"; shift 2 ;;
		--min-scenarios) MIN_SCENARIOS="$2"; shift 2 ;;
		--scenario-iters) SCENARIO_ITERS="$2"; shift 2 ;;
		--qa-iters) QA_ITERS="$2"; shift 2 ;;
		--adv-iters) ADV_ITERS="$2"; shift 2 ;;
		--shards) SHARDS="$2"; shift 2 ;;
		--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
		--temperature) TEMPERATURE="$2"; shift 2 ;;
		--types) TYPES="$2"; shift 2 ;;
		--pass-frac) PASS_FRAC="$2"; shift 2 ;;
		--disagree-frac) DISAGREE_FRAC="$2"; shift 2 ;;
		--re-shards) RE_SHARDS="$2"; shift 2 ;;
		--re-suffix) RE_SUFFIX="$2"; shift 2 ;;
		--cleanup-seed) CLEANUP_SEED="$2"; shift 2 ;;
		--jaccard) JACCARD="$2"; shift 2 ;;
		--drop-scope) DROP_SCOPE="$2"; shift 2 ;;
		--combined) COMBINED="$2"; shift 2 ;;
		--seed-batch) SEED_BATCH="$2"; shift 2 ;;
		-h|--help) usage ;;
		*) echo "Unknown option: $1"; usage ;;
	esac
done

[[ -z "$STEM" ]] && { echo "ERROR: --stem is required"; usage; }
[[ -z "$MODEL" ]] && { echo "ERROR: -m/--model is required"; usage; }
[[ -f ./run_pipeline.sh ]] || { echo "ERROR: ./run_pipeline.sh not found in $(pwd)"; exit 1; }
for f in count_pass.py dedup_batches.py split_combined.py shard_folder.py merge_rocketeval.py checklist_generator.py judge_panel.py poll_aggregate.py; do
	[[ -f "$f" ]] || { echo "ERROR: $f not found in $(pwd)"; exit 1; }
done

for v in "$TARGET_TOTAL" "$MAX_BATCHES" "$MIN_SCENARIOS" "$SCENARIO_ITERS"; do
	[[ "$v" =~ ^[0-9]+$ ]] || { echo "ERROR: '$v' must be a non-negative integer"; exit 1; }
done
[[ "$MAX_BATCHES" -ge 1 ]] || { echo "ERROR: --max-batches must be at least 1"; exit 1; }
case "$TARGET_TRACK" in reg|generic|all|cleaned) ;; *) echo "ERROR: bad --target-track"; exit 1 ;; esac
case "$DROP_SCOPE" in none|cross_batch|all) ;; *) echo "ERROR: bad --drop-scope"; exit 1 ;; esac
case "$SEED_BATCH" in first|none) ;; *) echo "ERROR: bad --seed-batch"; exit 1 ;; esac
[[ "$RE_SHARDS" =~ ^[0-9]+$ ]] || { echo "ERROR: --re-shards must be a non-negative integer"; exit 1; }
[[ "$RE_SHARDS" -eq 0 ]] && RE_SHARDS="$SHARDS"
[[ -n "$RE_SUFFIX" ]] || { echo "ERROR: --re-suffix cannot be empty"; exit 1; }
if [[ "$MAX_BATCHES" -gt 1 && "$SCENARIO_ITERS" -eq 0 ]]; then
	echo "ERROR: --scenario-iters is 0 with --max-batches $MAX_BATCHES."
	echo "Stage 1 would emit the same seeded scenarios in every batch."
	exit 1
fi

# Batch 2 onward regenerates batch 1's scenarios unless generator.py can be told
# what already exists. Not fatal, dedup catches the result, but it wastes spend.
if ! python generator.py --help 2>/dev/null | grep -q -- "-prior_scenarios"; then
	echo "WARNING: generator.py has no -prior_scenarios. Later batches will not"
	echo "         know about earlier scenarios and may regenerate them. Dedup"
	echo "         removes the duplicate questions after the fact, at full cost."
	echo ""
fi

TYPE_ARR=($TYPES)
[[ ${#TYPE_ARR[@]} -gt 0 ]] || { echo "ERROR: --types is empty"; exit 1; }

# The judge panel runs after a full batch of generation. A key that is only
# missing at that point costs the whole batch in wall clock, so say so now.
for k in ELM_API_KEY ELM_BASE_URL GEMINI_API_KEY; do
	if [[ -z "${!k:-}" ]]; then
		echo "WARNING: $k is not set. Generation may still run, but the judge"
		echo "         panel runs after each batch and will fail then, after the"
		echo "         batch has already been paid for."
	fi
done

echo "Stem:       $STEM"
echo "Model:      $MODEL ($MODEL_TYPE)"
echo "Target:     $TARGET_TOTAL on track '$TARGET_TRACK'"
echo "Batches:    up to $MAX_BATCHES, scenario-iters $SCENARIO_ITERS, qa-iters $QA_ITERS"
echo "Types:      ${TYPE_ARR[*]}"

CLEAN_STEM="programmatic_data_modified_verified_cleaned.tsv"
docs_dir() { echo "${PREFIX}-docs-type$1-$2"; }

# Runs run_pipeline.sh for one batch. run_pipeline.sh exits 1 if any shard fails,
# and it does so BEFORE the 9b merge and stage 10, so a single dead shard would
# otherwise strand the whole batch and, under set -e, kill this loop. Recovery is
# what you would do by hand: rewrite the manifest to the shards that reached
# stage 9, then rerun --from 9 --to 10. The dead shard's questions are lost; the
# top-up loop compensates on the next batch.
#
# Returns 0 if the batch produced cleaned corpora, 1 if nothing could be salvaged.
run_pipeline_recover() {
	local RUN="$1"; shift
	local rc=0
	bash ./run_pipeline.sh -n "$RUN" "$@" || rc=$?
	[[ "$rc" -eq 0 ]] && return 0

	echo ""
	echo "-- run_pipeline.sh exited $rc for $RUN, attempting shard recovery"
	local MAN="generation_outputs/${PREFIX}-scenarios-${RUN}/shards.manifest"
	if [[ ! -f "$MAN" ]]; then
		echo "   no manifest at $MAN. The failure was before sharding, or --shards is 1."
		return 1
	fi

	# Keep a shard only if it produced stage 9 output for all three types.
	# run_pipeline.sh always generates types 1-3 (this wrapper never sets
	# --qa-iters-tN), and it merges all three, so a shard missing any one of them
	# cannot be merged.
	local keep=() sfx t good total=0
	while IFS= read -r sfx; do
		[[ -n "$sfx" ]] || continue
		total=$((total + 1))
		good=1
		for t in 1 2 3; do
			[[ -f "generation_outputs/$(docs_dir "$t" "${RUN}${sfx}")/programmatic_data_modified_verified.tsv" ]] || good=0
		done
		[[ "$good" -eq 1 ]] && keep+=("$sfx")
	done < "$MAN"

	if [[ ${#keep[@]} -eq 0 ]]; then
		echo "   no shard reached stage 9 for all three types; nothing to salvage"
		return 1
	fi
	if [[ ${#keep[@]} -eq "$total" ]]; then
		echo "   every shard reached stage 9, so the failure was at merge or cleanup."
		echo "   retrying --from 9 --to 10 once."
	else
		echo "   salvaging ${#keep[@]} of $total shard(s): ${keep[*]}"
	fi
	[[ -f "${MAN}.orig" ]] || cp "$MAN" "${MAN}.orig"
	printf '%s\n' "${keep[@]}" > "$MAN"

	# Stage 9 reruns, but verification.py resumes from _verified.tsv, so this
	# costs no API calls. 9b then merges and stage 10 cleans.
	rc=0
	bash ./run_pipeline.sh -n "$RUN" "$@" --from 9 --to 10 || rc=$?
	if [[ "$rc" -ne 0 ]]; then
		echo "   recovery run also failed (exit $rc)"
		return 1
	fi
	echo "-- recovered $RUN from ${#keep[@]} shard(s)"
	return 0
}

DONE=()
FINAL=0

for b in $(seq 1 "$MAX_BATCHES"); do
	RUN="${STEM}-b${b}"
	echo ""
	echo "##################################################################"
	echo "  BATCH $b of at most $MAX_BATCHES   run=$RUN"
	echo "##################################################################"

	# ---- generation, skipped if this batch is already complete
	have=0
	miss=""
	for t in "${TYPE_ARR[@]}"; do
		if [[ -f "generation_outputs/$(docs_dir "$t" "$RUN")/${CLEAN_STEM}" ]]; then
			have=$((have + 1))
		else
			miss="${miss} type$t"
		fi
	done
	if [[ "$have" -eq "${#TYPE_ARR[@]}" ]]; then
		echo "-- cleaned corpora already present for every type, skipping generation"
	elif [[ "$have" -gt 0 ]]; then
		echo "ERROR: batch $RUN is partially complete. Missing:${miss}"
		echo "Rerunning the pipeline would rerun stage 1 and replace this batch's"
		echo "scenarios, leaving the already-cleaned types grounded in scenarios"
		echo "that no longer exist. Finish the batch by hand, e.g."
		echo "  ./run_pipeline.sh -n $RUN -m $MODEL -t $MODEL_TYPE -p $PREFIX \\"
		echo "      --shards $SHARDS --from <failing stage>"
		echo "then rerun this command; the completed batch will be skipped."
		exit 1
	else
		# Seeds are emitted once at most. Every later batch compares against every
		# earlier batch's scenarios so stage 1 does not regenerate them.
		if [[ "$SEED_BATCH" == "first" && "$b" -eq 1 ]]; then EMIT="all"; else EMIT="new"; fi
		PRIOR=""
		if [[ ${#DONE[@]} -gt 0 ]]; then
			for pb in "${DONE[@]}"; do
				PRIOR="${PRIOR:+$PRIOR,}generation_outputs/${PREFIX}-scenarios-${pb}/scenarios.tsv"
			done
		fi
		PIPE_EXTRA=(--emit-scope "$EMIT")
		if [[ -n "$PRIOR" ]]; then
			PIPE_EXTRA+=(--prior-scenarios "$PRIOR")
		fi
		echo "-- emit_scope $EMIT, prior scenario files: ${#DONE[@]}"
		if ! run_pipeline_recover "$RUN" -m "$MODEL" -t "$MODEL_TYPE" -p "$PREFIX" \
			--shards "$SHARDS" --max-parallel "$MAX_PARALLEL" \
			--scenario-iters "$SCENARIO_ITERS" \
			"${PIPE_EXTRA[@]}" \
			--qa-iters "$QA_ITERS" --adv-iters "$ADV_ITERS" \
			--temperature "$TEMPERATURE" --threshold 0 \
			--cleanup-seed "$CLEANUP_SEED"; then
			echo "ERROR: batch $RUN produced nothing usable. Stopping rather than"
			echo "burning further batches; a total failure usually means the API, the"
			echo "keys, or a code path, not bad luck on one shard."
			exit 1
		fi

		SCEN="generation_outputs/${PREFIX}-scenarios-${RUN}/scenarios.tsv"
		# Count expanded rows, not total. Seed_Type separates the seeded rows
		# (reg, generic) from stage 1 output (dynamic_reg, dynamic_generic), so
		# this means the same thing whether or not the seeds were emitted.
		read -r n_scen n_total <<<"$(python -c "
import pandas as pd, sys
d = pd.read_csv(sys.argv[1], sep='\t', dtype=str, keep_default_na=False)
if 'Seed_Type' not in d.columns:
    sys.exit('no Seed_Type column in ' + sys.argv[1])
st = d['Seed_Type'].str.strip()
print(int(st.str.startswith('dynamic_').sum()), len(d))
" "$SCEN")"
		echo "-- stage 1 wrote $n_scen expanded scenarios ($n_total rows in scenarios.tsv)"
		if [[ "$n_scen" -lt "$MIN_SCENARIOS" ]]; then
			echo "ERROR: $n_scen expanded scenarios is below --min-scenarios $MIN_SCENARIOS."
			echo "Stage 1 expansion produced nothing usable. Scenarios missing a"
			echo "REGULATORY_AREA_ID line are discarded silently by the parser."
			exit 1
		fi
	fi

	# ---- RocketEval, per type, skipped where already judged
	for t in "${TYPE_ARR[@]}"; do
		D="$(docs_dir "$t" "$RUN")"
		RE="generation_outputs/${D}/rocketeval"
		if [[ -f "${RE}/poll_scores.tsv" ]]; then
			echo "-- type$t already judged, skipping"
			continue
		fi
		if [[ -d "$RE" ]]; then
			echo "ERROR: ${RE} exists but has no poll_scores.tsv."
			echo "judge_panel.py appends, so rerunning over a partial directory"
			echo "duplicates rows. Delete ${RE} and rerun this command."
			echo "That discards the checklists too, which were paid for. If you"
			echo "know judge_panel.py's output filename, deleting only that file"
			echo "and rerunning the last two steps by hand is cheaper."
			exit 1
		fi
		echo "-- RocketEval type$t across $RE_SHARDS shard(s)"
		RLOG="generation_outputs/logs-${RUN}"
		mkdir -p "$RLOG"
		if [[ "$RE_SHARDS" -le 1 ]]; then
			python checklist_generator.py -out_dir generation_outputs/ -folder_name "$D" \
				-qa_type "type$t" -model_type "$MODEL_TYPE" -model "$MODEL"
			python judge_panel.py -out_dir generation_outputs/ -folder_name "$D"
		else
			# checklist_generator.py and judge_panel.py are sequential, so judging is
			# the dominant cost per batch. shard_folder.py splits the corpus into
			# independent folders, each with its own rocketeval/ directory, so there
			# is no shared file and the mode='a' append cannot interleave.
			# -suffix uses the = form: its value starts with a dash and argparse
			# would otherwise read it as another option.
			python shard_folder.py -out_dir generation_outputs/ -folder_name "$D" \
				-data "${CLEAN_STEM%.tsv}" -shards "$RE_SHARDS" -suffix="$RE_SUFFIX"
			RPIDS=(); RNAMES=(); RFAIL=()
			for rs in $(seq 1 "$RE_SHARDS"); do
				rsfx=$(printf -- "%s%02d" "$RE_SUFFIX" "$rs")
				[[ -d "generation_outputs/${D}${rsfx}" ]] || continue
				if [[ "$MAX_PARALLEL" -gt 0 ]]; then
					while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do sleep 5; done
				fi
				(
					python checklist_generator.py -out_dir generation_outputs/ \
						-folder_name "${D}${rsfx}" -qa_type "type$t" \
						-model_type "$MODEL_TYPE" -model "$MODEL" &&
					python judge_panel.py -out_dir generation_outputs/ \
						-folder_name "${D}${rsfx}"
				) < /dev/null > "${RLOG}/rocketeval-type${t}${rsfx}.log" 2>&1 &
				RPIDS+=($!); RNAMES+=("$rsfx")
			done
			[[ ${#RPIDS[@]} -gt 0 ]] || { echo "ERROR: shard_folder.py produced no shard folders for $D"; exit 1; }
			for ri in "${!RPIDS[@]}"; do
				rrc=0
				wait "${RPIDS[$ri]}" || rrc=$?
				[[ "$rrc" -eq 0 ]] || RFAIL+=("${RNAMES[$ri]}:$rrc")
			done
			if [[ ${#RFAIL[@]} -gt 0 ]]; then
				echo "ERROR: RocketEval shard failures for type$t: ${RFAIL[*]}"
				echo "Logs in ${RLOG}/. Partial judgments would give a wrong PASS count,"
				echo "so this stops rather than merging. Delete generation_outputs/${D}${RE_SUFFIX}*"
				echo "and rerun; completed batches are skipped."
				exit 1
			fi
			python merge_rocketeval.py -out_dir generation_outputs/ -folder_name "$D" \
				-suffix="$RE_SUFFIX"
		fi
		python poll_aggregate.py -out_dir generation_outputs/ -folder_name "$D" \
			-unsure exclude -pass_frac "$PASS_FRAC" -disagree_frac "$DISAGREE_FRAC"
	done

	DONE+=("$RUN")

	# ---- dedup across every batch so far, then count
	RUN_CSV="$(IFS=,; echo "${DONE[*]}")"
	TYPE_CSV="$(IFS=,; echo "${TYPE_ARR[*]}")"
	echo ""
	echo "-- dedup across: $RUN_CSV"
	rm -f "$COMBINED" "${COMBINED%.tsv}_dropped.tsv"
	python dedup_batches.py \
		-out_dir generation_outputs/ \
		-prefix "$PREFIX" \
		-run_names "$RUN_CSV" \
		-types "$TYPE_CSV" \
		-jaccard "$JACCARD" \
		-drop_scope "$DROP_SCOPE" \
		-out_file "$COMBINED" \
		-strict
	[[ -f "$COMBINED" ]] || {
		echo "ERROR: dedup_batches.py did not write $COMBINED."
		echo "Check its flags and output path, then set --combined accordingly."
		exit 1
	}

	echo ""
	echo "-- cumulative count"
	FINAL=$(python count_pass.py --runs "$RUN_CSV" --types "$TYPE_CSV" \
		--prefix "$PREFIX" --track "$TARGET_TRACK" --pass-frac "$PASS_FRAC" \
		--combined "$COMBINED")

	echo ""
	echo "  BATCH $b DONE.  cumulative $TARGET_TRACK = $FINAL / $TARGET_TOTAL"
	if [[ "$FINAL" -ge "$TARGET_TOTAL" ]]; then
		echo ""
		echo "TARGET REACHED after $b batch(es)."
		break
	fi
done

echo ""
echo "=================================================================="
echo "  batches run: ${DONE[*]}"
echo "  cumulative $TARGET_TRACK = $FINAL, target $TARGET_TOTAL"
echo "  combined corpus: $COMBINED"
echo "=================================================================="
echo ""
python split_combined.py -combined "$COMBINED"
echo ""
echo "Next, one run_eval.sh per (type, model, solver prompt) over those files."
echo "Type 3 needs -e type3-causal-check. Results join back positionally, so the"
echo "split files carry QID and Batch for the analysis frame."
if [[ "$FINAL" -lt "$TARGET_TOTAL" ]]; then
	echo ""
	echo "SHORT of target at --max-batches $MAX_BATCHES."
	echo "Rerun the same command with a higher --max-batches to add more batches;"
	echo "completed batches are skipped. The split files above are still usable."
	exit 2
fi