#!/bin/bash
#SBATCH --job-name=policies_mt_grid
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/policies_trades/mt/slurm-%j.out
# Pin to a specific node with a free GPU to avoid the CUBLAS error on saturated GPUs.
# Before launching, check which node is idle:  sinfo -N -o "%N %t" | awk '$2=="idle"'
#SBATCH --nodelist=vrhpc6.dsic.upv.es

# MT EMISSION POLICY GRID on cascade_2026 with partial MBR
# (MBR applied only at utterance close), fixed segment size of 0.96 s.
#
# ASR policy is FIXED at tolerant_agreement, tau=1 (best from the ASR sweep).
# The only variable is the target emission policy:
#
#   - wait_k            : fixed delay of k source words before emitting.
#   - tolerant_agreement: Tolerant Local Agreement (Levenshtein threshold tau).
#   - hybrid_waitk_tla  : wait-k guard at sentence start + TLA cruise (tau).
#
# Strict Local Agreement (baseline) is NOT included: already exists in the system.
#
# A failure in one combination does NOT stop the rest of the grid (handled with ||).
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_ROOT="${REPO_ROOT}/experiments/policies_trades/mt"

# ── MT policy grid ────────────────────────────────────────────────────────────
# Format: "policy|k|tau"  (k = wait_k; tau = mt_tolerant_tau; empty if not applicable)
RUNS=(
    "wait_k|1|"
    "wait_k|3|"
    "wait_k|5|"
    "tolerant_agreement||1"
    "tolerant_agreement||2"
    "hybrid_waitk_tla|1|1"
    "hybrid_waitk_tla|1|2"
    "hybrid_waitk_tla|3|1"
    "hybrid_waitk_tla|3|2"
    "hybrid_waitk_tla|5|1"
    "hybrid_waitk_tla|5|2"
)

# To run a subset in parallel: POLICY_FILTER = space-separated list of policies.
# Empty = run all.
POLICY_FILTER="${POLICY_FILTER:-}"
if [[ -n "${POLICY_FILTER}" ]]; then
    _filtered=()
    for _run in "${RUNS[@]}"; do
        _p="${_run%%|*}"
        for _keep in ${POLICY_FILTER}; do
            [[ "${_p}" == "${_keep}" ]] && _filtered+=("${_run}")
        done
    done
    RUNS=("${_filtered[@]}")
    echo "[filter] POLICY_FILTER='${POLICY_FILTER}' -> ${#RUNS[@]} runs: ${RUNS[*]}"
fi

# ── MBR decoders (partial, applied at utterance close only) ───────────────────
DECODING_METHODS=("rambr_chrf" "prunembr_xcomet" "rerank_kiwi")
N_VALUES=(8)   # MBR n_samples

# ── Fixed segment size ────────────────────────────────────────────────────────
SEGMENT_SIZE="0.96"
SEG_TAG="seg960"

PROCESSOR_TYPE="cascade_2026.agent_simulstream_mt_policies.CascadeSpeechProcessor"

# ── Fixed ASR policy (best from the ASR policy sweep) ─────────────────────────
ASR_POLICY="tolerant_agreement"
ASR_TOLERANT_TAU="1"

# ── Fixed parameters ──────────────────────────────────────────────────────────
ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
ASR_GPU_MEM="0.36"
LLM_GPU_MEM="0.36"
SOURCE_LANG="English"
TARGET_LANG="German"
LATENCY_UNIT="word"
MIN_START_SECONDS="5.0"
MAX_HISTORY_UTTERANCES="0"
MAX_NEW_TOKENS="100"
TEMPERATURE="1.0"
REPETITION_PENALTY="1.05"
EPSILON="0.02"
NER_RESULTS_PATH="null"

XCOMET_MODEL="myyycroft/XCOMET-lite"
XCOMET_BATCH_SIZE="16"
KIWI_MODEL="Unbabel/wmt22-cometkiwi-da"
KIWI_BATCH_SIZE="8"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${EXP_ROOT}"
cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Grid summary file ─────────────────────────────────────────────────────────
SUMMARY_FILE="${EXP_ROOT}/grid_summary.txt"
echo "MT policy grid: seg=${SEGMENT_SIZE}s, partial-MBR, epsilon=${EPSILON}, ASR=tolerant_tau${ASR_TOLERANT_TAU}" > "${SUMMARY_FILE}"
echo "Start: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
printf "%-28s %-16s %-6s %-10s %s\n" "POLICY_TAG" "METHOD" "N" "STATUS" "OUTPUT" >> "${SUMMARY_FILE}"
echo "────────────────────────────────────────────────────────────" >> "${SUMMARY_FILE}"

echo "════════════════════════════════════════════════════════════"
echo "  MT policy grid: segment_size=${SEGMENT_SIZE}s, partial MBR"
echo "  Processor : ${PROCESSOR_TYPE}"
echo "  Fixed ASR : ${ASR_POLICY} (tau=${ASR_TOLERANT_TAU})"
echo "  Policies  : ${RUNS[*]}"
echo "  Decoders  : ${DECODING_METHODS[*]}  |  N: ${N_VALUES[*]}"
echo "  Total     : $(( ${#RUNS[@]} * ${#DECODING_METHODS[@]} * ${#N_VALUES[@]} )) combinations"
echo "════════════════════════════════════════════════════════════"

N_OK=0; N_SKIP=0; N_FAIL=0

# ── Main loop: policy x decoder x N ──────────────────────────────────────────
for run in "${RUNS[@]}"; do
    policy="${run%%|*}"
    rest="${run#*|}"
    k="${rest%%|*}"
    tau="${rest##*|}"

    # Defaults per policy (irrelevant fields are fixed but unused).
    WAIT_K=3
    MT_TOLERANT_TAU=1
    case "${policy}" in
        wait_k)
            WAIT_K="${k}"
            policy_tag="mt_waitk${k}"
            ;;
        tolerant_agreement)
            MT_TOLERANT_TAU="${tau}"
            policy_tag="mt_tla_tau${tau}"
            ;;
        hybrid_waitk_tla)
            WAIT_K="${k}"
            MT_TOLERANT_TAU="${tau}"
            policy_tag="mt_hybrid_k${k}_tau${tau}"
            ;;
        *)
            echo "[ERROR] Unknown policy: ${policy}"; continue ;;
    esac

    for method in "${DECODING_METHODS[@]}"; do
        for n in "${N_VALUES[@]}"; do

            run_name="mt_${policy_tag}_${method}_n${n}_${SEG_TAG}"

            RUN_DIR="${EXP_ROOT}/${policy}"
            CONFIG_DIR="${RUN_DIR}/configs"
            INFERENCES_DIR="${RUN_DIR}/inferences"
            TRANSCRIPTS_DIR="${RUN_DIR}/transcripts"
            mkdir -p "${CONFIG_DIR}" "${INFERENCES_DIR}" "${TRANSCRIPTS_DIR}"

            config_file="${CONFIG_DIR}/${run_name}.yaml"
            metrics_file="${INFERENCES_DIR}/${run_name}.jsonl"
            transcript_file="${TRANSCRIPTS_DIR}/${run_name}.transcript.jsonl"

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo "  Policy: ${policy} (${policy_tag})  |  Decoder: ${method}  N=${n}"
            echo "  Config: ${config_file}"
            echo "────────────────────────────────────────────────────────────"

            if [[ -f "${metrics_file}" ]]; then
                echo "[SKIP] Already exists: ${metrics_file}"
                printf "%-28s %-16s %-6s %-10s %s\n" "${policy_tag}" "${method}" "${n}" "SKIP" "${metrics_file}" >> "${SUMMARY_FILE}"
                (( N_SKIP++ )) || true
                continue
            fi

            # Write YAML config for this combination
            cat > "${config_file}" <<CONFIG
type: "${PROCESSOR_TYPE}"
speech_chunk_size: ${SEGMENT_SIZE}
latency_unit: "${LATENCY_UNIT}"
detokenizer_type: "simuleval"

# Models
asr_model_name: "${ASR_MODEL}"
llm_model_name: "${LLM_MODEL}"
asr_gpu_memory_utilization: ${ASR_GPU_MEM}
llm_gpu_memory_utilization: ${LLM_GPU_MEM}

# Languages
source_lang: "${SOURCE_LANG}"
target_lang: "${TARGET_LANG}"

# Segmentation parameters
min_start_seconds: ${MIN_START_SECONDS}
max_history_utterances: ${MAX_HISTORY_UTTERANCES}

# LLM generation (MBR only at utterance close; greedy when utt_finished=False)
max_new_tokens: ${MAX_NEW_TOKENS}
temperature: ${TEMPERATURE}
repetition_penalty: ${REPETITION_PENALTY}
epsilon: ${EPSILON}
n_samples: ${n}

# MBR decoder
decoding_method: "${method}"
xcomet_model: "${XCOMET_MODEL}"
xcomet_batch_size: ${XCOMET_BATCH_SIZE}
kiwi_model: "${KIWI_MODEL}"
kiwi_batch_size: ${KIWI_BATCH_SIZE}

# ASR emission policy (fixed in this study)
asr_policy: "${ASR_POLICY}"
hold_n_chunks: 2
tolerant_tau: ${ASR_TOLERANT_TAU}
asr_transcript_log: "${transcript_file}"

# MT emission policy (the variable in this study)
mt_policy: "${policy}"
wait_k: ${WAIT_K}
mt_tolerant_tau: ${MT_TOLERANT_TAU}

# External context
ner_results_path: ${NER_RESULTS_PATH}
CONFIG

            echo "[RUN] Running inference (data/dev/src.txt)..."
            PYTHONUNBUFFERED=1 simulstream_inference \
                --speech-processor-config "${config_file}" \
                --wav-list-file "${REPO_ROOT}/data/dev/src.txt" \
                --src-lang "${SOURCE_LANG}" \
                --tgt-lang "${TARGET_LANG}" \
                --metrics-log-file "${metrics_file}" \
            && {
                echo "[OK] ${policy_tag}, ${method}, N=${n} done."
                printf "%-28s %-16s %-6s %-10s %s\n" "${policy_tag}" "${method}" "${n}" "OK" "${metrics_file}" >> "${SUMMARY_FILE}"
                (( N_OK++ )) || true
            } || {
                exit_code=$?
                echo "[ERROR] ${policy_tag}, ${method}, N=${n} failed (exit ${exit_code}). Continuing..."
                [[ -f "${metrics_file}" ]] && rm -f "${metrics_file}"
                printf "%-28s %-16s %-6s %-10s exit=%s\n" "${policy_tag}" "${method}" "${n}" "FAIL" "${exit_code}" >> "${SUMMARY_FILE}"
                (( N_FAIL++ )) || true
            }
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Grid complete.  OK=${N_OK}  SKIP=${N_SKIP}  FAIL=${N_FAIL}"
echo "  Results in: ${EXP_ROOT}"
echo "════════════════════════════════════════════════════════════"
{
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "End: $(date)"
    echo "OK=${N_OK}, SKIP=${N_SKIP}, FAIL=${N_FAIL}"
} >> "${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"
