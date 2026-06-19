#!/bin/bash
#SBATCH --job-name=mbr_seg1280_grid
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/mbr_trades/pruebas_1.28/slurm-%j.out

# NOTE: We do not use "set -e" globally here because we want a failure in one
# combination (method, N) to NOT stop the rest of the grid.
# Errors are handled explicitly with || inside the loop.
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_ROOT="${REPO_ROOT}/experiments/mbr_trades/pruebas_1.28"

# ── Grid ──────────────────────────────────────────────────────────────────────
DECODING_METHODS=("rambr_chrf" "prunembr_xcomet" "rerank_kiwi")
N_VALUES=(4 8 16 32)

# ── Fixed segment size ────────────────────────────────────────────────────────
SEGMENT_SIZE="1.28"
SEG_TAG="seg1280"

# ── Fixed parameters ──────────────────────────────────────────────────────────
ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
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

# Method-specific parameters
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
echo "MBR grid: seg=${SEGMENT_SIZE}s, epsilon=${EPSILON}" > "${SUMMARY_FILE}"
echo "Start: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
printf "%-25s %-6s %-10s %s\n" "METHOD" "N" "STATUS" "OUTPUT" >> "${SUMMARY_FILE}"
echo "────────────────────────────────────────────────────────────" >> "${SUMMARY_FILE}"

# ── Launch info ───────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  MBR grid: segment_size=${SEGMENT_SIZE}s (fixed), epsilon=${EPSILON}"
echo "  Methods : ${DECODING_METHODS[*]}"
echo "  N values: ${N_VALUES[*]}"
echo "  Total   : $((${#DECODING_METHODS[@]} * ${#N_VALUES[@]})) combinations"
echo "════════════════════════════════════════════════════════════"

# ── Counters ──────────────────────────────────────────────────────────────────
N_OK=0
N_SKIP=0
N_FAIL=0

# ── Main loop: method x N ─────────────────────────────────────────────────────
for method in "${DECODING_METHODS[@]}"; do
    for n in "${N_VALUES[@]}"; do

        run_name="mbr_${method}_n${n}_${SEG_TAG}"

        # Each (method, N) has its own subdirectory to avoid mixing configs.
        RUN_DIR="${EXP_ROOT}/${method}"
        CONFIG_DIR="${RUN_DIR}/configs"
        INFERENCES_DIR="${RUN_DIR}/inferences"
        mkdir -p "${CONFIG_DIR}" "${INFERENCES_DIR}"

        config_file="${CONFIG_DIR}/${run_name}.yaml"
        metrics_file="${INFERENCES_DIR}/${run_name}.jsonl"

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "  Method : ${method}"
        echo "  N      : ${n}"
        echo "  Config : ${config_file}"
        echo "────────────────────────────────────────────────────────────"

        # Skip if output already exists (auto-resume)
        if [[ -f "${metrics_file}" ]]; then
            echo "[SKIP] Already exists: ${metrics_file}"
            printf "%-25s %-6s %-10s %s\n" "${method}" "${n}" "SKIP" "${metrics_file}" >> "${SUMMARY_FILE}"
            (( N_SKIP++ )) || true
            continue
        fi

        # Write YAML config for this combination
        cat > "${config_file}" <<CONFIG
type: "cascade_2026.agent_simulstream_mbr.CascadeSpeechProcessor"
speech_chunk_size: ${SEGMENT_SIZE}
latency_unit: "${LATENCY_UNIT}"
detokenizer_type: "simuleval"

# Models
asr_model_name: "${ASR_MODEL}"
llm_model_name: "${LLM_MODEL}"

# Languages
source_lang: "${SOURCE_LANG}"
target_lang: "${TARGET_LANG}"

# Segmentation parameters
min_start_seconds: ${MIN_START_SECONDS}
max_history_utterances: ${MAX_HISTORY_UTTERANCES}

# LLM generation parameters
max_new_tokens: ${MAX_NEW_TOKENS}
temperature: ${TEMPERATURE}
repetition_penalty: ${REPETITION_PENALTY}
epsilon: ${EPSILON}
n_samples: ${n}

# MBR decoding method
# Options: none | rambr_chrf | prunembr_xcomet | rerank_kiwi
decoding_method: "${method}"

# Method-specific parameters
xcomet_model: "${XCOMET_MODEL}"
xcomet_batch_size: ${XCOMET_BATCH_SIZE}
kiwi_model: "${KIWI_MODEL}"
kiwi_batch_size: ${KIWI_BATCH_SIZE}

# External context
ner_results_path: ${NER_RESULTS_PATH}
CONFIG

        # Inference with error handling.
        # On failure (OOM, timeout, decoder error...) log and continue.
        echo "[RUN] Running inference..."

        PYTHONUNBUFFERED=1 simulstream_inference \
            --speech-processor-config "${config_file}" \
            --wav-list-file "${REPO_ROOT}/data/dev/src.txt" \
            --src-lang "${SOURCE_LANG}" \
            --tgt-lang "${TARGET_LANG}" \
            --metrics-log-file "${metrics_file}" \
        && {
            echo "[OK] ${method}, N=${n} done."
            printf "%-25s %-6s %-10s %s\n" "${method}" "${n}" "OK" "${metrics_file}" >> "${SUMMARY_FILE}"
            (( N_OK++ )) || true
        } || {
            exit_code=$?
            echo ""
            echo "[ERROR] ${method}, N=${n} failed with exit code ${exit_code}."
            echo "        Check the logs. Moving to next combination..."
            echo ""
            # Remove partial output file if created
            [[ -f "${metrics_file}" ]] && rm -f "${metrics_file}"
            printf "%-25s %-6s %-10s exit=%s\n" "${method}" "${n}" "FAIL" "${exit_code}" >> "${SUMMARY_FILE}"
            (( N_FAIL++ )) || true
        }

    done
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Grid complete."
echo "  OK   : ${N_OK}"
echo "  SKIP : ${N_SKIP}"
echo "  FAIL : ${N_FAIL}"
echo "  Results in: ${EXP_ROOT}"
echo "════════════════════════════════════════════════════════════"

{
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "End: $(date)"
    echo "OK=${N_OK}, SKIP=${N_SKIP}, FAIL=${N_FAIL}"
} >> "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"
