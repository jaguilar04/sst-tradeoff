#!/bin/bash
#SBATCH --job-name=cascade_2026_infer
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/baselines_trades/cascade_2026/slurm-%j.out

set -eo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_DIR="${REPO_ROOT}/experiments/baselines_trades/cascade_2026"
CONFIG_DIR="${EXP_DIR}/configs"
INFERENCES_DIR="${EXP_DIR}/inferences"

# ── Grid: segment sizes (seconds) ─────────────────────────────────────────────
SEGMENT_SIZES=("0.64" "0.96" "1.28" "1.60" "1.92")

# ── Fixed parameters ──────────────────────────────────────────────────────────
ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
SOURCE_LANG="English"
TARGET_LANG="German"
LATENCY_UNIT="word"
MIN_START_SECONDS="5.0"
MAX_HISTORY_UTTERANCES="0"
MAX_NEW_TOKENS="100"
TEMPERATURE="0.0"
REPETITION_PENALTY="1.05"
NER_RESULTS_PATH="null"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${CONFIG_DIR}"
mkdir -p "${INFERENCES_DIR}"

cd "${REPO_ROOT}"

source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Segment size sweep ────────────────────────────────────────────────────────
for seg in "${SEGMENT_SIZES[@]}"; do

    # Convert seconds to milliseconds for the run name (e.g. 0.64 -> 640)
    seg_ms=$(awk "BEGIN { printf \"%d\", ${seg} * 1000 }")
    run_name="cascade_2026_seg${seg_ms}"

    config_file="${CONFIG_DIR}/${run_name}.yaml"
    metrics_file="${INFERENCES_DIR}/${run_name}.jsonl"

    # Auto-generate config
    cat > "${config_file}" <<CONFIG
type: "cascade_2026.agent_simulstream.CascadeSpeechProcessor"
speech_chunk_size: ${seg}
latency_unit: "${LATENCY_UNIT}"
detokenizer_type: "simuleval"
asr_model_name: "${ASR_MODEL}"
llm_model_name: "${LLM_MODEL}"
source_lang: "${SOURCE_LANG}"
target_lang: "${TARGET_LANG}"
min_start_seconds: ${MIN_START_SECONDS}
max_history_utterances: ${MAX_HISTORY_UTTERANCES}
max_new_tokens: ${MAX_NEW_TOKENS}
temperature: ${TEMPERATURE}
repetition_penalty: ${REPETITION_PENALTY}
ner_results_path: ${NER_RESULTS_PATH}
CONFIG

    # Skip if output already exists
    if [[ -f "${metrics_file}" ]]; then
        echo "[cascade_2026] $(basename ${metrics_file}) already exists, skipping..."
        continue
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "[cascade_2026] segment_size=${seg}s (${seg_ms}ms)"
    echo "════════════════════════════════════════════════════════════"

    PYTHONUNBUFFERED=1 simulstream_inference \
        --speech-processor-config "${config_file}" \
        --wav-list-file "${REPO_ROOT}/data/dev/src.txt" \
        --src-lang "${SOURCE_LANG}" \
        --tgt-lang "${TARGET_LANG}" \
        --metrics-log-file "${metrics_file}"

    echo "[cascade_2026] Done segment_size=${seg}s"
    echo "────────────────────────────────────────────────────────────"

done

echo "Inference complete. Results in: ${INFERENCES_DIR}"
