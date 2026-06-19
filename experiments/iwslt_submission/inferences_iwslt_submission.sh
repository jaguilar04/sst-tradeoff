#!/bin/bash
#SBATCH --job-name=iwslt_submission_infer
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/iwslt_submission/slurm-%j.out

# ─────────────────────────────────────────────────────────────────────────────
# Inference for the IWSLT submission: 3 regimes (one run each).
#
#   high_latency → partial MBR + policies (cascade_2026.agent_simulstream_mt_policies)
#                  ASR: tolerant_agreement, tau=1
#                  MT : tolerant_agreement, mt_tau=1
#                  PruneMBR-XCOMET, epsilon=0.02, temp=1.0, N=8, seg=0.96.
#                  Produces asr_transcript_log -> used to compute WER.
#
#   low_latency  → identical to high_latency except MT policy:
#                  MT : wait_k, wait_k=3  (only difference)
#                  Also produces transcript -> WER.
#
#   baseline     → base cascade: strict local agreement + pure greedy (temp=0.0),
#                  1 sample, no epsilon, no MBR (decoding_method=none).
#                  Uses the asr_policies processor (instead of vanilla) ONLY to
#                  dump asr_transcript_log and compute WER; behavior is identical
#                  to the vanilla cascade_2026 baseline. seg=0.96.
#
# Input data: always data/test/ (src.txt).
# A failure in one regime does NOT stop the rest (handled with ||).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_ROOT="${REPO_ROOT}/experiments/iwslt_submission"
DATA_DIR="${REPO_ROOT}/data/test"
WAV_LIST="${DATA_DIR}/src.txt"

# ── Regimes ───────────────────────────────────────────────────────────────────
REGIMES=("high_latency" "low_latency" "baseline")

# To run a subset in parallel: REGIME_FILTER = space-separated list of regimes.
# Empty = run all.
REGIME_FILTER="${REGIME_FILTER:-}"
if [[ -n "${REGIME_FILTER}" ]]; then
    _filtered=()
    for _r in "${REGIMES[@]}"; do
        for _keep in ${REGIME_FILTER}; do
            [[ "${_r}" == "${_keep}" ]] && _filtered+=("${_r}")
        done
    done
    REGIMES=("${_filtered[@]}")
    echo "[filter] REGIME_FILTER='${REGIME_FILTER}' -> ${REGIMES[*]}"
fi

# ── Processors ────────────────────────────────────────────────────────────────
PROCESSOR_POLICIES="cascade_2026.agent_simulstream_mt_policies.CascadeSpeechProcessor"
# The baseline uses the ASR policies processor (not the vanilla one) solely to
# dump asr_transcript_log and compute WER; configured with local_agreement +
# decoding_method=none + greedy, its output is identical to the vanilla baseline.
PROCESSOR_BASELINE="cascade_2026.agent_simulstream_asr_policies.CascadeSpeechProcessor"

# ── Shared parameters for all 3 regimes ──────────────────────────────────────
ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
SOURCE_LANG="English"
TARGET_LANG="German"
LATENCY_UNIT="word"
MIN_START_SECONDS="5.0"
MAX_HISTORY_UTTERANCES="0"
MAX_NEW_TOKENS="100"
REPETITION_PENALTY="1.05"
SEGMENT_SIZE="0.96"
NER_RESULTS_PATH="null"

# ── Parameters for policy + partial-MBR regimes (high/low latency) ───────────
ASR_GPU_MEM="0.36"
LLM_GPU_MEM="0.36"
TEMPERATURE_POLICIES="1.0"
EPSILON="0.02"
N_SAMPLES="8"
DECODING_METHOD="prunembr_xcomet"
XCOMET_MODEL="myyycroft/XCOMET-lite"
XCOMET_BATCH_SIZE="16"
KIWI_MODEL="Unbabel/wmt22-cometkiwi-da"
KIWI_BATCH_SIZE="8"
ASR_POLICY="tolerant_agreement"
ASR_TOLERANT_TAU="1"

# ── Parameters for the vanilla baseline ──────────────────────────────────────
TEMPERATURE_BASELINE="0.0"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${EXP_ROOT}"
cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "════════════════════════════════════════════════════════════"
echo "  IWSLT submission inference  |  data: ${DATA_DIR}"
echo "  Regimes: ${REGIMES[*]}"
echo "════════════════════════════════════════════════════════════"

N_OK=0; N_SKIP=0; N_FAIL=0

# ── Regime loop ───────────────────────────────────────────────────────────────
for regime in "${REGIMES[@]}"; do

    run_name="iwslt_${regime}"
    RUN_DIR="${EXP_ROOT}/${regime}"
    CONFIG_DIR="${RUN_DIR}/configs"
    INFERENCES_DIR="${RUN_DIR}/inferences"
    TRANSCRIPTS_DIR="${RUN_DIR}/transcripts"
    mkdir -p "${CONFIG_DIR}" "${INFERENCES_DIR}" "${TRANSCRIPTS_DIR}"

    config_file="${CONFIG_DIR}/${run_name}.yaml"
    metrics_file="${INFERENCES_DIR}/${run_name}.jsonl"
    transcript_file="${TRANSCRIPTS_DIR}/${run_name}.transcript.jsonl"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Regime: ${regime}  ->  ${run_name}"
    echo "────────────────────────────────────────────────────────────"

    if [[ -f "${metrics_file}" ]]; then
        echo "[SKIP] Already exists: ${metrics_file}"
        (( N_SKIP++ )) || true
        continue
    fi

    # ── Write config per regime ───────────────────────────────────────────────
    if [[ "${regime}" == "baseline" ]]; then
        # Base cascade: strict local agreement + pure greedy, no MBR.
        # (decoding_method=none, does not load XCOMET/KIWI)
        # Uses asr_policies processor only to dump asr_transcript_log for WER.
        cat > "${config_file}" <<CONFIG
type: "${PROCESSOR_BASELINE}"
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

# LLM generation: pure greedy, single sample, no epsilon or MBR
max_new_tokens: ${MAX_NEW_TOKENS}
temperature: ${TEMPERATURE_BASELINE}
repetition_penalty: ${REPETITION_PENALTY}
epsilon: 0.0
n_samples: 1
decoding_method: "none"

# ASR emission policy: strict local agreement (= cascade base).
# Transcript log allows WER computation for the baseline too.
asr_policy: "local_agreement"
asr_transcript_log: "${transcript_file}"

# External context
ner_results_path: ${NER_RESULTS_PATH}
CONFIG

    else
        # high_latency / low_latency: policies + partial MBR.
        # The only difference between them is the MT emission policy.
        if [[ "${regime}" == "high_latency" ]]; then
            MT_POLICY="tolerant_agreement"
        else
            MT_POLICY="wait_k"
        fi
        MT_WAIT_K="3"
        MT_TOLERANT_TAU="1"

        cat > "${config_file}" <<CONFIG
type: "${PROCESSOR_POLICIES}"
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
temperature: ${TEMPERATURE_POLICIES}
repetition_penalty: ${REPETITION_PENALTY}
epsilon: ${EPSILON}
n_samples: ${N_SAMPLES}

# MBR decoder (PruneMBR-XCOMET)
decoding_method: "${DECODING_METHOD}"
xcomet_model: "${XCOMET_MODEL}"
xcomet_batch_size: ${XCOMET_BATCH_SIZE}
kiwi_model: "${KIWI_MODEL}"
kiwi_batch_size: ${KIWI_BATCH_SIZE}

# ASR emission policy (fixed across high/low latency)
asr_policy: "${ASR_POLICY}"
hold_n_chunks: 2
tolerant_tau: ${ASR_TOLERANT_TAU}
asr_transcript_log: "${transcript_file}"

# MT emission policy (the only difference between high and low latency)
mt_policy: "${MT_POLICY}"
wait_k: ${MT_WAIT_K}
mt_tolerant_tau: ${MT_TOLERANT_TAU}

# External context
ner_results_path: ${NER_RESULTS_PATH}
CONFIG
    fi

    # ── Run inference ─────────────────────────────────────────────────────────
    echo "[RUN] Running inference (${WAV_LIST})..."
    PYTHONUNBUFFERED=1 simulstream_inference \
        --speech-processor-config "${config_file}" \
        --wav-list-file "${WAV_LIST}" \
        --src-lang "${SOURCE_LANG}" \
        --tgt-lang "${TARGET_LANG}" \
        --metrics-log-file "${metrics_file}" \
    && {
        echo "[OK] ${run_name} done."
        (( N_OK++ )) || true
    } || {
        exit_code=$?
        echo "[ERROR] ${run_name} failed (exit ${exit_code}). Continuing..."
        [[ -f "${metrics_file}" ]] && rm -f "${metrics_file}"
        (( N_FAIL++ )) || true
    }

done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Inference complete.  OK=${N_OK}  SKIP=${N_SKIP}  FAIL=${N_FAIL}"
echo "  Results in: ${EXP_ROOT}"
echo "════════════════════════════════════════════════════════════"
