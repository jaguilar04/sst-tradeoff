#!/bin/bash
#SBATCH --job-name=cascade_2026_eval
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/baselines_trades/cascade_2026/slurm-eval-%j.out

set -eo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_DIR="${REPO_ROOT}/experiments/baselines_trades/cascade_2026"
INFERENCES_DIR="${EXP_DIR}/inferences"
CONFIGS_DIR="${EXP_DIR}/configs"
DATA_DIR="${REPO_ROOT}/data/dev"

AUDIO_DEFINITION="${DATA_DIR}/audio_definition.yaml"
REFERENCE="${DATA_DIR}/tgt_segments.txt"
TRANSCRIPTS="${DATA_DIR}/transcript_segments.txt"
CSV_OUT="${EXP_DIR}/results_cascade_2026.csv"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/simulstream:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"

# ── CSV header ────────────────────────────────────────────────────────────────
echo "jsonl_file,stream_laal_nca,stream_laal_ca,long_yaal_nca,long_yaal_ca,bleu,chrf_pp,comet_xl" \
    > "${CSV_OUT}"

# ── Loop over all JSONL files ─────────────────────────────────────────────────
for jsonl_file in "${INFERENCES_DIR}"/cascade_2026_*.jsonl; do

    basename=$(basename "${jsonl_file}" .jsonl)
    config_file="${CONFIGS_DIR}/${basename}.yaml"

    if [[ ! -f "${config_file}" ]]; then
        echo "[eval] [WARN] Config not found for ${basename}, skipping..."
        continue
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "[eval] Evaluating ${basename}"
    echo "════════════════════════════════════════════════════════════"

    # ── StreamLAAL ────────────────────────────────────────────────────────────
    echo "[eval] StreamLAAL..."
    stream_laal_out=$(simulstream_score_latency \
        --scorer stream_laal \
        --eval-config "${config_file}" \
        --log-file "${jsonl_file}" \
        --reference "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    echo "${stream_laal_out}"
    stream_laal_nca=$(echo "${stream_laal_out}" \
        | grep -oP "ideal_latency=\K[0-9.]+" || true)
    stream_laal_ca=$(echo "${stream_laal_out}" \
        | grep -oP "computational_aware_latency=\K[0-9.]+" || true)
    stream_laal_nca="${stream_laal_nca:-NA}"
    stream_laal_ca="${stream_laal_ca:-NA}"

    # ── LongYAAL (OmniSTEval) ─────────────────────────────────────────────────
    echo "[eval] LongYAAL (OmniSTEval)..."
    long_yaal_out=$(omnisteval longform \
        --speech_segmentation "${AUDIO_DEFINITION}" \
        --ref_sentences_file "${REFERENCE}" \
        --hypothesis_file "${jsonl_file}" \
        --hypothesis_format simulstream \
        --simulstream_config_file "${config_file}" \
        --word_level \
        --no_quality \
        --output_folder "${EXP_DIR}/results/omnisteval_${basename}" \
        2>&1) || true
    echo "${long_yaal_out}"
    long_yaal_nca_ms=$(echo "${long_yaal_out}" \
        | grep -oP "LongYAAL \(CU\)\s+\K[0-9.]+" || true)
    long_yaal_ca_ms=$(echo "${long_yaal_out}" \
        | grep -oP "LongYAAL \(CA\)\s+\K[0-9.]+" || true)
    if [[ -n "${long_yaal_nca_ms}" ]]; then
        long_yaal_nca=$(python3 -c "print(${long_yaal_nca_ms} / 1000)")
    else
        long_yaal_nca="NA"
    fi
    if [[ -n "${long_yaal_ca_ms}" ]]; then
        long_yaal_ca=$(python3 -c "print(${long_yaal_ca_ms} / 1000)")
    else
        long_yaal_ca="NA"
    fi

    # ── BLEU ──────────────────────────────────────────────────────────────────
    echo "[eval] BLEU..."
    bleu_out=$(simulstream_score_quality \
        --scorer sacrebleu \
        --eval-config "${config_file}" \
        --log-file "${jsonl_file}" \
        --references "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    echo "${bleu_out}"
    bleu=$(echo "${bleu_out}" \
        | grep -oiP "sacrebleu score: \K[0-9.]+" || true)
    bleu="${bleu:-NA}"

    # ── chrF++ ────────────────────────────────────────────────────────────────
    echo "[eval] chrF++..."
    chrf_out=$(simulstream_score_quality \
        --scorer chrf \
        --word-order 2 \
        --eval-config "${config_file}" \
        --log-file "${jsonl_file}" \
        --references "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    echo "${chrf_out}"
    chrf_pp=$(echo "${chrf_out}" \
        | grep -oiP "chrf score: \K[0-9.]+" || true)
    chrf_pp="${chrf_pp:-NA}"

    # ── COMET-XL ─────────────────────────────────────────────────────────────
    echo "[eval] COMET-XL..."
    comet_out=$(simulstream_score_quality \
        --scorer comet \
        --model "Unbabel/XCOMET-XL" \
        --eval-config "${config_file}" \
        --log-file "${jsonl_file}" \
        --references "${REFERENCE}" \
        --transcripts "${TRANSCRIPTS}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    echo "${comet_out}"
    comet_xl=$(echo "${comet_out}" \
        | grep -oiP "comet score: \K[0-9.]+" | head -n 1 || true)
    comet_xl="${comet_xl:-NA}"

    # ── Debug ─────────────────────────────────────────────────────────────────
    echo "[debug] stream_laal_nca='${stream_laal_nca}' stream_laal_ca='${stream_laal_ca}'"
    echo "[debug] long_yaal_nca='${long_yaal_nca}' long_yaal_ca='${long_yaal_ca}'"
    echo "[debug] bleu='${bleu}' chrf_pp='${chrf_pp}' comet_xl='${comet_xl}'"

    # ── Write CSV row ─────────────────────────────────────────────────────────
    echo "${basename},${stream_laal_nca},${stream_laal_ca},${long_yaal_nca},${long_yaal_ca},${bleu},${chrf_pp},${comet_xl}" \
        >> "${CSV_OUT}"

    echo "[eval] Done ${basename}"
    echo "────────────────────────────────────────────────────────────"

done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Results saved to: ${CSV_OUT}"
echo "════════════════════════════════════════════════════════════"
cat "${CSV_OUT}"
