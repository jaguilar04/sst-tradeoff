#!/bin/bash
#SBATCH --job-name=cascade_2026_mbr_eval
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/mbr_trades/cascade_2026_mbr/slurm-eval-%j.out

# NOTE: We do not use "set -e" globally — each scorer has its own || true so
# a single scorer failure does not abort the whole loop.
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

# ── Paths ─────────────────────────────────────────────────────────────────────
EXP_ROOT="${REPO_ROOT}/experiments/mbr_trades/cascade_2026_mbr"
DATA_DIR="${REPO_ROOT}/data/dev"

AUDIO_DEFINITION="${DATA_DIR}/audio_definition.yaml"
REFERENCE="${DATA_DIR}/tgt_segments.txt"
TRANSCRIPTS="${DATA_DIR}/transcript_segments.txt"
CSV_OUT="${EXP_ROOT}/results_cascade_2026_mbr.csv"

# ── Grid (must match inferences_cascade_2026_mbr.sh) ─────────────────────────
DECODING_METHODS=("rambr_chrf" "prunembr_xcomet" "rerank_kiwi")
N_VALUES=(4 8 16 32)

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/simulstream:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"

# ── Write CSV header only if file doesn't exist ───────────────────────────────
if [[ ! -f "${CSV_OUT}" ]]; then
    echo "method,n_samples,jsonl_basename,stream_laal_nca,stream_laal_ca,long_yaal_nca,long_yaal_ca,bleu,chrf_pp,comet_xl" \
        > "${CSV_OUT}"
fi

# Returns 0 (skip) if the row exists and has no NA values; 1 otherwise.
row_is_complete() {
    local key="$1"
    local row
    row=$(grep "^${key}," "${CSV_OUT}" 2>/dev/null || true)
    [[ -z "${row}" ]] && return 1
    echo "${row}" | grep -q ",NA" && return 1
    return 0
}

# Removes a row from the CSV so it can be rewritten cleanly.
remove_row() {
    local key="$1"
    local tmp
    tmp=$(mktemp)
    grep -v "^${key}," "${CSV_OUT}" > "${tmp}" || true
    mv "${tmp}" "${CSV_OUT}"
}

# ── Counters ──────────────────────────────────────────────────────────────────
N_OK=0
N_SKIP=0
N_MISSING=0

# ── Main loop: method x N ─────────────────────────────────────────────────────
for method in "${DECODING_METHODS[@]}"; do
    for n in "${N_VALUES[@]}"; do

        run_name="mbr_${method}_n${n}_seg960"
        jsonl_file="${EXP_ROOT}/${method}/inferences/${run_name}.jsonl"
        config_file="${EXP_ROOT}/${method}/configs/${run_name}.yaml"

        csv_key="${method},${n},${run_name}"

        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Method : ${method}, N : ${n}"
        echo "  JSONL  : ${jsonl_file}"
        echo "════════════════════════════════════════════════════════════"

        if [[ ! -f "${jsonl_file}" ]]; then
            echo "[MISSING] Inference not found, skipping..."
            (( N_MISSING++ )) || true
            continue
        fi

        if [[ ! -f "${config_file}" ]]; then
            echo "[WARN] Config not found for ${run_name}, skipping..."
            (( N_MISSING++ )) || true
            continue
        fi

        if row_is_complete "${csv_key}"; then
            echo "[SKIP] Already evaluated and complete."
            (( N_SKIP++ )) || true
            continue
        fi

        remove_row "${csv_key}"

        # ── StreamLAAL ────────────────────────────────────────────────────────
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

        # ── LongYAAL (OmniSTEval) ─────────────────────────────────────────────
        echo "[eval] LongYAAL (OmniSTEval)..."
        long_yaal_out=$(omnisteval longform \
            --speech_segmentation "${AUDIO_DEFINITION}" \
            --ref_sentences_file "${REFERENCE}" \
            --hypothesis_file "${jsonl_file}" \
            --hypothesis_format simulstream \
            --simulstream_config_file "${config_file}" \
            --word_level \
            --no_quality \
            --output_folder "${EXP_ROOT}/${method}/results/omnisteval_${run_name}" \
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

        # ── BLEU ──────────────────────────────────────────────────────────────
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

        # ── chrF++ ────────────────────────────────────────────────────────────
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

        # ── COMET-XL ─────────────────────────────────────────────────────────
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

        # ── Debug ─────────────────────────────────────────────────────────────
        echo "[debug] stream_laal_nca='${stream_laal_nca}' stream_laal_ca='${stream_laal_ca}'"
        echo "[debug] long_yaal_nca='${long_yaal_nca}' long_yaal_ca='${long_yaal_ca}'"
        echo "[debug] bleu='${bleu}' chrf_pp='${chrf_pp}' comet_xl='${comet_xl}'"

        # ── Write CSV row ─────────────────────────────────────────────────────
        echo "${method},${n},${run_name},${stream_laal_nca},${stream_laal_ca},${long_yaal_nca},${long_yaal_ca},${bleu},${chrf_pp},${comet_xl}" \
            >> "${CSV_OUT}"

        echo "[eval] Done ${run_name}"
        (( N_OK++ )) || true

    done
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Evaluation complete."
echo "  OK      : ${N_OK}"
echo "  SKIP    : ${N_SKIP}"
echo "  MISSING : ${N_MISSING}  (inference not yet generated)"
echo "  CSV     : ${CSV_OUT}"
echo "════════════════════════════════════════════════════════════"
echo ""
cat "${CSV_OUT}"
