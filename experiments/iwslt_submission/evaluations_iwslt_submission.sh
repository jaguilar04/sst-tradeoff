#!/bin/bash
#SBATCH --job-name=iwslt_submission_eval
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/iwslt_submission/slurm-eval-%j.out

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation of the IWSLT submission: one row per regime (high/low/baseline).
#
# Metrics per run:
#   - StreamLAAL (NCA/CA)  : streaming latency    (simulstream_score_latency)
#   - LongYAAL  (NCA/CA)   : longform latency      (omnisteval longform)
#   - BLEU, chrF++         : MT quality            (simulstream_score_quality)
#   - COMET-XL             : Unbabel/XCOMET-XL
#   - COMET-DA             : Unbabel/wmt22-comet-da
#   - WER                  : committed ASR WER      (compute_wer.py)
#                            (all 3 regimes dump asr_transcript_log)
#
# Input data: always data/test/.
# Each scorer uses || true to avoid aborting the loop on failure.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_ROOT="${REPO_ROOT}/experiments/iwslt_submission"
DATA_DIR="${REPO_ROOT}/data/test"

AUDIO_DEFINITION="${DATA_DIR}/audio_definition.yaml"
REFERENCE="${DATA_DIR}/tgt_segments.txt"
TRANSCRIPTS="${DATA_DIR}/transcript_segments.txt"
WAV_LIST="${DATA_DIR}/src.txt"
CSV_OUT="${EXP_ROOT}/results_iwslt_submission.csv"
COMPUTE_WER="${REPO_ROOT}/experiments/policies_trades/asr/compute_wer.py"

# ── COMET models ──────────────────────────────────────────────────────────────
COMET_XL_MODEL="Unbabel/XCOMET-XL"
COMET_DA_MODEL="Unbabel/wmt22-comet-da"

# ── Regimes ───────────────────────────────────────────────────────────────────
REGIMES=("high_latency" "low_latency" "baseline")

REGIME_FILTER="${REGIME_FILTER:-}"
if [[ -n "${REGIME_FILTER}" ]]; then
    _filtered=()
    for _r in "${REGIMES[@]}"; do
        for _keep in ${REGIME_FILTER}; do
            [[ "${_r}" == "${_keep}" ]] && _filtered+=("${_r}")
        done
    done
    REGIMES=("${_filtered[@]}")
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/simulstream:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"

# ── CSV header ────────────────────────────────────────────────────────────────
echo "regime,run_name,mt_policy,wait_k,mt_tau,decoding_method,n_samples,streamlaal_nca,streamlaal_ca,long_yaal_nca,long_yaal_ca,bleu,chrf_pp,comet_xl,comet_da,wer" \
    > "${CSV_OUT}"

# Returns regime metadata for CSV columns.
# Format: "mt_policy|wait_k|mt_tau|decoding_method|n_samples"
meta_for_regime() {
    case "$1" in
        high_latency) echo "tolerant_agreement|NA|1|prunembr_xcomet|8" ;;
        low_latency)  echo "wait_k|3|NA|prunembr_xcomet|8" ;;
        baseline)     echo "local_agreement|NA|NA|greedy|1" ;;
        *)            echo "NA|NA|NA|NA|NA" ;;
    esac
}

N_OK=0; N_MISSING=0

# ── Regime loop ───────────────────────────────────────────────────────────────
for regime in "${REGIMES[@]}"; do

    run_name="iwslt_${regime}"
    jsonl_file="${EXP_ROOT}/${regime}/inferences/${run_name}.jsonl"
    config_file="${EXP_ROOT}/${regime}/configs/${run_name}.yaml"
    transcript_file="${EXP_ROOT}/${regime}/transcripts/${run_name}.transcript.jsonl"

    IFS='|' read -r mt_policy wait_k mt_tau decoding_method n_samples \
        <<< "$(meta_for_regime "${regime}")"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ${run_name}  (mt_policy=${mt_policy})"
    echo "════════════════════════════════════════════════════════════"

    if [[ ! -f "${jsonl_file}" ]]; then
        echo "[MISSING] Inference not found: ${jsonl_file}, skipping..."
        (( N_MISSING++ )) || true
        continue
    fi
    if [[ ! -f "${config_file}" ]]; then
        echo "[WARN] Config not found: ${config_file}, skipping..."
        (( N_MISSING++ )) || true
        continue
    fi

    # ── StreamLAAL ────────────────────────────────────────────────────────────
    echo "[eval] StreamLAAL..."
    stream_laal_out=$(simulstream_score_latency \
        --scorer stream_laal \
        --eval-config "${config_file}" \
        --log-file "${jsonl_file}" \
        --reference "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    streamlaal_nca=$(echo "${stream_laal_out}" | grep -oP "ideal_latency=\K[0-9.]+" || true)
    streamlaal_ca=$(echo "${stream_laal_out}"  | grep -oP "computational_aware_latency=\K[0-9.]+" || true)
    streamlaal_nca="${streamlaal_nca:-NA}"
    streamlaal_ca="${streamlaal_ca:-NA}"

    # ── LongYAAL (OmniSTEval) ─────────────────────────────────────────────────
    echo "[eval] LongYAAL (OmniSTEval)..."
    long_yaal_out=$(omnisteval longform \
        --speech_segmentation "${AUDIO_DEFINITION}" \
        --ref_sentences_file "${REFERENCE}" \
        --hypothesis_file "${jsonl_file}" \
        --hypothesis_format simulstream \
        --simulstream_config_file "${config_file}" \
        --word_level --no_quality \
        --output_folder "${EXP_ROOT}/${regime}/results/omnisteval_${run_name}" 2>&1) || true
    long_yaal_nca_ms=$(echo "${long_yaal_out}" | grep -oP "LongYAAL \(CU\)\s+\K[0-9.]+" || true)
    long_yaal_ca_ms=$(echo "${long_yaal_out}"  | grep -oP "LongYAAL \(CA\)\s+\K[0-9.]+" || true)
    if [[ -n "${long_yaal_nca_ms}" ]]; then long_yaal_nca=$(python3 -c "print(${long_yaal_nca_ms}/1000)"); else long_yaal_nca="NA"; fi
    if [[ -n "${long_yaal_ca_ms}" ]];  then long_yaal_ca=$(python3 -c "print(${long_yaal_ca_ms}/1000)");   else long_yaal_ca="NA"; fi

    # ── BLEU ──────────────────────────────────────────────────────────────────
    echo "[eval] BLEU..."
    bleu_out=$(simulstream_score_quality \
        --scorer sacrebleu --eval-config "${config_file}" \
        --log-file "${jsonl_file}" --references "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    bleu=$(echo "${bleu_out}" | grep -oiP "sacrebleu score: \K[0-9.]+" || true)
    bleu="${bleu:-NA}"

    # ── chrF++ ────────────────────────────────────────────────────────────────
    echo "[eval] chrF++..."
    chrf_out=$(simulstream_score_quality \
        --scorer chrf --word-order 2 --eval-config "${config_file}" \
        --log-file "${jsonl_file}" --references "${REFERENCE}" \
        --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    chrf_pp=$(echo "${chrf_out}" | grep -oiP "chrf score: \K[0-9.]+" || true)
    chrf_pp="${chrf_pp:-NA}"

    # ── COMET-XL ──────────────────────────────────────────────────────────────
    echo "[eval] COMET-XL..."
    comet_xl_out=$(simulstream_score_quality \
        --scorer comet --model "${COMET_XL_MODEL}" --eval-config "${config_file}" \
        --log-file "${jsonl_file}" --references "${REFERENCE}" \
        --transcripts "${TRANSCRIPTS}" --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    comet_xl=$(echo "${comet_xl_out}" | grep -oiP "comet score: \K[0-9.]+" | head -n 1 || true)
    comet_xl="${comet_xl:-NA}"

    # ── COMET-DA ──────────────────────────────────────────────────────────────
    echo "[eval] COMET-DA..."
    comet_da_out=$(simulstream_score_quality \
        --scorer comet --model "${COMET_DA_MODEL}" --eval-config "${config_file}" \
        --log-file "${jsonl_file}" --references "${REFERENCE}" \
        --transcripts "${TRANSCRIPTS}" --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
    comet_da=$(echo "${comet_da_out}" | grep -oiP "comet score: \K[0-9.]+" | head -n 1 || true)
    comet_da="${comet_da:-NA}"

    # ── WER (committed ASR) — all 3 regimes produce a transcript log ──────────
    echo "[eval] WER (ASR)..."
    wer="NA"
    if [[ -f "${transcript_file}" ]]; then
        wer_out=$(python3 "${COMPUTE_WER}" \
            --transcript "${transcript_file}" \
            --reference "${TRANSCRIPTS}" \
            --audio-definition "${AUDIO_DEFINITION}" \
            --wav-list "${WAV_LIST}" 2>&1) || true
        echo "${wer_out}"
        wer=$(echo "${wer_out}" | grep -oP "CORPUS_WER=\K[0-9.]+" || true)
        wer="${wer:-NA}"
    else
        echo "[WARN] No transcript log found: ${transcript_file} -> WER=NA"
    fi

    echo "[debug] streamlaal=${streamlaal_nca}/${streamlaal_ca} yaal=${long_yaal_nca}/${long_yaal_ca} bleu=${bleu} chrf=${chrf_pp} comet_xl=${comet_xl} comet_da=${comet_da} wer=${wer}"

    echo "${regime},${run_name},${mt_policy},${wait_k},${mt_tau},${decoding_method},${n_samples},${streamlaal_nca},${streamlaal_ca},${long_yaal_nca},${long_yaal_ca},${bleu},${chrf_pp},${comet_xl},${comet_da},${wer}" \
        >> "${CSV_OUT}"

    echo "[eval] Done ${run_name}"
    (( N_OK++ )) || true
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Evaluation complete.  OK=${N_OK}  MISSING=${N_MISSING}"
echo "  CSV: ${CSV_OUT}"
echo "════════════════════════════════════════════════════════════"
cat "${CSV_OUT}"
