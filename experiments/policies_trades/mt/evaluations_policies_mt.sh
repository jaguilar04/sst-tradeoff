#!/bin/bash
#SBATCH --job-name=policies_mt_eval
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/jagucam1@alumno.upv.es/MAIN/experiments/policies_trades/mt/slurm-eval-%j.out

# Evaluation of the MT policy grid: latency (LongYAAL), MT quality
# (BLEU, chrF++, COMET-XL), and committed ASR WER.
# Each scorer has its own || true so a single failure does not abort the loop.
set -uo pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
REPO_ROOT="/data/jagucam1@alumno.upv.es/MAIN"
CONDA_ENV="/data/jagucam1@alumno.upv.es/MAIN/env"
CONDA_ACTIVATE="/data/jagucam1@alumno.upv.es/miniconda3/bin/activate"

EXP_ROOT="${REPO_ROOT}/experiments/policies_trades/mt"
DATA_DIR="${REPO_ROOT}/data/dev"

AUDIO_DEFINITION="${DATA_DIR}/audio_definition.yaml"
REFERENCE="${DATA_DIR}/tgt_segments.txt"
TRANSCRIPTS="${DATA_DIR}/transcript_segments.txt"
WAV_LIST="${DATA_DIR}/src.txt"
CSV_OUT="${EXP_ROOT}/results_policies_mt.csv"

# ── Grid (must match inferences_policies_mt.sh) ───────────────────────────────
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
fi

DECODING_METHODS=("rambr_chrf" "prunembr_xcomet" "rerank_kiwi")
N_VALUES=(8)
SEG_TAG="seg960"

# ── Setup ─────────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/simulstream:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$HOME/.local/bin:$PATH"

# ── Write CSV header only if file doesn't exist ───────────────────────────────
if [[ ! -f "${CSV_OUT}" ]]; then
    echo "policy_tag,policy,k,tau,method,n_samples,run_name,long_yaal_nca,long_yaal_ca,bleu,chrf_pp,comet_xl,wer" \
        > "${CSV_OUT}"
fi

# Returns 0 (skip) if the row exists and has no NA values; 1 otherwise.
row_is_complete() {
    local key="$1"
    local row
    row=$(grep ",${key}," "${CSV_OUT}" 2>/dev/null || true)
    [[ -z "${row}" ]] && return 1
    echo "${row}" | grep -q ",NA" && return 1
    return 0
}

# Removes a row from the CSV so it can be rewritten cleanly.
remove_row() {
    local key="$1"
    local tmp; tmp=$(mktemp)
    grep -v ",${key}," "${CSV_OUT}" > "${tmp}" || true
    mv "${tmp}" "${CSV_OUT}"
}

N_OK=0; N_SKIP=0; N_MISSING=0

# ── Main loop: policy x method x N ───────────────────────────────────────────
for run in "${RUNS[@]}"; do
    policy="${run%%|*}"
    rest="${run#*|}"
    k="${rest%%|*}"
    tau="${rest##*|}"

    case "${policy}" in
        wait_k)            policy_tag="mt_waitk${k}" ;;
        tolerant_agreement) policy_tag="mt_tla_tau${tau}" ;;
        hybrid_waitk_tla)  policy_tag="mt_hybrid_k${k}_tau${tau}" ;;
        *) echo "[ERROR] Unknown policy: ${policy}"; continue ;;
    esac

    for method in "${DECODING_METHODS[@]}"; do
        for n in "${N_VALUES[@]}"; do

            run_name="mt_${policy_tag}_${method}_n${n}_${SEG_TAG}"
            jsonl_file="${EXP_ROOT}/${policy}/inferences/${run_name}.jsonl"
            config_file="${EXP_ROOT}/${policy}/configs/${run_name}.yaml"
            transcript_file="${EXP_ROOT}/${policy}/transcripts/${run_name}.transcript.jsonl"

            echo ""
            echo "════════════════════════════════════════════════════════════"
            echo "  ${run_name}"
            echo "════════════════════════════════════════════════════════════"

            if [[ ! -f "${jsonl_file}" ]]; then
                echo "[MISSING] Inference not found, skipping..."
                (( N_MISSING++ )) || true
                continue
            fi
            if [[ ! -f "${config_file}" ]]; then
                echo "[WARN] Config not found, skipping..."
                (( N_MISSING++ )) || true
                continue
            fi
            if row_is_complete "${run_name}"; then
                echo "[SKIP] Already evaluated and complete."
                (( N_SKIP++ )) || true
                continue
            fi
            remove_row "${run_name}"

            # ── LongYAAL (OmniSTEval) ─────────────────────────────────────────
            echo "[eval] LongYAAL..."
            long_yaal_out=$(omnisteval longform \
                --speech_segmentation "${AUDIO_DEFINITION}" \
                --ref_sentences_file "${REFERENCE}" \
                --hypothesis_file "${jsonl_file}" \
                --hypothesis_format simulstream \
                --simulstream_config_file "${config_file}" \
                --word_level --no_quality \
                --output_folder "${EXP_ROOT}/${policy}/results/omnisteval_${run_name}" 2>&1) || true
            long_yaal_nca_ms=$(echo "${long_yaal_out}" | grep -oP "LongYAAL \(CU\)\s+\K[0-9.]+" || true)
            long_yaal_ca_ms=$(echo "${long_yaal_out}" | grep -oP "LongYAAL \(CA\)\s+\K[0-9.]+" || true)
            if [[ -n "${long_yaal_nca_ms}" ]]; then long_yaal_nca=$(python3 -c "print(${long_yaal_nca_ms}/1000)"); else long_yaal_nca="NA"; fi
            if [[ -n "${long_yaal_ca_ms}" ]];  then long_yaal_ca=$(python3 -c "print(${long_yaal_ca_ms}/1000)");  else long_yaal_ca="NA"; fi

            # ── BLEU ──────────────────────────────────────────────────────────
            echo "[eval] BLEU..."
            bleu_out=$(simulstream_score_quality \
                --scorer sacrebleu --eval-config "${config_file}" \
                --log-file "${jsonl_file}" --references "${REFERENCE}" \
                --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
            bleu=$(echo "${bleu_out}" | grep -oiP "sacrebleu score: \K[0-9.]+" || true)
            bleu="${bleu:-NA}"

            # ── chrF++ ────────────────────────────────────────────────────────
            echo "[eval] chrF++..."
            chrf_out=$(simulstream_score_quality \
                --scorer chrf --word-order 2 --eval-config "${config_file}" \
                --log-file "${jsonl_file}" --references "${REFERENCE}" \
                --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
            chrf_pp=$(echo "${chrf_out}" | grep -oiP "chrf score: \K[0-9.]+" || true)
            chrf_pp="${chrf_pp:-NA}"

            # ── COMET-XL ──────────────────────────────────────────────────────
            echo "[eval] COMET-XL..."
            comet_out=$(simulstream_score_quality \
                --scorer comet --model "Unbabel/XCOMET-XL" --eval-config "${config_file}" \
                --log-file "${jsonl_file}" --references "${REFERENCE}" \
                --transcripts "${TRANSCRIPTS}" --audio-definition "${AUDIO_DEFINITION}" 2>&1) || true
            comet_xl=$(echo "${comet_out}" | grep -oiP "comet score: \K[0-9.]+" | head -n 1 || true)
            comet_xl="${comet_xl:-NA}"

            # ── WER (committed ASR) ───────────────────────────────────────────
            echo "[eval] WER (ASR)..."
            wer="NA"
            if [[ -f "${transcript_file}" ]]; then
                wer_out=$(python3 "${EXP_ROOT}/../asr/compute_wer.py" \
                    --transcript "${transcript_file}" \
                    --reference "${TRANSCRIPTS}" \
                    --audio-definition "${AUDIO_DEFINITION}" \
                    --wav-list "${WAV_LIST}" 2>&1) || true
                echo "${wer_out}"
                wer=$(echo "${wer_out}" | grep -oP "CORPUS_WER=\K[0-9.]+" || true)
                wer="${wer:-NA}"
            else
                echo "[WARN] No transcript log found: ${transcript_file}"
            fi

            echo "[debug] yaal=${long_yaal_nca}/${long_yaal_ca} bleu=${bleu} chrf=${chrf_pp} comet=${comet_xl} wer=${wer}"

            echo "${policy_tag},${policy},${k},${tau},${method},${n},${run_name},${long_yaal_nca},${long_yaal_ca},${bleu},${chrf_pp},${comet_xl},${wer}" \
                >> "${CSV_OUT}"

            echo "[eval] Done ${run_name}"
            (( N_OK++ )) || true
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Evaluation complete.  OK=${N_OK}  SKIP=${N_SKIP}  MISSING=${N_MISSING}"
echo "  CSV: ${CSV_OUT}"
echo "════════════════════════════════════════════════════════════"
cat "${CSV_OUT}"
