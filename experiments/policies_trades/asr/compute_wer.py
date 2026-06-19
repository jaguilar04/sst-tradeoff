#!/usr/bin/env python3
"""Calcula el WER del ASR comprometido por una política de emisión.

Lee el log de transcript que genera ``agent_simulstream_asr_policies`` (líneas
``{"type":"final","speech_id":k,"transcript":...}``, una por audio) y lo compara
con la transcripción de referencia, agrupada por audio a partir del
``audio_definition.yaml``.

WER: usa ``jiwer`` si está instalado; si no, un cálculo propio word-level
(distancia de edición / nº de palabras de referencia). Ambos sobre texto
normalizado (minúsculas + sin puntuación), que es el protocolo habitual de WER.

Uso:
    python compute_wer.py \
        --transcript <run>.transcript.jsonl \
        --reference  data/transcript_segments.txt \
        --audio-definition data/audio_definition.yaml \
        --wav-list   data/src.txt \
        [--csv results.csv --tag run_name]
"""

import argparse
import json
import string
from collections import OrderedDict

import yaml

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def _edit_distance_words(ref, hyp) -> int:
    """Distancia de edición word-level (S+D+I)."""
    if ref == hyp:
        return 0
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i]
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[m]


def wer_counts(ref_text: str, hyp_text: str):
    """Devuelve (errores, n_palabras_ref) con texto ya normalizado."""
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    return _edit_distance_words(ref_words, hyp_words), len(ref_words)


def load_reference_by_wav(audio_def_path, reference_path):
    with open(audio_def_path) as f:
        audio_def = yaml.safe_load(f)
    with open(reference_path, encoding="utf-8") as f:
        ref_lines = [line.rstrip("\n") for line in f]
    if len(audio_def) != len(ref_lines):
        raise SystemExit(
            f"audio_definition ({len(audio_def)}) y reference "
            f"({len(ref_lines)}) tienen distinto nº de segmentos."
        )
    ref_by_wav = OrderedDict()
    for entry, ref in zip(audio_def, ref_lines):
        ref_by_wav.setdefault(entry["wav"], []).append(ref)
    return ref_by_wav


def load_hyp_by_speech(transcript_path):
    hyp = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "final":
                hyp[rec["speech_id"]] = rec.get("transcript", "")
    return hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--audio-definition", required=True)
    ap.add_argument("--wav-list", required=True,
                    help="src.txt: define el orden de audios = speech_id")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    try:
        from jiwer import wer as jiwer_wer
        backend = "jiwer"
    except Exception:
        jiwer_wer = None
        backend = "builtin"

    ref_by_wav = load_reference_by_wav(args.audio_definition, args.reference)
    hyp_by_speech = load_hyp_by_speech(args.transcript)

    with open(args.wav_list, encoding="utf-8") as f:
        wavs = [line.strip() for line in f if line.strip()]

    total_err, total_words = 0, 0
    per_audio = []
    ref_list, hyp_list = [], []
    for speech_id, wav in enumerate(wavs):
        ref_text = normalize(" ".join(ref_by_wav.get(wav, [])))
        hyp_text = normalize(hyp_by_speech.get(speech_id, ""))
        # err/ref_words para las columnas (idénticos a S+D+I de jiwer)
        err, nwords = wer_counts(ref_text, hyp_text)
        total_err += err
        total_words += nwords
        # tasa WER: jiwer si está disponible, si no fallback propio
        if jiwer_wer is not None and nwords:
            audio_wer = jiwer_wer(ref_text, hyp_text)
        else:
            audio_wer = (err / nwords) if nwords else float("nan")
        per_audio.append((speech_id, wav.split("/")[-1], audio_wer, err, nwords))
        if nwords:
            ref_list.append(ref_text)
            hyp_list.append(hyp_text)

    # WER de corpus (micro-promedio): jiwer agrega sobre todas las frases
    if jiwer_wer is not None and ref_list:
        corpus_wer = jiwer_wer(ref_list, hyp_list)
    else:
        corpus_wer = (total_err / total_words) if total_words else float("nan")

    print(f"# WER backend: {backend}")
    print(f"{'speech_id':>9}  {'audio':<28} {'WER':>8}  {'err':>5} {'ref_words':>9}")
    for sid, name, w, err, nwords in per_audio:
        print(f"{sid:>9}  {name:<28} {w*100:>7.2f}%  {err:>5} {nwords:>9}")
    print("-" * 64)
    print(f"{'CORPUS':>9}  {'':<28} {corpus_wer*100:>7.2f}%  "
          f"{total_err:>5} {total_words:>9}")

    if args.csv:
        import os
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", encoding="utf-8") as f:
            if write_header:
                f.write("tag,corpus_wer,total_err,total_ref_words,backend\n")
            f.write(f"{args.tag},{corpus_wer:.6f},{total_err},"
                    f"{total_words},{backend}\n")

    # Para captura en bash: última línea = WER de corpus (fracción).
    print(f"CORPUS_WER={corpus_wer:.6f}")


if __name__ == "__main__":
    main()
