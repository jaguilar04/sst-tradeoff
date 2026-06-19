"""SimulStream con MBR parcial + políticas de emisión de ASR configurables.

Hereda toda la lógica del agente de MBR *parcial*
(``cascade_2026.agent_simulstream_partial_mbr.CascadeSpeechProcessor``): MBR solo
al cerrar utterance (utt_finished=True), greedy/local agreement en la región de
streaming (utt_finished=False). Lo único que cambia aquí es la **política de
emisión del ASR**, es decir, cómo se decide qué parte de la transcripción es
estable y cuándo se emite.

Políticas (clave de config ``asr_policy``):
    - "local_agreement"    : prefijo común exacto (comportamiento base).
    - "hold_n"             : baseline estático; espera n chunks antes de emitir.
    - "tolerant_agreement" : LACP; tolera tau ediciones (Levenshtein) por palabra.

Además, si se indica ``asr_transcript_log`` en el config, se vuelca un JSONL con
la transcripción de cada step (para calcular WER offline) y, al cerrar cada
audio, una línea ``{"type":"final", ...}`` con la transcripción comprometida.
"""

import json
import logging
import os
from types import SimpleNamespace

import numpy as np
from simulstream.server.speech_processors.incremental_output import IncrementalOutput

from cascade_2026.agent_simulstream_partial_mbr import (
    CascadeSpeechProcessor as _BasePartialMbrProcessor,
)
from cascade_2026.asr_policies import build_policy

logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


class CascadeSpeechProcessor(_BasePartialMbrProcessor):
    """Partial-MBR + política de emisión de ASR seleccionable por config."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        self.asr_policy = build_policy(config)
        logging.info("[ASR-POLICY] política activa: %r", self.asr_policy)

        # Contador de chunks (steps) por audio. Se reinicia al cambiar de audio.
        self._step = 0
        self._policy_speech_id = self._state.speech_id

        # Prefijo estable calculado en el último _transcribe_audio (para el log).
        self._last_stable_prefix = None

        # Fuentes comprometidas del audio en curso (para la transcripción final).
        self._committed_this_speech = []

        # Log de transcript por step (opcional, para WER offline).
        self._transcript_log_path = getattr(config, "asr_transcript_log", None)
        self._transcript_fh = None
        if self._transcript_log_path:
            log_dir = os.path.dirname(self._transcript_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._transcript_fh = open(
                self._transcript_log_path, "w", encoding="utf-8"
            )

    # ------------------------------------------------------------------ #
    #  Hook de prefijo estable: delega en la política activa               #
    # ------------------------------------------------------------------ #
    def _asr_stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        prefix = self.asr_policy.stable_prefix(prev_hypo, curr_hypo)
        self._last_stable_prefix = prefix
        return prefix

    # ------------------------------------------------------------------ #
    #  Gating temporal (hold-n) + contador de steps por audio              #
    # ------------------------------------------------------------------ #
    def _reset_step_if_new_speech(self) -> None:
        if self._state.speech_id != self._policy_speech_id:
            self._policy_speech_id = self._state.speech_id
            self._step = 0
            self._committed_this_speech = []

    def process_chunk(self, waveform):
        self._reset_step_if_new_speech()
        self._step += 1

        if not self.asr_policy.should_transcribe(self._step):
            # hold-n: acumulamos el audio del chunk pero no transcribimos/emitimos.
            if waveform is not None and len(waveform) > 0:
                self._state.source = np.concatenate(
                    [self._state.source, np.asarray(waveform, dtype=np.float32)]
                )
            self._log_skip()
            return IncrementalOutput([], "", [], "")

        return super().process_chunk(waveform)

    # ------------------------------------------------------------------ #
    #  Transcripción: envuelve la base y vuelca el log por step            #
    # ------------------------------------------------------------------ #
    def _transcribe_audio(self, state):
        self._last_stable_prefix = None
        result = super()._transcribe_audio(state)
        self._log_step(state, result)
        return result

    def end_of_stream(self):
        finished_speech_id = self._state.speech_id
        result = super().end_of_stream()
        self._log_final(finished_speech_id)
        return result

    # ------------------------------------------------------------------ #
    #  Volcado del transcript (JSONL)                                      #
    # ------------------------------------------------------------------ #
    def _write_record(self, record: dict) -> None:
        if self._transcript_fh is None:
            return
        self._transcript_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._transcript_fh.flush()

    def _log_skip(self) -> None:
        if self._transcript_fh is None:
            return
        self._write_record(
            {
                "step": self._step,
                "speech_id": self._state.speech_id,
                "skipped": True,
            }
        )

    def _log_step(self, state, result) -> None:
        if self._transcript_fh is None:
            return
        asr_to_translate, utt_finished = result
        committed = ""
        if utt_finished and state.utt_sources:
            committed = state.utt_sources[-1]
            self._committed_this_speech.append(committed)
        self._write_record(
            {
                "step": self._step,
                "speech_id": state.speech_id,
                "policy": self.asr_policy.name,
                "curr_hypo": state.asr_hypotheses[-1] if state.asr_hypotheses else "",
                "stable_prefix": self._last_stable_prefix,
                "utt_finished": bool(utt_finished),
                "asr_to_translate": asr_to_translate or "",
                "committed_source": committed,
            }
        )

    def _log_final(self, finished_speech_id: int) -> None:
        if self._transcript_fh is None:
            return
        # Transcripción comprometida del audio + cola no cerrada (último parcial).
        trailing = ""
        if self._state.asr_hypotheses:
            trailing = self._state.asr_hypotheses[-1].strip()
        parts = [p for p in self._committed_this_speech if p]
        if trailing:
            parts.append(trailing)
        self._write_record(
            {
                "type": "final",
                "speech_id": finished_speech_id,
                "policy": self.asr_policy.name,
                "transcript": " ".join(parts).strip(),
            }
        )
