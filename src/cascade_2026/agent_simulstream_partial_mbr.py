"""
SimulStream speech processor con MBR *parcial*.

Variante de ``cascade_2026.agent_simulstream_mbr.CascadeSpeechProcessor`` en la
que el coste del epsilon-sampling + MBR solo se paga al **final de utterance**,
es decir, cuando el sistema NO está aplicando local agreement.

Lógica de generación:
    - utt_finished == False  (región de local agreement):
          generación greedy con temperatura 0 y una sola muestra.
          Comportamiento idéntico al sistema sin MBR: se emite el prefijo
          acordado por local agreement entre chunks consecutivos.
    - utt_finished == True   (cierre de utterance, sin local agreement):
          se generan N muestras con epsilon-sampling y se selecciona la mejor
          con el decoder MBRS configurado (rambr_chrf / prunembr_xcomet /
          rerank_kiwi). Así, el segmento que NO pasa por local agreement se
          apoya en el MBR para mejorar su calidad.

Reutiliza el resto de la lógica (ASR, prompts, segmentación, local agreement,
carga de decoders MBRS) de la clase base sin cambios.
"""

import logging
from types import SimpleNamespace
from typing import List

from vllm import SamplingParams

from cascade_2026.agent_simulstream_mbr import (
    CascadeSpeechProcessor as _BaseCascadeSpeechProcessor,
)

logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


class CascadeSpeechProcessor(_BaseCascadeSpeechProcessor):
    """ASR (Qwen3) + LLM (Qwen3) con MBR aplicado solo al final de utterance."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        # Parámetros de muestreo greedy para la región de local agreement
        # (utt_finished=False): una sola muestra, temperatura 0 (argmax),
        # sin epsilon. Equivale al comportamiento del sistema sin MBR.
        self.greedy_sampling_params = SamplingParams(
            n=1,
            top_k=-1,
            temperature=0.0,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
            stop=["\n"],
            seed=123,
        )

        # Flag interno: indica si la generación/selección actual corresponde a
        # un cierre de utterance (MBR activo) o a la región de local agreement
        # (greedy, sin MBR). Lo fija ``_translate_segment`` antes de llamar a
        # ``_llm_generate`` / ``_select_hypothesis``.
        self._mbr_active = False

        # Contadores de uso: cuántos segmentos se traducen con MBR
        # (fin de utterance) vs con local agreement (greedy, sin MBR).
        self._mbr_count = 0
        self._la_count = 0

    # ------------------------------------------------------------------ #
    #  Enrutado MBR-parcial: fija el flag y delega en la clase base        #
    # ------------------------------------------------------------------ #
    def _translate_segment(
        self, state, asr_segment: str, utt_finished: bool
    ) -> str:
        # Solo aplicamos epsilon-sampling + MBR cuando cerramos utterance.
        self._mbr_active = utt_finished

        # Contabilizamos únicamente los segmentos que generan algo (la clase
        # base hace early-return si asr_segment == "").
        if asr_segment != "":
            if utt_finished:
                self._mbr_count += 1
            else:
                self._la_count += 1
            logging.info(
                "[MBR-PARCIAL] modo=%s | mbr_total=%d local_agreement_total=%d",
                "MBR(fin_utt)" if utt_finished else "local_agreement",
                self._mbr_count,
                self._la_count,
            )

        return super()._translate_segment(state, asr_segment, utt_finished)

    # ------------------------------------------------------------------ #
    #  Resumen de uso al cerrar el stream                                   #
    # ------------------------------------------------------------------ #
    def end_of_stream(self):
        result = super().end_of_stream()
        total = self._mbr_count + self._la_count
        pct_mbr = 100.0 * self._mbr_count / total if total else 0.0
        logging.info(
            "[MBR-PARCIAL][RESUMEN] segmentos=%d | MBR(fin_utt)=%d (%.1f%%) | "
            "local_agreement=%d (%.1f%%)",
            total,
            self._mbr_count,
            pct_mbr,
            self._la_count,
            100.0 - pct_mbr if total else 0.0,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Generación LLM: greedy en local agreement, epsilon al cerrar        #
    # ------------------------------------------------------------------ #
    def _llm_generate(self, prompt: str) -> List[str]:
        sampling_params = (
            self.sampling_params if self._mbr_active else self.greedy_sampling_params
        )

        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=sampling_params.temperature,
                top_p=self._top_p,
                stop=["\n"],
                n=sampling_params.n,
                extra_body={"repetition_penalty": self._repetition_penalty},
            )
            return [c.text.replace("…", "") for c in response.choices]

        outputs = self.llm.generate(
            [prompt], sampling_params=sampling_params, use_tqdm=False
        )
        return [o.text.replace("…", "") for o in outputs[0].outputs]

    # ------------------------------------------------------------------ #
    #  Selección de hipótesis: MBR solo al cerrar utterance                #
    # ------------------------------------------------------------------ #
    def _select_hypothesis(self, hypotheses: List[str], source: str) -> str:
        if not self._mbr_active:
            # Región de local agreement: sin MBR, primera hipótesis no vacía.
            for h in hypotheses:
                if h.strip():
                    return h
            return ""
        return super()._select_hypothesis(hypotheses, source)
