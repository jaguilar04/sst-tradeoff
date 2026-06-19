"""Política de emisión de MT: Tolerant Local Agreement (TLA).

Idéntica al Local Agreement estricto salvo en el test de igualdad de tokens: dos
tokens ``a`` (de H_{t-1}) y ``b`` (de H_t) en la misma posición se consideran
equivalentes si su distancia de Levenshtein a nivel de carácter es <= ``tau``. El
prefijo común se calcula bajo esa igualdad relajada y los tokens emitidos son los
de la hipótesis ACTUAL H_t. Esto neutraliza oscilaciones por variación superficial
(mayúsculas, diacríticos, terminaciones morfológicas, artefactos de tokenización)
que atascarían el LA estricto.

``tau`` controla la agresividad: tau=1 absorbe solo formas casi idénticas, tau=2
también cambios flexivos cortos.

DIFERENCIA CLAVE con la versión de ASR (``asr_policies.tolerant_agreement``): en
MT NO se normaliza (ni minúsculas ni quitar puntuación) antes de comparar. En
traducción una distancia de edición de 1-2 puede cambiar el significado (p.ej.
partículas de negación, palabras cortas distintas), no solo la forma, así que se
compara el token tal cual y se vigila la calidad conforme crece tau.
"""

from __future__ import annotations

from cascade_2026.mt_policies.base import MtPolicy
from cascade_2026.mt_policies.text_utils import char_prefix_n_words, levenshtein


class TolerantAgreementPolicy(MtPolicy):
    name = "tolerant_agreement"

    def __init__(self, tau: int = 1):
        if tau < 0:
            raise ValueError(f"mt_tolerant_tau debe ser >= 0, recibido: {tau}")
        self.tau = int(tau)

    def _agreed_word_count(self, prev_words, curr_words) -> int:
        k = 0
        for pw, cw in zip(prev_words, curr_words):
            if pw == cw:
                k += 1
                continue
            if levenshtein(pw, cw) <= self.tau:
                k += 1
                continue
            break
        return k

    def commit(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        prev_words = prev_hypo.split()
        curr_words = curr_hypo.split()
        if not prev_words or not curr_words:
            return committed

        k = self._agreed_word_count(prev_words, curr_words)
        prefix = char_prefix_n_words(curr_hypo, k)

        # Monotonía: nunca devolver menos de lo ya comprometido.
        if len(prefix) < len(committed):
            return committed
        return prefix

    def __repr__(self) -> str:  # pragma: no cover
        return f"TolerantAgreementPolicy(tau={self.tau})"
