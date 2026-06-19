"""Política de emisión: Local Agreement clásico (LCP exacto).

Solo se emite como estable el prefijo que coincide **exactamente**
(carácter a carácter) entre la predicción del instante t-1 y la de t.
Es el comportamiento original del cascade_2026; sirve de referencia para
comprobar que las políticas nuevas no rompen nada.
"""

from __future__ import annotations

from cascade_2026.asr_policies.base import AsrPolicy


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


class LocalAgreementPolicy(AsrPolicy):
    name = "local_agreement"

    def stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        return longest_common_prefix(prev_hypo, curr_hypo)
