"""Política de emisión de MT: Local Agreement estricto (LCP exacto).

Solo se compromete como estable el prefijo que coincide **exactamente**
(carácter a carácter) entre la hipótesis de traducción del instante t-1 y la de
t. Es el comportamiento original del cascade_2026; se incluye aquí como baseline
adaptativo y referencia (el sistema ya lo tenía).

Propiedad esperada: buena calidad (solo se compromete contenido estable) pero
vulnerable a picos de latencia: cuando dos hipótesis consecutivas oscilan en un
único token (sinónimo, puntuación, mayúscula, flexión), el LCP se atasca en esa
posición y no se emite nada durante varios steps, generando una cola pesada en
la distribución de lag por token.
"""

from __future__ import annotations

from cascade_2026.mt_policies.base import MtPolicy
from cascade_2026.mt_policies.text_utils import longest_common_prefix


class LocalAgreementPolicy(MtPolicy):
    name = "local_agreement"

    def commit(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        return longest_common_prefix(prev_hypo, curr_hypo)
