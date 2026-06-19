"""Política de emisión de MT: Wait-k (política fija, content-agnostic).

Espera a haber leído ``k`` palabras de fuente antes de emitir el primer token de
target y, a partir de ahí, mantiene un desfase constante de ``k`` palabras entre
los flujos de fuente y target: en cada paso el nº de palabras de traducción
comprometidas es ``max(0, palabras_fuente_leidas - k)``, recortadas a las que ya
existen en la hipótesis actual.

Es determinista y NO mira la hipótesis: solo cuenta palabras de fuente leídas.
Propiedades: latencia plana y predecible, sin picos (el lag es constante por
construcción), pero sin capacidad de esperar más cuando la fuente es ambigua ni
de emitir más rápido cuando la traducción es localmente segura. Con k pequeño
(k=1) el modelo compromete casi sin contexto de fuente, lo que puede causar
errores de anticipación o salidas inestables al inicio de cada frase; k mayor
cambia latencia por estabilidad.

Al cerrar la frase/utterance (``utt_finished=True``, gestionado por el agente),
los tokens de target restantes se vuelcan (flush).
"""

from __future__ import annotations

from cascade_2026.mt_policies.base import MtPolicy
from cascade_2026.mt_policies.text_utils import char_prefix_n_words


class WaitKPolicy(MtPolicy):
    name = "wait_k"

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError(f"wait_k debe ser >= 1, recibido: {k}")
        self.k = int(k)

    def commit(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        curr_words = curr_hypo.split()

        # Desfase fijo k: nº de palabras de target a comprometer.
        target_words = max(0, n_source_words - self.k)
        # Monotonía: nunca por debajo de lo ya comprometido.
        target_words = max(target_words, len(committed.split()))
        # No comprometer más palabras de las que existen en la hipótesis actual.
        target_words = min(target_words, len(curr_words))

        return char_prefix_n_words(curr_hypo, target_words)

    def __repr__(self) -> str:  # pragma: no cover
        return f"WaitKPolicy(k={self.k})"
