"""Política de emisión de MT: híbrido Wait-k + Local Agreement flexible (TLA).

Política en dos fases, a nivel de frase:

  Fase 1 (guardia): al comienzo de cada nueva frase de target (tras comprometer
    un límite de frase, o al inicio del stream) se impone una condición wait-k:
    no se puede emitir ningún token de la nueva frase hasta haber leído al menos
    ``k`` palabras de fuente de la frase en curso. Esto evita el fallo conocido
    de las políticas de acuerdo al inicio de frase, donde el modelo casi no tiene
    contexto de fuente y dos hipótesis consecutivas pueden coincidir en aperturas
    alucinadas.

  Fase 2 (crucero): una vez satisfecha la guardia, la emisión la gobierna el
    Tolerant Local Agreement (``tau``).

Los dos parámetros (``k`` para la guardia, ``tau`` para el crucero) son
independientes y se barren por separado. El diseño sigue el principio del sistema
MLLP-VRAIN de IWSLT 2025, que combinaba RALCP con un wait-k activo solo al inicio
de cada frase para prevenir alucinaciones.

Nota de implementación: en este agente una "frase nueva" equivale a una utterance
recién cerrada, tras lo cual ``committed`` se reinicia y ``n_source_words`` vuelve
a contar desde la frase en curso, de modo que la guardia se re-arma sola en cada
frase.
"""

from __future__ import annotations

from cascade_2026.mt_policies.base import MtPolicy
from cascade_2026.mt_policies.tolerant_agreement import TolerantAgreementPolicy


class HybridWaitKTlaPolicy(MtPolicy):
    name = "hybrid_waitk_tla"

    def __init__(self, k: int = 3, tau: int = 1):
        if k < 1:
            raise ValueError(f"wait_k debe ser >= 1, recibido: {k}")
        self.k = int(k)
        self.tau = int(tau)
        self._tla = TolerantAgreementPolicy(tau=self.tau)

    def commit(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        # Guardia wait-k: hasta leer k palabras de fuente de la frase en curso no
        # se emite nada. n_source_words se reinicia por frase, así que esto solo
        # bloquea el arranque de cada frase nueva.
        if n_source_words < self.k:
            return committed

        # Crucero: Tolerant Local Agreement.
        return self._tla.commit(prev_hypo, curr_hypo, committed, n_source_words)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HybridWaitKTlaPolicy(k={self.k}, tau={self.tau})"
