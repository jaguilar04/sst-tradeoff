"""Política de emisión: Fixed-Chunk / Hold-n (baseline estático).

Espera ciegamente a que se acumulen ``n`` chunks de audio (de 0.96 s cada uno)
antes de transcribir y pasar el texto al traductor. No usa ningún acuerdo
entre hipótesis: cuando le toca emitir, confía en la hipótesis completa actual.

Funciona como grupo de control para demostrar por qué las políticas dinámicas
(local agreement / tolerant agreement) son necesarias.
"""

from __future__ import annotations

from cascade_2026.asr_policies.base import AsrPolicy


class HoldNPolicy(AsrPolicy):
    name = "hold_n"

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"hold_n_chunks debe ser >= 1, recibido: {n}")
        self.n = int(n)

    def should_transcribe(self, step: int) -> bool:
        # Solo se transcribe/emite cada n chunks (step empieza en 1).
        return step % self.n == 0

    def stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        # Sin acuerdo: se compromete la hipótesis completa actual.
        return curr_hypo

    def __repr__(self) -> str:  # pragma: no cover
        return f"HoldNPolicy(n={self.n})"
