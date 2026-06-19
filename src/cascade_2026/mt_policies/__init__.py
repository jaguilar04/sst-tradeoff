"""Políticas de emisión de la TRADUCCIÓN (MT) para el cascade_2026.

Espejo de ``cascade_2026.asr_policies`` pero del lado del target. Cada política
decide, en la región de streaming, qué prefijo de la traducción es estable y se
compromete. Se selecciona desde el config con la clave ``mt_policy``:

    mt_policy: "local_agreement" | "tolerant_agreement" | "wait_k"
             | "hybrid_waitk_tla"

Parámetros por política:
    tolerant_agreement -> mt_tolerant_tau (int, umbral Levenshtein; default 1)
    wait_k             -> wait_k          (int, palabras de desfase; default 3)
    hybrid_waitk_tla   -> wait_k (guardia) + mt_tolerant_tau (crucero)

El Local Agreement estricto es el baseline ya existente del sistema.
"""

from __future__ import annotations

from cascade_2026.mt_policies.base import MtPolicy
from cascade_2026.mt_policies.hybrid_waitk_tla import HybridWaitKTlaPolicy
from cascade_2026.mt_policies.local_agreement import LocalAgreementPolicy
from cascade_2026.mt_policies.tolerant_agreement import TolerantAgreementPolicy
from cascade_2026.mt_policies.wait_k import WaitKPolicy

__all__ = [
    "MtPolicy",
    "HybridWaitKTlaPolicy",
    "LocalAgreementPolicy",
    "TolerantAgreementPolicy",
    "WaitKPolicy",
    "build_policy",
]


def build_policy(config) -> MtPolicy:
    """Construye la política de emisión de MT a partir del config."""
    name = getattr(config, "mt_policy", "local_agreement")

    if name == "local_agreement":
        return LocalAgreementPolicy()

    if name == "tolerant_agreement":
        return TolerantAgreementPolicy(tau=int(getattr(config, "mt_tolerant_tau", 1)))

    if name == "wait_k":
        return WaitKPolicy(k=int(getattr(config, "wait_k", 3)))

    if name == "hybrid_waitk_tla":
        return HybridWaitKTlaPolicy(
            k=int(getattr(config, "wait_k", 3)),
            tau=int(getattr(config, "mt_tolerant_tau", 1)),
        )

    raise ValueError(
        f"mt_policy desconocida: {name!r}. Opciones: local_agreement | "
        f"tolerant_agreement | wait_k | hybrid_waitk_tla"
    )
