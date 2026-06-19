"""Políticas de emisión del ASR para el cascade_2026.

Cada política decide qué parte de la transcripción es estable y cuándo se
emite. Se selecciona desde el config con la clave ``asr_policy``:

    asr_policy: "local_agreement" | "hold_n" | "tolerant_agreement"

Parámetros por política:
    hold_n            -> hold_n_chunks (int, nº de chunks de 0.96 s a esperar)
    tolerant_agreement-> tolerant_tau (int, umbral Levenshtein; default 2)
"""

from __future__ import annotations

from cascade_2026.asr_policies.base import AsrPolicy
from cascade_2026.asr_policies.hold_n import HoldNPolicy
from cascade_2026.asr_policies.local_agreement import LocalAgreementPolicy
from cascade_2026.asr_policies.tolerant_agreement import TolerantAgreementPolicy

__all__ = [
    "AsrPolicy",
    "HoldNPolicy",
    "LocalAgreementPolicy",
    "TolerantAgreementPolicy",
    "build_policy",
]


def build_policy(config) -> AsrPolicy:
    """Construye la política de emisión del ASR a partir del config."""
    name = getattr(config, "asr_policy", "local_agreement")

    if name == "local_agreement":
        return LocalAgreementPolicy()

    if name == "hold_n":
        return HoldNPolicy(int(getattr(config, "hold_n_chunks", 2)))

    if name == "tolerant_agreement":
        return TolerantAgreementPolicy(tau=int(getattr(config, "tolerant_tau", 2)))

    raise ValueError(
        f"asr_policy desconocida: {name!r}. "
        f"Opciones: local_agreement | hold_n | tolerant_agreement"
    )
