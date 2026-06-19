"""Interfaz común de las políticas de emisión del ASR.

Una política decide, en cada step del streaming:

  - ``should_transcribe(step)``: si en este chunk se ejecuta el ASR y se emite
    algo. Por defecto siempre ``True`` (se transcribe en cada chunk). El
    baseline ``hold_n`` lo usa para esperar n chunks antes de transcribir.

  - ``stable_prefix(prev_hypo, curr_hypo)``: dado el texto transcrito en el
    instante anterior y el actual, devuelve la parte que se considera
    "estable" y puede comprometerse (enviarse al traductor / segmentar
    utterance). Es exactamente el punto donde el sistema base aplicaba el
    local agreement (``longest_common_prefix``).
"""

from __future__ import annotations


class AsrPolicy:
    """Clase base de las políticas de emisión del ASR."""

    name: str = "base"

    def should_transcribe(self, step: int) -> bool:  # noqa: D401
        """¿Se transcribe/emite en este step? Por defecto siempre."""
        return True

    def stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - solo logging
        return f"{self.__class__.__name__}(name={self.name!r})"
