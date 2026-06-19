"""Interfaz común de las políticas de emisión de la TRADUCCIÓN (MT).

Espejo de ``cascade_2026.asr_policies`` pero del lado del target. Una política
decide, en la región de streaming (``utt_finished=False``), qué prefijo de la
traducción se considera estable y puede comprometerse/emitirse.

En el agente base la decisión es un ``longest_common_prefix`` entre la hipótesis
de traducción del instante anterior y la actual (Local Agreement estricto). Aquí
ese punto se generaliza a una política configurable.

Contrato de ``commit``:

  - ``prev_hypo``       : hipótesis de traducción COMPLETA del step t-1.
  - ``curr_hypo``       : hipótesis de traducción COMPLETA del step t.
  - ``committed``       : prefijo ya comprometido hasta ahora (es prefijo de
                          ambas hipótesis; nunca se retracta).
  - ``n_source_words``  : nº de palabras de fuente (ASR) leídas para la frase en
                          curso. Lo usan las políticas content-agnostic (wait-k).

  Debe devolver el NUEVO prefijo comprometido COMPLETO, que ha de ser:
    1. un prefijo de caracteres real de ``curr_hypo``, y
    2. de longitud >= ``committed`` (monotonía: no se borra lo ya emitido).

  El agente calcula el incremento emitido como ``nuevo[len(committed):]``.
"""

from __future__ import annotations


class MtPolicy:
    """Clase base de las políticas de emisión de la traducción."""

    name: str = "base"

    def commit(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - solo logging
        return f"{self.__class__.__name__}(name={self.name!r})"
