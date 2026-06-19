"""Política de emisión: Tolerant Agreement (LACP).

Variante flexible del local agreement. En vez de exigir coincidencia exacta
carácter a carácter, permite un pequeño margen de error por palabra mediante un
umbral de Distancia de Levenshtein (``tau``). Así, las correcciones menores que
el ASR hace al final de la frase (mayúsculas, puntuación, faltas de ortografía,
plurales...) no rompen el prefijo común y no provocan picos de latencia.

Algoritmo (sobre H_{t-1} y H_t):

  1. Normalización: minúsculas + eliminación de toda la puntuación.
  2. Segmentación en palabras (token a token, preservando el índice).
  3. Comparación secuencial de izquierda a derecha. Para cada posición i:
       - si las palabras normalizadas coinciden → aceptada.
       - si no, se calcula Levenshtein(char-level) entre ambas:
            * distancia <= tau → se tolera y se acepta.
            * distancia  > tau → se rompe el bucle.
  4. El prefijo común son todas las palabras aceptadas. Por defecto se emite la
     forma de H_{t-1} (``emit_from='previous'``), fiel a la definición del LACP;
     con ``emit_from='current'`` se emite la forma más reciente del modelo.

En las configuraciones de IWSLT 2025/2026 el umbral por defecto es tau = 2.
"""

from __future__ import annotations

import string

from cascade_2026.asr_policies.base import AsrPolicy

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_token(token: str) -> str:
    """Minúsculas + sin puntuación, para un único token."""
    return token.lower().translate(_PUNCT_TABLE)


def _char_prefix_n_words(text: str, k: int) -> str:
    """Prefijo de caracteres de ``text`` que cubre exactamente sus k primeras
    palabras (separadas por espacios). Garantiza que el resultado es un prefijo
    real de ``text`` (necesario para la alineación de timestamps de la base)."""
    if k <= 0:
        return ""
    count = 0
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] == " ":
            i += 1
        start = i
        while i < n and text[i] != " ":
            i += 1
        if i > start:
            count += 1
            if count == k:
                return text[:i]
    return text


def levenshtein(a: str, b: str) -> int:
    """Distancia de edición a nivel de carácter (programación dinámica)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[len(b)]


class TolerantAgreementPolicy(AsrPolicy):
    name = "tolerant_agreement"

    def __init__(self, tau: int = 2):
        if tau < 0:
            raise ValueError(f"tolerant_tau debe ser >= 0, recibido: {tau}")
        self.tau = int(tau)

    def _agreed_word_count(self, prev_norm, curr_norm) -> int:
        k = 0
        for pw, cw in zip(prev_norm, curr_norm):
            if pw == cw:
                k += 1
                continue
            if levenshtein(pw, cw) <= self.tau:
                k += 1
                continue
            break
        return k

    def stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        prev_words = prev_hypo.split()
        curr_words = curr_hypo.split()
        if not prev_words or not curr_words:
            return ""

        # Normalización token a token (preserva la alineación de índices con
        # las palabras originales, aunque un token quede vacío tras normalizar).
        prev_norm = [_normalize_token(w) for w in prev_words]
        curr_norm = [_normalize_token(w) for w in curr_words]

        k = self._agreed_word_count(prev_norm, curr_norm)
        if k == 0:
            return ""

        # IMPORTANTE: devolvemos un prefijo de CARACTERES real de la hipótesis
        # ACTUAL que cubre exactamente las k palabras acordadas. La base
        # (find_end_time) indexa los timestamps sobre la hipótesis actual, así
        # que el prefijo estable debe ser un prefijo de caracteres de curr_hypo
        # —igual que el longest_common_prefix original— o los índices se van de
        # rango (IndexError). Las palabras toleradas usan, por tanto, la forma
        # más reciente del ASR (normalmente la corregida).
        return _char_prefix_n_words(curr_hypo, k)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TolerantAgreementPolicy(tau={self.tau})"
