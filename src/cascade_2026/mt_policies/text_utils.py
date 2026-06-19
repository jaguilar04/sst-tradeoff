"""Utilidades de texto compartidas por las políticas de emisión de MT."""

from __future__ import annotations


def longest_common_prefix(s1: str, s2: str) -> str:
    """Prefijo común exacto (a nivel de carácter) entre dos cadenas."""
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


def char_prefix_n_words(text: str, k: int) -> str:
    """Prefijo de caracteres de ``text`` que cubre exactamente sus k primeras
    palabras (separadas por espacios).

    Garantiza que el resultado es un prefijo REAL de ``text`` (no reconstruido
    con join), condición necesaria para que el cálculo del incremento
    ``nuevo[len(comprometido):]`` del agente sea correcto.
    """
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
