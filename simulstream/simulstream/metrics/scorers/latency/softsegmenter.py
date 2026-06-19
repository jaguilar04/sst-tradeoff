# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
SoftSegmenter-based latency scorer for simulstream.

Ports the SoftSegmenter alignment algorithm from OmniSTEval
(https://github.com/pe-trik/OmniSTEval) to replace MWER-based resegmentation.
The algorithm uses a dynamic programming alignment (Needleman-Wunsch-like) with
Jaccard character-set similarity instead of edit distance.

Reference:
    "Better Late Than Never: Evaluation of Latency Metrics for Simultaneous
    Speech-to-Text Translation" (https://arxiv.org/abs/2509.17349)
"""

import unicodedata
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from simulstream.metrics.readers import (
    ReferenceSentenceDefinition, OutputWithDelays, text_items)
from simulstream.metrics.scorers.latency import LatencyScorer, LatencyScoringSample, LatencyScores


_INF = float("inf")
_ALL_PUNCT = set(
    [".", "!", "?", ",", ";", ":", "-", "(", ")",
     "。", "！", "？", "，", "；", "：", "—", "（", "）", "ー"])


class _AlignOp(IntEnum):
    MATCH = 0
    DELETE = 1
    INSERT = 2
    NONE = 3


@dataclass
class _Word:
    text: str
    seq_id: Optional[int] = None
    emission_cu: Optional[float] = None
    emission_ca: Optional[float] = None


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _similarity(ref_text: str, hyp_text: str, char_level: bool) -> float:
    """Jaccard character-set similarity with punctuation mismatch penalty."""
    ref_is_punct = ref_text in _ALL_PUNCT
    hyp_is_punct = hyp_text in _ALL_PUNCT
    if ref_is_punct ^ hyp_is_punct:
        return -_INF
    if char_level:
        return float(ref_text == hyp_text)
    ref_set = set(ref_text)
    hyp_set = set(hyp_text)
    inter = len(ref_set & hyp_set)
    union = len(ref_set) + len(hyp_set) - inter
    return (inter / union) if union else 0.0


def _align_sequences(
        seq1: List[_Word], seq2: List[_Word], char_level: bool):
    """
    Dynamic programming alignment maximising Jaccard similarity (no gap penalties).
    seq1 = reference words, seq2 = hypothesis words.
    Returns two aligned lists with None for gaps.
    """
    n = len(seq1) + 1
    m = len(seq2) + 1
    dp = [[0.0] * m for _ in range(n)]
    dp_back = [[_AlignOp.NONE] * m for _ in range(n)]

    for i in range(n):
        dp_back[i][0] = _AlignOp.DELETE
    for j in range(m):
        dp_back[0][j] = _AlignOp.INSERT
    dp_back[0][0] = _AlignOp.MATCH

    for i in range(1, n):
        for j in range(1, m):
            match = dp[i - 1][j - 1] + _similarity(seq1[i - 1].text, seq2[j - 1].text, char_level)
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            if match >= delete and match >= insert:
                dp[i][j] = match
                dp_back[i][j] = _AlignOp.MATCH
            elif delete >= insert:
                dp[i][j] = delete
                dp_back[i][j] = _AlignOp.DELETE
            else:
                dp[i][j] = insert
                dp_back[i][j] = _AlignOp.INSERT

    aligned1: List[Optional[_Word]] = []
    aligned2: List[Optional[_Word]] = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        op = dp_back[i][j]
        if op == _AlignOp.MATCH:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif op == _AlignOp.DELETE:
            aligned1.append(seq1[i - 1])
            aligned2.append(None)
            i -= 1
        else:
            aligned1.append(None)
            aligned2.append(seq2[j - 1])
            j -= 1
    aligned1.reverse()
    aligned2.reverse()
    return aligned1, aligned2


def _process_alignment(
        ref_words: List[Optional[_Word]],
        hyp_words: List[Optional[_Word]],
        char_level: bool) -> List[_Word]:
    """
    Assign seq_ids to hypothesis words based on the alignment.
    Hypothesis words aligned to reference gaps are re-assigned to the nearest
    reference segment using similarity lookahead/lookback.
    """
    assert len(ref_words) == len(hyp_words)

    def get_next_non_none_ref(idx):
        while idx < len(ref_words) and ref_words[idx] is None:
            idx += 1
        return (idx, ref_words[idx]) if idx < len(ref_words) else (idx, None)

    result: List[_Word] = []
    last_ref: Optional[_Word] = None
    nexti = 0
    lookahead_ref: Optional[_Word] = None

    for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
        if ref is None and i >= nexti:
            if hyp is not None:
                nexti, next_ref = get_next_non_none_ref(i)
                assert next_ref is not None or last_ref is not None, \
                    "No reference word found for unaligned hypothesis word."
                next_score = _similarity(hyp.text, next_ref.text, char_level) \
                    if next_ref is not None else -_INF
                prev_score = _similarity(hyp.text, last_ref.text, char_level) \
                    if last_ref is not None else -_INF
                if next_score > prev_score:
                    ref = next_ref
                    lookahead_ref = next_ref
                else:
                    ref = last_ref
                    nexti = i
                    lookahead_ref = None
        elif ref is None and hyp is not None:
            # In the lookahead zone: assign to the same ref chosen on entry.
            ref = lookahead_ref

        if ref is not None and i >= nexti:
            last_ref = ref
        if hyp is not None:
            if ref is None:
                continue
            hyp.seq_id = ref.seq_id
            result.append(hyp)

    return result


@dataclass
class ResegmentedLatencyScoringSample:
    """
    Latency scoring sample after SoftSegmenter resegmentation.

    Attributes:
        audio_name: Identifier of the audio file.
        hypothesis: One OutputWithDelays per reference sentence,
            containing only the delays assigned to that segment.
        reference: Reference sentence definitions.
    """
    audio_name: str
    hypothesis: List[OutputWithDelays]
    reference: List[ReferenceSentenceDefinition]


class SoftSegmenterBasedLatencyScorer(LatencyScorer):
    """
    Abstract base class for latency scorers that use SoftSegmenter alignment.

    Replaces MWER-based resegmentation with the Jaccard-similarity dynamic
    programming alignment from OmniSTEval, following "Better Late Than Never:
    Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation"
    (https://arxiv.org/abs/2509.17349).
    """

    def __init__(self, args):
        super().__init__(args)
        self.latency_unit = args.latency_unit
        self.char_level = (args.latency_unit == "char")

    def requires_reference(self) -> bool:
        return True

    @abstractmethod
    def _do_score(self, samples: List[ResegmentedLatencyScoringSample]) -> LatencyScores:
        ...

    def score(self, samples: List[LatencyScoringSample]) -> LatencyScores:
        resegmented_samples = []
        for sample in samples:
            assert sample.reference is not None, \
                "Cannot run SoftSegmenter alignment without reference."

            # Build reference Word objects: one seq_id per sentence
            ref_words: List[_Word] = []
            for seq_id, ref_sent in enumerate(sample.reference):
                for item in text_items(ref_sent.content, self.latency_unit):
                    ref_words.append(_Word(text=_normalize(item), seq_id=seq_id))

            # Build hypothesis Word objects with their absolute delays
            hyp_items = text_items(sample.hypothesis.final_text, self.latency_unit)
            ideal_delays = sample.hypothesis.ideal_delays
            ca_delays = sample.hypothesis.computational_aware_delays
            assert len(hyp_items) == len(ideal_delays), (
                f"{sample.audio_name}: hypothesis words ({len(hyp_items)}) "
                f"!= ideal delays ({len(ideal_delays)})")
            hyp_words: List[_Word] = [
                _Word(text=_normalize(item), emission_cu=cu, emission_ca=ca)
                for item, cu, ca in zip(hyp_items, ideal_delays, ca_delays)
            ]

            # Align and assign seq_ids to hypothesis words
            aligned_ref, aligned_hyp = _align_sequences(ref_words, hyp_words, self.char_level)
            aligned_hyp_with_ids = _process_alignment(aligned_ref, aligned_hyp, self.char_level)

            # Group delays by reference segment
            n_segs = len(sample.reference)
            seg_ideal: List[List[float]] = [[] for _ in range(n_segs)]
            seg_ca: List[List[float]] = [[] for _ in range(n_segs)]
            for word in aligned_hyp_with_ids:
                if word.seq_id is not None and 0 <= word.seq_id < n_segs:
                    seg_ideal[word.seq_id].append(word.emission_cu)
                    seg_ca[word.seq_id].append(word.emission_ca)

            hyp_segments = [
                OutputWithDelays("", seg_ideal[i], seg_ca[i])
                for i in range(n_segs)
            ]
            resegmented_samples.append(ResegmentedLatencyScoringSample(
                sample.audio_name, hyp_segments, sample.reference))

        return self._do_score(resegmented_samples)
