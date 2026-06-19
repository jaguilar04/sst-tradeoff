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

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from mweralign import mweralign
from mweralign.segmenter import CJSegmenter

from simulstream.metrics.scorers.quality import QualityScorer, QualityScoringSample


@dataclass
class ResegmentedQualityScoringSample:
    """
    A sample containing realigned hypotheses and references.

    Attributes:
        audio_name (str): The identifier of the audio file.
        hypothesis (List[str]): Hypothesis lines after realignment.
        reference (List[str]): Reference lines aligned to the hypothesis.
        source (Optional[List[str]]): Source text (if available).
    """
    audio_name: str
    hypothesis: List[str]
    reference: List[str]
    source: Optional[List[str]] = None


class MWERSegmenterBasedQualityScorer(QualityScorer):
    """
    Abstract base class for scorers that require aligned system outputs and references through
    MWER Segmenter alignment.

    This class wraps a quality scorer and applies the MWER Segmenter alignment by `"Effects of
    automatic alignment on speech translation metrics"
    <https://aclanthology.org/2025.iwslt-1.7/>`_ to hypotheses before scoring.

    Subclasses must implement :meth:`_do_score`, which receives
    :class:`ResegmentedQualityScoringSample` instances, where output and references are aligned.

    Example:
        >>> class CustomQualityScorer(MWERSegmenterBasedQualityScorer):
        ...     def _do_score(self, samples):
        ...         # Compute a custom quality score
        ...         return ...
    """

    def __init__(self, args):
        super().__init__(args)
        self.segmenter = CJSegmenter() if args.latency_unit == "char" else None

    def requires_reference(self) -> bool:
        return True

    @abstractmethod
    def _do_score(self, samples: List[ResegmentedQualityScoringSample]) -> float:
        """
        Compute the final score on resegmented samples.

        This method must be implemented by subclasses.

        Args:
            samples (List[ResegmentedQualityScoringSample]): The aligned
                hypothesis–reference pairs, plus optional sources.

        Returns:
            float: The computed score.
        """
        ...

    def _tokenize(self, text: List[str]) -> List[str]:
        """
        Tokenize text using the segmenter.

        Borrowed from
        https://github.com/mjpost/mweralign/blob/d23a5479/mweralign/mweralign.py#L147
        """
        if self.segmenter is not None:
            tokenized_text = []
            for i in range(len(text)):
                if " ### " in text[i]:
                    pieces = text[i].strip().split(" ### ")
                    encoded = [" ".join(self.segmenter.encode(p)) for p in pieces]
                    tokenized_text.append(" ### ".join(encoded))
                elif "\t" in text[i]:
                    pieces = text[i].strip().split("\t")
                    # underlying C++ binary still uses ###
                    encoded = [" ".join(self.segmenter.encode(p)) for p in pieces]
                    tokenized_text.append(" ### ".join(encoded))
                else:
                    tokenized_text.append(" ".join(self.segmenter.encode(text[i].strip())))
            return "\n".join(tokenized_text)
        else:
            return "\n".join(text)

    def score(self, samples: List[QualityScoringSample]) -> float:
        resegmented_samples = []
        for sample in samples:
            assert sample.reference is not None, "Cannot realign hypothesis to missing reference"
            hypo = self._tokenize([sample.hypothesis])
            refs = self._tokenize(sample.reference)
            resegmented_hypos = mweralign.align_texts(refs, hypo).split("\n")

            assert len(sample.reference) == len(resegmented_hypos), \
                f"Reference ({sample.audio_name}) has mismatched number of target " \
                f"({len(sample.reference)}) and resegmented lines ({len(resegmented_hypos)})"

            if self.segmenter is not None:
                # segmenter.decode will strip() the spaces, but we need them to align with delays
                resegmented_hypos = [
                    hypo.replace(" ", "").replace("_", " ") for hypo in resegmented_hypos]

            resegmented_samples.append(ResegmentedQualityScoringSample(
                sample.audio_name,
                resegmented_hypos,
                sample.reference,
                sample.source
            ))
        return self._do_score(resegmented_samples)
