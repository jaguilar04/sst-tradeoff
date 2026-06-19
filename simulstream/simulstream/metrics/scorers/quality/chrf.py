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

import argparse
import logging
from typing import List

from sacrebleu import CHRF

from simulstream.metrics.scorers.quality import register_quality_scorer
from simulstream.metrics.scorers.quality.mwersegmenter import MWERSegmenterBasedQualityScorer, \
    ResegmentedQualityScoringSample


LOGGER = logging.getLogger('simulstream.metrics.scorers.quality.chrf')


@register_quality_scorer("chrf")
class ChrFScorer(MWERSegmenterBasedQualityScorer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.chrf = CHRF(char_order=args.char_order, word_order=args.word_order)

    def _do_score(self, samples: List[ResegmentedQualityScoringSample]) -> float:
        hypotheses = []
        references = []
        for sample in samples:
            hypotheses.extend(sample.hypothesis)
            references.extend(sample.reference)
        score = self.chrf.corpus_score(hypotheses, [references])
        LOGGER.info(f"chrF detailed score: {score}")
        return score.score

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--char-order", type=int, default=6)
        parser.add_argument("--word-order", type=int, default=0)

    def requires_source(self) -> bool:
        return False
