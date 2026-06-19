# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import copy
import unittest
from argparse import Namespace

from simulstream.metrics.scorers.quality.mwersegmenter import (
    MWERSegmenterBasedQualityScorer,
)
from simulstream.metrics.scorers.latency.mwersegmenter import (
    MWERSegmenterBasedLatencyScorer,
)
from simulstream.metrics.scorers.latency import LatencyScores


class TokenizeNoInplaceModificationTestCase(unittest.TestCase):
    """
    Ensures that _tokenize does not alter the references.
    See https://github.com/hlt-mt/simulstream/pull/20#issuecomment-3960951980
    """

    def _make_quality_scorer(self, latency_unit="char"):
        """Create a concrete subclass of the abstract quality scorer."""
        class _Scorer(MWERSegmenterBasedQualityScorer):
            def _do_score(self, samples):
                return 0.0

            @classmethod
            def add_arguments(cls, parser):
                pass

            def requires_source(self):
                return False

        args = Namespace(latency_unit=latency_unit)
        return _Scorer(args)

    def _make_latency_scorer(self, latency_unit="char"):
        """Create a concrete subclass of the abstract latency scorer."""
        class _Scorer(MWERSegmenterBasedLatencyScorer):
            def _do_score(self, samples):
                return LatencyScores(0.0, [])

            @classmethod
            def add_arguments(cls, parser):
                pass

            def requires_source(self):
                return False

        args = Namespace(latency_unit=latency_unit)
        return _Scorer(args)

    def test_quality_tokenize_does_not_modify_input(self):
        scorer = self._make_quality_scorer(latency_unit="char")
        text = ["你好世界", "这是测试"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_latency_tokenize_does_not_modify_input(self):
        scorer = self._make_latency_scorer(latency_unit="char")
        text = ["你好世界", "这是测试"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_quality_tokenize_no_modify_with_separator(self):
        scorer = self._make_quality_scorer(latency_unit="char")
        text = ["你好 ### 世界"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_quality_tokenize_no_modify_with_tab(self):
        scorer = self._make_quality_scorer(latency_unit="char")
        text = ["你好\t世界"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_quality_tokenize_does_not_modify_input_english(self):
        scorer = self._make_quality_scorer(latency_unit="word")
        text = ["hello world", "this is a test"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_latency_tokenize_does_not_modify_input_english(self):
        scorer = self._make_latency_scorer(latency_unit="word")
        text = ["hello world", "this is a test"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_quality_tokenize_no_modify_with_separator_english(self):
        scorer = self._make_quality_scorer(latency_unit="word")
        text = ["hello ### world"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)

    def test_quality_tokenize_no_modify_with_tab_english(self):
        scorer = self._make_quality_scorer(latency_unit="word")
        text = ["hello\tworld"]
        original = copy.deepcopy(text)
        scorer._tokenize(text)
        self.assertEqual(text, original)


if __name__ == '__main__':
    unittest.main()
