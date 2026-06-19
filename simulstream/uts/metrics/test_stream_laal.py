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

import unittest
from argparse import Namespace

from simulstream.metrics.readers import OutputWithDelays, ReferenceSentenceDefinition
from simulstream.metrics.scorers.latency import LatencyScoringSample
from simulstream.metrics.scorers.latency.stream_laal import StreamLaal


class StreamLaalTestCase(unittest.TestCase):
    def test_basic(self):
        reference = [
            ReferenceSentenceDefinition(
                "A New York, sono a capo di un'associazione no profit, chiamata Robin Hood.",
                12.61,
                4.07,
            ),
            ReferenceSentenceDefinition(
                "Quando non combatto la povertà, combatto gli incendi come assistente capitano di "
                "una brigata di pompieri volontari.",
                16.9,
                5.14,
            )
        ]
        hypothesis = OutputWithDelays(
            "Tornando a New York, sono il capo dello sviluppo per un non-profit chiamato Robin "
            "Hood. Quando non sto combattendo la povertà, sto combattendo i fuochi.",
            [14.0, 14.0, 14.0, 14.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 18.0,
             18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 20.0, 20.0, 20.0, 20.0],
            [18.22, 18.22, 18.22, 18.22, 19.93, 19.93, 19.93, 19.93, 19.93, 19.93, 19.93, 19.93,
             19.93, 23.01, 23.01, 23.01, 23.01, 23.01, 23.01, 23.01, 23.01, 27.30, 27.30, 27.30,
             27.30,]
        )
        scorer = StreamLaal(Namespace(latency_unit="word"))
        score = scorer.score([LatencyScoringSample("a", hypothesis, reference)])
        self.assertAlmostEqual(score.ideal_latency, 0.868587, 4)
        self.assertAlmostEqual(score.computational_aware_latency, 5.86, 4)

    def test_with_characters(self):
        reference = [
            ReferenceSentenceDefinition(
                "今天她看起很好，",
                12.61,
                3.07,
            ),
            ReferenceSentenceDefinition(
                "我们一起去公园散步吧。",
                16.9,
                3.14,
            ),
            ReferenceSentenceDefinition(
                "Amy",
                21.0,
                0.5,
            ),
            ReferenceSentenceDefinition(
                "今天心情很好",
                21.5,
                2.0,
            ),
        ]
        hypothesis = OutputWithDelays(
            "今天她很漂亮，我们一起去花园跑步吧。Amy 今天心情很好",
            [14.0, 14.0, 14.0, 15.0, 15.0, 16.0, 17.0,
             17.0, 17.0, 18.0, 18.0, 19.0, 19.0, 20.0, 20.0, 21.0, 21.0, 21.0,
             22.0, 22.0, 22.0, 22.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
            [14.5, 14.5, 14.5, 15.2, 15.2, 16.8, 17.5,
             18.0, 18.5, 18.5, 18.5, 20.1, 20.1, 21.3, 21.3, 22.0, 22.0, 22.0,
             23.0, 23.0, 23.0, 23.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        )
        scorer = StreamLaal(Namespace(latency_unit="char"))
        score = scorer.score([LatencyScoringSample("a", hypothesis, reference)])
        self.assertAlmostEqual(score.ideal_latency, 1.333312, 4)
        self.assertAlmostEqual(score.computational_aware_latency, 2.074095, 4)


if __name__ == '__main__':
    unittest.main()
