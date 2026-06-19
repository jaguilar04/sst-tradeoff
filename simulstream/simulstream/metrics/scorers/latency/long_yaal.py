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
import statistics
from typing import List

from simulstream.metrics.readers import text_items
from simulstream.metrics.scorers.latency import register_latency_scorer, LatencyScores
from simulstream.metrics.scorers.latency.softsegmenter import SoftSegmenterBasedLatencyScorer, \
    ResegmentedLatencyScoringSample


LOGGER = logging.getLogger('simulstream.metrics.scorers.latency.long_yaal')


@register_latency_scorer("long_yaal")
class LongYAAL(SoftSegmenterBasedLatencyScorer):
    """
    Computes LongYAAL using OmniSTEval's YAALScorer (https://github.com/pe-trik/OmniSTEval).

    Unlike StreamLAAL, which stops accumulating lag when the delay exceeds the segment duration,
    LongYAAL processes all emitted tokens regardless of segment boundaries. This makes it
    suitable for long-form simultaneous speech translation where the system continues emitting
    tokens beyond the nominal segment end time.
    """

    def _do_score(self, samples: List[ResegmentedLatencyScoringSample]) -> LatencyScores:
        # Import perezoso: omnisteval solo es necesario si se usa LongYAAL.
        # Importar este módulo lo hace el registro de scorers de latencia, así
        # que mantener el import a nivel de módulo rompía otros scorers (p.ej.
        # stream_laal) cuando omnisteval no está disponible en el intérprete.
        from omnisteval import YAALScorer, Instance
        cu_scorer = YAALScorer(computation_aware=False, is_longform=True)
        ca_scorer = YAALScorer(computation_aware=True, is_longform=True)

        instances = []
        skipped = 0

        for sample in samples:
            for sentence_output, sentence_ref in zip(sample.hypothesis, sample.reference):
                if not sentence_output.ideal_delays:
                    skipped += 1
                    continue

                # OmniSTEval expects delays in milliseconds relative to segment start.
                # SimulStream stores absolute delays in seconds, so we convert both.
                start_s = sentence_ref.start_time
                instances.append(Instance(
                    prediction=sentence_ref.content,
                    reference=sentence_ref.content,
                    source_length=sentence_ref.duration * 1000.0,
                    emission_cu=[(d - start_s) * 1000.0 for d in sentence_output.ideal_delays],
                    emission_ca=[(d - start_s) * 1000.0 for d in sentence_output.computational_aware_delays],
                    latency_unit=self.latency_unit,
                    longform=True,
                ))

        if skipped:
            LOGGER.warning(
                f"{skipped} sentences skipped in LongYAAL computation (empty hypothesis)")

        cu_scores = [s for s in (cu_scorer.compute(ins) for ins in instances) if s is not None]
        ca_scores = [s for s in (ca_scorer.compute(ins) for ins in instances) if s is not None]

        # Convert from milliseconds back to seconds to keep output consistent with other scorers.
        ideal_latency = statistics.mean(cu_scores) / 1000.0 if cu_scores else 0.0
        ca_latency = statistics.mean(ca_scores) / 1000.0 if ca_scores else 0.0
        return LatencyScores(ideal_latency, ca_latency)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass
