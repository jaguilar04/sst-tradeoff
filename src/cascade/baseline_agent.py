import argparse

from cascade.baseline_iwslt.cascade_agent import CascadeAgent
from cascade.baseline_iwslt.fixed_length_segmenter import FixedLengthSegmenter
from simuleval.utils import entrypoint


def _enrich_agent_args(args):
    """
    SimulEvalWrapper passes a SimpleNamespace directly from YAML.
    This helper parses optional ``agent_args`` (CLI-like list) and injects
    defaults/overrides expected by the baseline agent.
    """
    parser = argparse.ArgumentParser(add_help=False)
    FixedLengthSegmenter.add_args(parser)
    CascadeAgent.add_args(parser)

    raw_agent_args = getattr(args, "agent_args", None) or []
    parsed = parser.parse_args(raw_agent_args)

    for key, value in vars(parsed).items():
        setattr(args, key, value)
    return args


@entrypoint
class CascadeAgentWitFixedLengthSegmenter(CascadeAgent):
    """
    WhisperAgent that uses FixedLengthSegmenter to segment the speech
    """
    def __init__(self, args):
        args = _enrich_agent_args(args)
        super().__init__(args, FixedLengthSegmenter(args))

    @staticmethod
    def add_args(parser):
        """
        Add arguments to the parser
        """
        FixedLengthSegmenter.add_args(parser)
        CascadeAgent.add_args(parser)
