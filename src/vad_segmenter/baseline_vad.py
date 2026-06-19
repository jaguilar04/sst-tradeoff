import argparse

from simuleval.utils import entrypoint
from vad_segmenter.baseline_iwslt.vad_segmenter import VADSegmenter
from vad_segmenter.baseline_iwslt.seamless_m4t_agent import SeamlessM4TAgent


def _enrich_agent_args(args):
    """
    SimulEvalWrapper passes a SimpleNamespace directly from YAML.
    Parse optional ``agent_args`` and inject defaults/overrides expected
    by both VAD segmenter and SeamlessM4T agent.
    """
    parser = argparse.ArgumentParser(add_help=False)
    VADSegmenter.add_args(parser)
    SeamlessM4TAgent.add_args(parser)

    # Promote nested YAML ``config`` values to top-level attrs when missing.
    nested = getattr(args, "config", None)
    if nested is not None:
        for key, value in vars(nested).items():
            if not hasattr(args, key):
                setattr(args, key, value)

    # Backward-compatible key aliases from old configs.
    if not hasattr(args, "max_unvoiced_length") and hasattr(args, "pause_length"):
        args.max_unvoiced_length = args.pause_length
    if not hasattr(args, "voice_threshold") and hasattr(args, "vad_threshold"):
        args.voice_threshold = args.vad_threshold

    raw_agent_args = getattr(args, "agent_args", None) or []
    parsed = parser.parse_args(raw_agent_args)

    for key, value in vars(parsed).items():
        setattr(args, key, value)
    return args


@entrypoint
class SeamlessM4TAgentWithVADSegmenter(SeamlessM4TAgent):
    """
    SeamlessM4TAgent that uses VADSegmenter to segment the speech
    """
    def __init__(self, args):
        args = _enrich_agent_args(args)
        super().__init__(args, VADSegmenter(args))

    @staticmethod
    def add_args(parser):
        """
        Add arguments to the parser
        """
        VADSegmenter.add_args(parser)
        SeamlessM4TAgent.add_args(parser)
