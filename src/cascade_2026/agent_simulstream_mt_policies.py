"""SimulStream con MBR parcial + política de emisión de ASR + política de
emisión de MT (traducción) configurables.

Hereda del agente de políticas de ASR
(``cascade_2026.agent_simulstream_asr_policies.CascadeSpeechProcessor``), que ya
aporta MBR parcial (MBR solo al cerrar utterance) y la política de emisión del
ASR seleccionable. Lo único que añade aquí es la **política de emisión de la
traducción**: cómo se decide qué prefijo del target es estable y se compromete en
la región de streaming (``utt_finished=False``).

Para el estudio de políticas de MT se fija la política de ASR en
``tolerant_agreement`` con ``tolerant_tau=1`` (la mejor del barrido de ASR) y se
varía ``mt_policy``:

    mt_policy: "local_agreement"     -> Local Agreement estricto (baseline ya
                                        existente; prefijo común exacto).
             | "tolerant_agreement"  -> Tolerant LA (mt_tolerant_tau).
             | "wait_k"              -> Wait-k fijo (wait_k palabras de desfase).
             | "hybrid_waitk_tla"    -> guardia wait-k (wait_k) + crucero TLA
                                        (mt_tolerant_tau).

El MBR al cerrar utterance no se ve afectado: la política de MT solo gobierna el
prefijo estable de la región de streaming.
"""

import logging
from types import SimpleNamespace

from cascade_2026.agent_simulstream_asr_policies import (
    CascadeSpeechProcessor as _BaseAsrPolicyProcessor,
)
from cascade_2026.mt_policies import build_policy

logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


class CascadeSpeechProcessor(_BaseAsrPolicyProcessor):
    """MBR parcial + política de ASR + política de emisión de MT seleccionables."""

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)

        self.mt_policy = build_policy(config)
        logging.info("[MT-POLICY] política activa: %r", self.mt_policy)

    # ------------------------------------------------------------------ #
    #  Hook de prefijo estable del target: delega en la política activa    #
    # ------------------------------------------------------------------ #
    def _mt_stable_prefix(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        return self.mt_policy.commit(
            prev_hypo, curr_hypo, committed, n_source_words
        )
