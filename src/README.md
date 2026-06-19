# src — Custom Agents and Policies

This directory contains all custom streaming translation agents and decoding policies implemented for the thesis. Each subdirectory corresponds to a distinct system variant.

---

## Directory Structure

```
src/
├── cascade/              # Original cascade baseline
├── cascade_2026/         # Enhanced cascade with advanced policies and MBR
│   ├── asr_policies/     # ASR emission policies
│   └── mt_policies/      # MT decoding policies
├── fixed_segmenter/      # Fixed-chunk segmentation baseline
└── vad_segmenter/        # Voice Activity Detection baseline
```

---

## cascade

The original cascade system from the IWSLT 2025 baseline. Uses **Whisper (small)** for ASR and **m2m100\_418M** for MT. Segmentation is controlled by a `FixedLengthSegmenter` that accumulates audio until a `step_length` threshold is reached.

| File | Description |
|---|---|
| `baseline_agent.py` | Entry point; wires `CascadeAgent` with `FixedLengthSegmenter` |
| `baseline_iwslt/cascade_agent.py` | Core streaming agent logic |
| `baseline_iwslt/fixed_length_segmenter.py` | Accumulates audio chunks up to a fixed segment length |
| `baseline_iwslt/speech_segmentation.py` | Utility functions for segment boundary detection |

---

## cascade_2026

The enhanced cascade system used in the main experiments. Replaces Whisper with **Qwen3ASR** and uses **vLLM** for MT inference, enabling the use of large language models for translation. Implements a stateful `CascadeState` object to track streaming context across steps.

| File | Description |
|---|---|
| `agent_simulstream.py` | Main agent; stateful streaming loop with Qwen3ASR + vLLM |
| `agent_simulstream_asr_policies.py` | Agent variant that applies pluggable ASR emission policies |
| `agent_simulstream_mt_policies.py` | Agent variant that applies pluggable MT decoding policies |
| `agent_simulstream_mbr.py` | Full MBR variant: generates N hypotheses, re-ranks at commit time |
| `agent_simulstream_partial_mbr.py` | Partial MBR: re-ranks only the uncommitted suffix |
| `agent_simulstream_epsilon.py` | Epsilon-greedy exploration over hypothesis candidates |
| `latencies_distribution.py` | Utility to compute and log per-segment latency distributions |

### asr_policies

Controls **when** the ASR module commits a stable transcription prefix to the MT module.

| Policy | File | Description |
|---|---|---|
| **Hold-N** | `hold_n.py` | Waits for N consecutive identical prefixes before emitting |
| **Local Agreement** | `local_agreement.py` | Emits when the new hypothesis agrees with the previous one over a local window |
| **Tolerant Agreement** | `tolerant_agreement.py` | Variant of local agreement with tolerance for minor edits |

All policies inherit from `base.py`, which defines the interface:

```python
class ASRPolicy:
    def should_transcribe(self, step: int) -> bool: ...
    def stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str: ...
```

### mt_policies

Controls **how much** of the current transcription the MT module translates at each step.

| Policy | File | Description |
|---|---|---|
| **Wait-K** | `wait_k.py` | Waits for K source tokens before starting translation |
| **Hybrid Wait-K/TLA** | `hybrid_waitk_tla.py` | Combines Wait-K startup with Token-Level Agreement for refinement |
| **Local Agreement** | `local_agreement.py` | Commits translation tokens that are stable across consecutive steps |
| **Tolerant Agreement** | `tolerant_agreement.py` | Tolerant variant; allows minor token-level disagreement before re-translation |

All policies inherit from `base.py`. The `text_utils.py` module provides shared utilities for prefix comparison and tokenization.

---

## fixed_segmenter

Baseline using **SeamlessM4T** as an end-to-end speech translation model with fixed-size audio chunks. No ASR/MT split; the model processes fixed-length windows directly.

| File | Description |
|---|---|
| `baseline_fixed.py` | Entry point |
| `baseline_iwslt/seamless_m4t_agent.py` | SeamlessM4T streaming agent |
| `baseline_iwslt/fixed_length_segmenter.py` | Fixed-chunk audio splitter |
| `baseline_iwslt/speech_segmenter.py` | Segment boundary utilities |

---

## vad_segmenter

Baseline using **SeamlessM4T** with **Voice Activity Detection (VAD)**-based segmentation. Instead of fixed chunks, audio is split at detected speech pauses, producing more natural segments at the cost of unpredictable latency.

| File | Description |
|---|---|
| `baseline_vad.py` | Entry point |
| `baseline_iwslt/seamless_m4t_agent.py` | SeamlessM4T streaming agent |
| `baseline_iwslt/vad_segmenter.py` | VAD-based silence detection and segmentation |
| `baseline_iwslt/speech_segmenter.py` | Shared segment boundary utilities |

---

## Adding a New Policy

1. Create a new file in `cascade_2026/asr_policies/` or `cascade_2026/mt_policies/`.
2. Subclass the appropriate `base.py` abstract class and implement the required methods.
3. Register the policy in the corresponding `__init__.py`.
4. Reference it by class name in the experiment YAML config.
