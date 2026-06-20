# experiments — Experimental Pipeline

This directory contains all experiment configurations, inference outputs, evaluation results, and analysis scripts. The workflow for every experiment follows the same four-stage pattern:

```
Config (YAML) → Inference (JSONL) → Evaluation → Results (CSV)
```

---

## Directory Structure

```
experiments/
├── baselines_trades/         # Baseline system latency–quality sweeps
│   ├── cascade/              # Cascade (Whisper + m2m100), 4×5 config grid
│   ├── cascade_2026/         # Cascade 2026 (Qwen3ASR + vLLM), 5 segment lengths
│   ├── fixed/                # Fixed-chunk (SeamlessM4T), 5 chunk sizes
│   └── vad/                  # VAD-based (SeamlessM4T), 5×5 threshold grid
│
├── iwslt_submission/         # IWSLT 2025 shared task submissions
│   ├── baseline/
│   ├── low_latency/
│   └── high_latency/
│
├── mbr_trades/               # MBR decoding experiments on Cascade 2026
│   ├── cascade_2026_epsilon/     # Epsilon-greedy hypothesis selection
│   ├── cascade_2026_mbr/         # Full MBR with three ranking functions
│   │   ├── prunembr_xcomet/      # Pruning MBR + XCOMET scoring
│   │   ├── rambr_chrf/           # RAM-based MBR + chrF scoring
│   │   └── rerank_kiwi/          # Reranking with KIWI quality estimation
│   └── cascade_2026_partial_mbr/ # Partial MBR (uncommitted suffix only)
│       ├── prunembr_xcomet/
│       ├── rambr_chrf/
│       └── rerank_kiwi/
│
├── policies_trades/          # Streaming policy comparison experiments
│   ├── asr/                  # ASR emission policies (hold_n, local_agreement, tolerant)
│   │   ├── hold_n/
│   │   ├── local_agreement/
│   │   └── tolerant_agreement/
│   └── mt/                   # MT decoding policies (wait_k, hybrid, tolerant)
│       ├── wait_k/
│       ├── hybrid_waitk_tla/
│       └── tolerant_agreement/
│
```

Each experiment subfolder has the same internal layout:

```
<experiment>/
├── configs/       # YAML configuration files (one per run)
├── inferences/    # JSONL output files from simulstream
├── results/       # Per-run evaluation folders (JSONL only, one subfolder per config)
└── results_*.csv  # Aggregated metrics table
```

---

## Running Experiments

### Stage 1 — Inference

Each experiment folder contains an `inferences_*.sh` script that:
1. Generates YAML config files for each point in the parameter grid.
2. Submits SLURM jobs (one per config) that call `simulstream.inference`.
3. Writes output to `inferences/<run_name>.jsonl`.

```bash
cd experiments/baselines_trades/cascade
sbatch inferences_cascade.sh
```

To run without SLURM (single config, for local testing):

```bash
python -m simulstream.inference \
    --config experiments/baselines_trades/cascade/configs/cascade_step1.0_la1.yaml \
    --data-config data/dev/audio_definition.yaml \
    --output experiments/baselines_trades/cascade/inferences/cascade_step1.0_la1.jsonl
```

### Stage 2 — Evaluation

Once inference outputs are ready, the `evaluations_*.sh` script:
1. Iterates over all JSONL files in `inferences/`.
2. Runs OmniSTEval to compute LongYAAL.
3. Runs SacreBLEU for BLEU and chrF++.
4. Runs xCOMET-XL for neural quality scores.
5. Writes per-run results to `results/<run_name>/` and aggregates into `results_*.csv`.

```bash
cd experiments/baselines_trades/cascade
sbatch evaluations_cascade.sh
```

---

## Experiment Descriptions

### baselines_trades

Sweeps over the configuration space of each baseline system to trace out the latency–quality Pareto frontier.

| System | Parameter Grid | # Configs |
|---|---|---|
| `cascade` | `step_length` ∈ {1, 2, 4, 8} s × `latency_policy` ∈ {1, 2, 5, 10} | 20 |
| `cascade_2026` | `segment_length` ∈ {640, 960, 1280, 1600, 1920} tokens | 5 |
| `fixed` | `chunk_size` ∈ {5 values} | 5 |
| `vad` | `pause_threshold` × `vad_threshold` ∈ 5×5 grid | 25 |

### iwslt_submission

The three system variants submitted to the IWSLT 2025 simultaneous speech translation shared task. `low_latency` and `high_latency` variants are tuned by adjusting the latency policy parameter of the cascade baseline.

### mbr_trades

Experiments applying **Minimum Bayes Risk (MBR)** decoding to the Cascade 2026 system. At commit time, the system generates N translation hypotheses and selects the one that maximises expected utility under a given metric:

- **XCOMET** — neural translation quality estimator (Unbabel)
- **chrF** — character-level F-score
- **KIWI** — quality estimation without reference

Two MBR strategies are compared:
- **Full MBR** (`cascade_2026_mbr`) — applies MBR to the entire current hypothesis at each commit.
- **Partial MBR** (`cascade_2026_partial_mbr`) — applies MBR only to the uncommitted suffix, preserving already-emitted tokens.
- **Epsilon** (`cascade_2026_epsilon`) — epsilon-greedy exploration; emits the top-ranked hypothesis with probability 1−ε, and a random candidate otherwise.

### policies_trades

Experiments isolating the effect of **ASR emission policies** and **MT decoding policies** on the latency–quality trade-off, using Cascade 2026 as the base system.

**ASR policies** control when a stable transcript prefix is committed to MT:
- `hold_n` — waits for N identical consecutive hypotheses
- `local_agreement` — agreement-based commitment over a local window
- `tolerant_agreement` — allows minor edits before disagreeing

**MT policies** control how much source context is translated at each step:
- `wait_k` — translates only after K source tokens are available
- `hybrid_waitk_tla` — Wait-K startup, then token-level agreement
- `tolerant_agreement` — commits tokens that are stable with tolerance for small changes

---

## Config File Format

All configs are YAML files consumed by `simulstream.inference`. Example:

```yaml
type: "simulstream.server.speech_processors.simuleval_wrapper.SimulEvalWrapper"
simuleval_agent: "cascade.baseline_agent.CascadeAgentWitFixedLengthSegmenter"

latency_unit: "word"
speech_chunk_size: 0.5

config:
  device: "cuda"
  segment_length: 24.0
  step_length: 1.0
  whisper_model: "small"
  whisper_language: "en"
  translation_model: "facebook/m2m100_418M"
  translation_language: "de"
  translation_la_policy: 1
  transcript_context: 30
  translation_max_input_length_soft: 30
  translation_max_input_length_hard: 60
```

---

## Output Format

Inference produces one `.jsonl` file per run. Each line is a JSON object with segment-level information:

```json
{
  "index": 0,
  "prediction": "The model was trained on ...",
  "delays": [1.2, 1.4, 1.8],
  "elapsed": [0.12, 0.13, 0.15],
  "source_length": 5.3
}
```

Evaluation reads these files and writes metric summaries to the `results/` subfolder.
