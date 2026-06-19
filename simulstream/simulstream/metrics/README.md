# simulstream/metrics

This module contains the full evaluation stack of `simulstream`: log reading, final-output reconstruction, quality scoring, latency scoring, and global statistics.

## Purpose

`simulstream/metrics` takes inference logs (typically `metrics.jsonl`) and turns them into comparable evaluation outputs:

- Text quality (`score_quality`)
- Emission latency (`score_latency`)
- System behavior/cost statistics (`stats`)

## End-To-End Evaluation Flow

1. During inference, the system writes JSONL events with:
   - `generated_tokens`
   - `deleted_tokens` (retractions/retranslation)
   - `total_audio_processed`
   - `computation_time`
2. `LogReader` (in `readers.py`) reconstructs the final hypothesis for each audio file.
3. Delays are assigned per unit (`word` or `char`) for the final text:
   - `ideal_delays`: processed-audio time only
   - `computational_aware_delays`: processed-audio time + computation time
4. CLI scripts (`score_quality.py`, `score_latency.py`, `stats.py`) consume these structures and compute final metrics.

## File Structure

- `logger.py`
  - Configures `METRICS_LOGGER` to write JSONL entries.
- `readers.py`
  - Parses logs and references.
  - Reconstructs final output text and per-unit delays.
- `detokenizers.py`
  - Converts token sequences back into text, depending on the tokenizer/detokenizer setup.
- `score_quality.py`
  - CLI for quality metrics.
- `score_latency.py`
  - CLI for latency metrics.
- `stats.py`
  - CLI for aggregate statistics.
- `scorers/quality/*`
  - Concrete quality scorers.
- `scorers/latency/*`
  - Concrete latency scorers.

## Required Inputs

### 1) JSONL Log (`--log-file`)

Produced by server/inference. It must contain per-audio metadata (`wav_name`) and generation events (`generated_tokens`/`deleted_tokens`).

### 2) Evaluation Config (`--eval-config`)

YAML file containing detokenization-related settings (for example `detokenizer_type` and model/tokenizer parameters).

### 3) References / Audio Definition (metric-dependent)

Two supported modes:

- Segmented mode:
  - `--audio-definition` + reference/transcript file(s)
  - Used when segment-level alignment requires `offset` and `duration`.
- File-per-audio mode:
  - File lists via `--references` (and optionally `--transcripts`) without `--audio-definition`.
  - File stems must match audio names present in logs.

## `score_quality`: Available Scorer Types

Quality scorers are registered dynamically in `QUALITY_SCORER_REGISTRY`.

### `sacrebleu`

- File: `scorers/quality/sacrebleu.py`
- Class: `SacreBLEUScorer`
- Measures: corpus-level BLEU on final hypotheses.
- Requirements:
  - References: yes
  - Source/transcript (`src`): no
- Main arguments:
  - `--tokenizer` (BLEU tokenizer)

### `comet`

- File: `scorers/quality/comet.py`
- Class: `CometScorer`
- Measures: semantic translation quality with COMET.
- Requirements:
  - References: yes
  - Source/transcript (`src`): yes
- Main arguments:
  - `--model` (default: `Unbabel/wmt22-comet-da`)
  - `--batch-size`

### Important Note: MWER Segmenter Alignment

Both `sacrebleu` and `comet` inherit from `MWERSegmenterBasedQualityScorer` (`scorers/quality/mwersegmenter.py`).

Before scoring, hypotheses are realigned to references using `mweralign`, reducing segmentation bias (same translation content, different sentence splits).

## `score_latency`: Available Scorer Types

Latency scorers are registered in `LATENCY_SCORER_REGISTRY`.

### `stream_laal`

- File: `scorers/latency/stream_laal.py`
- Class: `StreamLaal`
- Measures: StreamLAAL (Length-Adaptive Average Lagging), averaged at sentence level.
- Output:
  - `ideal_latency`
  - `computational_aware_latency`
- Requirements:
  - Segmented references: yes (`--audio-definition` and `--reference`)
- Units:
  - `--latency-unit word|char`

### Important Note: MWER Segmenter Alignment For Latency

`StreamLaal` inherits from `MWERSegmenterBasedLatencyScorer` (`scorers/latency/mwersegmenter.py`), which:

1. Realigns hypothesis text to reference segments.
2. Re-distributes final-hypothesis delays to aligned segments.
3. Enables consistent sentence-level latency computation.

## `stats`: Available Metrics

Implemented in `stats.py`:

### `normalized_erasure`

- Definition: total deleted tokens / total final tokens.
- Interpretation:
  - Higher values indicate more flickering/re-writing (less stable retranslation).

### `real_time_factor`

- Definition: total computation time / total audio duration.
- Interpretation:
  - `> 1`: slower than real-time.
  - `< 1`: faster than real-time.

## Example Commands

### Quality (`sacrebleu`)

```bash
simulstream_score_quality \
  --scorer sacrebleu \
  --eval-config config/speech_processor.yaml \
  --log-file metrics.jsonl \
  --references refs.tgt \
  --transcripts src.src \
  --audio-definition audio_def.yaml
```

### Quality (`comet`)

```bash
simulstream_score_quality \
  --scorer comet \
  --eval-config config/speech_processor.yaml \
  --log-file metrics.jsonl \
  --references refs.tgt \
  --transcripts src.src \
  --audio-definition audio_def.yaml \
  --model Unbabel/wmt22-comet-da \
  --batch-size 16
```

### Latency (`stream_laal`)

```bash
simulstream_score_latency \
  --scorer stream_laal \
  --eval-config config/speech_processor.yaml \
  --log-file metrics.jsonl \
  --reference refs.tgt \
  --audio-definition audio_def.yaml \
  --latency-unit word
```

### Statistics

```bash
simulstream_stats \
  --eval-config config/speech_processor.yaml \
  --log-file metrics.jsonl \
  --latency-unit word
```

## How To Add A New Scorer

### New quality scorer

1. Create a file in `scorers/quality/`.
2. Implement a class inheriting from `QualityScorer` (or `MWERSegmenterBasedQualityScorer`).
3. Register it with `@register_quality_scorer("my_scorer")`.
4. Implement:
   - `score(...)`
   - `add_arguments(...)`
   - `requires_source()`
   - `requires_reference()`

### New latency scorer

1. Create a file in `scorers/latency/`.
2. Implement a class inheriting from `LatencyScorer` (or `MWERSegmenterBasedLatencyScorer`).
3. Register it with `@register_latency_scorer("my_latency")`.
4. Implement:
   - `score(...)`
   - `add_arguments(...)`
   - `requires_reference()`

Because registration is dynamic (`pkgutil.walk_packages`), new scorers automatically appear in `--scorer` choices for the CLIs.
