# data — Dev and Test Sets

This directory contains the evaluation data used throughout all experiments: a set of English academic conference talks with German reference translations, drawn from ACL 2022 proceedings.

---

## Directory Structure

```
data/
├── dev/                      # Development set (used for tuning and analysis)
│   ├── wavs/                 # Source audio files (.wav, 16 kHz mono)
│   ├── src.txt               # Paths to source audio files (one per line)
│   ├── tgt.txt               # Full reference translations (German, document-level)
│   ├── tgt_segments.txt      # Reference translations aligned to segments
│   ├── transcript.txt        # Reference transcripts (English, document-level)
│   ├── transcript_segments.txt  # Reference transcripts aligned to segments
│   └── audio_definition.yaml # Segment metadata (audio file, offset, duration)
└── test/                     # Test set (same structure as dev/)
```

---

## Source Data

| Property | Value |
|---|---|
| **Source language** | English |
| **Target language** | German |
| **Domain** | Academic NLP conference talks (ACL 2022) |
| **Audio format** | WAV, 16 kHz, mono |
| **Segments per talk** | ~30 utterances |
| **Talks in dev** | 5 |

The audio files are segments of recorded presentations from ACL 2022. Each WAV file may contain multiple non-overlapping utterances, identified by their `offset` and `duration` in `audio_definition.yaml`.

---

## File Formats

### `audio_definition.yaml`

Used by `simulstream.inference` to iterate over evaluation segments. Each entry maps to one utterance:

```yaml
- wav: /absolute/path/to/2022.acl-long.110.wav
  offset: 0.59       # start time in seconds
  duration: 2.82     # duration in seconds
- wav: /absolute/path/to/2022.acl-long.110.wav
  offset: 3.96
  duration: 6.12
```

> **Note:** The `wav` paths in `audio_definition.yaml` are absolute. If you move the repository, update these paths to match your local audio file locations.

### `src.txt`

One audio file path per line (document-level, not segment-level):

```
/path/to/2022.acl-long.110.wav
/path/to/2022.acl-long.117.wav
```

### `tgt.txt` / `tgt_segments.txt`

Reference translations in German. `tgt.txt` is document-level (one talk per line). `tgt_segments.txt` is segment-aligned (one utterance per line), used by evaluation tools that operate at the segment level.

### `transcript.txt` / `transcript_segments.txt`

Reference ASR transcripts in English, in the same document-level and segment-level formats as the translation references.

---

## Usage

Pass the `audio_definition.yaml` to the inference script via `--data-config`:

```bash
python -m simulstream.inference \
    --config experiments/baselines_trades/cascade/configs/cascade_step1.0_la1.yaml \
    --data-config data/dev/audio_definition.yaml \
    --output my_run.jsonl
```

Evaluation tools (OmniSTEval, SacreBLEU) use `tgt_segments.txt` as the reference file.
