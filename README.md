# Jeopardy with NLP

The notebooks preserve the original experiments. New work uses a shared Python preprocessing pipeline with no answer-frequency restrictions and no tokenization.

## Usage

Create the project environment (Python 3.12, including notebook and RAG dependencies):

```shell
conda env create -f environment.yml
conda activate jeopardy-nlp
```

To update an existing environment after the file changes:

```shell
conda env update -f environment.yml
```

Select the `jeopardy-nlp` environment in your editor, or launch `jupyter lab` from the activated environment. This is a dependency specification, not a complete lockfile. Python and PyTorch are pinned; other versions resolve at installation time.

On Windows, `environment.yml` installs PyTorch **2.11.0 with CUDA 12.8** from the official
PyTorch wheel index. It supports NVIDIA GPU execution and CPU execution. Other platforms
use the corresponding standard PyTorch package; hardware-specific builds may differ.
The Windows build was verified on an RTX 4060 Ti. A separate CUDA toolkit is not needed
for this model; an appropriate NVIDIA driver is required. See the
[official installation guide](https://pytorch.org/get-started/locally/).

If you previously installed CPU-only PyTorch, update the environment using the command
above. Verify the active interpreter with:

```shell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The preprocessing pipeline alone requires Python 3.10+ and no third-party dependencies. Run from the repository root:

```shell
python -m src.jeopardy.data.prepare
python -m unittest discover -s tests -v
```

Optional arguments: `--input PATH`, `--output-dir PATH`, `--seed INTEGER` (default 42).

## Processing rules

- Remove clues containing hyperlinks, URLs, or embedded-media markup. This conservative heuristic can remove answerable linked clues and can miss plain-text references to unseen media.
- Reject missing/non-string required fields and text empty after cleaning.
- Decode HTML entities, strip markup, unescape apostrophes, remove dataset wrapper quotes, and collapse whitespace. Preserve capitalization, punctuation, numbers, and articles in model-facing text.
- Keep every remaining answer regardless of frequency, including answers unseen in training. No tokenization, vocabulary construction, truncation, padding, or training occurs.
- Group case-insensitive, whitespace-normalized cleaned clue texts, preserving punctuation. Assign whole groups to seeded random 80/10/10 splits. Proportions are approximate; duplicate clues cannot cross splits. This does not detect paraphrases or hold out entire episodes.
- Keep duplicates within their split. Flag groups with differing answers instead of discarding them; category differences can legitimately explain such conflicts.

## Outputs

`data/processed/` contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`, ignored by Git. Each row retains the original record and source index, cleaned category/clue/answer, metadata, duplicate-group ID, conflict flag, and split.

`answer_normalized` is for evaluation only: case folding, punctuation removal, whitespace normalization, and leading-article removal. It never filters records or replaces the original answer. The scoring policy can be changed separately from cleaning; normalization alone does not handle aliases or establish semantic equivalence.

The manifest records the source checksum, pipeline version, settings, exclusive removal counts, split sizes, duplicate statistics, and output checksums. Source IDs refer to positions in the input file; the checksum identifies that file. Rerunning with identical input, settings, and code reproduces the outputs. Runs overwrite the named output files.

## Tokenizer

Update the environment before using the tokenizer: `conda env update -f environment.yml`.
The `tokenizers` dependency is required for tokenization, but not for preprocessing.

Train an 8,000-token byte-level BPE vocabulary on training categories, clues, and answers:

```shell
python -m src.jeopardy.tokenization.train --vocab-size 8000 --min-frequency 2
```

The command reads only `data/processed/train.jsonl` and rejects rows without `split="train"`.
Optional `--input` and `--output-dir` paths allow separate experiments. Artifacts default to
`artifacts/tokenizer/` (Git-ignored). `tokenizer.json` stores the vocabulary, merges, and
byte encoder/decoder; `config.json` stores settings, library version, checksums, and corpus metadata.

```python
from src.jeopardy.tokenization import JeopardyTokenizer

tokenizer = JeopardyTokenizer.load("artifacts/tokenizer")
ids = tokenizer.encode("Category: HISTORY. Clue: This president...", add_eos=True)
text = tokenizer.decode(ids)
batch = tokenizer.batch_encode(["A short clue", "Another clue"], add_eos=True)
# batch contains Python lists: input_ids and attention_mask.
# Initialize models with tokenizer.vocab_size and tokenizer.pad_token_id.
```

For direct training: `JeopardyTokenizer(vocab_size=8000, min_frequency=2).train(training_texts)`.
Training is one-shot; create a new instance for a new vocabulary. The actual `vocab_size`
can be smaller than the requested target. Minimum size is 260: all 256 bytes plus
`<pad>`, `<unk>`, `<bos>`, `<eos>` (IDs 0–3). Reserved strings are rejected in literal text.
The full byte alphabet represents unseen Unicode characters without unknown tokens.
Case, punctuation, and whitespace are preserved; no evaluation normalization is applied.

Encoding adds no boundary tokens by default; use `add_bos` and `add_eos` as needed.
Right padding is to the longest batch item by default; `padding=False` returns ragged lists,
and `padding="max_length"` requires an explicit `max_length`. Masks use 1 for real tokens
(including BOS/EOS), 0 for padding. Overlong sequences raise unless `truncation=True`;
truncation reserves room for requested BOS/EOS. Byte truncation can cut a Unicode character,
so exact decoding round-trips apply to untruncated sequences. No tensors or positional
encodings are constructed. Model-specific label shifts and padding-loss masks belong in
the training code. Use native tokenizers for pretrained Qwen/MiniLM, not this vocabulary.

Keep the saved tokenizer paired with the model trained on its IDs. The generic `.train()`
method cannot know a string's split; callers must supply training text only. The CLI enforces
the split field. Saved artifacts, rather than retraining across library versions, define
the stable token IDs used by experiments.

Implementation follows [Hugging Face Tokenizers byte-level components](https://huggingface.co/docs/tokenizers/main/en/components)
and [BPE trainer configuration](https://huggingface.co/docs/tokenizers/main/en/api/trainers).

## Code layout

```text
src/jeopardy/
    data/
        preprocessing.py   # Cleaning and record validation
        splitting.py       # Grouped split assignments
        prepare.py         # Dataset preparation CLI
    evaluation/
        answers.py         # Answer scoring normalization
    tokenization/
        bpe.py             # Reusable tokenizer implementation
        train.py           # Training-only tokenizer CLI
        metrics.py         # Coverage, lengths, usage, speed
        compare.py         # Vocabulary comparison CLI
        report.py          # HTML/Markdown/JSON reports
    models/
        gru_seq2seq/        # Model, dataset, training, prediction, checkpoints
        gru_attention/      # GRU encoder-decoder with additive attention
        t5_qa/              # Fine-tuned pretrained T5 answering model
```

Public imports stay short: `from src.jeopardy.tokenization import JeopardyTokenizer`
and `from src.jeopardy.evaluation import normalize_answer`. The old top-level
preprocessing/splitting imports and preparation/training commands remain as small
compatibility wrappers. Functional implementations live in the subpackages.

## Tokenizer inspection and vocabulary comparison

```shell
python -m src.jeopardy.tokenization.compare --vocab-sizes 4000 8000 16000
```

This trains separate vocabularies using training data only and evaluates the entire
validation split. It checks record IDs and clue groups for train/validation overlap;
the test set is never read. The existing default tokenizer is not replaced.

Open `artifacts/tokenizer_comparison/report.html` in a browser for a standalone report
with tables, a length comparison chart, expandable token examples with IDs and decoded
text, and longest sequences. `report.md` contains the summary tables; `metrics.json`
contains all measurements, settings, and source/artifact checksums. The `vocab_4000/`,
`vocab_8000/`, and `vocab_16000/` directories hold the saved tokenizer candidates.

Measurements include round-trip fidelity, unknown-token count, per-field length
percentiles, tokens per whitespace-separated word, fractions exceeding possible length
limits, training token usage, artifact sizes, illustrative embedding costs, and a
three-repeat warmed-up encoding benchmark. QA input lengths include category/clue
labels and BOS/EOS; answer lengths exclude special tokens. Speed depends on hardware
and background load. No answer-model quality or semantic quality is measured.

Change `--train`, `--validation`, `--output-dir`, or `--min-frequency` for other runs.
Candidate artifacts and reports are Git-ignored. This comparison adds no dependencies
beyond the existing `tokenizers` requirement; no further environment update is needed.

The existing exploration, MLP, and RAG notebooks have not yet been migrated. Their historical splits and frequency-filtered MLP metrics differ from this pipeline.

## GRU answering model

The first model is a small GRU encoder-decoder with shared learned embeddings. Its
model-specific code, training CLI, prediction CLI, and checkpoint helpers live together
under `src/jeopardy/models/gru_seq2seq/`. See that folder's `README.md` for all settings,
checkpoint behavior, and the optional clue-generation/joint task.

```shell
python -m src.jeopardy.models.gru_seq2seq.train --run-dir artifacts/models/gru_seq2seq/baseline --epochs 5
python -m src.jeopardy.models.gru_seq2seq.predict --checkpoint artifacts/models/gru_seq2seq/baseline/best.pt --category HISTORY --clue "This president signed the Emancipation Proclamation."
python -m src.jeopardy.models.gru_seq2seq.train --resume artifacts/models/gru_seq2seq/baseline/last.pt --epochs 10
```

Training prints batch progress, losses, throughput/ETA, validation predictions, and saved
checkpoint paths. It automatically selects CUDA when available, otherwise CPU; override
with `--device cpu` or `--device cuda`. Saved runs include the exact tokenizer and can be
used for inference without retraining. New runs require an empty output directory.

The attention successor lives in `src/jeopardy/models/gru_attention/` and uses the same
workflow:

```shell
python -m src.jeopardy.models.gru_attention.train --run-dir artifacts/models/gru_attention/attention_fast_v1 --epochs 20 --device auto --progress line
```

The optimized trainer vectorizes teacher-forced attention, uses CUDA mixed precision,
length-bucketed batches, batch size 128, attention width 128, and a 256-example monitoring
sample by default. Existing sequential-attention checkpoints remain loadable. See its
README for ablations, attention-trace prediction, and evaluation.

## Validation input ablations

Measure whether any saved answer model actually uses its clue and category inputs without
retraining or reading the test set:

```shell
python -m src.jeopardy.models.validation_ablation --checkpoint artifacts/models/gru_attention/attention_fast_v1/best.pt --device auto --progress line
```

The command evaluates the same validation examples three times: intact, with categories
shuffled between records, and with clues shuffled between records. Original answers remain
the targets. It reports accuracy, token F1, teacher-forced loss, prediction-change rate,
and paired correct-to-wrong/wrong-to-correct counts. Use `--limit 2000` for a quick seeded
diagnostic or omit it for the complete validation split. Results are written beside the
checkpoint under `validation_ablation_epoch_NNN/`; choose `--output-dir` for repeated runs.

## Pretrained T5 answering model

The transfer-learning model under `src/jeopardy/models/t5_qa/` fine-tunes
`google/flan-t5-small` with its native tokenizer. It saves separate best-loss and
best-generated-exact checkpoints. See that folder's README for training, resuming,
prediction, and validation commands.
