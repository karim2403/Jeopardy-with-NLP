# GRU sequence-to-sequence baseline

One encoder GRU and one decoder GRU, shared learned token embeddings, no attention.
Defaults: 128-dimensional embeddings, 256 hidden units, Adam at 0.001, batch size 64,
gradient norm clipping at 1.0. The 8K tokenizer gives 3,672,896 parameters. Packed encoder
sequences exclude padding from the final state. Training uses teacher forcing with
shifted targets and token-averaged cross-entropy excluding padding. Prediction is greedy.

Run commands from the repository root after `conda activate jeopardy-nlp`.

## Training

```shell
python -m src.jeopardy.models.gru_seq2seq.train --run-dir artifacts/models/gru_seq2seq/baseline --epochs 5
```

`--device auto` chooses CUDA if available, otherwise CPU. Force either with `--device cpu`
or `--device cuda`; explicitly requesting unavailable CUDA raises a clear error. CPU and
GPU use the same model/checkpoint format. GPU floating-point results can differ from CPU.

Startup prints device/GPU name, settings, parameter count, tokenizer checksum, and dataset
counts including over-length examples. `--log-every 20` updates cumulative token-weighted
loss (`avg`), token-weighted loss since the previous update (`recent`), examples/second,
and remaining batch ETA. `--progress auto` replaces the current batch line in interactive
terminals and prints normal lines when redirected to a file. Use `--progress line` to
force replacement in terminals not detected correctly, or `--progress plain` to retain
every update. Epoch summaries remain visible; `metrics.jsonl` always retains every logged
update, including cumulative/recent losses and elapsed time. Each epoch prints validation loss,
generated exact match on a fixed seeded subset, three prediction examples, and saved paths.
Epoch elapsed time includes validation/generation but excludes checkpoint writes.

The full validation split is used for loss. `--eval-samples 128` controls the fixed subset
used for generated exact match; this is an estimate, not a full validation or test score.
The test set is never loaded. The best checkpoint minimizes validation loss, which may
not coincide with the best generated exact match.

Length caps include BOS/EOS: source 128, target 64 by default. Over-length data raises
before training, with counts. Prefer increasing `--max-source-length` or
`--max-target-length`; `--truncate` explicitly permits truncation and preserves EOS.
No records are dropped. Generated answers are always scored against the original target,
including when training targets were truncated. Empty normalized targets use raw exact
match rather than counting empty generations as correct.

For a quick smoke run:

```shell
python -m src.jeopardy.models.gru_seq2seq.train --run-dir artifacts/models/gru_seq2seq/my_smoke --epochs 2 --train-limit 4096 --validation-limit 256 --eval-samples 64
```

Limits select seeded random records, not the first rows. Split/duplicate-group checks run
before sampling. Other knobs include `--embedding-dim`, `--hidden-dim`, `--batch-size`,
`--learning-rate`, `--seed`, `--threads`, `--train`, `--validation`, and `--tokenizer`.

## Saved runs and resuming

Each run directory contains:

- `config.json`: settings, source checksums, tokenizer identity, model dimensions.
- `tokenizer/`: complete copy of the tokenizer used by the run.
- `metrics.jsonl`: batch progress and epoch summaries.
- `validation_epoch_NNN.json`: generated predictions for that epoch's fixed subset.
- `best.pt`: lowest-validation-loss checkpoint.
- `last.pt`: last completed-epoch checkpoint for resuming.

Checkpoints contain weights, optimizer state, epoch, best score, configuration, Python/
Torch/CUDA RNG states, and data-loader shuffle-generator state. Writes use a temporary
file and atomic replacement. New runs refuse to overwrite nonempty run directories.

```shell
python -m src.jeopardy.models.gru_seq2seq.train --resume artifacts/models/gru_seq2seq/baseline/last.pt --epochs 10
```

`--epochs` is the desired total, not additional epochs. Resume restores saved settings;
do not supply different model/data settings. Device, threads, log frequency, and total
epochs can change. Resume rechecks data/tokenizer checksums. Incomplete-epoch progress
logs are discarded before replaying that epoch. Interruptions before the first completed
epoch have no checkpoint to resume. Resume expects the original run location and data
paths; prediction permits moving a run directory together with its tokenizer.

Exact CPU continuation is tested on a tiny deterministic run. GPU/backend differences
mean bit-for-bit reproducibility across devices or versions is not promised.

## Early stopping

New runs stop by default after two consecutive epochs without lower validation loss:

```shell
python -m src.jeopardy.models.gru_seq2seq.train --run-dir artifacts/models/gru_seq2seq/early_stop_run --epochs 20 --early-stopping-patience 2
```

`--epochs` remains the maximum. `--early-stopping-min-delta 0.01` requires at least that
much validation-loss improvement to reset patience. Use `--early-stopping-patience 0` to
disable stopping. The counter and reference loss are checkpointed and restored on resume.
`best.pt` remains the checkpoint with the numerically lowest validation loss, even when a
nonzero minimum delta controls patience. Stopping uses teacher-forced validation loss;
it does not monitor noisy generated exact match or embedding probes.

## Prediction without training

```shell
python -m src.jeopardy.models.gru_seq2seq.predict --checkpoint artifacts/models/gru_seq2seq/baseline/best.pt --category HISTORY --clue "This president signed the Emancipation Proclamation."
```

Prediction loads no training data. It validates the saved tokenizer checksum, generates
until EOS or the configured limit, and reports if EOS was not reached. Use
`--max-new-tokens` to override the generation limit. Over-length input raises rather than
silently truncating. Both `--device cpu` and `--device cuda` work with the same checkpoint.

Programmatic use:

```python
from src.jeopardy.models.gru_seq2seq.predict import Predictor

model = Predictor("artifacts/models/gru_seq2seq/baseline/best.pt", device="auto")
result = model.predict("HISTORY", "This president signed the Emancipation Proclamation.")
print(result["text"])
```

## Full held-out test evaluation (no training)

Run this after model and checkpoint selection is finished:

```shell
python -m src.jeopardy.models.gru_seq2seq.evaluate --checkpoint artifacts/models/gru_seq2seq/test1/best.pt --device auto --progress line
```

The primary overall accuracy is normalized exact match. The command also reports raw exact
match, normalized word-token F1, EOS/nonempty rates, teacher-forced loss/perplexity, and
separate exact-match accuracy for answers seen versus unseen in the training split. It
verifies that test clue groups do not overlap training groups.

Results default to `test_best_epoch_NNN/` beside the checkpoint. `summary.json` records
metrics and source/checkpoint hashes; `predictions.jsonl` records every clue, expected
answer, prediction, token IDs, and per-example scores. Existing nonempty result directories
are never overwritten. Use `--output-dir` for a distinct evaluation. This command freezes
the model and does not update weights or checkpoints.

## Clue generation and joint training

`--task answer` is the default. `--task clue` reverses the mapping; `--task joint` creates
both examples from each record without crossing splits. Prefixes are plain text:

```text
answer: category: HISTORY\nclue: ...     -> answer
clue: category: HISTORY\nanswer: ...    -> original clue
```

Train in a separate run directory, increase target length for clues, and use `--answer`
instead of `--clue` for prediction:

```shell
python -m src.jeopardy.models.gru_seq2seq.train --task joint --max-source-length 256 --max-target-length 256 --run-dir artifacts/models/gru_seq2seq/joint
python -m src.jeopardy.models.gru_seq2seq.predict --checkpoint artifacts/models/gru_seq2seq/joint/best.pt --category HISTORY --answer "Abraham Lincoln"
```

An answering-only checkpoint rejects clue-generation requests. Clue exact match is only
reference reconstruction, not a factuality or quality metric: many different clues are
valid for one answer. This baseline has no external knowledge retrieval and is intended
as a small learning experiment, not a reliable trivia system.

## Between-epoch embedding probes

By default training reports cosine similarity for `man/woman`, `Man/Woman`, `Paris/London`,
and `France/England` before training and after every epoch. Each row shows current cosine,
change since the previous measurement, and change from the reference. Token pieces/IDs
are printed once at startup. Probes consistently use a leading space, preserve case,
and average subword embeddings for multi-token words, like the standalone inspector.

Customize with repeated pairs, or disable the diagnostics:

```shell
python -m src.jeopardy.models.gru_seq2seq.train --run-dir artifacts/models/gru_seq2seq/probed --probe-pair man woman --probe-pair king queen
python -m src.jeopardy.models.gru_seq2seq.train --resume artifacts/models/gru_seq2seq/my_run/last.pt --epochs 12 --progress line
```

`--no-probes` disables the checks. Existing saved checkpoints remain compatible. New runs
use initialization (epoch 0) as reference; old checkpoints without probe history use the
loaded resume epoch, clearly labeled. New checkpoints preserve the reference and previous
scores. On resume, configured pairs are retained unless explicitly overridden; changing
pairs resets their reference to that resume epoch. Initial/resume measurements are stored
as `embedding_probes` events in `metrics.jsonl`; epoch summaries include `embedding_probes`.

Probes read a few embedding rows without gradients, model forward passes, or randomness.
They do not affect optimization or checkpoint selection. Increasing similarity is not an
accuracy objective and need not indicate better answering. No dependencies were added.

Implementation references: [packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
and [checkpoint serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## Inspect learned embeddings (no training)

```shell
python -m src.jeopardy.models.gru_seq2seq.inspect_embeddings --checkpoint artifacts/models/gru_seq2seq/my_run/best.pt --query Paris --compare Paris London France England --top-k 10
```

Prints checkpoint epoch and embedding shape, exact BPE pieces/IDs, ranked nearest-token
neighbors, and an optional numbered pairwise cosine matrix. Quote multiword queries.
`--query` supports several queries. `--json artifacts/embedding_inspection.json` saves
the full results to a new file, including segmentation and similarity values.

Runs on CPU and loads only checkpoint data: no model forward pass, training, or dataset
access. It checks that the saved tokenizer matches the checkpoint. The checkpoint is
not modified. No additional dependencies or environment update are needed.

The default `--spacing leading` prepends one space consistently to mimic words inside
text; `--spacing none` uses beginning-of-text tokenization. Query edge whitespace is
stripped. Case is preserved. For multiple pieces, the representation is the arithmetic
mean of their embedding vectors, not a separately learned whole-word vector. Neighbors
are individual BPE tokens, excluding special tokens and the query's constituent tokens;
the pairwise matrix compares the supplied word/phrase representations to each other.

Cosine ranges from -1 to 1, not a probability. Byte-level pieces may contain unusual
glyphs; Ġ commonly represents a space. Rare/unused entries can remain largely random.
These static embeddings were trained for our sequence task and are not contextual GRU
states; similarity does not establish semantic understanding or question-answer accuracy.
