# Optimized GRU with vectorized attention

This model keeps the baseline's tokenizer, teacher-forced objective, greedy generation,
progress display, embedding probes, early stopping, and checkpoint format. The optimized
training path runs the teacher-forced decoder over all answer tokens in one GRU call, then
computes projected dot-product attention for all decoder/source positions with batched
matrix operations. Generation remains sequential because each predicted token depends on
the previous prediction.

The first controlled experiment should change only attention-related behavior:

```shell
python -m src.jeopardy.models.gru_attention.train --run-dir artifacts/models/gru_attention/attention_fast_v1 --epochs 20 --device auto --progress line
```

Defaults:

```text
embedding_dim=128    hidden_dim=256      attention_dim=128
dropout=0.1          encoder_layers=1    decoder_layers=1
bidirectional=false  batch_size=128      learning_rate=0.001
precision=auto       bucket_multiplier=50
eval_samples=256     early_stopping_patience=3
```

Important options:

```shell
--attention-dim 128
--dropout 0.2
--bidirectional
--encoder-layers 2
--decoder-layers 2
--early-stopping-patience 3
--precision {auto,off,fp16,bf16}
--bucket-multiplier 50
```

Dropout is applied explicitly to embeddings and pre-output features, so it remains active
with one recurrent layer. With multiple layers it is also applied between recurrent layers.
A bidirectional encoder concatenates forward/backward states and bridges the final state
into the decoder width. Change one major choice at a time when comparing validation runs.

On CUDA, `precision=auto` selects BF16 when supported and otherwise FP16. FP16 uses gradient
scaling; BF16 does not need it. CPU automatically uses FP32. Length bucketing shuffles
examples reproducibly, groups similarly sized source sequences into batches, and reduces
attention work on padding. Validation is deterministically length-sorted. Set
`--bucket-multiplier 0` or `--precision off` for ablations.

The larger batch and smaller attention dimension are speed-oriented defaults, so this is
not a one-variable comparison with `attention_v1`. For a strict architecture comparison,
set `--batch-size 64 --attention-dim 256 --precision off --bucket-multiplier 0`. The older
sequential-additive checkpoints remain supported by prediction and evaluation, but cannot
be resumed with this new model type.

Prediction can show the highest-weighted source BPE positions for every generated token:

```shell
python -m src.jeopardy.models.gru_attention.predict --checkpoint artifacts/models/gru_attention/attention_fast_v1/best.pt --category HISTORY --clue "This president signed the Emancipation Proclamation." --show-attention
```

These weights are useful diagnostics, not guaranteed explanations. Inspect whether they
focus on clue-bearing words and compare correct clues against shuffled clues before making
strong claims about model reasoning.

After all architecture choices are made using validation data, run the held-out evaluator:

```shell
python -m src.jeopardy.models.gru_attention.evaluate --checkpoint artifacts/models/gru_attention/attention_fast_v1/best.pt --device auto --progress line
```

Do not use test results to choose attention dimensions, dropout, directionality, or layers.
No new dependencies are required. Training was not run during implementation.

Category-specific weighting is intentionally deferred. The first run lets us attribute any
change to attention. A later category experiment can encode the category separately, add a
learned category gate, or provide category tokens as a dedicated memory segment.
