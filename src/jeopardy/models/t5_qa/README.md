# Pretrained T5 Jeopardy QA

This model fine-tunes `google/flan-t5-small` as an open-vocabulary text-to-text
answering model. It uses the pretrained model's own SentencePiece tokenizer; the
project BPE tokenizer remains paired with the scratch GRU models.

## Train

```powershell
python -m src.jeopardy.models.t5_qa.train `
  --run-dir artifacts/models/t5_qa/flan_t5_small_v1 `
  --epochs 5 --device auto --progress line
```

Defaults use batch size 16, four-step gradient accumulation (effective batch 64),
BF16 when supported, learning rate `5e-5`, and a fixed 1,000-record generated
validation sample. No test records are read during training. The first run downloads
the pretrained model and tokenizer from Hugging Face.

The run contains `best_loss/`, `best_exact_match/`, and `last/` Hugging Face-compatible
checkpoint directories. Resume only from `last` and provide a larger total epoch count:

```powershell
python -m src.jeopardy.models.t5_qa.train `
  --resume artifacts/models/t5_qa/flan_t5_small_v1/last --epochs 8 `
  --device auto --progress line
```

## Predict and evaluate

```powershell
python -m src.jeopardy.models.t5_qa.predict `
  --checkpoint artifacts/models/t5_qa/flan_t5_small_v1/best_exact_match `
  --category "U.S. PRESIDENTS" `
  --clue "This president signed the Emancipation Proclamation."

python -m src.jeopardy.models.t5_qa.evaluate `
  --checkpoint artifacts/models/t5_qa/flan_t5_small_v1/best_exact_match `
  --split validation --device auto --progress line
```

Use the test split once after model and generation choices are fixed. Beam search is
available through `--num-beams`, but tune it on validation rather than test.
