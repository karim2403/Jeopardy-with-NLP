"""Reusable byte-level BPE for models trained from scratch.

Preserves cleaned text without lowercasing. Pretrained models must use their
native tokenizers. Padding/truncation are per-call, never persistent state.
"""
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class JeopardyTokenizer:
    def __init__(self, vocab_size=8000, min_frequency=2):
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 260:
            raise ValueError("vocab_size must be an integer >= 260 (256 bytes + 4 special tokens)")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._backend = None
        self.training_metadata = {}

    def _require_trained(self):
        if self._backend is None:
            raise RuntimeError("Train or load the tokenizer first")

    @property
    def vocab_size(self):
        """Actual vocabulary size, which can be below the requested target."""
        self._require_trained()
        return self._backend.get_vocab_size()

    def _special_id(self, token):
        self._require_trained()
        return self._backend.token_to_id(token)

    @property
    def pad_token_id(self):
        return self._special_id("<pad>")

    @property
    def unk_token_id(self):
        return self._special_id("<unk>")

    @property
    def bos_token_id(self):
        return self._special_id("<bos>")

    @property
    def eos_token_id(self):
        return self._special_id("<eos>")

    def train(self, texts, *, show_progress=False):
        """Fit once on an iterable of training strings; never refit in place."""
        if self._backend is not None:
            raise RuntimeError("Create a new tokenizer to train a different vocabulary")
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings, not one string")
        digest = hashlib.sha256()
        count = 0

        def checked_texts():
            nonlocal count
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError("Each training example must be a string")
                if any(token in text for token in SPECIAL_TOKENS):
                    raise ValueError("Training text contains a reserved special-token string")
                if not text:
                    continue
                data = text.encode("utf-8")
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
                count += 1
                yield text

        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        # Full byte alphabet covers Unicode even when a character was unseen.
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size, min_frequency=self.min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=show_progress,
        )
        backend.train_from_iterator(checked_texts(), trainer=trainer)
        if count == 0:
            raise ValueError("Training requires at least one nonempty string")
        self._backend = backend
        self.training_metadata = {"text_count": count, "text_sha256": digest.hexdigest()}
        return self

    def encode(self, text, *, add_bos=False, add_eos=False, max_length=None, truncation=False):
        self._require_trained()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if any(token in text for token in SPECIAL_TOKENS):
            raise ValueError("Text contains a reserved special-token string; use add_bos/add_eos")
        if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1):
            raise ValueError("max_length must be a positive integer")
        if truncation and max_length is None:
            raise ValueError("Explicit max_length is required for truncation")
        reserved = int(add_bos) + int(add_eos)
        if max_length is not None and max_length < reserved:
            raise ValueError("max_length cannot fit the requested special tokens")
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if max_length is not None and len(ids) + reserved > max_length:
            if not truncation:
                raise ValueError("Sequence exceeds max_length; opt into truncation explicitly")
            ids = ids[:max_length - reserved]
        return ([self.bos_token_id] if add_bos else []) + ids + ([self.eos_token_id] if add_eos else [])

    def decode(self, ids, *, skip_special_tokens=True):
        self._require_trained()
        ids = list(ids)
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("Token IDs must be integers within the vocabulary")
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts, *, padding=True, max_length=None, truncation=False,
                     add_bos=False, add_eos=False):
        """Return Python lists: input_ids and attention_mask (1=text, 0=padding).

        padding=True pads to longest item; 'max_length' pads to max_length;
        False returns ragged lists. Padding is on the right.
        """
        self._require_trained()
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings")
        if padding is not True and padding is not False and padding != "max_length":
            raise ValueError("padding must be True, False, or 'max_length'")
        if padding == "max_length" and max_length is None:
            raise ValueError("max_length is required for fixed-length padding")
        rows = [self.encode(t, add_bos=add_bos, add_eos=add_eos,
                            max_length=max_length, truncation=truncation) for t in texts]
        width = max_length if padding == "max_length" else max(map(len, rows), default=0)
        masks = []
        for row in rows:
            extra = width - len(row) if padding else 0
            masks.append([1] * len(row) + [0] * extra)
            row.extend([self.pad_token_id] * extra)
        return {"input_ids": rows, "attention_mask": masks}

    def token_pieces(self, text):
        """Exact stored byte-level pieces; use decode on full IDs to recover text."""
        return [self._backend.id_to_token(i) for i in self.encode(text)]

    def vocabulary(self):
        """Return a copy of the token-piece to ID mapping."""
        self._require_trained()
        return self._backend.get_vocab()

    def save(self, directory):
        self._require_trained()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"
        self._backend.save(str(path))
        config = {
            "format_version": 1, "type": "byte-level-bpe",
            "target_vocab_size": self.target_vocab_size, "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency, "special_tokens": list(SPECIAL_TOKENS),
            "tokenizers_version": tokenizers.__version__,
            "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training": self.training_metadata,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["format_version"] != 1 or config["type"] != "byte-level-bpe" or config["special_tokens"] != list(SPECIAL_TOKENS):
            raise ValueError("Unsupported tokenizer configuration")
        path = directory / "tokenizer.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["tokenizer_sha256"]:
            raise ValueError("Tokenizer checksum mismatch")
        result = cls(config["target_vocab_size"], config["min_frequency"])
        result._backend = Tokenizer.from_file(str(path))
        if result.vocab_size != config["vocab_size"] or any(result._special_id(t) != i for i, t in enumerate(SPECIAL_TOKENS)):
            raise ValueError("Vocabulary does not match configuration")
        result.training_metadata = config["training"]
        return result
