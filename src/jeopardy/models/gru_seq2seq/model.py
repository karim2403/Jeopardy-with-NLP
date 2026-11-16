"""One-layer encoder/decoder with shared embeddings and no attention."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_token_id=0, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(self.embedding(source), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return hidden

    def forward(self, source, lengths, decoder_input):
        hidden = self.encode(source, lengths)
        decoded, _ = self.decoder(self.embedding(decoder_input), hidden)
        return self.output(decoded)

    @torch.inference_mode()
    def generate(self, source, lengths, bos_id, eos_id, pad_id, unk_id, max_new_tokens=64):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        hidden = self.encode(source, lengths)
        token = torch.full((source.size(0), 1), bos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        output = []
        for _ in range(max_new_tokens):
            decoded, hidden = self.decoder(self.embedding(token), hidden)
            logits = self.output(decoded[:, -1])
            logits[:, [pad_id, bos_id, unk_id]] = -torch.inf
            next_token = logits.argmax(-1)
            next_token = torch.where(finished, pad_id, next_token)
            output.append(next_token)
            finished |= next_token.eq(eos_id)
            token = next_token.unsqueeze(1)
            if finished.all():
                break
        return torch.stack(output, dim=1)


def select_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
    return torch.device(requested)
