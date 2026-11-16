"""GRU sequence-to-sequence model with padding-masked additive attention."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_states, decoder_state, source_mask):
        energy = self.score(torch.tanh(
            self.encoder_projection(encoder_states) +
            self.decoder_projection(decoder_state).unsqueeze(1)
        )).squeeze(-1)
        energy = energy.masked_fill(~source_mask, -torch.inf)
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights


class GRUAttentionSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_token_id=0, embedding_dim=128, hidden_dim=256,
                 attention_dim=256, dropout=0.1, bidirectional=False,
                 encoder_layers=1, decoder_layers=1):
        super().__init__()
        if min(embedding_dim, hidden_dim, attention_dim, encoder_layers, decoder_layers) < 1:
            raise ValueError('Model dimensions and layer counts must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.pad_token_id = pad_token_id
        self.decoder_layers = decoder_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.embedding_dropout = nn.Dropout(dropout)
        recurrent_dropout = dropout if encoder_layers > 1 else 0.0
        self.encoder = nn.GRU(embedding_dim, hidden_dim, encoder_layers, batch_first=True,
                              bidirectional=bidirectional, dropout=recurrent_dropout)
        encoder_dim = hidden_dim * (2 if bidirectional else 1)
        self.hidden_bridge = nn.Linear(encoder_dim, hidden_dim)
        self.attention = AdditiveAttention(encoder_dim, hidden_dim, attention_dim)
        self.decoder = nn.GRU(embedding_dim + encoder_dim, hidden_dim, decoder_layers,
                              batch_first=True, dropout=dropout if decoder_layers > 1 else 0.0)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + encoder_dim + embedding_dim, vocab_size)
        self.bidirectional = bidirectional
        self.encoder_layers = encoder_layers

    def encode(self, source, lengths):
        embedded = self.embedding_dropout(self.embedding(source))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_states, hidden = self.encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True,
                                        total_length=source.size(1))
        directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.encoder_layers, directions, source.size(0), -1)[-1]
        final = torch.cat([hidden[i] for i in range(directions)], dim=-1)
        decoder_hidden = torch.tanh(self.hidden_bridge(final)).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        source_mask = source.ne(self.pad_token_id)
        return states, decoder_hidden, source_mask

    def decode_step(self, token, hidden, encoder_states, source_mask):
        embedded = self.embedding_dropout(self.embedding(token))
        query = hidden[-1]
        context, attention = self.attention(encoder_states, query, source_mask)
        decoder_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        decoded, hidden = self.decoder(decoder_input, hidden)
        features = torch.cat([decoded[:, 0], context, embedded], dim=-1)
        logits = self.output(self.output_dropout(features))
        return logits, hidden, attention

    def forward(self, source, lengths, decoder_input, return_attention=False):
        states, hidden, mask = self.encode(source, lengths)
        logits, attentions = [], []
        for step in range(decoder_input.size(1)):
            step_logits, hidden, weights = self.decode_step(decoder_input[:, step], hidden, states, mask)
            logits.append(step_logits)
            attentions.append(weights)
        output = torch.stack(logits, dim=1)
        if return_attention:
            return output, torch.stack(attentions, dim=1)
        return output

    @torch.inference_mode()
    def generate(self, source, lengths, bos_id, eos_id, pad_id, unk_id,
                 max_new_tokens=64, return_attention=False):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        states, hidden, mask = self.encode(source, lengths)
        token = torch.full((source.size(0),), bos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        output, attentions = [], []
        for _ in range(max_new_tokens):
            logits, hidden, weights = self.decode_step(token, hidden, states, mask)
            logits[:, [pad_id, bos_id, unk_id]] = -torch.inf
            next_token = logits.argmax(-1)
            next_token = torch.where(finished, pad_id, next_token)
            output.append(next_token)
            attentions.append(weights)
            finished |= next_token.eq(eos_id)
            token = next_token
            if finished.all():
                break
        ids = torch.stack(output, dim=1)
        if return_attention:
            return ids, torch.stack(attentions, dim=1)
        return ids


class VectorizedAttention(nn.Module):
    """Projected dot-product attention over all teacher-forced steps at once."""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.keys = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.queries = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.scale = attention_dim ** -0.5

    def forward(self, encoder_states, decoder_states, source_mask):
        keys = self.keys(encoder_states)
        queries = self.queries(decoder_states)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~source_mask.unsqueeze(1), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_states)
        return context, weights


class VectorizedGRUAttentionSeq2Seq(nn.Module):
    """Fast training path: parallel GRU decoding followed by batched attention."""
    def __init__(self, vocab_size, pad_token_id=0, embedding_dim=128, hidden_dim=256,
                 attention_dim=128, dropout=0.1, bidirectional=False,
                 encoder_layers=1, decoder_layers=1):
        super().__init__()
        if min(embedding_dim, hidden_dim, attention_dim, encoder_layers, decoder_layers) < 1:
            raise ValueError('Model dimensions and layer counts must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.pad_token_id = pad_token_id
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, encoder_layers, batch_first=True,
                              bidirectional=bidirectional,
                              dropout=dropout if encoder_layers > 1 else 0.0)
        encoder_dim = hidden_dim * (2 if bidirectional else 1)
        self.hidden_bridge = nn.Linear(encoder_dim, hidden_dim)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, decoder_layers, batch_first=True,
                              dropout=dropout if decoder_layers > 1 else 0.0)
        self.attention = VectorizedAttention(encoder_dim, hidden_dim, attention_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim + encoder_dim + embedding_dim, vocab_size)

    def encode(self, source, lengths):
        embedded = self.embedding_dropout(self.embedding(source))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_states, hidden = self.encoder(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True,
                                        total_length=source.size(1))
        directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.encoder_layers, directions, source.size(0), -1)[-1]
        final = torch.cat([hidden[i] for i in range(directions)], dim=-1)
        decoder_hidden = torch.tanh(self.hidden_bridge(final)).unsqueeze(0).repeat(
            self.decoder_layers, 1, 1)
        return states, decoder_hidden, source.ne(self.pad_token_id)

    def forward(self, source, lengths, decoder_input, return_attention=False):
        states, hidden, mask = self.encode(source, lengths)
        embedded = self.embedding_dropout(self.embedding(decoder_input))
        decoded, _ = self.decoder(embedded, hidden)
        context, attention = self.attention(states, decoded, mask)
        features = torch.cat([decoded, context, embedded], dim=-1)
        logits = self.output(self.output_dropout(features))
        return (logits, attention) if return_attention else logits

    @torch.inference_mode()
    def generate(self, source, lengths, bos_id, eos_id, pad_id, unk_id,
                 max_new_tokens=64, return_attention=False):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        states, hidden, mask = self.encode(source, lengths)
        token = torch.full((source.size(0), 1), bos_id, dtype=torch.long,
                           device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        output, attentions = [], []
        for _ in range(max_new_tokens):
            embedded = self.embedding(token)
            decoded, hidden = self.decoder(embedded, hidden)
            context, weights = self.attention(states, decoded, mask)
            logits = self.output(torch.cat([decoded[:, 0], context[:, 0], embedded[:, 0]], dim=-1))
            logits[:, [pad_id, bos_id, unk_id]] = -torch.inf
            next_token = logits.argmax(-1)
            next_token = torch.where(finished, pad_id, next_token)
            output.append(next_token)
            attentions.append(weights[:, 0])
            finished |= next_token.eq(eos_id)
            token = next_token.unsqueeze(1)
            if finished.all():
                break
        ids = torch.stack(output, dim=1)
        return (ids, torch.stack(attentions, dim=1)) if return_attention else ids
