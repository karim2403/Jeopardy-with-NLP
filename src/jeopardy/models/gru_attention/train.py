"""Train the vectorized attention GRU using the shared seq2seq workflow."""
from .model import VectorizedGRUAttentionSeq2Seq
from ..gru_seq2seq.train import build_parser, run

ATTENTION_DEFAULTS = dict(attention_dim=128, dropout=0.1, bidirectional=False,
                          encoder_layers=1, decoder_layers=1,
                          early_stopping_patience=3, batch_size=128,
                          eval_samples=256, precision='auto', bucket_multiplier=50)
MODEL_KEYS = ('attention_dim', 'dropout', 'bidirectional', 'encoder_layers', 'decoder_layers')


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--attention-dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--bidirectional', action='store_true', default=None)
    parser.add_argument('--encoder-layers', type=int)
    parser.add_argument('--decoder-layers', type=int)
    args = parser.parse_args()
    run(args, model_class=VectorizedGRUAttentionSeq2Seq, extra_defaults=ATTENTION_DEFAULTS,
        extra_model_keys=MODEL_KEYS, model_type='gru_attention_vectorized')


if __name__ == '__main__':
    main()
