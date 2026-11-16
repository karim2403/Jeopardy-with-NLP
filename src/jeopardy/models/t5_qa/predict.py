"""Generate a Jeopardy answer from a fine-tuned T5 checkpoint."""
import argparse
from pathlib import Path

import torch

from .common import autocast_context, model_device, resolve_precision
from .dataset import format_input


class T5Predictor:
    def __init__(self, checkpoint, device='auto', precision='auto'):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.checkpoint = Path(checkpoint)
        self.device = model_device(device)
        self.precision = resolve_precision(self.device, precision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device).eval()

    @torch.inference_mode()
    def answer(self, category, clue, max_input_length=128, max_new_tokens=32, num_beams=1):
        batch = self.tokenizer(format_input(category, clue), return_tensors='pt',
                               truncation=True, max_length=max_input_length).to(self.device)
        with autocast_context(self.device, self.precision):
            output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                         num_beams=num_beams, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--clue', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'off', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--max-input-length', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--num-beams', type=int, default=1)
    args = parser.parse_args()
    if min(args.max_input_length, args.max_new_tokens, args.num_beams) < 1:
        parser.error('lengths and beam count must be positive')
    predictor = T5Predictor(args.checkpoint, args.device, args.precision)
    print(predictor.answer(args.category, args.clue, args.max_input_length,
                           args.max_new_tokens, args.num_beams))


if __name__ == '__main__':
    main()
