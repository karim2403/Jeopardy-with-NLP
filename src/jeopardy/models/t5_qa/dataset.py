"""Formatting and batching for pretrained text-to-text models."""
import torch
from torch.utils.data import Dataset


def format_input(category, clue):
    return f"answer jeopardy clue\ncategory: {category}\nclue: {clue}"


class T5QADataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {'id': row['id'], 'category': row['category'], 'clue': row['clue'],
                'input_text': format_input(row['category'], row['clue']),
                'target_text': row['answer']}


class T5Collator:
    def __init__(self, tokenizer, max_input_length=128, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, items):
        inputs = self.tokenizer([x['input_text'] for x in items], padding=True,
                                truncation=True, max_length=self.max_input_length,
                                return_tensors='pt')
        targets = self.tokenizer(text_target=[x['target_text'] for x in items], padding=True,
                                 truncation=True, max_length=self.max_target_length,
                                 return_tensors='pt')
        labels = targets['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['items'] = items
        return inputs


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()
            if isinstance(value, torch.Tensor)}
