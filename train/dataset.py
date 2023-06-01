import random
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class DatasetForMLM(Dataset):
    """Dataset class for masked language modeling."""

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, mask_prob: float, max_tokens: int):
        """Initialize the DatasetForMLM.

        Args:
            texts (List[str]): List of sentences for training.
            tokenizer (transformers.AutoTokenizer): Tokenizer for tokenizing the input.
            mask_prob (float): Probability of masking tokens in the input sequence.
            max_tokens (int): Maximum number of tokens in a sequence.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing the masked input tokens, attention mask, and labels.
        """
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        masked_indices = [i for i, token in enumerate(input_ids) if token != self.tokenizer.pad_token_id]
        num_masked = max(1, int(len(masked_indices) * self.mask_prob))
        masked_indices = random.sample(masked_indices, num_masked)

        input_ids_masked = input_ids.detach().clone()
        for idx in masked_indices:
            input_ids_masked[idx] = self.tokenizer.mask_token_id

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids_masked}
