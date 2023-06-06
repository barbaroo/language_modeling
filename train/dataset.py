from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class DatasetForMLM(Dataset):
    """Dataset for the MLM model.

    Args:
        sentences (List[str]): The sentences in the dataset.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the sentences.
        max_tokens (int): The maximum number of tokens per sentence.

    Methods:
        __len__(): Returns the number of sentences in the dataset.
        __getitem__(): Returns a dictionary containing the input_ids, attention_mask, and labels for the given sentence.
    """

    def __init__(self, sentences: List[str], tokenizer: PreTrainedTokenizerBase, max_tokens: int):
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)

    def __len__(self) -> int:
        """Returns the number of sentences in the dataset.

        Returns:
            int: The number of sentences in the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a dictionary containing the input_ids, attention_mask, and labels for the given sentence.

        Args:
            idx (int): Index of the sentence.

        Returns:
            dict: Dictionary containing the input_ids, attention_mask, and labels for the given sentence.
        """
        try:
            sentence = self.sentences[idx]
            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            
            #return {"input_ids": input_ids, "attention_mask": attention_mask}
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}
        except Exception as e:
            raise Exception(f"Failed to process item at index {idx}: {str(e)}")
