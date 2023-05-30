import random

from torch.utils.data import Dataset


class DatasetForMLM(Dataset):
    def __init__(self, texts, tokenizer, mask_prob):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, padding="max_length", max_length=514, truncation=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        masked_indices = [i for i, token in enumerate(input_ids[0]) if token != self.tokenizer.pad_token_id]
        num_masked = max(1, int(len(masked_indices) * self.mask_prob))
        masked_indices = random.sample(masked_indices, num_masked)

        input_ids_masked = input_ids.detach().clone()
        for idx in masked_indices:
            input_ids_masked[0, idx] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": input_ids_masked.squeeze(),
        }
