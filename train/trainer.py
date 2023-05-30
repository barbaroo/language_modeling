import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from train.data_model import MLMTrainerParams
from train.dataset import DatasetForMLM


class MLMTrainer:
    def __init__(self, model, tokenizer, params):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params

    def load_dataset(self):
        data = pd.read_csv(self.params.dataset_path)
        sentences = data["Text"].tolist()[: self.params.n_examples]

        dataset = DatasetForMLM(sentences, self.tokenizer, self.params.mask_probability)

        train_length = int((1 - self.params.split) * len(dataset))
        test_length = len(dataset) - train_length
        dataset_train, dataset_test = random_split(dataset, [train_length, test_length])

        return dataset_train, dataset_test

    def train(self):
        dataset_train, dataset_test = self.load_dataset()
        dataloader = DataLoader(dataset_train, batch_size=self.params.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr)

        self.model.to(self.params.device)
        self.model.train()

        num_training_steps = self.params.epochs * len(dataloader)

        for epoch in range(self.params.epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.params.device)
                attention_mask = batch["attention_mask"].to(self.params.device)
                labels = batch["labels"].to(self.params.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch: {epoch}")
                loop.set_postfix(loss=loss.item())

    def test(self):
        self.model.eval()
        dataset_train, dataset_test = self.load_dataset()
        test_dataloader = DataLoader(dataset_test, batch_size=self.params.batch_size, shuffle=False)

        with torch.no_grad():
            total_loss = 0
            total_examples = 0

            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(self.params.device)
                attention_mask = batch["attention_mask"].to(self.params.device)
                labels = batch["labels"].to(self.params.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

        avg_loss = total_loss / total_examples
        print(f"Average Test Loss: {avg_loss}")
