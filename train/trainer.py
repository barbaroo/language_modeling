import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train.dataset import Dataset_for_MLM
from base import PATH_TO_DATA, BATCH_SIZE, LEARNING_RATE, EPOCHS, NUMBER_EXAMPLES, DEVICE

class MLMTrainer:
    def __init__(self, model, tokenizer, dataset_path = PATH_TO_DATA, batch_size=BATCH_SIZE, lr= LEARNING_RATE, epochs=EPOCHS, device = DEVICE, n_examples = NUMBER_EXAMPLES):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.n_examples = n_examples

    def load_dataset(self):
        data = pd.read_csv(self.dataset_path)
        sentences = data['Text'].tolist()[:self.n_examples]
        dataset = Dataset_for_MLM(sentences, self.tokenizer)
        return dataset

    def train(self):
        dataset = self.load_dataset()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)
        self.model.train()

        num_training_steps = self.epochs * len(dataloader)
        for epoch in range(self.epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                loop.set_description(f'Epoch: {epoch}')
                loop.set_postfix(loss=loss.item())



