import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from train.dataset import Dataset_for_MLM
from base import PATH_TO_DATA, BATCH_SIZE, LEARNING_RATE, EPOCHS, NUMBER_EXAMPLES, DEVICE, TRAIN_TEST_SPLIT

class MLMTrainer:
    def __init__(self, model, tokenizer, dataset_path = PATH_TO_DATA, batch_size=BATCH_SIZE, lr= LEARNING_RATE, epochs=EPOCHS, device = DEVICE, n_examples = NUMBER_EXAMPLES, split = TRAIN_TEST_SPLIT):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.n_examples = n_examples
        self.split = split

    def load_dataset(self):
        data = pd.read_csv(self.dataset_path)
        sentences = data['Text'].tolist()[:self.n_examples]
        sentences_train, sentences_test = train_test_split(sentences, test_size= self.split, random_state=42)
        dataset_train = Dataset_for_MLM(sentences_train, self.tokenizer)
        dataset_test = Dataset_for_MLM(sentences_test, self.tokenizer)
        
        return dataset_train, dataset_test

    def train(self):
        dataset_train, dataset_test = self.load_dataset()
        dataloader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
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
                
    
    
    def test(self):
        self.model.eval()
        dataset_train, dataset_test = self.load_dataset()
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            total_loss = 0
            total_examples = 0

            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

        avg_loss = total_loss / total_examples
        print(f'Average Test Loss: {avg_loss}')



