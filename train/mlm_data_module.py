import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from typing import List
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from .dataset import DatasetForMLM


class MLMDataModule(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: DictConfig):
        super().__init__()
        """
        Data module for Masked Language Modeling (MLM).

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the data.
            config (DictConfig): The configuration for the data module.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.sentences_train: List[str] = []
        self.sentences_val: List[str] = []
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None

    def prepare_data(self) -> None:
        """
        Prepares the data for the MLM model.

        This method will be called only once.

        Raises:
            DataPreparationException: If there is an error preparing the data.
        """
        try:
            data = pd.read_csv(self.config.sentences)
            sentences = data['Text'].tolist()#[:self.config.number_examples]
            self.sentences_train, self.sentences_val = train_test_split(
                sentences, test_size=self.config.train_test_split, random_state=42
            )

        except Exception as e:
            raise DataPreparationException(f"Data preparation failed: {str(e)}")

    def setup(self, stage: str = None) -> None:
        """
        Sets up the MLM data module.

        Args:
            stage (str, optional): Stage of training (fit/predict).

        Raises:
            DataSetupException: If there is an error setting up the data.
        """


        if stage == 'fit' or stage is None:
            try:
                self.train_dataset = DatasetForMLM(self.sentences_train, self.tokenizer, self.config.max_tokens)
                self.val_dataset = DatasetForMLM(self.sentences_val, self.tokenizer, self.config.max_tokens)
            except Exception as e:
                raise DataSetupException(f"Data setup failed: {str(e)}")
        elif stage == "test":
                self.val_dataset = DatasetForMLM(self.sentences_val, self.tokenizer, self.config.max_tokens)
        print(f'Setup called. Train dataset: {self.train_dataset}')


    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """

        if len(self.sentences_train) == 0:
            raise ValueError("No training data found.")
        print(f'train_dataloader called. Train dataset: {self.train_dataset}')
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        if len(self.sentences_val) == 0:
            raise ValueError("No validation data found.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def load_sentences(self, file_path: str) -> List[str]:
        """
        Load sentences from a file.

        Args:
            file_path (str): Path to the file containing sentences.

        Returns:
            List[str]: List of sentences.
        """
        try:
            with open(file_path, 'r') as f:
                texts = f.read().splitlines()
            return texts
        except Exception as e:
            raise Exception(f"Failed to load sentences from file: {str(e)}")


class DataPreparationException(Exception):
    """Exception raised when there is an error preparing the data."""

    def __init__(self, message: str):
        super().__init__(message)


class DataSetupException(Exception):
    """Exception raised when there is an error setting up the data."""

    def __init__(self, message: str):
        super().__init__(message)


class DataloaderCreationException(Exception):
    """Exception raised when there is an error creating the dataloader."""

    def __init__(self, message: str):
        super().__init__(message)
