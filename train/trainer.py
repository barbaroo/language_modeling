from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class MLMTrainer(pl.LightningModule):
    """LightningModule for training a masked language model."""

    def __init__(
        self,
        model: pl.LightningModule,
        tokenizer,
        dataset: Dataset,
        batch_size: int,
        lr: float,
        num_workers: int,
        gradient_clip_val: float,
        accumulate_grad_batches: int,
        train_test_split: float,
    ) -> None:
        """Initialize the MLMTrainer.

        Args:
            model (pl.LightningModule): Pre-trained masked language model.
            tokenizer: Tokenizer for tokenizing the input.
            dataset (Dataset): Dataset for masked language modeling.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for optimization.
            num_workers (int): Number of workers for data loading.
            gradient_clip_val (float): Maximum norm of the gradients.
            accumulate_grad_batches (int): The amount of gradient accumulation steps.
            train_test_split (float): The fraction of dataset to be used for training.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.split = train_test_split

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the dataset for training and validation.

        Args:
            stage (str, optional): The stage being setup (e.g., 'fit', 'test'). Defaults to None.
        """
        train_length = int((1 - self.split) * len(self.dataset))
        self.dataset_train, self.dataset_val = random_split(
            self.dataset, [train_length, len(self.dataset) - train_length]
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): Dictionary containing a batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = self.model(**batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step for a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): Dictionary containing a batch of data.
            batch_idx (int): Index of the batch.
        """
        val_loss = self.model(**batch)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, float, int]]:
        """Configure the optimizer and return it along with gradient clipping value and gradient accumulation settings.

        Returns:
            dict: A dictionary containing the optimizer, gradient clipping value, and gradient accumulation batches.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
        }

    def save_model(self, file_path: str) -> None:
        """Save the model to a file.

        Args:
            file_path (str): File path to save the model.
        """
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model from a file.

        Args:
            file_path (str): File path to load the model from.
        """
        self.model.load_state_dict(torch.load(file_path))
