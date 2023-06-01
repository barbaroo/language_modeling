import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MLMTrainer(pl.LightningModule):
    """LightningModule for training a masked language model."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        batch_size,
        lr,
        num_workers,
        gradient_clip_val,
        accumulate_grad_batches,
        train_test_split,
    ):
        """Initialize the MLMTrainer.

        Args:
            model (transformers.PreTrainedModel): Pre-trained masked language model.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for tokenizing the input.
            dataset (DatasetForMLM): Dataset for masked language modeling.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for optimization.
            num_workers (int): Number of workers for data loading.
            gradient_clip_val (float): Maximum norm of the gradients.
            accumulate_grad_batches (int): The amount of gradient accumulation steps.
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

        # Enable/Disable progress bar
        self.enable_progress_bar = True

    def setup(self, stage=None):
        """Setup the dataset for training and validation.

        Args:
            stage (str, optional): The stage being setup (e.g., 'fit', 'test'). Defaults to None.
        """
        train_length = int((1 - self.split) * len(self.dataset))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset, [train_length, len(self.dataset) - train_length]
        )

    def train_dataloader(self):
        """Get the training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Get the validation dataloader.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def forward(self, input_ids, attention_mask, labels):
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor containing the input token IDs.
            attention_mask (torch.Tensor): Tensor containing the attention mask.
            labels (torch.Tensor): Tensor containing the labels.

        Returns:
            torch.Tensor: Loss value.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        """Training step for a batch of data.

        Args:
            batch (dict): Dictionary containing a batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss = self.forward(input_ids, attention_mask, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for a batch of data.

        Args:
            batch (dict): Dictionary containing a batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss value.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss = self.forward(input_ids, attention_mask=attention_mask, labels=labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
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

    def save_model(self, file_path):
        """Save the model to a file.

        Args:
            file_path (str): File path to save the model.
        """
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        """Load the model from a file.

        Args:
            file_path (str): File path to load the model from.
        """
        self.model.load_state_dict(torch.load(file_path))
