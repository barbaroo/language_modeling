from typing import Dict, Optional, Union
from torch import nn, Tensor
from transformers import AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningModule

class MLMModel(LightningModule):
    """
    MLMModel Class.

    This class is a PyTorch Lightning module that encapsulates a Masked Language Model.
    """
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig):
        """
        MLMModel class constructor.

        Args:
            model_cfg (DictConfig): The configuration for the model.
            optimizer_cfg (DictConfig): The configuration for the optimizer.
        """
        super().__init__()

        # Model configuration
        self.save_hyperparameters()  # Saves model and optimizer configurations for checkpoints
        self.model = AutoModelForMaskedLM.from_pretrained(self.hparams.model_cfg.name)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None, labels: Tensor = None ) -> Dict[str, Tensor]:
        """
        Forward pass of the MLM model.

        Args:
            input_ids (Tensor): Input tensor of token IDs.
            attention_mask (Tensor): Attention mask tensor. Defaults to None.
            labels (torch.Tensor): Tensor containing the labels.

        Returns:
            outputs (Dict[str, Tensor]): Model outputs.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the MLM model.

        Args:
            batch (Dict[str, Tensor]): Batch of training data.
            batch_idx (int): Batch index.

        Returns:
            loss (Tensor): Training loss.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, int]]:
        """
        Validation step for the MLM model.

        Args:
            batch (Dict[str, Tensor]): Batch of validation data.
            batch_idx (int): Batch index.

        Returns:
            log (Dict[str, Union[Tensor, int]]): Dictionary with the validation loss to be logged.
        """
        outputs = self(**batch)
        val_loss = outputs.loss
        log = {"val_loss": val_loss}
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return log

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for the MLM model.

        Returns:
            dict: A dictionary with optimizer and learning rate scheduler.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optimizer_cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.optimizer_cfg.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.optimizer_cfg.warmup_steps, 
            num_training_steps=self.hparams.optimizer_cfg.total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
