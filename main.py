import logging
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train.dataset import DatasetForMLM
from train.trainer import MLMTrainer

class ModelTraining:
    """Class for training a masked language model."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the ModelTraining class.

        Args:
            cfg (DictConfig): Configuration options from Hydra.
        """
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.model.name)
        self.dataset = DatasetForMLM(cfg.dataset.sentences, self.tokenizer, cfg.dataset.mask_probability, cfg.dataset.max_tokens)

    def get_callbacks(self) -> List[Callback]:
        """
        Get the list of callbacks for the trainer.

        Returns:
            List[Callback]: List of PyTorch Lightning Callbacks.
        """
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        return [
            checkpoint_callback,
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=True),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    def get_trainer(self) -> pl.Trainer:
        """
        Get the PyTorch Lightning Trainer.

        Returns:
            pl.Trainer: PyTorch Lightning Trainer.
        """
        return pl.Trainer(
            **self.cfg.trainer,
            logger=pl.loggers.TensorBoardLogger(save_dir=f"{self.cfg.trainer.default_root_dir}/logs", name="mlm_logs"),
            callbacks=self.get_callbacks(),
        )

    def train(self) -> None:
        """
        Train the model.
        """
        trainer = self.get_trainer()
        mlm_trainer = MLMTrainer(
            self.model,
            self.tokenizer,
            self.dataset,
            self.cfg.batch_size,
            self.cfg.lr,
            self.cfg.num_workers,
            self.cfg.trainer.gradient_clip_val,
            self.cfg.trainer.accumulate_grad_batches,
            self.cfg.dataset.train_test_split,
        )
        trainer.fit(mlm_trainer)


@hydra.main(config_path="./config/", config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Main function for training a masked language model.

    Args:
        cfg (DictConfig): Configuration options from Hydra.
    """
    if cfg.num_workers is None:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        cfg.num_workers = num_cores

    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)  # Set a seed for reproducibility

    model_training = ModelTraining(cfg)

    try:
        model_training.train()
    except FileNotFoundError as e:
        logging.exception("An error occurred during training.")
        raise e


if __name__ == "__main__":
    main()
