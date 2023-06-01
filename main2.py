import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train.dataset import DatasetForMLM
from train.trainer import MLMTrainer


@hydra.main(config_path="./config/", config_name="base")
def main(cfg: DictConfig):
    """Main function for training a masked language model.

    Args:
        cfg (DictConfig): Configuration options.
    """
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)  # Set a seed for reproducibility

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model.name)

    dataset = DatasetForMLM(cfg.dataset.sentences, tokenizer, cfg.dataset.mask_probability, cfg.dataset.max_tokens)

    # Create the ModelCheckpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=pl.loggers.TensorBoardLogger(save_dir=cfg.trainer.default_root_dir, name="mlm_logs"),
        callbacks=[
            checkpoint_callback,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    )

    try:
        mlm_trainer = MLMTrainer(
            model,
            tokenizer,
            dataset,
            cfg.batch_size,
            cfg.lr,
            cfg.num_workers,
            cfg.trainer.gradient_clip_val,
            cfg.trainer.accumulate_grad_batches,
            cfg.dataset.train_test_split,
        )
        trainer.fit(mlm_trainer)
    except FileNotFoundError as e:
        logging.exception("An error occurred during training.")
        raise e


if __name__ == "__main__":
    main()
