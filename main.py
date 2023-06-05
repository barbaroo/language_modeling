import logging
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mlmodel import MLMModel
from train.mlm_data_module import MLMDataModule


class MLMPipeline:
    """Main class to handle Masked Language Model training using PyTorch Lightning."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

    def create_trainer(self) -> Trainer:
        """Function to create a PyTorch Lightning Trainer with given configurations.

        Returns:
            Trainer: Initialized PyTorch Lightning Trainer.
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.config.trainer.default_root_dir}/checkpoints",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=self.config.early_stopping_patience, mode="min"
        )

        # Before creating the DeepSpeedPlugin, validate the config
        # if 'config_file_or_dict' not in self.config.deepspeed_config:
        #    raise ValueError("Missing required key 'config_file_or_dict' in deepspeed config.")

        # deepspeed_plugin = DeepSpeedPlugin(
        #    **self.config.deepspeed_config
        # )

        return Trainer(
            logger=True,
            max_epochs=self.config.trainer.max_epochs,
            enable_progress_bar=self.config.trainer.enable_progress_bar,
            enable_model_summary=self.config.trainer.enable_model_summary,
            gradient_clip_val=self.config.trainer.gradient_clip_val,
            accumulate_grad_batches=self.config.trainer.accumulate_grad_batches,
            limit_train_batches=self.config.trainer.limit_train_batches,
            limit_val_batches=self.config.trainer.limit_val_batches,
            devices=self.config.trainer.devices,
            accelerator=self.config.trainer.accelerator,
            strategy=self.config.trainer.strategy,
            precision=self.config.trainer.precision,
            fast_dev_run=self.config.trainer.fast_dev_run,
            deterministic=self.config.trainer.deterministic,
            check_val_every_n_epoch=self.config.trainer.check_val_every_n_epoch,
            num_nodes=self.config.trainer.num_nodes,
            benchmark=self.config.trainer.benchmark,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

    def run(self) -> None:
        """Main function to run the MLM training pipeline.

        Raises:
            Exception: Any exception that occurs during the training process.
        """
        self.log.info(OmegaConf.to_yaml(self.config))

        torch.manual_seed(self.config.seed)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
            datamodule = MLMDataModule(tokenizer, self.config.dataset)
            datamodule.prepare_data()
            model = MLMModel(self.config.model, self.config.optimizer)
            model.model = AutoModelForMaskedLM.from_pretrained(self.config.model.name)
        except Exception as e:
            self.log.exception("An error occurred during data preparation or model initialization: %s", str(e))
            raise e

        trainer = self.create_trainer()

        self.log.info("Starting training.")
        try:
            trainer.fit(model, datamodule)
        except Exception as e:
            self.log.exception("Training failed with an exception: %s", str(e))
            raise e
        self.log.info("Training completed.")


@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:
    """Entry point for the script, creates an instance of MLMPipeline and runs the pipeline.

    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    pipeline = MLMPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
