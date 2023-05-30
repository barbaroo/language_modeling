import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train.data_model import MLMTrainerParams
from train.trainer import MLMTrainer


@hydra.main(config_path="./config/", config_name="config")
def main(cfg: DictConfig):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model.name
    path_to_data = cfg.dataset.sentences
    batch_size = cfg.trainer.batch_size
    train_test_split = cfg.dataset.train_test_split
    learning_rate = cfg.trainer.lr
    epochs = cfg.trainer.epochs
    max_tokens = cfg.dataset.max_tokens
    device = DEVICE
    number_examples = cfg.dataset.n_examples
    mask_probability = cfg.dataset.mask_probability

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    params = MLMTrainerParams(
        dataset_path=path_to_data,
        batch_size=batch_size,
        lr=learning_rate,
        epochs=epochs,
        device=device,
        n_examples=number_examples,
        split=train_test_split,
        mask_probability=mask_probability,
    )

    trainer = MLMTrainer(model, tokenizer, params)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
