import torch
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train.data_model import MLMTrainerParams
from train.trainer import MLMTrainer


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["MODEL_NAME"]
    path_to_data = config["PATH_TO_DATA"]

    batch_size = config["BATCH_SIZE"]
    train_test_split = config["TRAIN_TEST_SPLIT"]
    learning_rate = float(config["LEARNING_RATE"])
    epochs = config["EPOCHS"]
    max_tokens = config["MAX_TOKENS"]
    number_examples = config["NUMBER_EXAMPLES"]
    device = torch.device(DEVICE)
    mask_probability = config["MASK_PROBABILITY"]

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
