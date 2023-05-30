import torch
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train.trainer import MLMTrainer


def main():
    # Load YAML configuration file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Access parameters from the configuration file
    model_name = config["MODEL_NAME"]
    path_to_data = config["PATH_TO_DATA"]
    # output_dir = config['OUTPUT_DIR']  # Uncomment if needed

    batch_size = config["BATCH_SIZE"]
    train_test_split = config["TRAIN_TEST_SPLIT"]
    learning_rate = config["LEARNING_RATE"]
    epochs = config["EPOCHS"]
    max_tokens = config["MAX_TOKENS"]
    number_examples = config["NUMBER_EXAMPLES"]
    device = torch.device(config["DEVICE"])
    mask_probability = config["MASK_PROBABILITY"]

    # Initialize model and tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Train model
    trainer = MLMTrainer(model, tokenizer)
    trainer.train()

    # Test model
    trainer.test()


if __name__ == "__main__":
    main()
