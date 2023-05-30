from train.trainer import MLMTrainer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from base import MODEL_NAME

#Initialize model and tokenizers
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

#Train model
trainer = MLMTrainer(model, tokenizer)
trainer.train()

#Test model
trainer.test()

