import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(script_dir, "../models/roberta_baseline")

# Load trained model
tokenizer = AutoTokenizer.from_pretrained(weights_dir)
model = AutoModelForSequenceClassification.from_pretrained(weights_dir)

raw_dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_lengh', truncation=True, max_lengh=128)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

small_eval_dataset = tokenized_datasets["test"].select(range(64))


batch_size = 64
eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

device = torch.device("cuda")
model.to(device)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
	for batch in eval_dataloader:
		batch = {k: v.to(device) for k, v in batch.items()}
		outputs = model(**batch)
		
		# Logits are the raw scores
		logits = outputs.logits
		
		# Taking the index of the highest score (0 or 1)
		predictions = torch.argmax(logits, dim=-1)
		
		# Moving back to CPU and converting to numpy arrays
		all_preds.extend(predictions.cpu().numpy())
		all_labels.extend(batch["labels"].cpu().numpy())
  
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Macro-F1: {macro_f1:.4f}")