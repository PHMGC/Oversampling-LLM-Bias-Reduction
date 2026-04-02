import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(script_dir, "../models/roberta_baseline")

# Load trained model
tokenizer = AutoTokenizer.from_pretrained(weights_dir)
model = AutoModelForSequenceClassification.from_pretrained(weights_dir)

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