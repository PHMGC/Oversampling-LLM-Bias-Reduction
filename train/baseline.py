import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Loading a public sentiment dataset to serve as our initial test base
print("Loading dataset")
raw_dataset = load_dataset("imdb")

def tokenize_function(examples):
	# Truncating to 128 to speed up our initial testing phase
	return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing data")
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# Renaming the column so the model recognizes it as the targets
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Setting the format to PyTorch tensors
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(42))

# Setting batch size based on typical RoBERTa fine-tuning
batch_size = 16

# Creating the DataLoaders
train_dataloader = DataLoader(
	#tokenized_datasets["train"], 
    small_train_dataset,
    shuffle=True, batch_size=batch_size
    
)
eval_dataloader = DataLoader(
	tokenized_datasets["test"], batch_size=batch_size
)

# Instantiating the optimizer with a standard learning rate for fine-tuning
optimizer = AdamW(model.parameters(), lr=5e-5)

# Number of times the model will see the entire dataset
num_epocs = 3
device = torch.device("cuda")
model.to(device)

# Setting the model to training mode
model.train()
for epoch in range(num_epocs):
    for step, batch in enumerate(train_dataloader):
        # Moving the data to the same device as the model
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Clear previus gradients
        optimizer.zero_grad()
        
        # Forward pass and loss extraction
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        print(f"Batch {step + 1} | Loss {loss.item()}")


script_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(script_dir, "../models/roberta_baseline")
os.makedirs(weights_dir, exist_ok=True)

# Save the model and the tokenizer
print("Saving model to", weights_dir)
model.save_pretrained(weights_dir)
tokenizer.save_pretrained(weights_dir)