from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Load dataset
dataset = pd.read_csv(
    "C:/Users/Gicano Brothers/Documents/POP Repositories/Sentiment-Analysis/data/raw/IMDB Dataset.csv"
)

# Print column names to identify the correct column for text data
print(dataset.columns)


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


# Assuming the correct column name is 'review' based on typical IMDB dataset structure
dataset["input_ids"] = dataset["review"].apply(
    lambda x: tokenize_function(x)["input_ids"]
)
dataset["attention_mask"] = dataset["review"].apply(
    lambda x: tokenize_function(x)["attention_mask"]
)

# Prepare the dataset for training
dataset = dataset.rename(columns={"sentiment": "labels"})
dataset["labels"] = dataset["labels"].apply(lambda x: 1 if x == "positive" else 0)

# Convert to torch tensors
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(dataset["input_ids"].tolist(), dtype=torch.long),
    torch.tensor(dataset["attention_mask"].tolist(), dtype=torch.long),
    torch.tensor(dataset["labels"].tolist(), dtype=torch.long),
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Use train_dataset as eval_dataset for simplicity
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
