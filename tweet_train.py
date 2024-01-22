from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import re

def preprocess_text(text):
    text = str(text) if text is not None else ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'&\S+;', '', text)
    text = re.sub(r'[^\w\s,.!?\'\"]', '', text)
    
    return text

def preprocess_function(examples):
    examples["tweet_text"] = [preprocess_text(tweet) for tweet in examples["tweet_text"]]
    return examples

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["tweet_text"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = [row[1:] + [tokenizer.pad_token_id] for row in tokenized_inputs["input_ids"]]
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("StephanAkkerman/financial-tweets-crypto")

dataset = dataset.map(preprocess_function, batched=True)
tokenized_datasets = dataset["train"].map(tokenize_function, batched=True, remove_columns=["tweet_text"])

training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=0.1,
    evaluation_strategy="no",
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()

model.save_pretrained("/dev/shm/training/")
tokenizer.save_pretrained("/dev/shm/training/")

