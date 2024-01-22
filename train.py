from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch


torch.cuda.set_device(1)
def tokenize_function(examples):
    concatenated_texts = []
    for tweet_text in examples["tweet_text"]:
        if tweet_text is None:
            tweet_text = ""
    return tokenizer(examples, padding="max_length", truncation=True, max_length=512)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("StephanAkkerman/financial-tweets-crypto")
tokenized_datasets = dataset["train"].map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()

