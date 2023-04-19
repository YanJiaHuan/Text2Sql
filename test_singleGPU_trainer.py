import os
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import deepspeed
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = "translate English to SQL: " + example["question"]
        output_text = example["query"]

        input_tokenized = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        output_tokenized = self.tokenizer(output_text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        return {
            "input_ids": input_tokenized["input_ids"][0],
            "attention_mask": input_tokenized["attention_mask"][0],
            "labels": output_tokenized["input_ids"][0],
            "labels_attention_mask": output_tokenized["attention_mask"][0]
        }

def main():
    # Set up the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-3B")

    # Set up the dataset
    dataset = load_dataset("spider", split='train')
    dataset = dataset.shuffle(seed=42)

    # Create the custom dataset
    train_dataset = CustomDataset(tokenizer, dataset)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        save_total_limit=3,
        logging_dir="./logs",
        fp16=True,  # Enable mixed precision training
        learning_rate=5e-5,
    )



    deepspeed_engine, model, optimizer, train_dataloader = deepspeed.initialize(
        args=training_args,
        model=T5ForConditionalGeneration.from_pretrained("t5-3B"),
        model_parameters=None,
        training_data=train_dataset,
        config="deepspeed_config.json",
    )

    device = deepspeed_engine.local_rank

    # Calculate total steps
    total_steps = len(train_dataset) * training_args.num_train_epochs // deepspeed_engine.train_micro_batch_size_per_gpu

    # Train the model with tqdm progress bar
    progress_bar = tqdm(range(total_steps), desc="Training")
    deepspeed_engine.train()

    for step in progress_bar:
        loss = deepspeed_engine.step()
        progress_bar.set_description(f"Loss: {loss.item()}")

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}, Loss: {loss.item()}")

    deepspeed_engine.save_checkpoint(training_args.output_dir)


if __name__ == "__main__":
    main()
