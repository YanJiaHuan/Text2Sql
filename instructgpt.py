from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load your dataset
data = load_dataset('json', data_files='train_1890.json')
data = data['train'].train_test_split(test_size=0.1)

# Define accuracy metric (Replace with your own relevant metric)
accuracy_metric = load_metric('accuracy')

# Tokenize and format the dataset
def format_dataset(example):
    # Tokenize context and question
    tokenized_input = tokenizer(example['input'], padding='max_length', truncation=True, max_length=512)
    # Tokenize answer
    tokenized_label = tokenizer(example['output'], padding='max_length', truncation=True, max_length=512)
    return {'input_ids': tokenized_input['input_ids'], 'labels': tokenized_label['input_ids']}

# Tokenize and format the datasets
tokenized_train_dataset = data['train'].map(format_dataset, batched=True)
tokenized_eval_dataset = data['test'].map(format_dataset, batched=True)

# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Detokenize predictions
    detokenized_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    # Detokenize labels
    detokenized_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    # Print some predictions for the sake of example
    for i in range(5):  # Change this range according to your needs
        print(f"Input: {detokenized_labels[i]}")
        print(f"Prediction: {detokenized_predictions[i]}\n")

    # Compute accuracy (or other relevant metrics)
    return accuracy_metric.compute(predictions=detokenized_predictions, references=detokenized_labels)

# Set up training arguments
training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    save_strategy='epoch',
    num_train_epochs=5,  # specify the number of epochs you want here
    per_device_train_batch_size=16,  # specify the batch size you want here
    per_device_eval_batch_size=16,  # specify the evaluation batch size if you want it to be different from the training batch size
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

# trainer.train()

from transformers import pipeline

# Load your trained model and the corresponding tokenizer
model = GPT2LMHeadModel.from_pretrained('./test-clm/checkpoint-3645')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Create a pipeline for text generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)  # device=0 means it will run on the first GPU

# Run inference on the model
input_text = tokenized_eval_dataset[0]
print('input_text', input_text)
truncated_input = input_text['input'][:tokenizer.model_max_length - 1]  # -1 for special tokens
output = generator(truncated_input, max_length=512, num_return_sequences=1)


print(output)

# CUDA_VISIBLE_DEVICES=0 python instructgpt.py