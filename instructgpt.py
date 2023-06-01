from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load your dataset
data = load_dataset('json', data_files='train_1890.json')
data = data['train'].train_test_split(test_size=0.1)

# Tokenize and format the dataset
def format_dataset(example):
    # Tokenize context and question
    print(example['instruction'])
    tokenized_input = tokenizer(example['input'], padding='max_length', truncation=True, max_length=512)
    # Tokenize answer
    tokenized_label = tokenizer(example['output'], padding='max_length', truncation=True, max_length=512)
    return {'input_ids': tokenized_input['input_ids'], 'labels': tokenized_label['input_ids']}

# Tokenize and format the datasets
tokenized_train_dataset = data['train'].map(format_dataset, batched=True)
tokenized_eval_dataset = data['test'].map(format_dataset, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy='epoch'
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

trainer.train()

# CUDA_VISIBLE_DEVICES=0 python instructgpt.py