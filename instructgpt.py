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
bleu_metric = load_metric('sacrebleu')

# Tokenize and format the dataset
def format_dataset(example):
    # Combine instruction and input
    inputs = []
    for instruction, context in zip(example['instruction'], example['input']):
        input = 'Instruction: ' + instruction + ' Context: ' + context
        inputs.append(input)
    # Tokenize combined input and output
    tokenized_input = tokenizer(inputs, padding='max_length', truncation=True, max_length=512)
    tokenized_output = tokenizer(example['output'], padding='max_length', truncation=True, max_length=512)

    return {'input_ids': tokenized_input['input_ids'], 'labels': tokenized_output['input_ids'], 'attention_mask': tokenized_input['attention_mask']}

# Tokenize and format the datasets
tokenized_train_dataset = data['train'].map(format_dataset, batched=True)
tokenized_eval_dataset = data['test'].map(format_dataset, batched=True)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels, input_ids = eval_pred
    predictions = logits.argmax(-1)

    # Convert ids to tokens (do not skip special tokens)
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=False) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=False) for label in labels]
    decoded_inputs = [tokenizer.decode(input, skip_special_tokens=False) for input in input_ids]
    for i in range(min(5, len(decoded_preds))):  # print first 5 examples
        print(f"Example {i+1}:")
        print(f"Input: {decoded_inputs[i]}")
        print(f"Prediction: {decoded_preds[i]}")
        print(f"Label: {decoded_labels[i]}\n")

    # Tokenize on space to get list of words (required for BLEU)
    decoded_preds = [pred.split(' ') for pred in decoded_preds]
    decoded_labels = [[label.split(' ')] for label in decoded_labels]  # Note that it's a list of list


    return bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)


# Set up training arguments
training_args = TrainingArguments(
    output_dir = "AI_Tutor_Training",
    evaluation_strategy = "steps",
    eval_steps = 1,
    learning_rate=1e-4,
    weight_decay=0.01,
    save_strategy='steps',
    save_steps = 6000,
    num_train_epochs=20,  # specify the number of epochs you want here
    per_device_train_batch_size=16,  # specify the batch size you want here
    per_device_eval_batch_size=8,  # specify the evaluation batch size if you want it to be different from the training batch size
    gradient_accumulation_steps=1,  # dafault is 1
    eval_accumulation_steps=1,  # default is 1
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    deepspeed="./ds_config_zero3.json",
    fp16 = True,
    include_inputs_for_metrics=True,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# from transformers import pipeline
#
# # Load your trained model and the corresponding tokenizer
# model = GPT2LMHeadModel.from_pretrained('./test-clm/checkpoint-3645')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
# # Add padding token to the tokenizer
# tokenizer.pad_token = tokenizer.eos_token
#
# # Create a pipeline for text generation
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)  # device=0 means it will run on the first GPU
#
# # Run inference on the model
# input_text = tokenized_eval_dataset[0]
# print('input_text', input_text)
# truncated_input = input_text['input'][:tokenizer.model_max_length - 1]  # -1 for special tokens
# output = generator(truncated_input, max_length=512, num_return_sequences=1)
#
#
# print(output)


# deepspeed --include localhost:0,1,2,3 instructgpt.py --deepspeed ds_config_zero3.json