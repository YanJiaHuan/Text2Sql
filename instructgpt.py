from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import evaluate
import numpy as np
import torch
import random

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token
data_path = 'train_1890/'
task_name = 'title_train'
# authors_train, keywords_train, background_train, methodologies_train, conclusions_train
# Load your dataset
data = load_dataset('json', data_files=data_path+task_name+'.json')
data = data['train'].train_test_split(test_size=0.1)

# Define accuracy metric (Replace with your own relevant metric)
bleu_metric = load_metric('sacrebleu')
# bleu_metric = evaluate.load('sacrebleu')
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

    decoded_preds = [pred.replace('<|endoftext|>', '') for pred in decoded_preds]
    decoded_labels = [label.replace('<|endoftext|>', '') for label in decoded_labels]
    for i in range(min(5, len(decoded_preds))):  # print first 5 examples
        print(f"Example {i+1}:")
        print(f"Input: {decoded_inputs[i]}")
        print(f"Prediction: {decoded_preds[i]}")
        print(f"Label: {decoded_labels[i]}\n")

    # Tokenize on space to get list of words (required for BLEU)
    decoded_preds = [pred.split(' ') for pred in decoded_preds]
    decoded_labels = [[label.split(' ')] for label in decoded_labels]  # Note that it's a list of list

    eval_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {'eval_score': eval_score['score']}


# Set up training arguments
training_args = TrainingArguments(
    output_dir = "AI_Tutor_Training/"+task_name,
    evaluation_strategy = "steps",
    eval_steps = 200,
    learning_rate=1e-5,
    weight_decay=0.05,
    save_strategy='steps',
    save_steps = 600,
    num_train_epochs=1000,  # specify the number of epochs you want here
    per_device_train_batch_size=24,  # specify the batch size you want here
    per_device_eval_batch_size=8,  # specify the evaluation batch size if you want it to be different from the training batch size
    gradient_accumulation_steps=1,  # dafault is 1
    eval_accumulation_steps=1,  # default is 1
    # deepspeed="./ds_config_zero3.json",
    fp16 = False,
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


# from transformers import GPT2LMHeadModel, GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # Add padding token to the tokenizer
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = GPT2LMHeadModel.from_pretrained('./AI_Tutor_Training/'+task_name+'/checkpoint-2400')
# model = model.to(device)  # move model to the device
# model.eval()
#
#
# def generate_text(prompt_text):
#     # Define the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Move the model to the defined device
#     model.to(device)
#
#     # Encode the prompt text
#     encoded_input = tokenizer.encode(prompt_text, return_tensors='pt', max_length=512, truncation=True)
#     # Move the encoded input to the device
#     encoded_input = encoded_input.to(device)
#
#     # Generate text
#     output = model.generate(encoded_input, max_length=512, pad_token_id=tokenizer.eos_token_id)
#     # Decode the output
#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#     return decoded_output
#
#
# # Load your dataset
# data = load_dataset('json', data_files=data_path+task_name+'.json')
# data = data['train'].train_test_split(test_size=0.1)
#
# # Randomly select 5 samples
# random_indices = random.sample(range(len(data['train'])),1)
# selected_samples = data['train'].select(random_indices)
#
# # Generate text for each selected sample
# for example in selected_samples:
#     prompt_text = 'Instruction: ' + example['instruction'] + ' Context: ' + example['input']
#     # prompt_text = 'This is a test'
#     # print('Input:\n', prompt_text)
#     print('Prediction:\n', generate_text(prompt_text))
#     print('Label:\n', example['output'])
#     print('###'*10, '\n')



# deepspeed --include localhost:0,1,2,3 instructgpt.py --deepspeed ds_config_zero3.json
# deepspeed --include localhost:0 instructgpt.py --deepspeed ds_config_zero3.json
# CUDA_VISIBLE_DEVICES=0,1,2,3 python instructgpt.py