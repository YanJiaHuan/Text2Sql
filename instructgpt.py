from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import GPT2LMHeadModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import nltk
import evaluate
import numpy as np
import torch
import random

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

# new process
def preprocess_function(examples):
    inputs = ['Instruction: ' + instruction + ' Context: ' + context for instruction, context in zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["output"], max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

# new metrci
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir = "AI_Tutor_Training/"+task_name + "_round2",
    evaluation_strategy = "steps",
    eval_steps = 200,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy='steps',
    save_steps = 600,
    num_train_epochs=100,  # specify the number of epochs you want here
    per_device_train_batch_size=24,  # specify the batch size you want here
    per_device_eval_batch_size=8,  # specify the evaluation batch size if you want it to be different from the training batch size
    gradient_accumulation_steps=1,  # dafault is 1
    eval_accumulation_steps=1,  # default is 1
    # deepspeed="./ds_config_zero3.json",
    fp16 = True,
    predict_with_generate=True,
)

# Train the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
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