from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import GPT2LMHeadModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import nltk
import numpy as np
import torch
import random

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vicgalle/gpt2-open-instruct-v1")
model = GPT2LMHeadModel.from_pretrained("vicgalle/gpt2-open-instruct-v1")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token
data_path = 'train_1890/'
task_name = 'title_train'

# Load your dataset
data = load_dataset('json', data_files=data_path + task_name + '.json')
data = data['train'].train_test_split(test_size=0.1)

# Define accuracy metric
bleu_metric = load_metric('sacrebleu')


def preprocess_function(examples):
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
    inputs = [prompt + "### Instruction:\n" + instruction + "\n\n" + "### Input:\n" + context + "\n\n" for instruction, context in
              zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, padding="max_length", max_length=1024, truncation=True)
    labels = tokenizer(examples["output"], padding="max_length", max_length=1024, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_data = data.map(preprocess_function, batched=True)


def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = [tokenizer.batch_decode(label, skip_special_tokens=True) for label in labels]
    print(decoded_preds[:5])
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu_score": result['score']}  # Rename the score as 'bleu_score'

training_args = Seq2SeqTrainingArguments(
    output_dir="AI_Tutor_Training/" + task_name + "_round2",
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=1e-4,
    weight_decay=1e-5,
    save_strategy='steps',
    save_steps=600,
    num_train_epochs=500,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    fp16=True,
    predict_with_generate=True,
    logging_dir="./logs_forAT",     # Path to directory to save logs
    logging_strategy='steps',   # Log after every X steps
    logging_steps=100           # Set X to be 100
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    data_collator=data_collator,
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
# tensorboard dev upload --logdir ./logs_forAT
# --name yjh
# --description yjh

