import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments,T5Config
from datasets import load_dataset

with open('./tables_new_2.json', 'r') as f:
    tables_new = json.load(f)

db_id_to_content = {table['db_id']: table['content'] for table in tables_new}

# def preprocess_function(example, tokenizer, db_id_to_content):
#     content = db_id_to_content[example['db_id']]
#     question = example['question'] +' '+ content
#     query = example['query']
#
#     input_tokenized = tokenizer(question, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
#     output_tokenized = tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
#
#     return {
#         "input_ids": input_tokenized["input_ids"],
#         "attention_mask": input_tokenized["attention_mask"],
#         "labels": output_tokenized["input_ids"]
#     }
def preprocess_function(example, tokenizer, db_id_to_content):
    contents = [db_id_to_content[db_id] for db_id in example['db_id']]
    questions = [question + ' ' + content for question, content in zip(example['question'], contents)]
    queries = example['query']

    input_tokenized = tokenizer(questions, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    output_tokenized = tokenizer(queries, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    return {
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": output_tokenized["input_ids"]
    }


def main():
    model_name = "t5-3B"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    dataset = load_dataset("spider", split='train').shuffle(seed=42)
    dataset = dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)
    eval_dataset = load_dataset("spider", split='validation').shuffle(seed=42)
    eval_dataset = eval_dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)

    training_args = TrainingArguments(
        output_dir="checkpoints/T5-3B/batch2_zero3_epoch8_lr5e5_tokes",
        num_train_epochs=8,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        logging_steps=20,
        save_total_limit=1,  # Only save the best model
    )
    config = T5Config.from_pretrained(model_name, ignore_pad_token_for_loss=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()


## deepspeed --num_gpus=4 test_multiGPU_trainer.py |this code will always use the first 4 GPUs
## deepspeed --include localhost:1,2 test_multiGPU_trainer.py | use this code to choose GPUs to run
## Try to remove /.cache/pytorch_extensions if stuck somewhere
## add activation_checkpointing in ds_config if oom(batch = 16,without: 20GB/GPU, with: 26GB/GPU
## need to run the script provided by deepspeed to convert the model to normal torch model
## python zero_to_fp32.py --model_file /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 --output /home/jiahuan/test/checkpoints/T5-3B

## python zero_to_fp32.py /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 /home/jiahuan/test/checkpoints/T5-3B

