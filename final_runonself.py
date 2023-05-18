import glob
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer,T5Config, Seq2SeqTrainingArguments,Seq2SeqTrainer,TrainerCallback,Trainer,TrainingArguments
from datasets.arrow_dataset import Dataset
import nltk
import torch
import os
from Evaluation_self import evaluate,evaluate_test

from transformers import Trainer
with open('./tables_new_picard.json', 'r') as f:
    tables_new = json.load(f)

db_id_to_content = {table['db_id']: table['content'] for table in tables_new}
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

def load_data():
    # path
    data_path = "./dataset_final/data"
    schema_path = "./dataset_final/schemas"
    data_files = glob.glob(f'{data_path}/*.json')
    schema_files = glob.glob(f'{schema_path}/*.json')

    data =  []
    schemas = {}

    # Load data
    for file in data_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            # Here we use the filename as the db_id
            db_id = os.path.basename(file).replace('.json', '')
            for entry in file_data:
                entry['db_id'] = db_id
                data.append(entry)

    # Load schemas
    for file in schema_files:
        with open(file, 'r') as f:
            file_schema = json.load(f)
            db_id = os.path.basename(file).replace('.json', '')
            schemas[db_id] = []
            # Concatenate table schemas
            for table_name, table_schema in file_schema.items():
                table_schema_str = ', '.join([f"{col_name}: {col_type}" for col_name, col_type in table_schema.items()])
                schemas[db_id].append(f"{table_name} | {table_schema_str}")
            schemas[db_id] = ' || '.join(schemas[db_id])
    return data, schemas


def preprocess_function(example, tokenizer, db_id_to_content):
    contents = [db_id_to_content[db_id] for db_id in example['db_id']]
    questions = [question + ' ' + content for question, content in zip(example['question'], contents)]
    queries = example['query']
    input_tokenized = tokenizer(questions, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    output_tokenized = tokenizer(queries, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    return {
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": output_tokenized["input_ids"],
        "db_id": example["db_id"],
        "gold_query": example["query"]
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 't5-3b'
    tokenizer_name = 't5-3b'
    # model_name = "./checkpoints/T5-3B/batch2_zero3_epoch50_lr1e4_seq2seq/checkpoint-8000"
    # tokenizer_name = "tscholak/cxmefzzi"
    # config = T5Config.from_pretrained(model_name, ignore_pad_token_for_loss=True)
    config = T5Config.from_pretrained(model_name)
    config.max_length = 512
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config).to(device)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name,model_max_length=512)

       # Load data
    train_data, schemas = load_data()

    with open('./spider_local/dev.json', 'r') as f:
        eval_data = json.load(f)

    db_id_train = [entry["db_id"] for entry in train_data]
    query_train = [entry["query"] for entry in train_data]
    question_train = [entry["question"] for entry in train_data]


    dataset_train = Dataset.from_dict({
        "db_id": db_id_train,
        "query": query_train,
        "question": question_train,
    })
    db_id_eval = [entry["db_id"] for entry in eval_data]
    query_eval = [entry["query"] for entry in eval_data]
    question_eval = [entry["question"] for entry in eval_data]

    dataset_eval = Dataset.from_dict({
        "db_id": db_id_eval,
        "query": query_eval,
        "question": question_eval,
    })


    # Shuffle and select a subset of the data, if needed
    dataset_train = dataset_train.shuffle(seed=42)
    dataset_eval = dataset_eval

    # Preprocess the data
    dataset = dataset_train.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)
    eval_dataset = dataset_eval.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)

    # dataset = load_dataset("spider", split='train').shuffle(seed=42).select(range(20))
    # dataset = dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)
    # eval_dataset = load_dataset("spider", split='validation').shuffle(seed=42).select(range(20))
    # eval_dataset = eval_dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)


    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/T5-3B/batch2_zero3_epoch50_lr1e4_seq2seq_2",
        deepspeed="./deepspeed_config.json",
        num_train_epochs=50,
        learning_rate=8e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1, # dafault is 1
        max_grad_norm=1.0,
        evaluation_strategy="steps",  # Change evaluation_strategy to "steps"
        eval_steps=1000,
        save_steps=2000,# Add eval_steps parameter need to lower the log/eval/save steps to see the report results
        save_strategy="steps",
        disable_tqdm=False,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=4,
        include_inputs_for_metrics=True,
        # save_total_limit=1,  # Only save the best model
    )
    import re
    def compute_metric(eval_pred):
        preds=eval_pred.predictions
        labels=eval_pred.label_ids
        db_ids=eval_pred.inputs
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(db_ids, skip_special_tokens=True)
        db_id = []
        for question in decoded_inputs:
            result = re.search(r'\|(.+?)\|', question)
            db_id.append(result.group(1).strip())
        genetrated_queries = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]###########
        gold_queries_and_db_ids = []
        with open("./Evaluation_file/gold_example_1e4__checkpoint-10000.txt", 'r') as file:
            for line in file:
                # Split the line by the tab character '\t'
                query, db_id = line.strip().split('\t')

                # Append the query and db_id as a tuple to the list
                gold_queries_and_db_ids.append((query, db_id))
        db_dir = './database'
        etype = 'all'
        table = './tables.json'
        # print("now you see")
        score = evaluate(gold_queries_and_db_ids, genetrated_queries, db_dir, etype, table)
        print(f"Execution Accuracy: {score}")
        return {"exec":score}  # 必须返回字典

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric,
        # callbacks=[EvalCallback(model,tokenizer,eval_dataset)],
    )
    trainer.train()

if __name__ == "__main__":
    main()

## deepspeed --include localhost:0,1,2,3 finaltest_trainer_eval.py | use this code to choose GPUs to run
## Try to remove /.cache/pytorch_extensions if stuck somewhere
## add activation_checkpointing in ds_config if oom(batch = 16,without: 20GB/GPU, with: 26GB/GPU
## need to run the script provided by deepspeed to convert the model to normal torch model
## python zero_to_fp32.py --model_file /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 --output /home/jiahuan/test/checkpoints/T5-3B

## python zero_to_fp32.py /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 /home/jiahuan/test/checkpoints/T5-3B

