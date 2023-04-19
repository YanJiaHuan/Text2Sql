import json
from transformers import T5ForConditionalGeneration, T5Tokenizer,T5Config, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import os
import torch
import nltk
from Evaluation_self import evaluate

with open('./tables_new_picard.json', 'r') as f:
    tables_new = json.load(f)

db_id_to_content = {table['db_id']: table['content'] for table in tables_new}

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
    # model_name = 't5-3b'
    model_name = "./checkpoints/T5-3B/batch2_zero3_epoch30_lr5e5/checkpoint-33000"
    tokenizer_name = "tscholak/cxmefzzi"
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name,model_max_length=512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset("spider", split='train').shuffle(seed=42)
    dataset = dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)
    eval_dataset = load_dataset("spider", split='validation').shuffle(seed=42).select(range(500))
    eval_dataset = eval_dataset.map(lambda e: preprocess_function(e, tokenizer, db_id_to_content), batched=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/T5-3B/batch2_zero3_epoch30_lr1e4_seq2seq",
        deepspeed="./deepspeed_config.json",
        num_train_epochs=50,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        evaluation_strategy="steps",  # Change evaluation_strategy to "steps"
        eval_steps=10,
        save_steps=10000,# Add eval_steps parameter
        save_strategy="steps",
        disable_tqdm=False,
        load_best_model_at_end=True,
        predict_with_generate=True,
        # save_total_limit=1,  # Only save the best model
    )

    def compute_custom_metric(eval_pred):
        print('start computing metric...')
        # decoded_preds = tokenizer.decode(eval_pred.predictions, skip_special_tokens=True)
        decoded_preds = [''.join(tokenizer.decode(pred, skip_special_tokens=True).split()) for pred in
                         eval_pred.predictions]

        print(decoded_preds)
        decoded_labels = eval_dataset[:]['query']

        eval_dataset.set_format(type='torch', columns=['db_id'])
        db_ids = eval_dataset[:]['db_id']

        gold_queries_and_db_ids = list(zip(decoded_labels, db_ids))
        print(gold_queries_and_db_ids)
        db_dir = './database'
        etype = 'all'
        table = './tables.json'
        score = evaluate(gold_queries_and_db_ids, decoded_preds, db_dir, etype, table)
        return {"exec": score}

    config = T5Config.from_pretrained(model_name, ignore_pad_token_for_loss=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config).to(device)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_custom_metric,
        generation_kwargs={"max_length": 512},
    )

    trainer.train()
    # trainer.evaluate()
    # save tokenizer
    tokenizer_output_dir = training_args.output_dir + "/tokenizer"
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_output_dir)
if __name__ == "__main__":
    main()

## deepspeed --include localhost:0,1,2,3 finaltest_trainer_eval.py | use this code to choose GPUs to run
## Try to remove /.cache/pytorch_extensions if stuck somewhere
## add activation_checkpointing in ds_config if oom(batch = 16,without: 20GB/GPU, with: 26GB/GPU
## need to run the script provided by deepspeed to convert the model to normal torch model
## python zero_to_fp32.py --model_file /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 --output /home/jiahuan/test/checkpoints/T5-3B

## python zero_to_fp32.py /home/jiahuan/test/checkpoints/T5-3B/checkpoint-657 /home/jiahuan/test/checkpoints/T5-3B

