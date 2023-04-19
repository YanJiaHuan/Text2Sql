import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5Config
from datasets import load_dataset
import json
import nltk
from Evaluation_self import evaluate
import os
# load new table file
with open('./tables_new_picard.json', 'r') as f:
    tables_new = json.load(f)

db_id_to_content = {table['db_id']: table['content'] for table in tables_new}
def tokenize_function(examples, tokenizer, db_id_to_content):
    content = db_id_to_content[examples['db_id'][0]]
    # content = ''
    input_texts = [question+'  '+content for question in examples["question"]]
    # input_texts = [content + " " + question for question in examples["question"]]
    output_texts = examples["query"]
    input_tokenized = tokenizer(input_texts, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    output_tokenized = tokenizer(output_texts, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    return {
        "input_texts": input_texts,
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": output_tokenized["input_ids"]
    }
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./checkpoints/T5-3B/batch2_zero3_epoch30_lr5e5/checkpoint-33000"
    # model_path = "tscholak/cxmefzzi"
    tokenizer_path = "tscholak/cxmefzzi"


    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)


    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    dataset_validation = load_dataset("spider", split='validation').shuffle(seed=42)
    dataset_validation = dataset_validation.select(range(2))
    dataset = dataset_validation.map(lambda e: tokenize_function(e, tokenizer, db_id_to_content), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval",
        per_device_eval_batch_size=1,
        disable_tqdm=False,
        predict_with_generate=True,
    )

    def compute_custom_metric(eval_pred):
        # the sql of gold query
        print(eval_pred.label_ids)
        print(eval_pred.predictions)
        decoded_labels = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        gold = []
        gold.append(decoded_labels)
        gold.append('concert_singer')
        print("Decoded labels:", decoded_labels)
        print("Decoded output:", decoded_preds)
        db_dir = './database'
        etype = 'all'
        kmaps = './tables.json'
        score = evaluate(decoded_labels, decoded_preds,db_dir,etype,kmaps)
        return {"exec": score}


    # os.environ["MASTER_PORT"] = "29501"
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_custom_metric,
    )
    evaluation_result = trainer.evaluate()


if __name__ == "__main__":
    main()

# deepspeed --include localhost:3 evaluate.py
# CUDA_VISIBLE_DEVICES=4  python evaluate.py