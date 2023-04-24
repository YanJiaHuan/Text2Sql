import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

with open('./tables_new_picard.json', 'r') as f:
    tables_new = json.load(f)

db_id_to_content = {table['db_id']: table['content'] for table in tables_new}

# beam search
def generate_query(model, tokenizer, input_text, device, max_length=512, num_beams=4, num_return_sequences=1):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, padding="max_length",
                       truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                 max_length=512, num_beams=num_beams, num_return_sequences=num_return_sequences,
                                 repetition_penalty=2.0, length_penalty=0.5, early_stopping=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "./checkpoints/T5-3B/batch2_zero3_epoch50_lr1e4_seq2seq/"
    # Replace with the path to your model
    checkpoint = "checkpoint-50000"
    model_path = path+ checkpoint
    # tokenier_path = "./checkpoints/T5-3B/batch2_zero3_epoch4_lr5e5/tokenizer-7000"
    tokenier_path = "tscholak/cxmefzzi"
    # model_path = "tscholak/cxmefzzi"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenier_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # Load questions from JSON file
    with open("dev.json", "r") as f:
        data = json.load(f)

    input_texts = []
    gold_queries = []

    for item in data:
        question = item["question"]
        db_id = item["db_id"]
        content = db_id_to_content[db_id]
        input_texts.append(question + ' ' + content)
        gold_queries.append(item["query"])
    lr_mark = "1e4_"
    pred_file = f"./Evaluation_file/pred_example_{lr_mark}_{checkpoint}.txt"
    gold_file = f"./Evaluation_file/gold_example_{lr_mark}_{checkpoint}.txt"
    with open(pred_file, "w") as pred_f, open(gold_file, "w") as gold_f:
        for i, input_text in enumerate(input_texts):
            gold_query = gold_queries[i]

            generated_sql = generate_query(model, tokenizer, [input_text], device)

            print(f"Generated SQL: {generated_sql}")
            print(f"Gold SQL: {gold_query}")
            print('\n')
            pred_f.write(f"{generated_sql}\n")
            gold_f.write(f"{gold_query}\t{data[i]['db_id']}\n")


if __name__ == "__main__":
    main()




# CUDA_VISIBLE_DEVICES=3 python test_singleGPU_text2sql.py