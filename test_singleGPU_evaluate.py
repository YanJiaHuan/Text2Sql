import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Replace with the path to your model
    model_path = "./checkpoints/T5-3B/checkpoint-657"

    tokenizer = T5Tokenizer.from_pretrained('t5-3B')
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    # Load the dataset
    dataset_validation = load_dataset("spider", split='validation')
    dataset = dataset_validation.shuffle(seed=42)
    print("..................")
    # Tokenize the input and output
    def tokenize_function(examples):
        input_texts = ["translate English to SQL: " + question for question in examples["question"]]
        output_texts = examples["query"]
        input_tokenized = tokenizer(input_texts, return_tensors="pt", max_length=128, truncation=True,
                                    padding="max_length")
        output_tokenized = tokenizer(output_texts, return_tensors="pt", max_length=128, truncation=True,
                                     padding="max_length")
        return {"input_ids": input_tokenized["input_ids"], "attention_mask": input_tokenized["attention_mask"],
                "labels": output_tokenized["input_ids"], "labels_attention_mask": output_tokenized["attention_mask"]}

    dataset = dataset.map(tokenize_function, batched=True)

    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch, tokenizer):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        labels_attention_mask = [torch.tensor(item['labels_attention_mask']) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_attention_mask = pad_sequence(labels_attention_mask, batch_first=True,
                                             padding_value=tokenizer.pad_token_id)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
                'labels_attention_mask': labels_attention_mask}


    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            total_loss += loss.item()
            count += 1

            avg_loss = total_loss / count
            print(f"Average loss: {avg_loss}")

if __name__ == "__main__":
    main()


# Average loss: 0.30857391896472103 # epoch-3