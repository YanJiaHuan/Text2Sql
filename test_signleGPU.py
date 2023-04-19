############################################################################################################
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler
from transformers import T5Config,T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers.file_utils import is_torch_tpu_available
from transformers.optimization import AdamW
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from datasets import load_dataset
from tqdm.auto import tqdm
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the model
    model = T5ForConditionalGeneration.from_pretrained("t5-3B").to(device)
    # Set up the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-3B")

    # Set up the dataset
    dataset= load_dataset("spider", split='train')
    # dataset_validation = load_dataset("spider", split='validation')
    dataset = dataset.shuffle(seed=42)
    # dataset_validation = dataset_validation.shuffle(seed=42)

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

    #Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # use SGD if out of memory
    # import torch.optim as optim
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    scaler = GradScaler()

    num_epochs = 3
    num_batches_per_epoch = len(dataloader)
    num_training_steps = num_epochs * num_batches_per_epoch
    progress_bar = tqdm(range(num_training_steps))

    # Train the model
    model.train()
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    print('start training....')

    for epoch in range(num_epochs):  # Number of epochs
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss  # T5 model handles the loss internally
            epoch_loss += loss.item()  # Accumulate the loss
            # use half precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.update(1)

            # Print the loss for this batch
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

        # Save checkpoint at the end of each epoch
        checkpoint_dir = os.path.join(output_dir, f"epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        progress_bar.reset()


    progress_bar.close()


if __name__ == "__main__":
    main()


# use this to tun: python -m torch.distributed.launch --nproc_per_node=4 test4.py
# or this to run: torchrun --nproc_per_node=6 test4.py