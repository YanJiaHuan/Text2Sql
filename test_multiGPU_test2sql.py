from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

input_question = 'How many singers do we have?'
input_query = "SELECT count(*) FROM singer"


from transformers import T5Tokenizer

# Load the T5-3B tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-3B")

# Tokenize the input question and query
tokenized_question = tokenizer.tokenize(input_question)
tokenized_query = tokenizer.tokenize(input_query)

print("Tokenized question:", tokenized_question)
print("Tokenized query:", tokenized_query)
