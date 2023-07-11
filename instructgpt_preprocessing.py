import pandas as pd
from datasets import load_dataset

data = load_dataset('json', data_files='train_1890.json')

# Convert the 'train' split of the data to a Pandas dataframe
df = pd.DataFrame(data['train'])

# List of instructions
instruction_list = [
    'Mark the title of the given paper.',
    "List all the authors' names.",
    'Mark the keywords of this paper and give their definitions.',
    'Summarize the given introduction to generate the research background of this paper.',
    'List all the research methodologies proposed by this paper and summarize their details.',
    "Give a conclusion about this paper's major achievements and breakthroughs."
]

# Create a mapping from instruction to short names for the files
short_names = {
    'Mark the title of the given paper.': 'title',
    "List all the authors' names.": 'authors',
    'Mark the keywords of this paper and give their definitions.': 'keywords',
    'Summarize the given introduction to generate the research background of this paper.': 'background',
    'List all the research methodologies proposed by this paper and summarize their details.': 'methodologies',
    "Give a conclusion about this paper's major achievements and breakthroughs.": 'conclusions'
}
dataset_path = 'train_1890/'
# Loop over the instructions and save data for each instruction into a separate JSON file
for instruction in instruction_list:
    df_filtered = df[df['instruction'] == instruction]
    df_filtered.to_json(f'{dataset_path+short_names[instruction]}_train.json', orient='records', lines=True)

# python instructgpt_preprocessing.py