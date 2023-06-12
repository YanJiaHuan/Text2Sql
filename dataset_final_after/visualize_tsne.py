import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import spacy
import os

# Load Spacy's English language model
nlp = spacy.load('en_core_web_md')

# List of words to visualize
file_spider = './Spider_results.json'
file_ours = './results.json'
spider = pd.read_json(file_spider)
ours_schema = './schemas'

def get_table_names(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            schema = data['schema']
    table_names = []
    for table in schema:
        table_names.append(table['table_name'])
    return table_names

words_spider = spider['db_id'].tolist()
print(words_spider)
# Get the word vectors
def post_process(words,output):
    word_vectors = [nlp(word).vector for word in words]

    # Convert list of word vectors into a numpy array
    word_vectors = np.array(word_vectors)

    # Initialize and fit t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    # Create a DataFrame
    df = pd.DataFrame(word_vectors_2d, columns=['x', 'y'])
    df['word'] = words

    # Save the DataFrame to a CSV file
    df.to_csv(output, index=False)

post_process(words_spider, './spider_vector.csv')