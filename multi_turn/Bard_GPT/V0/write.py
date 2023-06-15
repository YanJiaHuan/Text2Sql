import json
import pandas as pd

path_to_Spider = "/Users/yan/Desktop/text2sql/spider"

file_1 = "/train_spider.json"
file_2 = "/dev.json"
file_3 = "/train_others.json"
with open(path_to_Spider + file_1, 'r') as f:
    Spider_train = pd.read_json(f)
with open(path_to_Spider + file_2, 'r') as f:
    Spider_dev = pd.read_json(f)
with open(path_to_Spider + file_3, 'r') as f:
    Spider_others = pd.read_json(f)

Spider = pd.concat([Spider_train , Spider_dev, Spider_others], ignore_index=True)

with open('./gold.txt', 'w') as f:
    for index, row in Spider.iterrows():
        f.write("%s\t%s\n" % (row['query'], row['db_id']))

# SQL \t db_id