import pandas as pd
path_to_CoSQL = "/Users/yan/Desktop/text2sql/cosql_dataset"
DATASET = path_to_CoSQL+"/sql_state_tracking/cosql_dev.json"
OUTPUT_FILE = './gold_sql.txt'
dataset = pd.read_json(DATASET)

gold = []
for index, row in dataset.iterrows():
    dict_round = {}
    dict_round['query'] = row['interaction'][0]['query']
    dict_round['db_id'] = row['database_id']
    gold.append(dict_round)
print(gold)
print(len(gold))

with open(OUTPUT_FILE, 'w') as f:
    for item in gold:
        f.write(f"{item['query']}\t{item['db_id']}\n")