import os
import json
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Parenthesis
from sqlparse.tokens import Keyword, DML, DDL, Newline, CTE, Wildcard
import pandas as pd

WHERE_OPS = ("NOT", "BETWEEN", "=", ">", "<", ">=", "<=", "!=", "IN", "LIKE", "IS", "EXISTS")
AGG_OPS = ("COUNT", "MAX", "MIN", "SUM", "AVG")
DDL_OPS = ("INSERT", "CREATE", "ALTER", "UPDATE", "DELETE", "DROP")
def get_nestedSQL(parsed):
    nested = []
    for token in parsed.tokens:
        if isinstance(token, Parenthesis):
            nested.append(token)
    return nested

def count_component1(parsed):
    count = 0
    for token in parsed.tokens:
        if token.ttype in Keyword and token.value.upper() in WHERE_OPS:
            count += 1
        elif token.ttype in Keyword and token.value.upper() in ["GROUP BY", "ORDER BY", "LIMIT", "JOIN", "OR", "LIKE", "HAVING"]:
            count += 1
    return count

def count_component2(parsed):
    nested = get_nestedSQL(parsed)
    return len(nested)

def count_others(parsed):
    count = 0
    agg_count = 0
    select_count = 0
    where_count = 0
    groupby_count = 0
    for token in parsed.tokens:
        if token.ttype in Keyword and token.value.upper() in AGG_OPS:
            agg_count += 1
        elif token.ttype in Keyword and token.value.upper() == "SELECT":
            select_count += 1
        elif token.ttype in Keyword and token.value.upper() == "WHERE":
            where_count += 1
        elif token.ttype in Keyword and token.value.upper() == "GROUP BY":
            groupby_count += 1
    if agg_count > 1:
        count += 1
    if select_count > 1:
        count += 1
    if where_count > 1:
        count += 1
    if groupby_count > 1:
        count += 1
    return count


def eval_hardness(sql):
    parsed = sqlparse.parse(sql)[0]
    count_comp1_ = count_component1(parsed)
    count_comp2_ = count_component2(parsed)
    count_others_ = count_others(parsed)

    # Check for DDL operations
    for token in parsed.tokens:
        if token.value.upper() in DDL_OPS:
            # return as 'others' rather than 'add' or 'drop'
            return "others"

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium"
    elif (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard"
    else:
        return "extra"


import random

def process_directory(directory_path):
    difficulties = {"easy": 0, "medium": 0, "hard": 0, "extra": 0, "others": 0}
    file_results = {}
    samples = {"easy": [], "medium": [], "hard": [], "extra": [], "others": []}
    selected_dbs = {"easy": set(), "medium": set(), "hard": set(), "extra": set(), "others": set()}

    filenames = os.listdir(directory_path)
    random.shuffle(filenames)  # shuffle filenames to randomize selection

    for filename in filenames:
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            sql_queries = [(item['sql'], item['text'], item['tables']) for item in data]

            file_difficulties = []
            for query, text, tables in sql_queries:
                hardness = eval_hardness(query)
                difficulties[hardness] += 1
                file_difficulties.append(hardness)

                # Save samples for each category if not yet selected from this database
                if len(samples[hardness]) < 3 and filename not in selected_dbs[hardness]:
                    samples[hardness].append({
                        'database': filename,
                        'sql': query,
                        'text': text,
                        'tables': tables,
                    })
                    selected_dbs[hardness].add(filename)  # mark this database as selected for this hardness

            file_results[filename] = file_difficulties

    return file_results, difficulties, samples

# path_Spider_test = '../dev.json'
# path_Spider_train = '../train_spider.json'
# with open(path_Spider_test, 'r') as f:
#     Spider_test = pd.read_json(f)
# with open(path_Spider_train, 'r') as f:
#     Spider_train = pd.read_json(f)
# Spider = pd.concat([Spider_test, Spider_train], ignore_index=True)
# sql_dev = [sql for sql in Spider['query']]

# difficulties = []
# for query in sql_dev:
#     difficulties.append(eval_hardness(query))
#
# print('Count of difficulties:')
# print(pd.Series(difficulties).value_counts())

path_self = './data'
file_results, total_counts, samples = process_directory(path_self)

# Output the results
print("Total counts:", total_counts)
print("File specific results:", file_results)
print("Samples:")
for hardness, samples_list in samples.items():
    print(f"{hardness} samples:")
    for sample in samples_list:
        print(sample)