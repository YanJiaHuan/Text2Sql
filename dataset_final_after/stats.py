import os
import json
import sqlite3
import argparse

from process_sql import Schema, get_sql, get_schema_from_json,get_schema
from collections import defaultdict
from glob import glob

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def condition_has_or(conds):
    return 'or' in conds[1::2]

def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def val_has_op(val_unit):
    return val_unit[0] != UNIT_OPS.index('none')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    return 0


def recall(count, total):
    if count == total:
        return 1
    return 0


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += Evaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count



def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True




import tiktoken
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

import re

def condition_count_meta(sql):
    record = {}
    record["orderBy"] = len(re.findall(r"(?i)\bORDER BY\b", sql))
    record["groupBy"] = len(re.findall(r"(?i)\bGROUP BY\b", sql))
    record["having"] = len(re.findall(r"(?i)\bHAVING\b", sql))
    record["nested"] = len(re.findall(r"\([^)]*\bSELECT\b", sql))
    record["join"] = len(re.findall(r"(?i)\bJOIN\b", sql))
    return record
def condition_count(sql):
    record = {}
    record["orderBy"] = 1 if re.search(r"(?i)\bORDER BY\b", sql) else 0
    record["groupBy"] = 1 if re.search(r"(?i)\bGROUP BY\b", sql) else 0
    record["having"] = 1 if re.search(r"(?i)\bHAVING\b", sql) else 0
    record["nested"] = 1 if re.search(r"\([^)]*\bSELECT\b", sql) else 0
    record["join"] = 1 if re.search(r"(?i)\bJOIN\b", sql) else 0
    return record

import random
def Count_ours():
    stats = defaultdict(int)
    examples = defaultdict(list)  # add a defaultdict of lists to store examples
    total_counts = 0
    total_tokens = 0

    # Aggregation statistics
    agg_stats = defaultdict(int)

    # New dictionary to hold per-database statistics
    database_stats = defaultdict(
        lambda: {'tokens': 0, 'count': 0, 'condition_count': defaultdict(int), 'num_tables': 0, 'num_columns': 0})

    for f in glob("schemas/*.schema.json"):
        topic = f.split("/")[-1].split(".")[0]
        sql_count = 0
        with open(f, 'r') as schema_file, open("data/%s.json" % topic) as data_file:
            schema = json.load(schema_file)
            data = json.load(data_file)
            sql_count = len(data)
            failed = 0
            database_stats[topic]['sql_count'] = sql_count
            # Count tables and columns
            database_stats[topic]['num_tables'] = len(schema)
            database_stats[topic]['num_columns'] = sum(len(cols) for cols in schema.values())
            schema = Schema(schema)
            for entry in data:
                sql = entry["sql"]
                record = condition_count(sql)  # record
                num_tokens = num_tokens_from_string(sql)
                total_tokens += num_tokens
                total_counts += 1

                # Update the stats for this database
                database_stats[topic]['tokens'] += num_tokens
                database_stats[topic]['count'] += 1
                for k, v in record.items():
                    database_stats[topic]['condition_count'][k] += v
                    agg_stats[k] += v

                try:
                    g_sql = get_sql(schema, sql)
                    hardness = eval_hardness(g_sql)
                    stats[hardness] += 1

                    # add the query to the list of examples for its hardness
                    examples[hardness].append((topic, schema.tables, sql))

                except:
                    failed += 1
                    stats["failed"] += 1

    if total_counts > 0:
        print('Total average tokens:', total_tokens / total_counts)

    # Print aggregation statistics
    print(total_counts)
    print('condition_count:', agg_stats)
    print('Database stats:', {db: {'avg_tokens': stats['tokens'] / stats['count'], 'num_tables': stats['num_tables'], 'num_columns': stats['num_columns'], 'sql_count': stats['sql_count'],'condition_count': dict(stats['condition_count'])} for db, stats in database_stats.items()})
    print('Hardness stats:', dict(stats))
    database_stats_for_json = {
        db: {
            'avg_tokens': stats['tokens'] / stats['count'],
            'num_tables': stats['num_tables'],
            'num_columns': stats['num_columns'],
            'sql_count': stats['sql_count'],
            'condition_count': dict(stats['condition_count'])
        } for db, stats in database_stats.items()
    }

    with open('./database_stats.json', 'w') as f:
        json.dump(database_stats_for_json, f)

    for hardness in ['easy', 'medium', 'hard', 'extra hard']:
        print(f"{hardness} examples:")

        # Get the list of examples for this hardness
        hardness_examples = examples[hardness]

        # Use random.sample() to select up to 3 examples
        # random.sample() will raise a ValueError if there are fewer than 3 examples,
        # so catch that exception and just use all examples in that case
        try:
            selected_examples = random.sample(hardness_examples, 3)
        except ValueError:
            selected_examples = hardness_examples

        for example in selected_examples:
            topic, tables, sql = example
            print(f"Topic: {topic}")
            print(f"Tables: {', '.join(tables)}")
            print(f"SQL: {sql}")
            print()

if __name__ == "__main__":
    Count_ours()
    # test = "SELECT Customers.CustomerName, SUM(Sales.Amount) AS TotalPurchases FROM Customers JOIN Sales ON Customers.CustomerID = Sales.CustomerID GROUP BY Customers.CustomerName ORDER BY TotalPurchases DESC;"
    # test2 = "SELECT count(*) ,  T1.year FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_winner  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' GROUP BY T1.year"
    # db2 = "/Users/yan/Desktop/text2sql/spider/database/baseball_1/baseball_1.sqlite"
    # db = "./schemas/accounting.schema.json"
    # # schema = Schema(json.load(open(db)))
    # # g_sql = get_sql(schema, test)
    # # print(g_sql)
    # schema = Schema(get_schema(db2))
    # g_sql = get_sql(schema, test2)
    # print(g_sql)
    ################ Spider ################
    stats = defaultdict(int)
    agg_stats = defaultdict(int)
    length_counts = defaultdict(int)  # New dictionary to keep track of cumulative SQL lengths per database
    num_queries = defaultdict(int)  # New dictionary to keep track of number of queries per database
    total_counts = 0
    total_length = 0  # Variable to keep track of total SQL length across all databases
    failed = 0
    db_path = '/Users/yan/Desktop/text2sql/spider/database'
    gold_path = '../multi_turn/Bard_GPT/V0/gold.txt'
    with open(gold_path) as f:
        gold = f.readlines()
    for index, sample in enumerate(gold):
        total_counts += 1
        sql, db_id = sample.rsplit("\t", 1)
        db_id = db_id.strip()  # remove potential trailing whitespace
        db = f"{db_path}/{db_id}/{db_id}.sqlite"
        schema = Schema(get_schema(db))
        record = condition_count(sql)  # record
        num_tokens = num_tokens_from_string(sql)

        for k, v in record.items():
            agg_stats[k] += v
        try:
            g_sql = get_sql(schema, sql)
            hardness = eval_hardness(g_sql)
            stats[hardness] += 1
        except:
            failed += 1
            stats["failed"] += 1

        # Add length of current SQL to the cumulative length for the current database
        length_counts[db_id] += num_tokens
        # Increase the number of queries for the current database
        num_queries[db_id] += 1
        # Add length of current SQL to the total length
        total_length += num_tokens

    print(stats)
    print(total_counts)
    print(failed)
    print(agg_stats)

    # Compute average length per database and print it
    avg_length_per_db = {db: length_counts[db] / num_queries[db] for db in length_counts}
    print('Average length per database:', avg_length_per_db)

    # Compute total average length and print it
    total_avg_length = total_length / total_counts if total_counts > 0 else 0
    print('Total average length:', total_avg_length)


