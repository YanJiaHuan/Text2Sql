import json

with open('./database_stats.json', 'r') as f:
    data_dict = json.load(f)

output = ''
for category, stats in data_dict.items():
    # Replace special characters
    category = category.replace("_", "\\_")

    line = "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
        category.capitalize(),
        round(stats['avg_tokens']),
        stats['num_tables'],
        stats['num_columns'],
        stats['sql_count'],
        stats['condition_count']['orderBy'],
        stats['condition_count']['groupBy'],
        stats['condition_count']['having'],
        stats['condition_count']['nested'],
        stats['condition_count']['join']
    )
    output += line + '\n'

print(output)
