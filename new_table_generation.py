import json

# Read input JSON file
with open('tables.json', 'r') as infile:
    input_json = json.load(infile)

output_json = []

for db in input_json:
    db_id = db['db_id']
    tables_dict = {}

    for column_info in db['column_names_original']:
        table_index = column_info[0]
        column_name = column_info[1]
        table_name = db['table_names_original'][table_index]

        if column_name == "*":
            continue

        if table_name not in tables_dict:
            tables_dict[table_name] = []

        tables_dict[table_name].append(column_name)

    content = []
    for table_name, columns in tables_dict.items():
        content.append(f"|| {table_name} | {', '.join(columns)}")
    content_str = f"||| {db_id} " + ' '.join(content)
    output_json.append({
        'db_id': db_id,
        'content': content_str
    })

# Write output JSON file
with open('./tables_new_picard.json', 'w') as outfile:
    json.dump(output_json, outfile, indent=2)
