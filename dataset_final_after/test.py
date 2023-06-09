import json
import os
import glob
import statistics
from collections import defaultdict

def count_tables_and_columns(json_file):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Count tables and columns
    table_set = set()
    column_count = 0
    for entry in data:
        table_id = entry["column_names"][0][0]
        if table_id != -1:
            table_set.add(table_id)
            column_count += 1

    return len(table_set), column_count

def process_directory(dir_path):
    # Ensure directory exists
    if not os.path.isdir(dir_path):
        print(f"Directory not found: {dir_path}")
        return

    results = {}

    # Loop through all JSON files in the directory
    for json_file in glob.glob(os.path.join(dir_path, '*.json')):
        table_count, column_count = count_tables_and_columns(json_file)
        results[json_file] = {
            "table_count": table_count,
            "column_count": column_count
        }
        print(f"File {json_file} contains {table_count} unique tables and {column_count} columns.")

    # Write results to a new JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def calculate_statistics(results):
    # Get all table counts and column counts
    table_counts = [v["table_count"] for v in results.values()]
    column_counts = [v["column_count"] for v in results.values()]

    # Calculate statistics
    avg_tables = statistics.mean(table_counts)
    max_tables = max(table_counts)
    var_tables = statistics.variance(table_counts) if len(table_counts) > 1 else 0

    avg_columns = statistics.mean(column_counts)
    max_columns = max(column_counts)
    var_columns = statistics.variance(column_counts) if len(column_counts) > 1 else 0

    print(f"Average number of tables: {avg_tables}")
    print(f"Maximum number of tables: {max_tables}")
    print(f"Variance of number of tables: {var_tables}")

    print(f"Average number of columns: {avg_columns}")
    print(f"Maximum number of columns: {max_columns}")
    print(f"Variance of number of columns: {var_columns}")

# Example usage:
# results = process_directory('../dataset_final_after/schemas')
# calculate_statistics(results)
