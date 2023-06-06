import pandas as pd
import time
import re
import openai
import os
from os import environ
import sys
import tiktoken
import sqlite3
from Bard import Chatbot

'''
This is the first version of GPT-Bard contrasive prompting model, in this demo, we still focus on Spider, 
but the difference will be: The few-shots are generated by Bard.
Version 1: 
    1. The few-shot is generated by Bard.
    2. Bard will generate SQL/question samples based on the real input SQL/question.
    3. GPT know the few-shots are generated by Bard.
    4. Bard know it is Spider task.
'''



####################  0. Prompt   ####################
SQL_generation_prompt = '''
You are an expert in SQL. I will give you a natural language question and a database schema, 
please help me generate the corresponding SQL query with no further explaination.
'''
few_shot_generation_prompt_Bard = '''
You are an expert in SQL. I will give you a database schema in Spider dataset, and you need to generate three
SQL queries with natural language questions based on the schema.
'''

three_shots_SQL_generation_prompt = '''
Here is some examples of EASY, MEDIUM and HARD SQL queries.
SELECT count(*) FROM singer 
SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype
SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')
'''

zero_shots_SQL_generation_prompt = '''
Sorry, I won't give you any examples. Please generate based on your own semantic parsing ability.
'''

three_shots_SQL_generation_prompt_from_Bard = '''
I will give you some expamples of how SQL is generated, please follow the instructions and generate your own answer(SQL).
'''

three_shot_Spider_prompt_without_explain = '''
Here is a sample of text2sql for you to understand the task.
Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]

foreign key:[course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
primary key:[classroom.building,department.dept_name,course.course_id,instructor.ID,section.course_id,teaches.ID,student.ID,takes.ID,advisor.s_ID,time_slot.time_slot_id,prereq.course_id]

example 1:
Question: Find out the average salary of professors?
SELECT avg ( salary )  FROM instructor

example 2:
Question: Find the average salary of the professors of each department?
SELECT avg ( salary ) , dept_name FROM instructor GROUP BY dept_name

example 3:
Question: Which department has the highest average salary of professors?
SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )  DESC LIMIT 1
'''

checker_prompt = '''
Please help me generate the corresponding SQL query with no further explaination.
'''

#################### 1. Set up  ####################
#----------------------------------------------------------------------------------------------------------

# API_KEY = "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" # 自己的
API_KEY = "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"  # 买的
# API_KEY = "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" # gpt4 孙哥
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

#changed
task = 'Spider' # 1 for CoSQL, 2 for Spider
if task == 'CoSQL':
    path_to_CoSQL = "./cosql_dataset"
    DATASET_SCHEMA = path_to_CoSQL+"/tables.json"
    DATASET = path_to_CoSQL+"/sql_state_tracking/cosql_dev.json"
    OUTPUT_FILE_1 = "./predicted_sql.txt"
    OUTPUT_FILE_2 = "./gold_sql.txt"
    DATABASE_PATH = path_to_CoSQL+"/database"
else:
    path_to_Spider = "/Users/yan/Desktop/text2sql/spider"
    DATASET_SCHEMA = path_to_Spider + "/tables.json"
    DATASET = path_to_Spider + "/dev.json"
    OUTPUT_FILE_1 = "./Spider/predicted_sql.txt"
    OUTPUT_FILE_2 = "./Spider/gold_sql.txt"
    DATABASE_PATH = path_to_Spider + "/database"


# set max tokens limit
MAX_TOKENS = 4096
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
encoding = tiktoken.encoding_for_model(model_name)
# count the token
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# load dataset
def load_data(DATASET):
    return pd.read_json(DATASET)


def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output
def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output
def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output
def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def SQL_checker(sql, database):
    # sql be like: "SELECT * FROM car_1 WHERE car_1.id = 1"
    # database is the path to local xxx.sqlite
    # the function of this part is to check if the sql is valid, if not, return the error message
    path = DATABASE_PATH + '/' + database + '/' + database + '.sqlite'
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(path)
        # Create a cursor object to execute the SQL query
        cursor = conn.cursor()
        # Execute the SQL query
        cursor.execute(sql)
        # Commit the transaction and close the connection
        conn.commit()
        conn.close()
        # Return a success message if the SQL query is valid
        prompt =  "The SQL query is valid in grammar."
        checker = False
    except sqlite3.Error as e:
        # Return the error message if the SQL query is not valid
        instruction = f"""#### the sql generated by you: {sql}, has error like :{e} , please fix the error and generate again. \n"""
        fields = find_fields_MYSQL_like(database)
        fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
        fields += "Primary_keys = " + find_primary_keys_MYSQL_like(database)
        prompt = instruction + fields + checker_prompt
        checker = True
    return prompt, checker

import time

def GPT4_generation(prompt):
    '''
    openai.error.RateLimitError: Rate limit reached for default-gpt-3.5-turbo
    in organization org-GFmlumrCZBB2Y40fVv7f8qgp on requests per min. Limit: 3 / min.
    Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues.
    Please add a payment method to your account to increase your rate limit.
    Visit https://platform.openai.com/account/billing to add a payment method.
    '''
    limit_marker = False
    fake_SQL = "SELECT COUNT(*) FROM singer"
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                n = 1,
                stream = False,
                temperature=0.0,
                max_tokens=600,
                top_p = 1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response['choices'][0]['message']['content'], limit_marker

        except openai.error.RateLimitError as e:
            print(f"RateLimitError: {e}")
            print("Sleeping for 20 seconds...")
            time.sleep(20)
            print("Retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return fake_SQL, limit_marker


# initial the chatbot


def extract_sql(response):
    matches = re.findall(r'```sql\n(.*?)\n```', response, re.DOTALL)
    return matches


tokens=(
    "WwiJN0oLURBx7gX_O8WVz9Fufj1iefdzkpt2fsbsb-e8al2Kvufapnj5mYa6vGo5P1ub9w.",
    "WwhXnsbFLxozhOKG1-NUO78iif9IiN5El3Qk9yk5fi70TMcaUMOwfWwjTyqAyNe6MCtiEA.",
    "Wwi1wxVyz-X2piJk8Ts84d08Fm1UmHDTOS7ftlD6LCXdbUVjFrQlJfl97an8UHhZQM8juQ.",
    "Wwj6xMcUvzQUaKwcRQ-qvwrIcZLDBRp9XP25HkEVBAJDVZBzujepzI_dttehdJiCAjCIMg.",
    "WwjMZ_TL9xIl4jREPppT5df6tAsjLLgjRo_GKK5iLslGOh5lMtstOMP_iJEADXq6gjFEKA.",
    "Wgj-oa5yHxfmjo0lLybtWGLiWYoKTZ07NXcUiaPiUHmtQQiAKlfzNTOA9lwqmCz2N0qGFg."
)


def Bard_generation(prompt):
    limit_marker = False
    token_index = 0
    chatbot = Chatbot(tokens[token_index])
    answer = chatbot.ask(prompt)
    print('whole answer', answer)

    while True:  # This loop will continue until a string is returned
        if isinstance(answer, dict):  # check if answer is a dictionary (error response)
            limit_marker = True
            print("Token limit reached, switching to a new token...")
            token_index += 1  # Move to the next token
            if token_index >= len(tokens):  # If we've used all tokens, start over
                token_index = 0
                print("exceeding total limit, Waiting 15 seconds...")
                time.sleep(15)  # freeze for 15s
            chatbot = Chatbot(tokens[token_index])  # Create a new chatbot with the new token
            answer = chatbot.ask(prompt)  # resend the request
        else:
            return answer[0][0], limit_marker
def save_breaker(breaker):
    with open("breaker.txt", "w") as f:
        f.write(str(breaker))

# Function to load the breaker value from a file
def load_breaker():
    if os.path.exists("breaker.txt"):
        with open("breaker.txt", "r") as f:
            breaker =  int(f.read())
            if breaker > 1037:
                breaker = 0
            else:
                breaker = breaker
            return breaker
    return 0




if __name__ == '__main__':
###########################################################################################
    # load the data
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    SQLs_temp_pred = []
    SQLs_temp_gold = []
    for index,sample in val_df.iterrows():
        print('index:',index)
        db_id = sample['db_id'] # e.g.'car_1'
        question = sample['question'] # e.g.'How many car models are produced by each maker? List the count and the maker full name.'
        SQL_gold = sample['query'] # e.g.'SELECT COUNT(*) FROM car_1 WHERE car_1.id = 1'
        print('SQL_gold:',SQL_gold)
        schema = find_fields_MYSQL_like(db_id) + '\n' + "foreign key:" + find_foreign_keys_MYSQL_like(
            db_id) + '\n' + "primary key:" + find_primary_keys_MYSQL_like(db_id)  #
        ###############################################
        '''message to Bard, to get few-shots'''
        message_Bard = few_shot_generation_prompt_Bard + \
                          "\ndatabase:" + db_id + \
                            "\ndatabase chema:" + schema
        print('message to Bard:', message_Bard)
        response_Bard, _ = Bard_generation(message_Bard)
        print('response_Bard:', response_Bard)
        ###############################################
        '''message to GPT, to get SQL'''
        message_GPT = three_shots_SQL_generation_prompt_from_Bard + \
                      response_Bard + \
                      SQL_generation_prompt + \
                        "\ndatabase:" + db_id + \
                        "\ndatabase chema:" + schema + \
                        "Just give me the plain SQL without any placeholders." + \
                        "\nquestion:" + question+ \
                        "\nYour SQL:"
        print('message to GPT3.5:', message_GPT)
        SQL, limit_marker = GPT4_generation(message_GPT)
        print('SQL:', SQL)
        SQL = SQL.replace('\n', ' ')
        print('\nGPT generated SQL:', SQL + '\n')
        SQLs_temp_pred.append(SQL)
        SQLs_temp_gold.append(SQL_gold+'\t'+db_id)

        with open ('./Eval/predicted_sql.txt','a') as f:
                f.write(SQL+'\n')
        with open ('./Eval/gold_sql.txt','a') as f:
                f.write(SQL_gold+'\t'+db_id+'\n')

# CUDA_VISIBLE_DEVICES=7 python read_cosql.py