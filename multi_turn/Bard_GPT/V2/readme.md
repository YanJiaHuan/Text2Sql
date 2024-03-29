'''
This is a GPT-Bard contrasive prompting method, in this demo, we still focus on Spider, 
but the difference will be: The few-shots are generated by Bard.
Version 2: 
    1. The few-shot is generated by Bard.
    2. Bard will generate SQL/question samples based on the real input SQL/question (with CoT).
    3. GPT know the few-shots are generated by Bard.
    4. Bard know it is Spider task.
'''
---
### Info
Bard 会根据一样的schema，给三个fewshot，并简单用CoT的策略解释一下，然后GPT会根据这三个fewshot和schema生成SQL。

---
### num/total score of test-suite
738/1034 0.611
1034/1034 0.596

---
### Prompt
####################  0. Prompt   ####################
SQL_generation_prompt = '''
You are an expert in SQL. I will give you a natural language question and a database schema, 
please help me generate the corresponding SQL query with no further explaination.
'''
few_shot_generation_prompt_Bard = '''
You are an expert in SQL. I will give you a database schema in Spider dataset, and you need to generate three
SQL queries with natural language questions based on the schema and explian the chain of thoughts logic. Also, I will give you some exapmles
of how this task is done.
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
Let's think step by step: 'salary of professors' -> 'salary of instructor' -> Go to Table instructor, find the column salary, and calculate the average value.
SELECT avg ( salary )  FROM instructor

example 2:
Question: Find the average salary of the professors of each department?
Let's think step by step: 'salary of professors of each department' -> 'salary of instructor of each department' -> Go to Table instructor, find the column salary, and calculate the average value. 'each department'->group by the department.
SELECT avg ( salary ) , dept_name FROM instructor GROUP BY dept_name

example 3:
Question: Which department has the highest average salary of professors?
Let's think step by step: 'highest average salary of professors' -> 'highest average salary of instructor' -> Go to Table instructor, find the column salary, and calculate the average value. 'highest' -> order by the average value. 'department' -> group by the department.
SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )  DESC LIMIT 1
'''

checker_prompt = '''
Please help me generate the corresponding SQL query with no further explaination.
'''