import pandas as pd
import time
import openai
import os
import sys
import tiktoken
import signal


#----------------------------------------------------prompts-----------------------------------------------
schema_linking_prompt = '''Table advisor, columns = [*,s_ID,i_ID]
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
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Find the buildings which have rooms with capacity more than 50."
A: Let’s think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.capacity]
"rooms with capacity" so we need column = [classroom.building]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "How many heads of the departments are older than 56 ?"
A: Let’s think step by step. In the question "How many heads of the departments are older than 56 ?", we are asked:
"How many heads of the departments" so we need column = [head.*]
"older" so we need column = [head.age]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [56]. So the Schema_links are:
Schema_links: [head.*,head.age,56]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
A: Let’s think step by step. In the question "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?", we are asked:
"distinct creation years of the departments" so we need column = [department.Creation]
"departments managed by" so we need column = [management.department_ID]
"born in" so we need column = [head.born_state]
Based on the columns and tables, we need these Foreign_keys = [department.Department_ID = management.department_ID,management.head_ID = head.head_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = ['Alabama']. So the Schema_links are:
Schema_links: [department.Creation,department.Department_ID = management.department_ID,head.head_ID = management.head_ID,head.born_state,'Alabama']

Table Addresses, columns = [*,address_id,line_1,line_2,city,zip_postcode,state_province_county,country]
Table Candidate_Assessments, columns = [*,candidate_id,qualification,assessment_date,asessment_outcome_code]
Table Candidates, columns = [*,candidate_id,candidate_details]
Table Courses, columns = [*,course_id,course_name,course_description,other_details]
Table People, columns = [*,person_id,first_name,middle_name,last_name,cell_mobile_number,email_address,login_name,password]
Table People_Addresses, columns = [*,person_address_id,person_id,address_id,date_from,date_to]
Table Student_Course_Attendance, columns = [*,student_id,course_id,date_of_attendance]
Table Student_Course_Registrations, columns = [*,student_id,course_id,registration_date]
Table Students, columns = [*,student_id,student_details]
Foreign_keys = [Students.student_id = People.person_id,People_Addresses.address_id = Addresses.address_id,People_Addresses.person_id = People.person_id,Student_Course_Registrations.course_id = Courses.course_id,Student_Course_Registrations.student_id = Students.student_id,Student_Course_Attendance.student_id = Student_Course_Registrations.student_id,Student_Course_Attendance.course_id = Student_Course_Registrations.course_id,Candidates.candidate_id = People.person_id,Candidate_Assessments.candidate_id = Candidates.candidate_id]
Q: "List the id of students who never attends courses?"
A: Let’s think step by step. In the question "List the id of students who never attends courses?", we are asked:
"id of students" so we need column = [Students.student_id]
"never attends courses" so we need column = [Student_Course_Attendance.student_id]
Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]

Table Country, columns = [*,id,name]
Table League, columns = [*,id,country_id,name]
Table Player, columns = [*,id,player_api_id,player_name,player_fifa_api_id,birthday,height,weight]
Table Player_Attributes, columns = [*,id,player_fifa_api_id,player_api_id,date,overall_rating,potential,preferred_foot,attacking_work_rate,defensive_work_rate,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,free_kick_accuracy,long_passing,ball_control,acceleration,sprint_speed,agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots,aggression,interceptions,positioning,vision,penalties,marking,standing_tackle,sliding_tackle,gk_diving,gk_handling,gk_kicking,gk_positioning,gk_reflexes]
Table Team, columns = [*,id,team_api_id,team_fifa_api_id,team_long_name,team_short_name]
Table Team_Attributes, columns = [*,id,team_fifa_api_id,team_api_id,date,buildUpPlaySpeed,buildUpPlaySpeedClass,buildUpPlayDribbling,buildUpPlayDribblingClass,buildUpPlayPassing,buildUpPlayPassingClass,buildUpPlayPositioningClass,chanceCreationPassing,chanceCreationPassingClass,chanceCreationCrossing,chanceCreationCrossingClass,chanceCreationShooting,chanceCreationShootingClass,chanceCreationPositioningClass,defencePressure,defencePressureClass,defenceAggression,defenceAggressionClass,defenceTeamWidth,defenceTeamWidthClass,defenceDefenderLineClass]
Table sqlite_sequence, columns = [*,name,seq]
Foreign_keys = [Player_Attributes.player_api_id = Player.player_api_id,Player_Attributes.player_fifa_api_id = Player.player_fifa_api_id,League.country_id = Country.id,Team_Attributes.team_api_id = Team.team_api_id,Team_Attributes.team_fifa_api_id = Team.team_fifa_api_id]
Q: "List the names of all left-footed players who have overall rating between 85 and 90."
A: Let’s think step by step. In the question "List the names of all left-footed players who have overall rating between 85 and 90.", we are asked:
"names of all left-footed players" so we need column = [Player.player_name,Player_Attributes.preferred_foot]
"players who have overall rating" so we need column = [Player_Attributes.overall_rating]
Based on the columns and tables, we need these Foreign_keys = [Player_Attributes.player_api_id = Player.player_api_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [left,85,90]. So the Schema_links are:
Schema_links: [Player.player_name,Player_Attributes.preferred_foot,Player_Attributes.overall_rating,Player_Attributes.player_api_id = Player.player_api_id,left,85,90]

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
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Give the title of the course offered in Chandler during the Fall of 2010."
A: Let’s think step by step. In the question "Give the title of the course offered in Chandler during the Fall of 2010.", we are asked:
"title of the course" so we need column = [course.title]
"course offered in Chandler" so we need column = [SECTION.building]
"during the Fall" so we need column = [SECTION.semester]
"of 2010" so we need column = [SECTION.year]
Based on the columns and tables, we need these Foreign_keys = [course.course_id = SECTION.course_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [Chandler,Fall,2010]. So the Schema_links are:
Schema_links: [course.title,course.course_id = SECTION.course_id,SECTION.building,SECTION.year,SECTION.semester,Chandler,Fall,2010]

Table city, columns = [*,City_ID,Official_Name,Status,Area_km_2,Population,Census_Ranking]
Table competition_record, columns = [*,Competition_ID,Farm_ID,Rank]
Table farm, columns = [*,Farm_ID,Year,Total_Horses,Working_Horses,Total_Cattle,Oxen,Bulls,Cows,Pigs,Sheep_and_Goats]
Table farm_competition, columns = [*,Competition_ID,Year,Theme,Host_city_ID,Hosts]
Foreign_keys = [farm_competition.Host_city_ID = city.City_ID,competition_record.Farm_ID = farm.Farm_ID,competition_record.Competition_ID = farm_competition.Competition_ID]
Q: "Show the status of the city that has hosted the greatest number of competitions."
A: Let’s think step by step. In the question "Show the status of the city that has hosted the greatest number of competitions.", we are asked:
"the status of the city" so we need column = [city.Status]
"greatest number of competitions" so we need column = [farm_competition.*]
Based on the columns and tables, we need these Foreign_keys = [farm_competition.Host_city_ID = city.City_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [city.Status,farm_competition.Host_city_ID = city.City_ID,farm_competition.*]

'''
###########
schema_linking_prompt_new = '''
# You are an expert in SQL and please use the necessary tables, columns, 
and foreign keys from the database schema to do the text2sql task. 
Return format like:
Schema_links: [city.Status,farm_competition.Host_city_ID = city.City_ID,farm_competition.*]
Schema_links: [course.title,course.course_id = SECTION.course_id,SECTION.building,SECTION.year,SECTION.semester,Chandler,Fall,2010]
#
'''

# 删了三个example
classification_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
A: Let’s think step by step. The SQL query for the question "Find the buildings which have rooms with capacity more than 50." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let’s think step by step. The SQL query for the question "What are the names of all instructors who advise students in the math depart sorted by total credits of the student." needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
A: Let’s think step by step. The SQL query for the question "Find the room number of the rooms which can sit 50 to 100 students and their buildings." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let’s think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the title of course that is provided by both Statistics and Psychology departments."
schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let’s think step by step. The SQL query for the question "Find the title of course that is provided by both Statistics and Psychology departments." needs these tables = [course], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the titles of courses that is provided by Psychology departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]
A: Let’s think step by step. The SQL query for the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." needs these tables = [teaches], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the id of instructors who taught a class in Spring 2010"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the name of the department that offers the highest total credits?"
schema_links: [course.dept_name,course.credits]
A: Let’s think step by step. The SQL query for the question "Find the name of the department that offers the highest total credits?." needs these tables = [course], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

'''
# 删了三个example

classification_prompt_new = " "
easy_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
Schema_links: [classroom.building,classroom.capacity,50]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity  >  50

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
Schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

Q: "Give the name of the student in the History department with the most credits."
Schema_links: [student.name,student.dept_name,student.tot_cred,History]
SQL: SELECT name FROM student WHERE dept_name  =  'History' ORDER BY tot_cred DESC LIMIT 1

Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance]
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Q: "Find the department name of the instructor whose name contains 'Soisalon'."
Schema_links: [instructor.dept_name,instructor.name,Soisalon]
SQL: SELECT dept_name FROM instructor WHERE name LIKE '%Soisalon%'

Q: "What is the name of the department with the most credits?"
Schema_links: [course.dept_name,course.credits]
SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY sum(credits) DESC LIMIT 1

Q: "How many instructors teach a course in the Spring of 2010?"
Schema_links: [teaches.ID,teaches.semester,teaches.YEAR,Spring,2010]
SQL: SELECT COUNT (DISTINCT ID) FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010

Q: "Find the name of the students and their department names sorted by their total credits in ascending order."
Schema_links: [student.name,student.dept_name,student.tot_cred]
SQL: SELECT name ,  dept_name FROM student ORDER BY tot_cred

Q: "Find the year which offers the largest number of courses."
Schema_links: [SECTION.YEAR,SECTION.*]
SQL: SELECT YEAR FROM SECTION GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1

Q: "What are the names and average salaries for departments with average salary higher than 42000?"
Schema_links: [instructor.dept_name,instructor.salary,42000]
SQL: SELECT dept_name ,  AVG (salary) FROM instructor GROUP BY dept_name HAVING AVG (salary)  >  42000

Q: "How many rooms in each building have a capacity of over 50?"
Schema_links: [classroom.*,classroom.building,classroom.capacity,50]
SQL: SELECT count(*) ,  building FROM classroom WHERE capacity  >  50 GROUP BY building

'''
easy_prompt_new = " "
# 删了三个example
medium_prompt = '''Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select sum(department.budget) from department  where  department.dept_name = \"Marketing\"  or  department.dept_name = \"Finance\"
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

Q: "What is the name and building of the departments whose budget is more than the average budget?"
Schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation:  select department.dept_name , department.building from department  where  @.@ > avg ( department.budget ) 
SQL: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

Q: "Find the total number of students and total number of instructors for each department."
Schema_links: [department.dept_name = student.dept_name,student.id,department.dept_name = instructor.dept_name,instructor.id]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [department,student,instructor]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: "select count( distinct student.ID) , count( distinct instructor.ID) , department.dept_name from department  group by instructor.dept_name
SQL: SELECT count(DISTINCT T2.id) ,  count(DISTINCT T3.id) ,  T3.dept_name FROM department AS T1 JOIN student AS T2 ON T1.dept_name  =  T2.dept_name JOIN instructor AS T3 ON T1.dept_name  =  T3.dept_name GROUP BY T3.dept_name

'''
medium_prompt_new = " "
# 删了三个example
hard_prompt = '''Q: "Find the title of courses that have two prerequisites?"
Schema_links: [course.title,course.course_id = prereq.course_id]
A: Let's think step by step. "Find the title of courses that have two prerequisites?" can be solved by knowing the answer to the following sub-question "What are the titles for courses with two prerequisites?".
The SQL query for the sub-question "What are the titles for courses with two prerequisites?" is SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2
So, the answer to the question "Find the title of courses that have two prerequisites?" is =
Intermediate_representation: select course.title from course  where  count ( prereq.* )  = 2  group by prereq.course_id
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.dept_name,department.building,department.budget]
A: Let's think step by step. "Find the name and building of the department with the highest budget." can be solved by knowing the answer to the following sub-question "What is the department name and corresponding building for the department with the greatest budget?".
The SQL query for the sub-question "What is the department name and corresponding building for the department with the greatest budget?" is SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

Q: "Find the title, credit, and department name of courses that have more than one prerequisites?"
Schema_links: [course.title,course.credits,course.dept_name,course.course_id = prereq.course_id]
A: Let's think step by step. "Find the title, credit, and department name of courses that have more than one prerequisites?" can be solved by knowing the answer to the following sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?".
The SQL query for the sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?" is SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select course.title , course.credits , course.dept_name from course  where  count ( prereq.* )  > 1  group by prereq.course_id 
SQL: SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1

Q: "Give the name and building of the departments with greater than average budget."
Schema_links: [department.dept_name,department.building,department.budget]
A: Let's think step by step. "Give the name and building of the departments with greater than average budget." can be solved by knowing the answer to the following sub-question "What is the average budget of departments?".
The SQL query for the sub-question "What is the average budget of departments?" is SELECT avg(budget) FROM department
So, the answer to the question "Give the name and building of the departments with greater than average budget." is =
Intermediate_representation: select department.dept_name , department.building from department  where  @.@ > avg ( department.budget )
SQL: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
Schema_links: [teaches.id,teaches.semester,teaches.YEAR,Fall,2009,Spring,2010]
A: Let's think step by step. "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." can be solved by knowing the answer to the following sub-question "Find the id of instructors who taught a class in Spring 2010".
The SQL query for the sub-question "Find the id of instructors who taught a class in Spring 2010" is SELECT id FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010
So, the answer to the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." is = 
Intermediate_representation: select teaches.ID from teaches  where  teaches.semester = \"Fall\"  and  teaches.year = 2009  and  teaches.semester != \"Spring\"  and  teaches.year = 2010 
SQL: SELECT id FROM teaches WHERE semester  =  'Fall' AND YEAR  =  2009 EXCEPT SELECT id FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010

Q: "Find the name of the courses that do not have any prerequisite?"
Schema_links: [course.title,course.course_id]
A: Let's think step by step. "Find the name of the courses that do not have any prerequisite?" can be solved by knowing the answer to the following sub-question "What are the courses that have any prerequisite?".
The SQL query for the sub-question "What are the courses that have any prerequisite?" is SELECT course_id FROM prereq
So, the answer to the question "Find the name of the courses that do not have any prerequisite?" is =
Intermediate_representation: select course.title from course  where  @.@ not in prereq.course_id 
SQL: SELECT title FROM course WHERE course_id NOT IN (SELECT course_id FROM prereq)

Q: "Find the salaries of all distinct instructors that are less than the largest salary."
Schema_links: [instructor.salary]
A: Let's think step by step. "Find the salaries of all distinct instructors that are less than the largest salary." can be solved by knowing the answer to the following sub-question "What is the largest salary of instructors".
The SQL query for the sub-question "What is the largest salary of instructors" is SELECT max(salary) FROM instructor
So, the answer to the question "Find the salaries of all distinct instructors that are less than the largest salary." is =
Intermediate_representation: select  distinct instructor.salary from instructor  where  @.@ < max ( instructor.salary )
SQL: SELECT DISTINCT salary FROM instructor WHERE salary  <  (SELECT max(salary) FROM instructor)

Q: "Find the names of students who have taken any course in the fall semester of year 2003."
Schema_links: [student.id,student.name,takes.id,takes.semester,fall,2003]
A: Let's think step by step. "Find the names of students who have taken any course in the fall semester of year 2003." can be solved by knowing the answer to the following sub-question "Find the students who have taken any course in the fall semester of year 2003.".
The SQL query for the sub-question "Find the students who have taken any course in the fall semester of year 2003." is SELECT id FROM takes WHERE semester  =  'Fall' AND YEAR  =  2003
So, the answer to the question "Find the names of students who have taken any course in the fall semester of year 2003." is =
Intermediate_representation: select student.name from student  where  takes.semester = \"Fall\"  and  takes.year = 2003
SQL: SELECT name FROM student WHERE id IN (SELECT id FROM takes WHERE semester  =  'Fall' AND YEAR  =  2003)

'''
hard_prompt_new = " "

# 删了三个exampple
#----------------------------------------------------------------------------------------------------------

API_KEY = "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" # 自己的
# API_KEY = "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"  # 买的
# API_KEY = "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" # gpt4 孙哥
os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

#changed
task = 'Spider' # 1 for CoSQL, 2 for Spider
if task == 'CoSQL':
    path_to_CoSQL = "/Users/yan/Desktop/text2sql/cosql_dataset"
    DATASET_SCHEMA = path_to_CoSQL+"/tables.json"
    DATASET = path_to_CoSQL+"/sql_state_tracking/cosql_dev.json"
    OUTPUT_FILE_1 = "./predicted_sql.txt"
    OUTPUT_FILE_2 = "./gold_sql.txt"
else:
    path_to_Spider = "/Users/yan/Desktop/text2sql/spider"
    DATASET_SCHEMA = path_to_Spider + "/tables.json"
    DATASET = path_to_Spider + "/dev.json"
    OUTPUT_FILE_1 = "./Spider/predicted_sql.txt"
    OUTPUT_FILE_2 = "./Spider/gold_sql.txt"

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


def load_data(DATASET):
    return pd.read_json(DATASET)

def hard_prompt_maker(test_sample_text,database,schema_links,sub_questions):
  instruction = "# Use the intermediate representation and the schema links to generate the SQL queries for each of the questions.\n"
  fields = find_fields_MYSQL_like("college_2")
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like("college_2") + '\n'
  fields += find_fields_MYSQL_like(database)
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  stepping = f'''\nA: Let's think step by step. "{test_sample_text}" can be solved by knowing the answer to the following sub-question "{sub_questions}".'''
  fields += "\n"
  prompt = instruction +fields + hard_prompt_new + 'Q: "' + test_sample_text + '"' + '\nschema_links: ' + schema_links + stepping +'\nThe SQL query for the sub-question"'
  return prompt
def medium_prompt_maker(test_sample_text,database,schema_links):
  instruction = "# Use the the schema links and Intermediate_representation to generate the SQL queries for each of the questions.\n"
  fields = find_fields_MYSQL_like("college_2")
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like("college_2") + '\n'
  fields += find_fields_MYSQL_like(database)
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  fields += "\n"
  prompt = instruction +fields + medium_prompt_new + 'Q: "' + test_sample_text + '\nSchema_links: ' + schema_links + '\nA: Let’s think step by step.'
  return prompt
def easy_prompt_maker(test_sample_text,database,schema_links):
  instruction = "# Use the the schema links to generate the SQL queries for each of the questions.\n"
  fields = find_fields_MYSQL_like("college_2")
  fields += find_fields_MYSQL_like(database)
  fields += "\n"
  prompt = 'Q: "' + test_sample_text+instruction +fields + easy_prompt_new + '\nSchema_links: ' + schema_links + '\nSQL:'
  return prompt
def classification_prompt_maker(test_sample_text,database,schema_links):
  instruction = "# For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.\n"
  instruction += "\nif need nested queries: predict NESTED\n"
  instruction += "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
  instruction += "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
  fields = find_fields_MYSQL_like("college_2")
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like("college_2") + '\n'
  fields += find_fields_MYSQL_like(database)
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  fields += "\n"
  prompt = 'Q: "' + test_sample_text +instruction + fields + classification_prompt_new + '\nschema_links: ' + schema_links + '\nA: Let’s think step by step.'
  return prompt
def schema_linking_prompt_maker(test_sample_text,database):
  instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
  fields = find_fields_MYSQL_like(database)
  foreign_keys = "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  prompt = instruction + schema_linking_prompt_new + fields +foreign_keys+ 'Q: "' + test_sample_text + """"\nA: Let’s think step by step."""
  return prompt
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
def debuger(test_sample_text,database,sql):
  instruction = """#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.

"""
  fields = find_fields_MYSQL_like(database)
  fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
  fields += "Primary_keys = " + find_primary_keys_MYSQL_like(database)
  prompt = '#### Question: ' + test_sample_text+instruction + fields + '\n#### SQLite SQL QUERY\n' + sql +'\n#### SQLite FIXED SQL QUERY\nSELECT'
  return prompt

#changed
def GPT4_generation(prompt):
    token_count = num_tokens_from_string(prompt,model_name)
    if token_count > MAX_TOKENS:
        print(f"Token count ({token_count}) exceeds the maximum limit ({MAX_TOKENS}). Skipping request.")
        prompt = prompt[:MAX_TOKENS]
    print("##"*10)
    print(prompt)
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
    stop = ["Q:"]
    )
    return response['choices'][0]['message']['content']

def GPT4_debug(prompt):
    token_count = num_tokens_from_string(prompt, model_name)
    if token_count > MAX_TOKENS:
        print(f"Token count ({token_count}) exceeds the maximum limit ({MAX_TOKENS}). Skipping request.")
        prompt= prompt[:MAX_TOKENS]
    print("##"*10)
    print(prompt)
    response = openai.ChatCompletion.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    n = 1,
    stream = False,
    temperature=0.0,
    max_tokens=350,
    top_p = 1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop = ["#", ";","\n\n"]
    )
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
###########################################################################################
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    print(val_df)
    print(f"Number of data samples {val_df.shape[0]}")
    # 293 for Corsql
    CODEX = []
    if task == 'CoSQL':
        for index, row in val_df.iterrows():
            #if index < 405: continue #for testing
            print(f"index is {index}")
            print(row['final'])
            print(row['interaction'][0]['utterance'])
            print(row['interaction'][0]['query'])
            question = row['interaction'][0]['utterance']
            sql = row['interaction'][0]['query']
            db_id = row['database_id']
            schema_links = None
            while schema_links is None:
                try:
                    schema_links = GPT4_generation(
                        schema_linking_prompt_maker(question, db_id))
                    print("schema_links generation done 1/6")
                except:
                    time.sleep(3)
                    pass
            try:
                schema_links = schema_links.split("Schema_links: ")[1]
                print("schema link split done 2/6")
            except:
                print("Slicing error for the schema_linking module")
                schema_links = schema_links
            #print(schema_links)
            classification = None
            while classification is None:
                try:
                    classification = GPT4_generation(
                        classification_prompt_maker(question, db_id, schema_links[1:]))
                    print("classification generation done 3/6")
                except:
                    time.sleep(3)
                    pass
            try:
                predicted_class = classification.split("Label: ")[1]
                print("classification split done 4/6")
            except:
                print("Slicing error for the classification module")
                predicted_class = '"NESTED"'
            #print(classification)
            if '"EASY"' in predicted_class:
                print("EASY")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(easy_prompt_maker(question, db_id, schema_links))
                        print("EASY generation done 5/6")
                    except:
                        time.sleep(3)
                        pass
            elif '"NON-NESTED"' in predicted_class:
                print("NON-NESTED")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(medium_prompt_maker(question, db_id, schema_links))
                        print("Medium generation done 5/6")
                    except:
                        time.sleep(3)
                        pass
                try:
                    SQL = SQL.split("SQL: ")[1]
                except:
                    print("SQL slicing error")
                    SQL = "SELECT"
            else:
                # changed
                # sub_questions = classification.split('questions = ["')[1].split('"]')[0]
                # print("NESTED")
                try:
                    sub_questions = classification.split('questions = ["')[1].split('"]')[0]
                except IndexError:
                    sub_questions = question
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(
                            hard_prompt_maker(question, db_id, schema_links, sub_questions))
                        print("hard generation done 5/6")
                    except:
                        time.sleep(3)
                        pass
                try:
                    SQL = SQL.split("SQL: ")[1]
                except:
                    print("SQL slicing error")
                    SQL = "SELECT"
            print(SQL)
            debugged_SQL = None
            while debugged_SQL is None:
                try:
                    debugged_SQL = GPT4_debug(debuger(question, db_id, SQL)).replace("\n", " ")
                    print("debugger generation done 6/6")
                except:
                    time.sleep(3)
                    pass
            SQL = "SELECT " + debugged_SQL
            print(SQL)
            CODEX.append([question, SQL, sql, db_id])
            #break
        df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
        results = df['PREDICTED SQL'].tolist()
        with open(OUTPUT_FILE_1, 'w') as f:
            for line in results:
                f.write(f"{line}\n")
    else:
        for index, row in val_df[4:7].iterrows():
            # if index < 405: continue #for testing
            print(f"index is {index}")
            question = row['question']
            db_id = row['db_id']
            schema_links = None
            while schema_links is None:
                try:
                    schema_links = GPT4_generation(
                        schema_linking_prompt_maker(question, db_id))
                except:
                    time.sleep(3)
                    pass
            try:
                schema_links = schema_links.split("Schema_links: ")[1]
            except:
                print("Slicing error for the schema_linking module")
                schema_links = "[]"
            # print(schema_links)
            classification = None
            while classification is None:
                try:
                    classification = GPT4_generation(
                        classification_prompt_maker(question, db_id, schema_links[1:]))
                except:
                    time.sleep(3)
                    pass
            try:
                predicted_class = classification.split("Label: ")[1]
            except:
                print("Slicing error for the classification module")
                predicted_class = '"NESTED"'
            # print(classification)
            if '"EASY"' in predicted_class:
                print("EASY")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(easy_prompt_maker(question, db_id, schema_links))
                    except:
                        time.sleep(3)
                        pass
            elif '"NON-NESTED"' in predicted_class:
                print("NON-NESTED")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(medium_prompt_maker(question, db_id, schema_links))
                    except:
                        time.sleep(3)
                        pass
                try:
                    SQL = SQL.split("SQL: ")[1]
                except:
                    print("SQL slicing error")
                    SQL = "SELECT"
            else:
                try:
                    sub_questions = classification.split('questions = ["')[1].split('"]')[0]
                except IndexError:
                    sub_questions = question
                print("NESTED")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(
                            hard_prompt_maker(question, db_id, schema_links, sub_questions))
                    except:
                        time.sleep(3)
                        pass
                try:
                    SQL = SQL.split("SQL: ")[1]
                except:
                    print("SQL slicing error")
                    SQL = "SELECT"
            print(SQL)
            debugged_SQL = None
            while debugged_SQL is None:
                try:
                    debugged_SQL = GPT4_debug(debuger(question, db_id, SQL)).replace("\n", " ")
                except:
                    time.sleep(3)
                    pass
            SQL = "SELECT " + debugged_SQL
            print(SQL)
            CODEX.append([question, SQL, row['query'], db_id])
            # break
        df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
        results = df['PREDICTED SQL'].tolist()
        with open(OUTPUT_FILE_1, 'w') as f:
            for line in results:
                f.write(f"{line}\n")
    task = 'Spider'
    if task == 'CoSQL':
        dataset = pd.read_json(DATASET)
        gold = []
        for index, row in dataset[:100].iterrows():
            dict_round = {}
            dict_round['query'] = row['interaction'][0]['query']
            dict_round['db_id'] = row['database_id']
            gold.append(dict_round)
    else:
        dataset = pd.read_json(DATASET)
        gold = []
        for index, row in dataset[:100].iterrows():
            dict_round = {}
            dict_round['query'] = row['query']
            dict_round['db_id'] = row['db_id']
            gold.append(dict_round)

    with open(OUTPUT_FILE_2, 'w') as f:
        for item in gold:
            f.write(f"{item['query']}\t{item['db_id']}\n")