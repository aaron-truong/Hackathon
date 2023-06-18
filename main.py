from langchain.llms import OpenAI
from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from dialogue_template import DialogueAgent, DialogueSimulator

# hyperparameters
difficulty = 0.5
company = "Google"
comp_descrip = '''
Google is a multinational technology company that specializes in internet-related services and products. 
It is one of the largest tech companies in the world and is known for its search engine, cloud computing, 
software, and hardware. Google also offers a wide range of services such as Gmail, Google Maps, YouTube, 
Google Play, and Google Drive. It is a leader in innovation and is constantly pushing the boundaries of 
technology to create better products and services for its users.
'''

self_descrip = '''
Objective:
Highly motivated and skilled Computer Science Masters student seeking a challenging position to leverage my technical expertise and academic accomplishments at UC Berkeley. Passionate about applying cutting-edge technologies and solving complex problems in the field of computer science.

Education:
Master of Science in Computer Science, University of California, Berkeley

Bachelor of Science in Computer Science, University of California, Berkeley

Relevant Coursework:

Advanced Algorithms
Artificial Intelligence
Machine Learning
Data Structures
Database Systems
Computer Networks
Operating Systems
Software Engineering

Skills:

Programming Languages: Python, Java, C++, JavaScript
Frameworks and Libraries: TensorFlow, PyTorch, Django, React
Database Management: SQL, MongoDB, PostgreSQL
Data Analysis and Visualization: Pandas, NumPy, Matplotlib, Tableau
Web Development: HTML, CSS, JavaScript, Node.js
Version Control: Git, GitHub
Operating Systems: Windows, Linux, macOS
Problem Solving and Algorithm Design
Strong Mathematical Foundation
'''

job_title = 'Software Engineering Intern, MS, Summer'
job_descrip = '''Join us for a 12-14 week paid internship that offers personal and professional development, 
an executive speaker series, and community-building. The Software Engineering Internship program will give you 
an opportunity to work on complex computer science...
Software Engineer, Software, Reliability Engineer, Summer, Intern, Engineer'''

rng = 0.3
focus = "technical"
role = "senior engineer"
name = "Nick"
my_role = "college student"





llm = OpenAI(openai_api_key="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", temperature=0.4)
chat_llm = ChatOpenAI(openai_api_key="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", temperature=rng)
# company = input("Please input the company name: ")

# comp_descrip = llm.predict(f"Please give me a 50 word description of the company {company}.").strip()

# print(comp_descrip)

# ans = input("\nIs this an accurate description? ")
# while True:
#     detect = llm.predict(f"Write '0' if the following answer resembles 'yes', '1' if it resembles 'no', or '2' if it resembles neither: {ans}. Write nothing else.").strip()
#     if detect == '0':
#         break
#     elif detect == '1':
#         comp_descrip = input(f"Please give your own description of {company}: ")
#         break
#     else:
#         ans = input("\nIs this an accurate description? ")

# job_descrip = input("Please input the job title: ")

# = input("Please input your resumé/self-description: ")

int_message = SystemMessage(content=f'''
You are a {role} at {company} named {name}. {job_descrip} You will conduct a {focus} interview for me, 
a {my_role} applying for the {job_title} job, which has description: "{job_descrip}". My resume is: "{self_descrip}". Speak in 
first person from the perspective of {name}. Do not change roles! Do not speak from the perspective
of anyone else. Remember you are {name}, a {role} at {company}. Do not add anything else. End the
interview when you feel that it is done.
''')

interviewer = DialogueAgent(name=name, system_message=int_message, model=chat_llm)
interviewer.reset()
# interviewer.receive(name, int_message)
idx = 0

for i in range(30):
    if idx % 2 == 0:
        message = interviewer.send()
        interviewer.receive(name, message)
        # print(message)
    else:
        message = input()
        interviewer.receive("me", message)
    idx += 1






