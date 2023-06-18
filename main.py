from langchain.llms import OpenAI
from typing import List, Dict, Callable, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from generate_task import BabyAGI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from dialogue_template import DialogueAgent, DialogueSimulator

# hyperparameters
key = "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"
difficulty = 0.5
company = "Jane Street"
word_limit = 50
comp_descrip = '''
Jane Street is a global proprietary trading firm that specializes in quantitative trading and operates 
in various financial markets. It was founded in 2000 and is headquartered in New York City. Jane Street 
is known for its expertise in algorithmic trading and high-frequency trading strategies.
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

job_title = 'Software Engineering Intern'
job_descrip = '''We are looking for Software Engineering Interns to join our Winter 2024 Co-op program. Our goal is to give you a real sense of what it’s like to work at Jane Street full time. Over the course of your internship, you will explore ways to approach and solve exciting problems within your field of interest through fun and challenging classes, interactive sessions, and group discussions—and then you will have the chance to put those lessons to practical use.

As an intern, you are paired with full-time employees who act as mentors, collaborating with you on real-world projects we actually need done. When you’re not working on your project, you will have plenty of time to use our office amenities, physical and virtual educational resources, attend social events, and engage with the parts of our work that excite you the most.

If you’ve never thought about a career in finance, you’re in good company. Many of us were in the same position before working here. If you have a curious mind, a collaborative spirit, and a passion for solving interesting problems, we have a feeling you’ll fit right in.

As a Software Engineering intern, you’ll learn how we use OCaml (our primary development language) in our day to day work, and gain exposure to the libraries and tools that are foundational to our internal systems.

During the program you’ll work on two projects, mentored closely by the full-time employees who designed them. Some projects consider big-picture questions that we’re still trying to figure out, while others involve building something new. Your mentors will work in two distinct areas, so you’ll gain a better understanding of the wide range of problems we solve every day, from high performance trading systems to programming language design and everything in between. 

If you’d like to learn more, you can read about our interview and team placement processes and get a sense of what our most recent intern projects looked like.

About You
We don’t expect you to have a background in finance, OCaml, functional programming, or any other specific field—we’re looking for smart people who enjoy solving interesting problems. We’re more interested in how you think and learn than what you currently know. You should be:

Must be enrolled in a Co-op program at your university
A top-notch programmer with a passion for technology
Collaborative and courteous with strong interpersonal and communication skills
Eager to ask questions, admit mistakes, and learn new things
Must be able to work in-person in our NYC office from January 2024 through April 2024
Fluent in English'''

rng = 1.0
focus = "technical"
role = "Senior Trader"
name = "Nick"
my_role = "college student"





llm = OpenAI(openai_api_key=key, temperature=0.4)
chat_llm = ChatOpenAI(openai_api_key=key, temperature=rng)
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

embeddings_model = OpenAIEmbeddings(openai_api_key=key)
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

OBJECTIVE = f'''
Conduct a {focus} interview as a {role} at {company}. The person you are interviewing
is a {my_role}, applying for the {job_title} position, which has the description: "{job_descrip}". 
The interviewee's resume is: "{self_descrip}".
'''

llm = OpenAI(openai_api_key=key, temperature=0)

# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 1
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})

task_list = baby_agi.task_list
task_list.appendleft({'task_id':'1', 'task_name':"Introduce yourself and the company, and ask the interviewee to introduce himself."})
print(baby_agi.task_list)

int_message = SystemMessage(content=f'''
Your name is {name}. You are a {role} at the company {company}, conducting a {focus} interview for
the {job_title} job, which has the job description: "{job_descrip}". Broadly follow the outline: "{task_list}". Ask
questions specific to the interviewee's resume: "{self_descrip}".
Speak only in first person from the perspective of {name}. Do not change roles! Do not speak from the 
perspective of anyone else. After you speak, wait for a response from the interviewee. Never forget to
keep your reponse to {word_limit} words. Remember you are {name} at company {company}. Do not add anything else.
''')

# sys_template = SystemMessagePromptTemplate.from_template(template=int_message)
# sys_message = sys_template.format_messages(a)

interviewer = DialogueAgent(
                name=name, 
                system_message=int_message,
                model=chat_llm)
interviewer.reset()
# interviewer.receive(name, int_message)
idx = 0

# for i in range(30):
#     if idx % 2 == 0:
#         message = interviewer.send()
#         interviewer.receive(name, message)
#         print(message)
#     else:
#         message = input()
#         interviewer.receive("me", message)
#     idx += 1





from tkinter import *

# GUI
root = Tk()
root.title("Interview Simulator")

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


# Send function

lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10, width=20, height=1).grid(
	row=0)

txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=120)
txt.grid(row=1, column=0, columnspan=2)

def send():
    send = "You -> " + e.get()
    txt.insert(END, "\n" + send)
    user = e.get().lower()
    interviewer.receive("me", user)
    
    e.delete(0, END)
    if send.split(' ')[-1] == 'END':
        #terminate and get diagnostic
        god_instruction = SystemMessage(content=f'''
            You work for {company} and are deciding whether or not to hire a candidate for the {job_title} job, which has the job description: "{job_descrip}".
            You are given a script of an interview between the interviewee and the candidate and must only use the candidates responces as reason to hire or reject them.
            You must decide whether or not to hire the candidate. You must explain and jusify
            your choice. Do not add anything else.
            ''')

        master = DialogueAgent(name="master", system_message=god_instruction, model=chat_llm)
        master.reset()
        master.receive('master', 'here is a script:\n'+str(interviewer.message_history) + '\nLog ends. What is your decision?')
        txt.insert(END, "\n BOT -> " + master.send())
        return
    message = interviewer.send()
    interviewer.receive(name, message)
    txt.insert(END, "\n" + "BOT -> " + message)

scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)

message = interviewer.send()
interviewer.receive(name, message)
txt.insert(END, "BOT -> " + message)

send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			command=send).grid(row=2, column=1)

root.mainloop()

